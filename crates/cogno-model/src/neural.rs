//! Bounded differentiable classifier backed by `cogno-scirust`.
//!
//! This module is the first production-facing bridge between `cogno-model`
//! and the differentiable SciRust substrate. It deliberately keeps the model
//! small and read-only at inference time: training may update weights, but a
//! frozen [`NeuralModel`] can only classify/rank/propose. Every proposal still
//! crosses the deterministic `cogno-core` validation and policy boundary.
//!
//! Training is bounded by fixed feature/label/parameter/payload caps. For each
//! class, a one-vs-rest sigmoid head is differentiated through the
//! `cogno_scirust::Tape` and updated by AdamW. No model output gains authority.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::readonly::ReadOnlyCapability;
use crate::training::{Corpus, CorpusSplit, Label};
use cogno_core::{EvidenceId, ProposalAction, ReasonCode, RuleCategory, RuleScope};
use cogno_scirust::{AdamW, Optimizer, SciRustError, Shape, Tape, Tensor};
use std::ops::Range;
use std::sync::Arc;

/// Minimum feature width accepted by the differentiable classifier.
pub const MIN_NEURAL_FEATURES: usize = 16;
/// Maximum feature width accepted by the differentiable classifier.
pub const MAX_NEURAL_FEATURES: usize = 4_096;
/// Maximum number of labels trained by one model.
pub const MAX_NEURAL_LABELS: u16 = 64;
/// Maximum number of trainable scalar weights.
pub const MAX_NEURAL_PARAMETERS: usize = 262_144;
/// Maximum input payload size consumed by this backend.
pub const MAX_NEURAL_PAYLOAD_BYTES: usize = 4_096;
/// Maximum number of candidates accepted by one ranking call.
pub const MAX_NEURAL_RANK_CANDIDATES: usize = 256;
/// Maximum epoch budget accepted by the trainer.
pub const MAX_NEURAL_EPOCHS: u32 = 1_024;

/// Configuration for bounded differentiable supervised training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuralConfig {
    pub input_dim: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub max_payload_bytes: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            epochs: 24,
            learning_rate: 0.01,
            max_payload_bytes: MAX_NEURAL_PAYLOAD_BYTES,
        }
    }
}

/// Fail-closed errors raised by differentiable model training/inference.
#[derive(Clone, Debug, PartialEq)]
pub enum NeuralModelError {
    InvalidConfig,
    EmptyTrainingSplit,
    InvalidSplitIndex { index: usize, corpus_len: usize },
    LabelOutOfRange { label: u16, maximum: u16 },
    PayloadTooLarge { actual: usize, maximum: usize },
    TooManyRankCandidates { actual: usize, maximum: usize },
    ArithmeticOverflow,
    SciRust(SciRustError),
}

impl From<SciRustError> for NeuralModelError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Integer-only summary of one deterministic training run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuralTrainingReport {
    pub examples: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_correct: usize,
    pub final_accuracy_bps: u16,
}

/// Frozen differentiable classifier weights.
#[derive(Clone, Debug)]
pub struct NeuralModel {
    input_dim: usize,
    num_labels: u16,
    max_payload_bytes: usize,
    weights: Vec<f32>,
}

impl NeuralModel {
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[must_use]
    pub const fn num_labels(&self) -> u16 {
        self.num_labels
    }

    #[must_use]
    pub const fn max_payload_bytes(&self) -> usize {
        self.max_payload_bytes
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }

    /// Exposes frozen weights for deterministic audit/replay comparisons.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Probability-like sigmoid score for one label.
    pub fn score(&self, label: u16, payload: &[u8]) -> Result<f32, NeuralModelError> {
        if label >= self.num_labels {
            return Err(NeuralModelError::LabelOutOfRange {
                label,
                maximum: self.num_labels,
            });
        }
        let features = encode_payload(self.input_dim, self.max_payload_bytes, payload)?;
        let range = weight_range(label, self.input_dim, self.weights.len())?;
        let class_weights = self
            .weights
            .get(range)
            .ok_or(NeuralModelError::ArithmeticOverflow)?;
        let mut logit = 0.0f32;
        for (&feature, &weight) in features.iter().zip(class_weights.iter()) {
            logit += feature * weight;
        }
        if !logit.is_finite() {
            return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
        }
        stable_sigmoid(logit)
    }

    /// Deterministic argmax. Equal scores select the lowest label id.
    pub fn classify(&self, payload: &[u8]) -> Result<Label, NeuralModelError> {
        if self.num_labels == 0 {
            return Err(NeuralModelError::EmptyTrainingSplit);
        }
        let mut best = Label(0);
        let mut best_score = f32::NEG_INFINITY;
        for label in 0..self.num_labels {
            let score = self.score(label, payload)?;
            if score > best_score {
                best = Label(label);
                best_score = score;
            }
        }
        Ok(best)
    }

    /// Confidence is the top-two probability margin, expressed in basis points.
    pub fn confidence_bps(&self, payload: &[u8]) -> Result<u16, NeuralModelError> {
        let mut best = f32::NEG_INFINITY;
        let mut runner_up = f32::NEG_INFINITY;
        for label in 0..self.num_labels {
            let score = self.score(label, payload)?;
            if score > best {
                runner_up = best;
                best = score;
            } else if score > runner_up {
                runner_up = score;
            }
        }
        let margin = if self.num_labels <= 1 {
            best.max(0.0)
        } else {
            (best - runner_up).max(0.0)
        };
        if !margin.is_finite() {
            return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
        }
        Ok((margin.clamp(0.0, 1.0) * 10_000.0).round() as u16)
    }
}

/// Trainer that differentiates each one-vs-rest head through `cogno-scirust`.
#[derive(Clone, Copy, Debug)]
pub struct NeuralTrainer {
    pub config: NeuralConfig,
}

impl NeuralTrainer {
    pub fn try_new(config: NeuralConfig) -> Result<Self, NeuralModelError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        corpus: &Corpus,
        split: &CorpusSplit,
    ) -> Result<(NeuralModel, NeuralTrainingReport), NeuralModelError> {
        if split.indices.is_empty() {
            return Err(NeuralModelError::EmptyTrainingSplit);
        }

        let mut max_label = 0u16;
        for &index in &split.indices {
            let example =
                corpus
                    .examples
                    .get(index)
                    .ok_or(NeuralModelError::InvalidSplitIndex {
                        index,
                        corpus_len: corpus.examples.len(),
                    })?;
            max_label = max_label.max(example.label.0);
            if example.payload.len() > self.config.max_payload_bytes {
                return Err(NeuralModelError::PayloadTooLarge {
                    actual: example.payload.len(),
                    maximum: self.config.max_payload_bytes,
                });
            }
        }

        let num_labels = max_label
            .checked_add(1)
            .ok_or(NeuralModelError::ArithmeticOverflow)?;
        if num_labels > MAX_NEURAL_LABELS {
            return Err(NeuralModelError::LabelOutOfRange {
                label: max_label,
                maximum: MAX_NEURAL_LABELS,
            });
        }
        let parameters = usize::from(num_labels)
            .checked_mul(self.config.input_dim)
            .ok_or(NeuralModelError::ArithmeticOverflow)?;
        if parameters > MAX_NEURAL_PARAMETERS {
            return Err(NeuralModelError::InvalidConfig);
        }

        let mut weights = vec![0.0f32; parameters];
        let mut optimizers = Vec::with_capacity(usize::from(num_labels));
        for _ in 0..num_labels {
            optimizers.push(AdamW::try_new(
                self.config.learning_rate,
                self.config.input_dim,
            )?);
        }

        for _ in 0..self.config.epochs {
            for &index in &split.indices {
                let example =
                    corpus
                        .examples
                        .get(index)
                        .ok_or(NeuralModelError::InvalidSplitIndex {
                            index,
                            corpus_len: corpus.examples.len(),
                        })?;
                let features = encode_payload(
                    self.config.input_dim,
                    self.config.max_payload_bytes,
                    &example.payload,
                )?;
                for label in 0..num_labels {
                    train_head(
                        &features,
                        label,
                        example.label,
                        self.config.input_dim,
                        &mut weights,
                        &mut optimizers[usize::from(label)],
                    )?;
                }
            }
        }

        let model = NeuralModel {
            input_dim: self.config.input_dim,
            num_labels,
            max_payload_bytes: self.config.max_payload_bytes,
            weights,
        };
        let (correct, total) = evaluate(&model, corpus, split)?;
        let accuracy_bps = if total == 0 {
            0
        } else {
            let scaled = correct
                .checked_mul(10_000)
                .ok_or(NeuralModelError::ArithmeticOverflow)?
                / total;
            u16::try_from(scaled).map_err(|_| NeuralModelError::ArithmeticOverflow)?
        };
        Ok((
            model,
            NeuralTrainingReport {
                examples: total,
                epochs: self.config.epochs,
                parameters,
                final_correct: correct,
                final_accuracy_bps: accuracy_bps,
            },
        ))
    }
}

/// Frozen read-only façade over a differentiably-trained model.
#[derive(Clone, Debug)]
pub struct SciRustReadOnlyModel {
    pub model: Arc<NeuralModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SCIRUST_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[
    ReadOnlyCapability::Classify,
    ReadOnlyCapability::Extract,
    ReadOnlyCapability::Rank,
    ReadOnlyCapability::Explain,
];

impl SciRustReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: NeuralModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SCIRUST_READ_ONLY_CAPS,
        }
    }

    pub fn classify(&self, payload: &[u8]) -> Result<Label, BackendError> {
        self.model.classify(payload).map_err(map_backend_error)
    }

    pub fn extract(
        &self,
        payload: &[u8],
        evidence_id: EvidenceId,
    ) -> Result<OwnedProposal, BackendError> {
        let _ = self.classify(payload)?;
        let confidence_bps = self
            .model
            .confidence_bps(payload)
            .map_err(map_backend_error)?;
        Ok(OwnedProposal {
            action: ProposalAction::ExtractPreference,
            category: RuleCategory::Behavior,
            scope: RuleScope::Session,
            confidence_bps,
            evidence_ids: vec![evidence_id],
            payload: payload.to_vec(),
        })
    }

    pub fn rank(&self, candidates: &[&[u8]]) -> Result<Vec<usize>, BackendError> {
        if candidates.len() > MAX_NEURAL_RANK_CANDIDATES {
            return Err(BackendError::InputTooLarge);
        }
        let mut order = Vec::with_capacity(candidates.len());
        for (index, payload) in candidates.iter().enumerate() {
            let label = self.classify(payload)?;
            let score = self
                .model
                .score(label.0, payload)
                .map_err(map_backend_error)?;
            order.push((index, score));
        }
        order.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(order.into_iter().map(|(index, _)| index).collect())
    }

    #[must_use]
    pub fn explain(&self, label: Label) -> ReasonCode {
        ReasonCode(label.0.min(u16::MAX - 1))
    }
}

impl ModelBackend for SciRustReadOnlyModel {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            phase: 4,
            read_only: true,
            tools_enabled: false,
            differentiable: true,
        }
    }

    fn next_proposal(&mut self) -> Result<OwnedProposal, BackendError> {
        Err(BackendError::ReadOnlyViolation)
    }
}

fn validate_config(config: NeuralConfig) -> Result<(), NeuralModelError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&config.input_dim)
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || config.max_payload_bytes == 0
        || config.max_payload_bytes > MAX_NEURAL_PAYLOAD_BYTES
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(NeuralModelError::InvalidConfig);
    }
    Ok(())
}

fn encode_payload(
    input_dim: usize,
    max_payload_bytes: usize,
    payload: &[u8],
) -> Result<Vec<f32>, NeuralModelError> {
    if payload.len() > max_payload_bytes {
        return Err(NeuralModelError::PayloadTooLarge {
            actual: payload.len(),
            maximum: max_payload_bytes,
        });
    }
    if input_dim < MIN_NEURAL_FEATURES {
        return Err(NeuralModelError::InvalidConfig);
    }
    let mut features = vec![0.0f32; input_dim];
    features[0] = 1.0;
    let bucket_span = input_dim - 1;
    let mut hash = 0xcbf29ce484222325u64;
    for (position, &byte) in payload.iter().enumerate() {
        hash ^= u64::from(byte).wrapping_add((position as u64).rotate_left(17));
        hash = hash.wrapping_mul(0x100000001b3);
        let bucket = 1 + (hash as usize % bucket_span);
        features[bucket] += 1.0;
    }
    if !payload.is_empty() {
        let norm = payload.len() as f32;
        for value in &mut features[1..] {
            *value /= norm;
        }
    }
    Ok(features)
}

fn train_head(
    features: &[f32],
    label: u16,
    target_label: Label,
    input_dim: usize,
    weights: &mut [f32],
    optimizer: &mut AdamW,
) -> Result<(), NeuralModelError> {
    let range = weight_range(label, input_dim, weights.len())?;
    let class_weights = weights
        .get(range.clone())
        .ok_or(NeuralModelError::ArithmeticOverflow)?
        .to_vec();

    let mut tape = Tape::new(8, input_dim);
    let feature_tensor =
        Tensor::try_new(Shape::try_new(&[input_dim])?, features.to_vec(), input_dim)?;
    let weight_tensor = Tensor::try_new(Shape::try_new(&[input_dim])?, class_weights, input_dim)?;
    let feature_var = tape.variable(feature_tensor)?;
    let weight_var = tape.variable(weight_tensor)?;
    let product = tape.mul(feature_var, weight_var)?;
    let logit = tape.sum(product)?;
    let probability = tape.sigmoid(logit)?;
    let target = if label == target_label.0 { 1.0 } else { 0.0 };
    let target_var = tape.variable(Tensor::try_scalar(target)?)?;
    let difference = tape.sub(probability, target_var)?;
    let loss = tape.mul(difference, difference)?;
    tape.backward(loss)?;
    let gradient = tape.grad_of(weight_var).to_vec();
    let class_weights = weights
        .get_mut(range)
        .ok_or(NeuralModelError::ArithmeticOverflow)?;
    optimizer.step(class_weights, &gradient)?;
    Ok(())
}

fn evaluate(
    model: &NeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<(usize, usize), NeuralModelError> {
    let mut correct = 0usize;
    for &index in &split.indices {
        let example = corpus
            .examples
            .get(index)
            .ok_or(NeuralModelError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            })?;
        if model.classify(&example.payload)? == example.label {
            correct = correct.saturating_add(1);
        }
    }
    Ok((correct, split.indices.len()))
}

fn weight_range(
    label: u16,
    input_dim: usize,
    weights_len: usize,
) -> Result<Range<usize>, NeuralModelError> {
    let start = usize::from(label)
        .checked_mul(input_dim)
        .ok_or(NeuralModelError::ArithmeticOverflow)?;
    let end = start
        .checked_add(input_dim)
        .ok_or(NeuralModelError::ArithmeticOverflow)?;
    if end > weights_len {
        return Err(NeuralModelError::ArithmeticOverflow);
    }
    Ok(start..end)
}

fn stable_sigmoid(value: f32) -> Result<f32, NeuralModelError> {
    let output = if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    };
    if output.is_finite() {
        Ok(output)
    } else {
        Err(NeuralModelError::SciRust(SciRustError::NonFinite))
    }
}

fn map_backend_error(error: NeuralModelError) -> BackendError {
    match error {
        NeuralModelError::PayloadTooLarge { .. }
        | NeuralModelError::TooManyRankCandidates { .. } => BackendError::InputTooLarge,
        NeuralModelError::InvalidConfig
        | NeuralModelError::EmptyTrainingSplit
        | NeuralModelError::InvalidSplitIndex { .. }
        | NeuralModelError::LabelOutOfRange { .. }
        | NeuralModelError::ArithmeticOverflow
        | NeuralModelError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{LabeledExample, SplitKind};
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn corpus() -> (Corpus, CorpusSplit) {
        let mut corpus = Corpus::with_seed(7);
        for payload in [b"aaaa-alpha".as_slice(), b"aaaa-style", b"aaaa-format"] {
            assert!(corpus.add(LabeledExample::new(
                Label(0),
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        for payload in [b"zzzz-beta".as_slice(), b"zzzz-behavior", b"zzzz-policy"] {
            assert!(corpus.add(LabeledExample::new(
                Label(1),
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        let split = CorpusSplit {
            kind: SplitKind::Train,
            indices: (0..corpus.examples.len()).collect(),
        };
        (corpus, split)
    }

    #[test]
    fn differentiable_training_is_repeatable_and_learns_training_set() {
        let (corpus, split) = corpus();
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 48,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("valid neural config");
        let (left, left_report) = trainer.train(&corpus, &split).expect("train left");
        let (right, right_report) = trainer.train(&corpus, &split).expect("train right");
        assert_eq!(left_report, right_report);
        assert_eq!(left.weights(), right.weights());
        assert_eq!(left_report.examples, 6);
        assert_eq!(left_report.final_correct, 6);
        assert_eq!(left_report.final_accuracy_bps, 10_000);
        assert_eq!(left.classify(b"aaaa-alpha").expect("classify"), Label(0));
        assert_eq!(left.classify(b"zzzz-beta").expect("classify"), Label(1));
    }

    #[test]
    fn read_only_scirust_backend_proposes_but_has_no_authority() {
        let (corpus, split) = corpus();
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 48,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("valid neural config");
        let (model, _) = trainer.train(&corpus, &split).expect("train");
        let mut backend = SciRustReadOnlyModel::from_trained(model);
        let info = backend.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
        let proposal = backend
            .extract(b"aaaa-alpha", EvidenceId::from_u64(9))
            .expect("extract");
        assert!(cogno_core::validate_proposal(&proposal.as_view()).is_ok());
        assert_eq!(
            backend.next_proposal(),
            Err(BackendError::ReadOnlyViolation)
        );
    }

    #[test]
    fn oversized_payload_and_invalid_split_fail_closed() {
        let (corpus, split) = corpus();
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 1,
            learning_rate: 0.02,
            max_payload_bytes: 8,
        })
        .expect("valid neural config");
        assert!(matches!(
            trainer.train(&corpus, &split),
            Err(NeuralModelError::PayloadTooLarge { .. })
        ));

        let bad_split = CorpusSplit {
            kind: SplitKind::Train,
            indices: vec![usize::MAX],
        };
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 1,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("valid neural config");
        assert!(matches!(
            trainer.train(&corpus, &bad_split),
            Err(NeuralModelError::InvalidSplitIndex { .. })
        ));
    }
}
