//! Bounded nonlinear Phase-4 model trained through `cogno-scirust`.
//!
//! The historical [`crate::neural::NeuralModel`] remains the canonical v1
//! artifact model. This module adds a parallel one-hidden-layer MLP so COGNO
//! can exercise a genuinely nonlinear trainable path before the persisted
//! artifact schema is migrated. The authority boundary is unchanged: training
//! updates weights, frozen inference is read-only, and every proposal still
//! crosses deterministic `cogno-core` validation and policy gates.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::{
    NeuralModelError, NeuralTrainingReport, MAX_NEURAL_EPOCHS, MAX_NEURAL_FEATURES,
    MAX_NEURAL_LABELS, MAX_NEURAL_PAYLOAD_BYTES, MAX_NEURAL_RANK_CANDIDATES, MIN_NEURAL_FEATURES,
};
use crate::readonly::ReadOnlyCapability;
use crate::training::{Corpus, CorpusSplit, Label};
use cogno_core::{EvidenceId, ProposalAction, ReasonCode, RuleCategory, RuleScope};
use cogno_scirust::{Mlp, MlpAdamW, MlpConfig, SciRustError, MAX_MLP_DIM, MAX_MLP_PARAMETERS};
use std::sync::Arc;

/// Minimum hidden width accepted by the nonlinear model.
pub const MIN_MLP_HIDDEN_FEATURES: usize = 4;
/// Maximum hidden width accepted by the model-facing configuration.
pub const MAX_MLP_HIDDEN_FEATURES: usize = 1_024;

/// Configuration for bounded nonlinear supervised training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpNeuralConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub max_payload_bytes: usize,
    pub initialization_seed: u64,
}

impl Default for MlpNeuralConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            hidden_dim: 64,
            epochs: 24,
            learning_rate: 0.01,
            max_payload_bytes: MAX_NEURAL_PAYLOAD_BYTES,
            initialization_seed: 0,
        }
    }
}

/// Frozen nonlinear classifier backed by the bounded SciRust MLP.
#[derive(Clone, Debug, PartialEq)]
pub struct MlpNeuralModel {
    network: Mlp,
    num_labels: u16,
    max_payload_bytes: usize,
}

impl MlpNeuralModel {
    pub(crate) fn from_verified_parts(
        input_dim: usize,
        hidden_dim: usize,
        num_labels: u16,
        max_payload_bytes: usize,
        initialization_seed: u64,
        input_hidden: Vec<f32>,
        hidden_bias: Vec<f32>,
        hidden_output: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> Result<Self, NeuralModelError> {
        validate_model_shape(input_dim, hidden_dim, num_labels, max_payload_bytes)?;
        let network = Mlp::from_parts(
            MlpConfig {
                input_dim,
                hidden_dim,
                output_dim: usize::from(num_labels),
                seed: initialization_seed,
            },
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        )?;
        Ok(Self {
            network,
            num_labels,
            max_payload_bytes,
        })
    }

    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.network.config().input_dim
    }

    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.network.config().hidden_dim
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
    pub const fn initialization_seed(&self) -> u64 {
        self.network.config().seed
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.network.parameter_count()
    }

    #[must_use]
    pub fn input_hidden(&self) -> &[f32] {
        self.network.input_hidden()
    }

    #[must_use]
    pub fn hidden_bias(&self) -> &[f32] {
        self.network.hidden_bias()
    }

    #[must_use]
    pub fn hidden_output(&self) -> &[f32] {
        self.network.hidden_output()
    }

    #[must_use]
    pub fn output_bias(&self) -> &[f32] {
        self.network.output_bias()
    }

    /// Return the raw class logits for a bounded payload.
    pub fn logits(&self, payload: &[u8]) -> Result<Vec<f32>, NeuralModelError> {
        let features = encode_payload(self.input_dim(), self.max_payload_bytes, payload)?;
        let logits = self.network.forward(&features)?;
        if logits.len() != usize::from(self.num_labels) {
            return Err(NeuralModelError::InvalidConfig);
        }
        Ok(logits)
    }

    /// Return normalized class probabilities using stable softmax.
    pub fn probabilities(&self, payload: &[u8]) -> Result<Vec<f32>, NeuralModelError> {
        stable_softmax(&self.logits(payload)?)
    }

    /// Probability assigned to one label.
    pub fn score(&self, label: u16, payload: &[u8]) -> Result<f32, NeuralModelError> {
        if label >= self.num_labels {
            return Err(NeuralModelError::LabelOutOfRange {
                label,
                maximum: self.num_labels,
            });
        }
        self.probabilities(payload)?
            .get(usize::from(label))
            .copied()
            .ok_or(NeuralModelError::ArithmeticOverflow)
    }

    /// Finite log-probability for held-out/meta review evidence.
    pub fn log_probability(&self, label: u16, payload: &[u8]) -> Result<f32, NeuralModelError> {
        let probability = self.score(label, payload)?;
        if probability <= 0.0 || !probability.is_finite() {
            return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
        }
        let value = probability.ln();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(NeuralModelError::SciRust(SciRustError::NonFinite))
        }
    }

    /// Deterministic argmax over nonlinear logits. Ties select the lowest id.
    pub fn classify(&self, payload: &[u8]) -> Result<Label, NeuralModelError> {
        let logits = self.logits(payload)?;
        let mut best = Label(0);
        let mut best_logit = f32::NEG_INFINITY;
        for (index, logit) in logits.into_iter().enumerate() {
            if logit > best_logit {
                best =
                    Label(u16::try_from(index).map_err(|_| NeuralModelError::ArithmeticOverflow)?);
                best_logit = logit;
            }
        }
        if best_logit.is_finite() {
            Ok(best)
        } else {
            Err(NeuralModelError::SciRust(SciRustError::NonFinite))
        }
    }

    /// Top-two softmax margin expressed in basis points.
    pub fn confidence_bps(&self, payload: &[u8]) -> Result<u16, NeuralModelError> {
        let probabilities = self.probabilities(payload)?;
        let mut best = f32::NEG_INFINITY;
        let mut runner_up = f32::NEG_INFINITY;
        for probability in probabilities {
            if probability > best {
                runner_up = best;
                best = probability;
            } else if probability > runner_up {
                runner_up = probability;
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

/// Trainer for the bounded nonlinear model.
#[derive(Clone, Copy, Debug)]
pub struct MlpNeuralTrainer {
    pub config: MlpNeuralConfig,
}

impl MlpNeuralTrainer {
    pub fn try_new(config: MlpNeuralConfig) -> Result<Self, NeuralModelError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        corpus: &Corpus,
        split: &CorpusSplit,
    ) -> Result<(MlpNeuralModel, NeuralTrainingReport), NeuralModelError> {
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
        validate_model_shape(
            self.config.input_dim,
            self.config.hidden_dim,
            num_labels,
            self.config.max_payload_bytes,
        )?;

        let mut network = Mlp::try_new(MlpConfig {
            input_dim: self.config.input_dim,
            hidden_dim: self.config.hidden_dim,
            output_dim: usize::from(num_labels),
            seed: self.config.initialization_seed,
        })?;
        let parameters = network.parameter_count();
        let mut optimizer = MlpAdamW::try_new(self.config.learning_rate, &network)?;
        let mut target = vec![0.0f32; usize::from(num_labels)];

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
                target.fill(0.0);
                let target_index = usize::from(example.label.0);
                let slot =
                    target
                        .get_mut(target_index)
                        .ok_or(NeuralModelError::LabelOutOfRange {
                            label: example.label.0,
                            maximum: num_labels,
                        })?;
                *slot = 1.0;
                network.train_step_squared(&mut optimizer, &features, &target)?;
            }
        }

        let model = MlpNeuralModel {
            network,
            num_labels,
            max_payload_bytes: self.config.max_payload_bytes,
        };
        let (correct, total) = evaluate(&model, corpus, split)?;
        let scaled = correct
            .checked_mul(10_000)
            .ok_or(NeuralModelError::ArithmeticOverflow)?
            .checked_div(total)
            .ok_or(NeuralModelError::ArithmeticOverflow)?;
        let final_accuracy_bps =
            u16::try_from(scaled).map_err(|_| NeuralModelError::ArithmeticOverflow)?;
        Ok((
            model,
            NeuralTrainingReport {
                examples: total,
                epochs: self.config.epochs,
                parameters,
                final_correct: correct,
                final_accuracy_bps,
            },
        ))
    }
}

/// Frozen read-only facade over the nonlinear SciRust model.
#[derive(Clone, Debug)]
pub struct SciRustMlpReadOnlyModel {
    pub model: Arc<MlpNeuralModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SCIRUST_MLP_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[
    ReadOnlyCapability::Classify,
    ReadOnlyCapability::Extract,
    ReadOnlyCapability::Rank,
    ReadOnlyCapability::Explain,
];

impl SciRustMlpReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: MlpNeuralModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SCIRUST_MLP_READ_ONLY_CAPS,
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

impl ModelBackend for SciRustMlpReadOnlyModel {
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

fn validate_config(config: MlpNeuralConfig) -> Result<(), NeuralModelError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&config.input_dim)
        || !(MIN_MLP_HIDDEN_FEATURES..=MAX_MLP_HIDDEN_FEATURES).contains(&config.hidden_dim)
        || config.input_dim > MAX_MLP_DIM
        || config.hidden_dim > MAX_MLP_DIM
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

fn validate_model_shape(
    input_dim: usize,
    hidden_dim: usize,
    num_labels: u16,
    max_payload_bytes: usize,
) -> Result<(), NeuralModelError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&input_dim)
        || !(MIN_MLP_HIDDEN_FEATURES..=MAX_MLP_HIDDEN_FEATURES).contains(&hidden_dim)
        || input_dim > MAX_MLP_DIM
        || hidden_dim > MAX_MLP_DIM
        || num_labels == 0
        || num_labels > MAX_NEURAL_LABELS
        || usize::from(num_labels) > MAX_MLP_DIM
        || max_payload_bytes == 0
        || max_payload_bytes > MAX_NEURAL_PAYLOAD_BYTES
    {
        return Err(NeuralModelError::InvalidConfig);
    }
    let parameters = MlpConfig {
        input_dim,
        hidden_dim,
        output_dim: usize::from(num_labels),
        seed: 0,
    }
    .parameter_count()?;
    if parameters > MAX_MLP_PARAMETERS {
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

fn stable_softmax(logits: &[f32]) -> Result<Vec<f32>, NeuralModelError> {
    if logits.is_empty() {
        return Err(NeuralModelError::InvalidConfig);
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
    }
    let mut probabilities = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let value = (logit - maximum).exp();
        if !value.is_finite() {
            return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
        }
        probabilities.push(value);
        sum += value;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
    }
    for probability in &mut probabilities {
        *probability /= sum;
        if !probability.is_finite() {
            return Err(NeuralModelError::SciRust(SciRustError::NonFinite));
        }
    }
    Ok(probabilities)
}

fn evaluate(
    model: &MlpNeuralModel,
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
        let mut corpus = Corpus::with_seed(17);
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

    fn config() -> MlpNeuralConfig {
        MlpNeuralConfig {
            input_dim: 64,
            hidden_dim: 16,
            epochs: 48,
            learning_rate: 0.02,
            max_payload_bytes: 128,
            initialization_seed: 23,
        }
    }

    #[test]
    fn nonlinear_training_is_repeatable_and_learns_training_set() {
        let (corpus, split) = corpus();
        let trainer = MlpNeuralTrainer::try_new(config()).expect("valid MLP config");
        let (left, left_report) = trainer.train(&corpus, &split).expect("train left");
        let (right, right_report) = trainer.train(&corpus, &split).expect("train right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 6);
        assert_eq!(left_report.final_correct, 6);
        assert_eq!(left_report.final_accuracy_bps, 10_000);
        assert_eq!(left.classify(b"aaaa-alpha").expect("classify"), Label(0));
        assert_eq!(left.classify(b"zzzz-beta").expect("classify"), Label(1));
        assert!(left
            .log_probability(0, b"aaaa-alpha")
            .expect("log p")
            .is_finite());
    }

    #[test]
    fn nonlinear_parameter_count_includes_both_layers_and_biases() {
        let (corpus, split) = corpus();
        let trainer = MlpNeuralTrainer::try_new(config()).expect("valid MLP config");
        let (model, report) = trainer.train(&corpus, &split).expect("train");
        let expected = 64 * 16 + 16 + 16 * 2 + 2;
        assert_eq!(model.parameter_count(), expected);
        assert_eq!(report.parameters, expected);
    }

    #[test]
    fn read_only_mlp_backend_preserves_authority_boundary() {
        let (corpus, split) = corpus();
        let trainer = MlpNeuralTrainer::try_new(config()).expect("valid MLP config");
        let (model, _) = trainer.train(&corpus, &split).expect("train");
        let mut backend = SciRustMlpReadOnlyModel::from_trained(model);
        let info = backend.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
        let proposal = backend
            .extract(b"aaaa-alpha", EvidenceId::from_u64(11))
            .expect("extract");
        assert!(cogno_core::validate_proposal(&proposal.as_view()).is_ok());
        assert_eq!(
            backend.next_proposal(),
            Err(BackendError::ReadOnlyViolation)
        );
    }

    #[test]
    fn hostile_mlp_shapes_and_payloads_fail_closed() {
        assert!(matches!(
            MlpNeuralTrainer::try_new(MlpNeuralConfig {
                hidden_dim: MAX_MLP_HIDDEN_FEATURES + 1,
                ..config()
            }),
            Err(NeuralModelError::InvalidConfig)
        ));

        let (corpus, split) = corpus();
        let trainer = MlpNeuralTrainer::try_new(MlpNeuralConfig {
            max_payload_bytes: 8,
            ..config()
        })
        .expect("valid bounded config");
        assert!(matches!(
            trainer.train(&corpus, &split),
            Err(NeuralModelError::PayloadTooLarge { .. })
        ));
    }
}
