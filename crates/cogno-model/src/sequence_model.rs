//! Bounded sequence-model bridge from `cogno-model` to `cogno-scirust`.
//!
//! Raw hostile payload bytes are first framed by [`crate::ByteTokenizer`], then
//! consumed by the differentiable [`cogno_scirust::SequenceClassifier`]. The
//! bridge owns training orchestration and the read-only proposal surface; it
//! does not grant the model authority over policy, tools, persistence or the
//! deterministic runtime decision boundary.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::{NeuralTrainingReport, MAX_NEURAL_EPOCHS, MAX_NEURAL_LABELS, MAX_NEURAL_RANK_CANDIDATES};
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use crate::training::{Corpus, CorpusSplit, Label};
use cogno_core::{EvidenceId, ProposalAction, ReasonCode, RuleCategory, RuleScope};
use cogno_scirust::{
    SciRustError, SequenceClassifier, SequenceClassifierAdamW, SequenceClassifierConfig,
    SequenceEncoderConfig, MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_EMBEDDING_DIM,
    MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_TOKENS, MAX_SEQUENCE_VOCAB,
};
use std::sync::Arc;

/// Minimum embedding width accepted by the model-facing sequence config.
pub const MIN_SEQUENCE_EMBEDDING_DIM: usize = 2;
/// Minimum hidden width accepted by the model-facing sequence config.
pub const MIN_SEQUENCE_HIDDEN_DIM: usize = 2;

/// Bounded configuration for supervised sequence training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceNeuralConfig {
    pub max_tokens: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub encoder_seed: u64,
    pub head_seed: u64,
}

impl Default for SequenceNeuralConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            embedding_dim: 32,
            hidden_dim: 32,
            epochs: 24,
            learning_rate: 0.01,
            encoder_seed: 0,
            head_seed: 1,
        }
    }
}

/// Fail-closed errors for the tokenizer-to-sequence-classifier bridge.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceNeuralError {
    InvalidConfig,
    EmptyTrainingSplit,
    InsufficientLabels,
    InvalidSplitIndex { index: usize, corpus_len: usize },
    LabelOutOfRange { label: u16, maximum: u16 },
    TooManyRankCandidates { actual: usize, maximum: usize },
    ArithmeticOverflow,
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequenceNeuralError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceNeuralError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Frozen sequence classifier plus the exact tokenizer bound used at training.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceNeuralModel {
    classifier: SequenceClassifier,
    tokenizer: ByteTokenizer,
    num_labels: u16,
}

impl SequenceNeuralModel {
    /// Reconstruct an already-validated classifier behind the exact tokenizer
    /// bound. Future hostile artifact loaders can use this after manifest/hash
    /// verification without exposing mutable weights.
    pub(crate) fn from_verified_parts(
        classifier: SequenceClassifier,
        max_tokens: usize,
        num_labels: u16,
    ) -> Result<Self, SequenceNeuralError> {
        let tokenizer = ByteTokenizer::try_new(max_tokens)?;
        if num_labels < 2
            || num_labels > MAX_NEURAL_LABELS
            || usize::from(num_labels) != classifier.config().num_classes
            || classifier.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
            || classifier.config().encoder.max_tokens != max_tokens
        {
            return Err(SequenceNeuralError::InvalidConfig);
        }
        Ok(Self {
            classifier,
            tokenizer,
            num_labels,
        })
    }

    #[must_use]
    pub const fn num_labels(&self) -> u16 {
        self.num_labels
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.classifier.config().encoder.max_tokens
    }

    #[must_use]
    pub const fn embedding_dim(&self) -> usize {
        self.classifier.config().encoder.embedding_dim
    }

    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.classifier.config().encoder.hidden_dim
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.classifier.parameter_count()
    }

    #[must_use]
    pub const fn classifier(&self) -> &SequenceClassifier {
        &self.classifier
    }

    /// Raw logits for one bounded arbitrary-byte payload.
    pub fn logits(&self, payload: &[u8]) -> Result<Vec<f32>, SequenceNeuralError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.classifier.logits(&tokens)?)
    }

    /// Stable normalized class probabilities.
    pub fn probabilities(&self, payload: &[u8]) -> Result<Vec<f32>, SequenceNeuralError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.classifier.probabilities(&tokens)?)
    }

    /// Probability assigned to one label.
    pub fn score(&self, label: u16, payload: &[u8]) -> Result<f32, SequenceNeuralError> {
        self.validate_label(label)?;
        self.probabilities(payload)?
            .get(usize::from(label))
            .copied()
            .ok_or(SequenceNeuralError::ArithmeticOverflow)
    }

    /// Finite log-probability for held-out/meta review.
    pub fn log_probability(&self, label: u16, payload: &[u8]) -> Result<f32, SequenceNeuralError> {
        let probability = self.score(label, payload)?;
        if probability <= 0.0 || !probability.is_finite() {
            return Err(SequenceNeuralError::SciRust(SciRustError::NonFinite));
        }
        let value = probability.ln();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(SequenceNeuralError::SciRust(SciRustError::NonFinite))
        }
    }

    /// Deterministic argmax. Exact ties are resolved by the SciRust classifier
    /// toward the lowest class id.
    pub fn classify(&self, payload: &[u8]) -> Result<Label, SequenceNeuralError> {
        let tokens = self.tokenizer.encode(payload)?;
        let label = self.classifier.classify(&tokens)?;
        let label = u16::try_from(label).map_err(|_| SequenceNeuralError::ArithmeticOverflow)?;
        self.validate_label(label)?;
        Ok(Label(label))
    }

    /// Top-two probability margin represented in basis points.
    pub fn confidence_bps(&self, payload: &[u8]) -> Result<u16, SequenceNeuralError> {
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
        let margin = (best - runner_up).max(0.0);
        if !margin.is_finite() {
            return Err(SequenceNeuralError::SciRust(SciRustError::NonFinite));
        }
        Ok((margin.clamp(0.0, 1.0) * 10_000.0).round() as u16)
    }

    fn validate_label(&self, label: u16) -> Result<(), SequenceNeuralError> {
        if label >= self.num_labels {
            return Err(SequenceNeuralError::LabelOutOfRange {
                label,
                maximum: self.num_labels,
            });
        }
        Ok(())
    }
}

/// Trainer for the bounded tokenized sequence model.
#[derive(Clone, Copy, Debug)]
pub struct SequenceNeuralTrainer {
    pub config: SequenceNeuralConfig,
}

impl SequenceNeuralTrainer {
    pub fn try_new(config: SequenceNeuralConfig) -> Result<Self, SequenceNeuralError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        corpus: &Corpus,
        split: &CorpusSplit,
    ) -> Result<(SequenceNeuralModel, NeuralTrainingReport), SequenceNeuralError> {
        if split.indices.is_empty() {
            return Err(SequenceNeuralError::EmptyTrainingSplit);
        }
        let tokenizer = ByteTokenizer::try_new(self.config.max_tokens)?;
        let mut max_label = 0u16;
        for &index in &split.indices {
            let example = corpus.examples.get(index).ok_or(
                SequenceNeuralError::InvalidSplitIndex {
                    index,
                    corpus_len: corpus.examples.len(),
                },
            )?;
            max_label = max_label.max(example.label.0);
            tokenizer.encode(&example.payload)?;
        }
        let num_labels = max_label
            .checked_add(1)
            .ok_or(SequenceNeuralError::ArithmeticOverflow)?;
        if num_labels < 2 {
            return Err(SequenceNeuralError::InsufficientLabels);
        }
        if num_labels > MAX_NEURAL_LABELS || usize::from(num_labels) > MAX_SEQUENCE_CLASSES {
            return Err(SequenceNeuralError::LabelOutOfRange {
                label: max_label,
                maximum: MAX_NEURAL_LABELS,
            });
        }

        let classifier_config = SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                max_tokens: self.config.max_tokens,
                embedding_dim: self.config.embedding_dim,
                hidden_dim: self.config.hidden_dim,
                seed: self.config.encoder_seed,
            },
            num_classes: usize::from(num_labels),
            head_seed: self.config.head_seed,
        };
        let mut classifier = SequenceClassifier::try_new(classifier_config)?;
        let parameters = classifier.parameter_count();
        let mut optimizer =
            SequenceClassifierAdamW::try_new(self.config.learning_rate, &classifier)?;

        for _ in 0..self.config.epochs {
            for &index in &split.indices {
                let example = corpus.examples.get(index).ok_or(
                    SequenceNeuralError::InvalidSplitIndex {
                        index,
                        corpus_len: corpus.examples.len(),
                    },
                )?;
                let tokens = tokenizer.encode(&example.payload)?;
                classifier.train_step(&mut optimizer, &tokens, usize::from(example.label.0))?;
            }
        }

        let model = SequenceNeuralModel::from_verified_parts(
            classifier,
            self.config.max_tokens,
            num_labels,
        )?;
        let (correct, total) = evaluate(&model, corpus, split)?;
        let scaled = correct
            .checked_mul(10_000)
            .ok_or(SequenceNeuralError::ArithmeticOverflow)?
            .checked_div(total)
            .ok_or(SequenceNeuralError::ArithmeticOverflow)?;
        let final_accuracy_bps =
            u16::try_from(scaled).map_err(|_| SequenceNeuralError::ArithmeticOverflow)?;
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

/// Frozen read-only façade over the sequence classifier.
#[derive(Clone, Debug)]
pub struct SciRustSequenceReadOnlyModel {
    pub model: Arc<SequenceNeuralModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SCIRUST_SEQUENCE_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[
    ReadOnlyCapability::Classify,
    ReadOnlyCapability::Extract,
    ReadOnlyCapability::Rank,
    ReadOnlyCapability::Explain,
];

impl SciRustSequenceReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequenceNeuralModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SCIRUST_SEQUENCE_READ_ONLY_CAPS,
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

impl ModelBackend for SciRustSequenceReadOnlyModel {
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

fn validate_config(config: SequenceNeuralConfig) -> Result<(), SequenceNeuralError> {
    if !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.max_tokens)
        || config.max_tokens > MAX_SEQUENCE_TOKENS
        || BYTE_TOKENIZER_VOCAB_SIZE > MAX_SEQUENCE_VOCAB
        || !(MIN_SEQUENCE_EMBEDDING_DIM..=MAX_SEQUENCE_EMBEDDING_DIM)
            .contains(&config.embedding_dim)
        || !(MIN_SEQUENCE_HIDDEN_DIM..=MAX_SEQUENCE_HIDDEN_DIM).contains(&config.hidden_dim)
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(SequenceNeuralError::InvalidConfig);
    }
    ByteTokenizer::try_new(config.max_tokens)?;
    Ok(())
}

fn evaluate(
    model: &SequenceNeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<(usize, usize), SequenceNeuralError> {
    let mut correct = 0usize;
    for &index in &split.indices {
        let example = corpus.examples.get(index).ok_or(
            SequenceNeuralError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            },
        )?;
        if model.classify(&example.payload)? == example.label {
            correct = correct
                .checked_add(1)
                .ok_or(SequenceNeuralError::ArithmeticOverflow)?;
        }
    }
    Ok((correct, split.indices.len()))
}

fn map_backend_error(error: SequenceNeuralError) -> BackendError {
    match error {
        SequenceNeuralError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        | SequenceNeuralError::Tokenizer(ByteTokenizerError::LengthOverflow)
        | SequenceNeuralError::TooManyRankCandidates { .. }
        | SequenceNeuralError::SciRust(SciRustError::CapacityExceeded { .. })
        | SequenceNeuralError::SciRust(SciRustError::BudgetExceeded { .. }) => {
            BackendError::InputTooLarge
        }
        SequenceNeuralError::Tokenizer(ByteTokenizerError::AllocationFailed)
        | SequenceNeuralError::Tokenizer(ByteTokenizerError::InvalidMaximum)
        | SequenceNeuralError::InvalidConfig
        | SequenceNeuralError::EmptyTrainingSplit
        | SequenceNeuralError::InsufficientLabels
        | SequenceNeuralError::InvalidSplitIndex { .. }
        | SequenceNeuralError::LabelOutOfRange { .. }
        | SequenceNeuralError::ArithmeticOverflow
        | SequenceNeuralError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::LabeledExample;
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn training_corpus() -> (Corpus, CorpusSplit) {
        let mut corpus = Corpus::with_seed(7);
        for (label, payload) in [
            (0, b"alpha-a".as_slice()),
            (0, b"alpha-b".as_slice()),
            (0, b"alpha-c".as_slice()),
            (1, b"omega-a".as_slice()),
            (1, b"omega-b".as_slice()),
            (1, b"omega-c".as_slice()),
        ] {
            assert!(corpus.add(LabeledExample::new(
                Label(label),
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        let split = CorpusSplit {
            kind: crate::training::SplitKind::Train,
            indices: (0..corpus.examples.len()).collect(),
        };
        (corpus, split)
    }

    fn test_config() -> SequenceNeuralConfig {
        SequenceNeuralConfig {
            max_tokens: 32,
            embedding_dim: 8,
            hidden_dim: 8,
            epochs: 64,
            learning_rate: 0.02,
            encoder_seed: 17,
            head_seed: 19,
        }
    }

    #[test]
    fn training_is_repeatable_and_produces_bounded_report() {
        let (corpus, split) = training_corpus();
        let trainer = SequenceNeuralTrainer::try_new(test_config()).expect("trainer");
        let (left, left_report) = trainer.train(&corpus, &split).expect("left");
        let (right, right_report) = trainer.train(&corpus, &split).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 6);
        assert!(left_report.parameters > 0);
        assert!(left_report.final_correct > 0);
        assert!(left_report.final_accuracy_bps <= 10_000);
    }

    #[test]
    fn tokenizer_capacity_is_enforced_before_sequence_inference() {
        let (corpus, split) = training_corpus();
        let trainer = SequenceNeuralTrainer::try_new(test_config()).expect("trainer");
        let (model, _) = trainer.train(&corpus, &split).expect("model");
        let oversized = vec![b'x'; model.max_tokens()];
        assert!(matches!(
            model.classify(&oversized),
            Err(SequenceNeuralError::Tokenizer(
                ByteTokenizerError::TokenCapacityExceeded { .. }
            ))
        ));
    }

    #[test]
    fn read_only_sequence_backend_never_generates_stateful_proposals() {
        let (corpus, split) = training_corpus();
        let trainer = SequenceNeuralTrainer::try_new(test_config()).expect("trainer");
        let (model, _) = trainer.train(&corpus, &split).expect("model");
        let mut read_only = SciRustSequenceReadOnlyModel::from_trained(model);
        let info = read_only.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
        assert_eq!(
            read_only.next_proposal(),
            Err(BackendError::ReadOnlyViolation)
        );
        let proposal = read_only
            .extract(b"alpha-a", EvidenceId::from_u64(1))
            .expect("proposal");
        assert!(cogno_core::validate_proposal(&proposal.as_view()).is_ok());
    }

    #[test]
    fn invalid_config_and_single_label_training_fail_closed() {
        assert_eq!(
            SequenceNeuralTrainer::try_new(SequenceNeuralConfig {
                max_tokens: 1,
                ..test_config()
            }),
            Err(SequenceNeuralError::InvalidConfig)
        );
        let mut corpus = Corpus::with_seed(1);
        assert!(corpus.add(LabeledExample::new(
            Label(0),
            b"only".to_vec(),
            InputOrigin::ExplicitUserInstruction,
            EvidenceOrigin::ExplicitUserApproval,
        )));
        let split = CorpusSplit {
            kind: crate::training::SplitKind::Train,
            indices: vec![0],
        };
        let trainer = SequenceNeuralTrainer::try_new(test_config()).expect("trainer");
        assert_eq!(
            trainer.train(&corpus, &split),
            Err(SequenceNeuralError::InsufficientLabels)
        );
    }
}
