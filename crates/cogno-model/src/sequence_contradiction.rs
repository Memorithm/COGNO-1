//! Byte-tokenized pairwise contradiction model backed by the connected
//! `cogno-scirust` sequence classifier.
//!
//! Input pairs are framed deterministically as `[BOS] left [SEP] right [EOS]`.
//! The model only estimates a soft contradiction class/probability; evidence
//! admission and all hard consistency decisions remain outside the model.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::MAX_NEURAL_EPOCHS;
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_scirust::{
    SciRustError, SequenceClassifier, SequenceClassifierAdamW, SequenceClassifierConfig,
    SequenceEncoderConfig,
};
use std::sync::Arc;

/// Non-contradictory pair class id.
pub const SEQUENCE_CONTRADICTION_CLEAR_CLASS: usize = 0;
/// Contradictory pair class id.
pub const SEQUENCE_CONTRADICTION_CLASS: usize = 1;
/// The contradiction head is deliberately binary.
pub const SEQUENCE_CONTRADICTION_CLASSES: usize = 2;
/// Maximum explicit pair observations accepted by one deterministic run.
pub const MAX_SEQUENCE_CONTRADICTION_EXAMPLES: usize = 4_096;

/// One host-labelled pairwise contradiction observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceContradictionExample {
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub contradicts: bool,
}

impl SequenceContradictionExample {
    #[must_use]
    pub fn new(left: Vec<u8>, right: Vec<u8>, contradicts: bool) -> Self {
        Self {
            left,
            right,
            contradicts,
        }
    }
}

/// Bounded deterministic contradiction-training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceContradictionConfig {
    pub classifier: SequenceClassifierConfig,
    pub epochs: u32,
    pub learning_rate: f32,
}

impl Default for SequenceContradictionConfig {
    fn default() -> Self {
        Self {
            classifier: SequenceClassifierConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                num_classes: SEQUENCE_CONTRADICTION_CLASSES,
                head_seed: 1,
            },
            epochs: 24,
            learning_rate: 0.01,
        }
    }
}

/// Fail-closed model-facing contradiction errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceContradictionError {
    InvalidConfig,
    EmptyTrainingSet,
    TooManyExamples { actual: usize, maximum: usize },
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequenceContradictionError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceContradictionError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Deterministic summary of one contradiction training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceContradictionTrainingReport {
    pub examples: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_mean_loss: f32,
}

/// Frozen byte-tokenized contradiction estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceContradictionModel {
    classifier: SequenceClassifier,
    tokenizer: ByteTokenizer,
}

impl SequenceContradictionModel {
    fn from_classifier(classifier: SequenceClassifier) -> Result<Self, SequenceContradictionError> {
        if classifier.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
            || classifier.config().num_classes != SEQUENCE_CONTRADICTION_CLASSES
        {
            return Err(SequenceContradictionError::InvalidConfig);
        }
        let tokenizer = ByteTokenizer::try_new(classifier.config().encoder.max_tokens)?;
        Ok(Self {
            classifier,
            tokenizer,
        })
    }

    #[must_use]
    pub const fn classifier(&self) -> &SequenceClassifier {
        &self.classifier
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.tokenizer.max_tokens()
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.classifier.parameter_count()
    }

    /// Stable `[clear, contradiction]` probabilities for one ordered pair.
    pub fn probabilities(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<[f32; SEQUENCE_CONTRADICTION_CLASSES], SequenceContradictionError> {
        let tokens = self.tokenizer.encode_pair(left, right)?;
        let probabilities = self.classifier.probabilities(&tokens)?;
        let probabilities: [f32; SEQUENCE_CONTRADICTION_CLASSES] = probabilities
            .try_into()
            .map_err(|_| SequenceContradictionError::InvalidConfig)?;
        Ok(probabilities)
    }

    /// Soft contradiction probability. This is non-authoritative evidence only.
    pub fn contradiction_probability(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<f32, SequenceContradictionError> {
        Ok(self.probabilities(left, right)?[SEQUENCE_CONTRADICTION_CLASS])
    }

    /// Deterministic binary class prediction; ties inherit the classifier's
    /// lowest-class-id rule and therefore resolve to non-contradiction.
    pub fn contradicts(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<bool, SequenceContradictionError> {
        let tokens = self.tokenizer.encode_pair(left, right)?;
        Ok(self.classifier.classify(&tokens)? == SEQUENCE_CONTRADICTION_CLASS)
    }
}

/// Deterministic trainer for explicit contradiction-labelled byte pairs.
#[derive(Clone, Copy, Debug)]
pub struct SequenceContradictionTrainer {
    pub config: SequenceContradictionConfig,
}

impl SequenceContradictionTrainer {
    pub fn try_new(
        config: SequenceContradictionConfig,
    ) -> Result<Self, SequenceContradictionError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        examples: &[SequenceContradictionExample],
    ) -> Result<
        (
            SequenceContradictionModel,
            SequenceContradictionTrainingReport,
        ),
        SequenceContradictionError,
    > {
        validate_examples(examples)?;
        let tokenizer = ByteTokenizer::try_new(self.config.classifier.encoder.max_tokens)?;

        // Validate every pair framing before constructing optimizer state or
        // applying any update.
        for example in examples {
            let _ = tokenizer.encode_pair(&example.left, &example.right)?;
        }

        let mut classifier = SequenceClassifier::try_new(self.config.classifier)?;
        let parameters = classifier.parameter_count();
        let mut optimizer =
            SequenceClassifierAdamW::try_new(self.config.learning_rate, &classifier)?;
        let mut final_loss_sum = 0.0f32;

        for epoch in 0..self.config.epochs {
            let last_epoch = epoch + 1 == self.config.epochs;
            if last_epoch {
                final_loss_sum = 0.0;
            }
            for example in examples {
                let tokens = tokenizer.encode_pair(&example.left, &example.right)?;
                let target = if example.contradicts {
                    SEQUENCE_CONTRADICTION_CLASS
                } else {
                    SEQUENCE_CONTRADICTION_CLEAR_CLASS
                };
                let loss = classifier.train_step(&mut optimizer, &tokens, target)?;
                if last_epoch {
                    final_loss_sum += loss;
                    if !final_loss_sum.is_finite() {
                        return Err(SequenceContradictionError::SciRust(SciRustError::NonFinite));
                    }
                }
            }
        }

        let divisor = examples.len() as f32;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(SequenceContradictionError::InvalidConfig);
        }
        let final_mean_loss = final_loss_sum / divisor;
        if !final_mean_loss.is_finite() {
            return Err(SequenceContradictionError::SciRust(SciRustError::NonFinite));
        }
        let model = SequenceContradictionModel::from_classifier(classifier)?;
        Ok((
            model,
            SequenceContradictionTrainingReport {
                examples: examples.len(),
                epochs: self.config.epochs,
                parameters,
                final_mean_loss,
            },
        ))
    }
}

/// Frozen read-only contradiction facade. Classification is its only exposed
/// capability; it cannot propose policy or mutate state.
#[derive(Clone, Debug)]
pub struct SciRustSequenceContradictionReadOnlyModel {
    pub model: Arc<SequenceContradictionModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SEQUENCE_CONTRADICTION_READ_ONLY_CAPS: &[ReadOnlyCapability] =
    &[ReadOnlyCapability::Classify];

impl SciRustSequenceContradictionReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequenceContradictionModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SEQUENCE_CONTRADICTION_READ_ONLY_CAPS,
        }
    }

    pub fn contradiction_probability(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<f32, BackendError> {
        self.model
            .contradiction_probability(left, right)
            .map_err(map_backend_error)
    }

    pub fn contradicts(&self, left: &[u8], right: &[u8]) -> Result<bool, BackendError> {
        self.model
            .contradicts(left, right)
            .map_err(map_backend_error)
    }
}

impl ModelBackend for SciRustSequenceContradictionReadOnlyModel {
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

fn validate_config(config: SequenceContradictionConfig) -> Result<(), SequenceContradictionError> {
    if config.classifier.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(3..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.classifier.encoder.max_tokens)
        || config.classifier.num_classes != SEQUENCE_CONTRADICTION_CLASSES
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(SequenceContradictionError::InvalidConfig);
    }
    let _ = ByteTokenizer::try_new(config.classifier.encoder.max_tokens)?;
    let _ = SequenceClassifier::try_new(config.classifier)?;
    Ok(())
}

fn validate_examples(
    examples: &[SequenceContradictionExample],
) -> Result<(), SequenceContradictionError> {
    if examples.is_empty() {
        return Err(SequenceContradictionError::EmptyTrainingSet);
    }
    if examples.len() > MAX_SEQUENCE_CONTRADICTION_EXAMPLES {
        return Err(SequenceContradictionError::TooManyExamples {
            actual: examples.len(),
            maximum: MAX_SEQUENCE_CONTRADICTION_EXAMPLES,
        });
    }
    Ok(())
}

fn map_backend_error(error: SequenceContradictionError) -> BackendError {
    match error {
        SequenceContradictionError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded {
            ..
        })
        | SequenceContradictionError::TooManyExamples { .. } => BackendError::InputTooLarge,
        SequenceContradictionError::InvalidConfig
        | SequenceContradictionError::EmptyTrainingSet
        | SequenceContradictionError::Tokenizer(_)
        | SequenceContradictionError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequenceContradictionConfig {
        SequenceContradictionConfig {
            classifier: SequenceClassifierConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 40,
                    embedding_dim: 8,
                    hidden_dim: 8,
                    seed: 1009,
                },
                num_classes: SEQUENCE_CONTRADICTION_CLASSES,
                head_seed: 1013,
            },
            epochs: 64,
            learning_rate: 0.02,
        }
    }

    fn examples() -> Vec<SequenceContradictionExample> {
        vec![
            SequenceContradictionExample::new(
                b"sky is blue".to_vec(),
                b"sky is blue".to_vec(),
                false,
            ),
            SequenceContradictionExample::new(
                b"sky is blue".to_vec(),
                b"sky is green".to_vec(),
                true,
            ),
            SequenceContradictionExample::new(
                b"water is wet".to_vec(),
                b"water is wet".to_vec(),
                false,
            ),
            SequenceContradictionExample::new(
                b"water is wet".to_vec(),
                b"water is dry".to_vec(),
                true,
            ),
        ]
    }

    #[test]
    fn deterministic_training_and_pair_inference_repeat_exactly() {
        let trainer = SequenceContradictionTrainer::try_new(config()).expect("trainer");
        let (left, left_report) = trainer.train(&examples()).expect("left");
        let (right, right_report) = trainer.train(&examples()).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 4);
        let probabilities = left
            .probabilities(b"sky is blue", b"sky is green")
            .expect("probabilities");
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities[0] + probabilities[1] - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn pair_framing_capacity_fails_before_training() {
        let mut config = config();
        config.classifier.encoder.max_tokens = 16;
        let trainer = SequenceContradictionTrainer::try_new(config).expect("trainer");
        let error = trainer
            .train(&[SequenceContradictionExample::new(
                vec![b'a'; 8],
                vec![b'b'; 8],
                true,
            )])
            .expect_err("pair exceeds framing capacity");
        assert!(matches!(
            error,
            SequenceContradictionError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        ));
    }

    #[test]
    fn non_binary_classifier_is_rejected() {
        let mut invalid = config();
        invalid.classifier.num_classes = 3;
        assert!(matches!(
            SequenceContradictionTrainer::try_new(invalid),
            Err(SequenceContradictionError::InvalidConfig)
        ));
    }

    #[test]
    fn frozen_contradiction_backend_is_classify_only_and_never_proposes() {
        let trainer = SequenceContradictionTrainer::try_new(config()).expect("trainer");
        let (model, _) = trainer.train(&examples()).expect("model");
        let mut readonly = SciRustSequenceContradictionReadOnlyModel::from_trained(model);
        assert_eq!(readonly.capabilities, &[ReadOnlyCapability::Classify]);
        let probability = readonly
            .contradiction_probability(b"sky is blue", b"sky is green")
            .expect("probability");
        assert!((0.0..=1.0).contains(&probability));
        assert_eq!(
            readonly.next_proposal(),
            Err(BackendError::ReadOnlyViolation)
        );
        let info = readonly.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
    }
}
