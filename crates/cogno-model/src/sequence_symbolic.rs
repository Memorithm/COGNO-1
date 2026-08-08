//! Byte-tokenized symbolic-satisfaction model backed by the connected
//! `cogno-scirust` sequence symbolic head.
//!
//! Training examples carry explicit host-provided rule truth labels. The model
//! learns only soft satisfaction estimates; it never becomes the authority for
//! rule truth, adoption, rejection or enforcement.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::MAX_NEURAL_EPOCHS;
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_scirust::{
    SciRustError, SequenceEncoderConfig, SequenceSymbolicAdamW, SequenceSymbolicConfig,
    SequenceSymbolicHead, MAX_SEQUENCE_SYMBOLIC_RULES,
};
use std::sync::Arc;

/// Maximum explicit symbolic examples accepted by one deterministic run.
pub const MAX_SEQUENCE_SYMBOLIC_EXAMPLES: usize = 4_096;

/// One host-labelled symbolic observation. Each boolean corresponds to one
/// configured rule head and is converted to an exact `0.0`/`1.0` target only
/// at the numerical training boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceSymbolicExample {
    pub payload: Vec<u8>,
    pub rule_satisfied: Vec<bool>,
}

impl SequenceSymbolicExample {
    #[must_use]
    pub fn new(payload: Vec<u8>, rule_satisfied: Vec<bool>) -> Self {
        Self {
            payload,
            rule_satisfied,
        }
    }
}

/// Bounded deterministic symbolic training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceSymbolicModelConfig {
    pub symbolic: SequenceSymbolicConfig,
    pub epochs: u32,
    pub learning_rate: f32,
}

impl Default for SequenceSymbolicModelConfig {
    fn default() -> Self {
        Self {
            symbolic: SequenceSymbolicConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                num_rules: 4,
                head_seed: 1,
            },
            epochs: 24,
            learning_rate: 0.01,
        }
    }
}

/// Fail-closed model-facing symbolic errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceSymbolicModelError {
    InvalidConfig,
    EmptyTrainingSet,
    TooManyExamples { actual: usize, maximum: usize },
    RuleTargetCountMismatch { actual: usize, expected: usize },
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequenceSymbolicModelError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceSymbolicModelError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Deterministic summary of one symbolic training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceSymbolicTrainingReport {
    pub examples: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_mean_loss: f32,
}

/// Frozen byte-tokenized soft symbolic model.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSymbolicModel {
    symbolic: SequenceSymbolicHead,
    tokenizer: ByteTokenizer,
}

impl SequenceSymbolicModel {
    fn from_symbolic(symbolic: SequenceSymbolicHead) -> Result<Self, SequenceSymbolicModelError> {
        if symbolic.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE {
            return Err(SequenceSymbolicModelError::InvalidConfig);
        }
        let tokenizer = ByteTokenizer::try_new(symbolic.config().encoder.max_tokens)?;
        Ok(Self {
            symbolic,
            tokenizer,
        })
    }

    #[must_use]
    pub const fn symbolic(&self) -> &SequenceSymbolicHead {
        &self.symbolic
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.tokenizer.max_tokens()
    }

    #[must_use]
    pub const fn num_rules(&self) -> usize {
        self.symbolic.config().num_rules
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.symbolic.parameter_count()
    }

    /// Return one non-authoritative soft satisfaction per configured rule.
    pub fn satisfactions(&self, payload: &[u8]) -> Result<Vec<f32>, SequenceSymbolicModelError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.symbolic.satisfactions(&tokens)?)
    }

    /// Return the soft conjunction diagnostic. This value is never a hard-rule
    /// admission decision.
    pub fn soft_conjunction(&self, payload: &[u8]) -> Result<f32, SequenceSymbolicModelError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.symbolic.soft_conjunction(&tokens)?)
    }
}

/// Deterministic trainer for explicit byte payload / rule-truth observations.
#[derive(Clone, Copy, Debug)]
pub struct SequenceSymbolicTrainer {
    pub config: SequenceSymbolicModelConfig,
}

impl SequenceSymbolicTrainer {
    pub fn try_new(
        config: SequenceSymbolicModelConfig,
    ) -> Result<Self, SequenceSymbolicModelError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        examples: &[SequenceSymbolicExample],
    ) -> Result<(SequenceSymbolicModel, SequenceSymbolicTrainingReport), SequenceSymbolicModelError>
    {
        validate_examples(examples, self.config.symbolic.num_rules)?;
        let tokenizer = ByteTokenizer::try_new(self.config.symbolic.encoder.max_tokens)?;

        // Validate the complete hostile corpus before constructing optimizer
        // state or applying any update.
        for example in examples {
            let _ = tokenizer.encode(&example.payload)?;
        }

        let mut symbolic = SequenceSymbolicHead::try_new(self.config.symbolic)?;
        let parameters = symbolic.parameter_count();
        let mut optimizer = SequenceSymbolicAdamW::try_new(self.config.learning_rate, &symbolic)?;
        let mut final_loss_sum = 0.0f32;

        for epoch in 0..self.config.epochs {
            let last_epoch = epoch + 1 == self.config.epochs;
            if last_epoch {
                final_loss_sum = 0.0;
            }
            for example in examples {
                let tokens = tokenizer.encode(&example.payload)?;
                let targets: Vec<f32> = example
                    .rule_satisfied
                    .iter()
                    .map(|&satisfied| if satisfied { 1.0 } else { 0.0 })
                    .collect();
                let loss = symbolic.train_target_step(&mut optimizer, &tokens, &targets)?;
                if last_epoch {
                    final_loss_sum += loss;
                    if !final_loss_sum.is_finite() {
                        return Err(SequenceSymbolicModelError::SciRust(SciRustError::NonFinite));
                    }
                }
            }
        }

        let divisor = examples.len() as f32;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(SequenceSymbolicModelError::InvalidConfig);
        }
        let final_mean_loss = final_loss_sum / divisor;
        if !final_mean_loss.is_finite() {
            return Err(SequenceSymbolicModelError::SciRust(SciRustError::NonFinite));
        }
        let model = SequenceSymbolicModel::from_symbolic(symbolic)?;
        Ok((
            model,
            SequenceSymbolicTrainingReport {
                examples: examples.len(),
                epochs: self.config.epochs,
                parameters,
                final_mean_loss,
            },
        ))
    }
}

/// Frozen read-only facade for soft symbolic classification. It cannot create,
/// adopt, delete or enforce rules.
#[derive(Clone, Debug)]
pub struct SciRustSequenceSymbolicReadOnlyModel {
    pub model: Arc<SequenceSymbolicModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SEQUENCE_SYMBOLIC_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[ReadOnlyCapability::Classify];

impl SciRustSequenceSymbolicReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequenceSymbolicModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SEQUENCE_SYMBOLIC_READ_ONLY_CAPS,
        }
    }

    pub fn satisfactions(&self, payload: &[u8]) -> Result<Vec<f32>, BackendError> {
        self.model.satisfactions(payload).map_err(map_backend_error)
    }

    pub fn soft_conjunction(&self, payload: &[u8]) -> Result<f32, BackendError> {
        self.model
            .soft_conjunction(payload)
            .map_err(map_backend_error)
    }
}

impl ModelBackend for SciRustSequenceSymbolicReadOnlyModel {
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

fn validate_config(config: SequenceSymbolicModelConfig) -> Result<(), SequenceSymbolicModelError> {
    if config.symbolic.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.symbolic.encoder.max_tokens)
        || config.symbolic.num_rules == 0
        || config.symbolic.num_rules > MAX_SEQUENCE_SYMBOLIC_RULES
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(SequenceSymbolicModelError::InvalidConfig);
    }
    let _ = ByteTokenizer::try_new(config.symbolic.encoder.max_tokens)?;
    let _ = SequenceSymbolicHead::try_new(config.symbolic)?;
    Ok(())
}

fn validate_examples(
    examples: &[SequenceSymbolicExample],
    expected_rules: usize,
) -> Result<(), SequenceSymbolicModelError> {
    if examples.is_empty() {
        return Err(SequenceSymbolicModelError::EmptyTrainingSet);
    }
    if examples.len() > MAX_SEQUENCE_SYMBOLIC_EXAMPLES {
        return Err(SequenceSymbolicModelError::TooManyExamples {
            actual: examples.len(),
            maximum: MAX_SEQUENCE_SYMBOLIC_EXAMPLES,
        });
    }
    for example in examples {
        if example.rule_satisfied.len() != expected_rules {
            return Err(SequenceSymbolicModelError::RuleTargetCountMismatch {
                actual: example.rule_satisfied.len(),
                expected: expected_rules,
            });
        }
    }
    Ok(())
}

fn map_backend_error(error: SequenceSymbolicModelError) -> BackendError {
    match error {
        SequenceSymbolicModelError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded {
            ..
        })
        | SequenceSymbolicModelError::TooManyExamples { .. } => BackendError::InputTooLarge,
        SequenceSymbolicModelError::InvalidConfig
        | SequenceSymbolicModelError::EmptyTrainingSet
        | SequenceSymbolicModelError::RuleTargetCountMismatch { .. }
        | SequenceSymbolicModelError::Tokenizer(_)
        | SequenceSymbolicModelError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequenceSymbolicModelConfig {
        SequenceSymbolicModelConfig {
            symbolic: SequenceSymbolicConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 32,
                    embedding_dim: 8,
                    hidden_dim: 8,
                    seed: 907,
                },
                num_rules: 2,
                head_seed: 911,
            },
            epochs: 64,
            learning_rate: 0.02,
        }
    }

    fn examples() -> Vec<SequenceSymbolicExample> {
        vec![
            SequenceSymbolicExample::new(b"safe evidence".to_vec(), vec![true, true]),
            SequenceSymbolicExample::new(b"violating evidence".to_vec(), vec![false, true]),
        ]
    }

    #[test]
    fn deterministic_training_and_inference_repeat_exactly() {
        let trainer = SequenceSymbolicTrainer::try_new(config()).expect("trainer");
        let (left, left_report) = trainer.train(&examples()).expect("left");
        let (right, right_report) = trainer.train(&examples()).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 2);
        assert!(left_report.final_mean_loss.is_finite());
        let satisfactions = left.satisfactions(b"safe evidence").expect("satisfactions");
        assert_eq!(satisfactions.len(), 2);
        assert!(satisfactions
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn rule_target_shape_fails_before_training() {
        let trainer = SequenceSymbolicTrainer::try_new(config()).expect("trainer");
        assert_eq!(
            trainer.train(&[SequenceSymbolicExample::new(
                b"payload".to_vec(),
                vec![true],
            )]),
            Err(SequenceSymbolicModelError::RuleTargetCountMismatch {
                actual: 1,
                expected: 2,
            })
        );
    }

    #[test]
    fn hostile_token_lengths_fail_before_training() {
        let trainer = SequenceSymbolicTrainer::try_new(config()).expect("trainer");
        let oversized = vec![b'x'; 31];
        let error = trainer
            .train(&[SequenceSymbolicExample::new(oversized, vec![true, false])])
            .expect_err("framing exceeds max tokens");
        assert!(matches!(
            error,
            SequenceSymbolicModelError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        ));
    }

    #[test]
    fn frozen_symbolic_backend_is_classify_only_and_never_proposes() {
        let trainer = SequenceSymbolicTrainer::try_new(config()).expect("trainer");
        let (model, _) = trainer.train(&examples()).expect("model");
        let mut readonly = SciRustSequenceSymbolicReadOnlyModel::from_trained(model);
        assert_eq!(readonly.capabilities, &[ReadOnlyCapability::Classify]);
        assert_eq!(
            readonly
                .satisfactions(b"safe evidence")
                .expect("soft output")
                .len(),
            2
        );
        assert_eq!(
            readonly.next_proposal(),
            Err(BackendError::ReadOnlyViolation)
        );
        let info = readonly.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
    }

    #[test]
    fn wrong_tokenizer_vocabulary_is_rejected() {
        let mut invalid = config();
        invalid.symbolic.encoder.vocab_size = BYTE_TOKENIZER_VOCAB_SIZE + 1;
        assert!(matches!(
            SequenceSymbolicTrainer::try_new(invalid),
            Err(SequenceSymbolicModelError::InvalidConfig)
        ));
    }
}
