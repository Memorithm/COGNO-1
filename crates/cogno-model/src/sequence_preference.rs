//! Byte-tokenized pairwise preference model backed by the connected
//! `cogno-scirust` sequence scorer.
//!
//! This is a model-facing bridge for accepted/edited/rejected preference
//! learning. Training may update the bounded sequence encoder and scalar score
//! head, but frozen inference can only compare/rank already-provided payloads.
//! It cannot create policy, install weights, enable tools, or bypass any hard
//! `cogno-core` gate.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::{MAX_NEURAL_EPOCHS, MAX_NEURAL_RANK_CANDIDATES};
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_scirust::{
    SciRustError, SequenceEncoderConfig, SequenceScorer, SequenceScorerAdamW, SequenceScorerConfig,
};
use std::cmp::Ordering;
use std::sync::Arc;

/// Maximum explicit preference pairs accepted by one deterministic run.
pub const MAX_SEQUENCE_PREFERENCE_PAIRS: usize = 4_096;

/// One explicit preference observation. Both sides remain non-authoritative
/// byte payloads; provenance/admission policy belongs to the caller/runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencePreferencePair {
    pub preferred: Vec<u8>,
    pub dispreferred: Vec<u8>,
}

impl SequencePreferencePair {
    #[must_use]
    pub fn new(preferred: Vec<u8>, dispreferred: Vec<u8>) -> Self {
        Self {
            preferred,
            dispreferred,
        }
    }
}

/// Bounded deterministic pairwise-training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequencePreferenceConfig {
    pub scorer: SequenceScorerConfig,
    pub epochs: u32,
    pub learning_rate: f32,
    pub margin: f32,
}

impl Default for SequencePreferenceConfig {
    fn default() -> Self {
        Self {
            scorer: SequenceScorerConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                head_seed: 1,
            },
            epochs: 24,
            learning_rate: 0.01,
            margin: 1.0,
        }
    }
}

/// Fail-closed model-facing preference errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SequencePreferenceError {
    InvalidConfig,
    EmptyTrainingSet,
    TooManyPairs { actual: usize, maximum: usize },
    TooManyRankCandidates { actual: usize, maximum: usize },
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequencePreferenceError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequencePreferenceError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Deterministic summary of one pairwise training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequencePreferenceTrainingReport {
    pub pairs: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_mean_loss: f32,
}

/// Frozen byte-tokenized scalar preference model.
#[derive(Clone, Debug, PartialEq)]
pub struct SequencePreferenceModel {
    scorer: SequenceScorer,
    tokenizer: ByteTokenizer,
}

impl SequencePreferenceModel {
    fn from_scorer(scorer: SequenceScorer) -> Result<Self, SequencePreferenceError> {
        let tokenizer = ByteTokenizer::try_new(scorer.config().encoder.max_tokens)?;
        if scorer.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE {
            return Err(SequencePreferenceError::InvalidConfig);
        }
        Ok(Self { scorer, tokenizer })
    }

    #[must_use]
    pub const fn scorer(&self) -> &SequenceScorer {
        &self.scorer
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.tokenizer.max_tokens()
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.scorer.parameter_count()
    }

    /// Scalar preference score for one arbitrary byte payload.
    pub fn score(&self, payload: &[u8]) -> Result<f32, SequencePreferenceError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.scorer.score(&tokens)?)
    }

    /// Compare two payloads by learned preference score. Equal scores remain
    /// equal; callers that need a total ranking use stable input-index ties.
    pub fn compare(&self, left: &[u8], right: &[u8]) -> Result<Ordering, SequencePreferenceError> {
        let left = self.score(left)?;
        let right = self.score(right)?;
        Ok(left.total_cmp(&right))
    }

    /// Rank already-produced payloads, best first. Ties preserve ascending
    /// original index for deterministic replay.
    pub fn rank(&self, candidates: &[&[u8]]) -> Result<Vec<usize>, SequencePreferenceError> {
        if candidates.len() > MAX_NEURAL_RANK_CANDIDATES {
            return Err(SequencePreferenceError::TooManyRankCandidates {
                actual: candidates.len(),
                maximum: MAX_NEURAL_RANK_CANDIDATES,
            });
        }
        let mut scored = Vec::with_capacity(candidates.len());
        for (index, payload) in candidates.iter().enumerate() {
            scored.push((index, self.score(payload)?));
        }
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(scored.into_iter().map(|(index, _)| index).collect())
    }
}

/// Deterministic trainer for explicit preferred/dispreferred byte pairs.
#[derive(Clone, Copy, Debug)]
pub struct SequencePreferenceTrainer {
    pub config: SequencePreferenceConfig,
}

impl SequencePreferenceTrainer {
    pub fn try_new(config: SequencePreferenceConfig) -> Result<Self, SequencePreferenceError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        pairs: &[SequencePreferencePair],
    ) -> Result<(SequencePreferenceModel, SequencePreferenceTrainingReport), SequencePreferenceError>
    {
        if pairs.is_empty() {
            return Err(SequencePreferenceError::EmptyTrainingSet);
        }
        if pairs.len() > MAX_SEQUENCE_PREFERENCE_PAIRS {
            return Err(SequencePreferenceError::TooManyPairs {
                actual: pairs.len(),
                maximum: MAX_SEQUENCE_PREFERENCE_PAIRS,
            });
        }

        let tokenizer = ByteTokenizer::try_new(self.config.scorer.encoder.max_tokens)?;
        let mut scorer = SequenceScorer::try_new(self.config.scorer)?;
        let parameters = scorer.parameter_count();
        let mut optimizer = SequenceScorerAdamW::try_new(self.config.learning_rate, &scorer)?;
        let mut final_loss_sum = 0.0f32;

        for epoch in 0..self.config.epochs {
            let last_epoch = epoch + 1 == self.config.epochs;
            if last_epoch {
                final_loss_sum = 0.0;
            }
            for pair in pairs {
                let preferred = tokenizer.encode(&pair.preferred)?;
                let dispreferred = tokenizer.encode(&pair.dispreferred)?;
                let loss = scorer.train_pairwise_step(
                    &mut optimizer,
                    &preferred,
                    &dispreferred,
                    self.config.margin,
                )?;
                if last_epoch {
                    final_loss_sum += loss;
                    if !final_loss_sum.is_finite() {
                        return Err(SequencePreferenceError::SciRust(SciRustError::NonFinite));
                    }
                }
            }
        }

        let divisor = pairs.len() as f32;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(SequencePreferenceError::InvalidConfig);
        }
        let final_mean_loss = final_loss_sum / divisor;
        if !final_mean_loss.is_finite() {
            return Err(SequencePreferenceError::SciRust(SciRustError::NonFinite));
        }
        let model = SequencePreferenceModel::from_scorer(scorer)?;
        Ok((
            model,
            SequencePreferenceTrainingReport {
                pairs: pairs.len(),
                epochs: self.config.epochs,
                parameters,
                final_mean_loss,
            },
        ))
    }
}

/// Frozen read-only facade. Ranking is the only exposed model capability.
#[derive(Clone, Debug)]
pub struct SciRustSequencePreferenceReadOnlyModel {
    pub model: Arc<SequencePreferenceModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SEQUENCE_PREFERENCE_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[ReadOnlyCapability::Rank];

impl SciRustSequencePreferenceReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequencePreferenceModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SEQUENCE_PREFERENCE_READ_ONLY_CAPS,
        }
    }

    pub fn score(&self, payload: &[u8]) -> Result<f32, BackendError> {
        self.model.score(payload).map_err(map_backend_error)
    }

    pub fn compare(&self, left: &[u8], right: &[u8]) -> Result<Ordering, BackendError> {
        self.model.compare(left, right).map_err(map_backend_error)
    }

    pub fn rank(&self, candidates: &[&[u8]]) -> Result<Vec<usize>, BackendError> {
        self.model.rank(candidates).map_err(map_backend_error)
    }
}

impl ModelBackend for SciRustSequencePreferenceReadOnlyModel {
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

fn validate_config(config: SequencePreferenceConfig) -> Result<(), SequencePreferenceError> {
    if config.scorer.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.scorer.encoder.max_tokens)
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
        || !config.margin.is_finite()
        || config.margin < 0.0
    {
        return Err(SequencePreferenceError::InvalidConfig);
    }
    let _ = ByteTokenizer::try_new(config.scorer.encoder.max_tokens)?;
    let _ = SequenceScorer::try_new(config.scorer)?;
    Ok(())
}

fn map_backend_error(error: SequencePreferenceError) -> BackendError {
    match error {
        SequencePreferenceError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded {
            ..
        })
        | SequencePreferenceError::TooManyRankCandidates { .. }
        | SequencePreferenceError::TooManyPairs { .. } => BackendError::InputTooLarge,
        SequencePreferenceError::Tokenizer(_)
        | SequencePreferenceError::InvalidConfig
        | SequencePreferenceError::EmptyTrainingSet
        | SequencePreferenceError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequencePreferenceConfig {
        SequencePreferenceConfig {
            scorer: SequenceScorerConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 32,
                    embedding_dim: 8,
                    hidden_dim: 8,
                    seed: 607,
                },
                head_seed: 613,
            },
            epochs: 64,
            learning_rate: 0.02,
            margin: 1.0,
        }
    }

    fn pairs() -> Vec<SequencePreferencePair> {
        vec![
            SequencePreferencePair::new(b"accepted alpha".to_vec(), b"rejected omega".to_vec()),
            SequencePreferencePair::new(b"accepted beta".to_vec(), b"rejected delta".to_vec()),
        ]
    }

    #[test]
    fn deterministic_training_learns_explicit_preference_pairs() {
        let trainer = SequencePreferenceTrainer::try_new(config()).expect("trainer");
        let (left, left_report) = trainer.train(&pairs()).expect("left");
        let (right, right_report) = trainer.train(&pairs()).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.pairs, 2);
        assert_eq!(left_report.epochs, 64);
        assert_eq!(
            left.compare(b"accepted alpha", b"rejected omega")
                .expect("compare"),
            Ordering::Greater
        );
        assert_eq!(
            left.rank(&[b"rejected omega", b"accepted alpha"])
                .expect("rank"),
            vec![1, 0]
        );
    }

    #[test]
    fn byte_tokenizer_bounds_apply_before_pairwise_scoring() {
        let trainer = SequencePreferenceTrainer::try_new(config()).expect("trainer");
        let oversized = vec![b'x'; 31];
        let error = trainer
            .train(&[SequencePreferencePair::new(oversized, b"small".to_vec())])
            .expect_err("framing exceeds max tokens");
        assert!(matches!(
            error,
            SequencePreferenceError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        ));
    }

    #[test]
    fn frozen_preference_backend_exposes_rank_only_and_never_proposes() {
        let trainer = SequencePreferenceTrainer::try_new(config()).expect("trainer");
        let (model, _) = trainer.train(&pairs()).expect("model");
        let mut readonly = SciRustSequencePreferenceReadOnlyModel::from_trained(model);
        assert_eq!(readonly.capabilities, &[ReadOnlyCapability::Rank]);
        assert_eq!(
            readonly.rank(&[b"rejected omega", b"accepted alpha"]),
            Ok(vec![1, 0])
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
    fn invalid_byte_vocab_is_rejected_before_training() {
        let mut invalid = config();
        invalid.scorer.encoder.vocab_size = BYTE_TOKENIZER_VOCAB_SIZE - 1;
        assert!(matches!(
            SequencePreferenceTrainer::try_new(invalid),
            Err(SequencePreferenceError::InvalidConfig)
        ));
    }
}
