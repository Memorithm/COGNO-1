//! Byte-tokenized bridge for the joint shared sequence cognitive objective.
//!
//! One [`cogno_scirust::SequenceCognitiveHeads`] instance owns the shared
//! sequence encoder used by classification, preference, symbolic satisfaction,
//! contradiction and retrieval. Training preflights the complete hostile byte
//! corpus before optimizer construction, then performs one connected multi-task
//! update per observation.
//!
//! Every output remains non-authoritative model data. Hard constraints, rule
//! truth, evidence admission, measured runtime cost, persistence and tools stay
//! outside this bridge.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::{MAX_NEURAL_EPOCHS, MAX_NEURAL_RANK_CANDIDATES};
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_scirust::{
    CognitiveClassification, CognitiveContradiction, CognitivePreference, CognitiveRetrieval,
    CognitiveSymbolic, SciRustError, SequenceCognitiveAdamW, SequenceCognitiveBatch,
    SequenceCognitiveConfig, SequenceCognitiveHeads, SequenceCognitiveLossReport,
    SequenceCognitiveLossWeights, SequenceEncoderConfig, MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
};
use std::cmp::Ordering;
use std::sync::Arc;

/// Maximum explicit joint cognitive observations in one deterministic run.
pub const MAX_SEQUENCE_COGNITIVE_EXAMPLES: usize = 4_096;

/// One owned multi-signal supervision observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveExample {
    pub classification_payload: Vec<u8>,
    pub classification_target: usize,
    pub preferred: Vec<u8>,
    pub dispreferred: Vec<u8>,
    pub symbolic_payload: Vec<u8>,
    pub rule_satisfied: Vec<bool>,
    pub contradiction_left: Vec<u8>,
    pub contradiction_right: Vec<u8>,
    pub contradicts: bool,
    pub retrieval_query: Vec<u8>,
    pub retrieval_candidates: Vec<Vec<u8>>,
    pub retrieval_positive_idx: usize,
}

/// Bounded deterministic joint-training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceCognitiveModelConfig {
    pub cognitive: SequenceCognitiveConfig,
    pub epochs: u32,
    pub learning_rate: f32,
    pub loss_weights: SequenceCognitiveLossWeights,
    pub preference_margin: f32,
    pub retrieval_temperature: f32,
    pub max_retrieval_candidates: usize,
}

impl Default for SequenceCognitiveModelConfig {
    fn default() -> Self {
        Self {
            cognitive: SequenceCognitiveConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                num_classes: 3,
                num_rules: 4,
                classification_seed: 1,
                preference_seed: 2,
                symbolic_seed: 3,
                contradiction_seed: 4,
            },
            epochs: 24,
            learning_rate: 0.01,
            loss_weights: SequenceCognitiveLossWeights::default(),
            preference_margin: 1.0,
            retrieval_temperature: 0.2,
            max_retrieval_candidates: 8,
        }
    }
}

/// Fail-closed model-facing joint cognitive errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceCognitiveModelError {
    InvalidConfig,
    EmptyTrainingSet,
    TooManyExamples { actual: usize, maximum: usize },
    ClassificationTargetOutOfRange { target: usize, classes: usize },
    RuleTargetCountMismatch { actual: usize, expected: usize },
    TooFewRetrievalCandidates { actual: usize },
    TooManyRetrievalCandidates { actual: usize, maximum: usize },
    InvalidRetrievalPositiveIndex { idx: usize, len: usize },
    TooManyRankCandidates { actual: usize, maximum: usize },
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequenceCognitiveModelError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceCognitiveModelError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Mean component losses from the final deterministic epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceCognitiveTrainingReport {
    pub examples: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_mean_classification: f32,
    pub final_mean_preference: f32,
    pub final_mean_symbolic: f32,
    pub final_mean_contradiction: f32,
    pub final_mean_retrieval: f32,
    pub final_mean_weighted_total: f32,
}

/// Frozen byte-tokenized shared cognitive model.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceCognitiveModel {
    heads: SequenceCognitiveHeads,
    tokenizer: ByteTokenizer,
    max_retrieval_candidates: usize,
}

impl SequenceCognitiveModel {
    pub(crate) fn from_heads(
        heads: SequenceCognitiveHeads,
        max_retrieval_candidates: usize,
    ) -> Result<Self, SequenceCognitiveModelError> {
        if heads.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
            || !(2..=MAX_SEQUENCE_RETRIEVAL_CANDIDATES).contains(&max_retrieval_candidates)
        {
            return Err(SequenceCognitiveModelError::InvalidConfig);
        }
        let tokenizer = ByteTokenizer::try_new(heads.config().encoder.max_tokens)?;
        Ok(Self {
            heads,
            tokenizer,
            max_retrieval_candidates,
        })
    }

    #[must_use]
    pub const fn heads(&self) -> &SequenceCognitiveHeads {
        &self.heads
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.tokenizer.max_tokens()
    }

    #[must_use]
    pub const fn max_retrieval_candidates(&self) -> usize {
        self.max_retrieval_candidates
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.heads.parameter_count()
    }

    pub fn classification_probabilities(
        &self,
        payload: &[u8],
    ) -> Result<Vec<f32>, SequenceCognitiveModelError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.heads.classification_probabilities(&tokens)?)
    }

    pub fn classify(&self, payload: &[u8]) -> Result<usize, SequenceCognitiveModelError> {
        let probabilities = self.classification_probabilities(payload)?;
        let mut best_index = 0usize;
        let mut best_value = probabilities[0];
        for (index, &value) in probabilities.iter().enumerate().skip(1) {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        Ok(best_index)
    }

    pub fn preference_score(&self, payload: &[u8]) -> Result<f32, SequenceCognitiveModelError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.heads.preference_score(&tokens)?)
    }

    pub fn preference_compare(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<Ordering, SequenceCognitiveModelError> {
        Ok(self
            .preference_score(left)?
            .total_cmp(&self.preference_score(right)?))
    }

    pub fn preference_rank(
        &self,
        candidates: &[&[u8]],
    ) -> Result<Vec<usize>, SequenceCognitiveModelError> {
        if candidates.len() > MAX_NEURAL_RANK_CANDIDATES {
            return Err(SequenceCognitiveModelError::TooManyRankCandidates {
                actual: candidates.len(),
                maximum: MAX_NEURAL_RANK_CANDIDATES,
            });
        }
        let mut scored = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            scored.push((index, self.preference_score(candidate)?));
        }
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(scored.into_iter().map(|(index, _)| index).collect())
    }

    pub fn symbolic_satisfactions(
        &self,
        payload: &[u8],
    ) -> Result<Vec<f32>, SequenceCognitiveModelError> {
        let tokens = self.tokenizer.encode(payload)?;
        Ok(self.heads.symbolic_satisfactions(&tokens)?)
    }

    pub fn contradiction_probability(
        &self,
        left: &[u8],
        right: &[u8],
    ) -> Result<f32, SequenceCognitiveModelError> {
        let pair = self.tokenizer.encode_pair(left, right)?;
        Ok(self.heads.contradiction_probabilities(&pair)?[1])
    }

    pub fn retrieval_similarities(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<Vec<f32>, SequenceCognitiveModelError> {
        validate_retrieval_count(candidates.len(), self.max_retrieval_candidates)?;
        let query = self.tokenizer.encode(query)?;
        let mut encoded = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            encoded.push(self.tokenizer.encode(candidate)?);
        }
        let refs: Vec<&[u16]> = encoded.iter().map(Vec::as_slice).collect();
        Ok(self.heads.retrieval_similarities(&query, &refs)?)
    }

    pub fn retrieval_select_best(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<usize, SequenceCognitiveModelError> {
        validate_retrieval_count(candidates.len(), self.max_retrieval_candidates)?;
        let query = self.tokenizer.encode(query)?;
        let mut encoded = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            encoded.push(self.tokenizer.encode(candidate)?);
        }
        let refs: Vec<&[u16]> = encoded.iter().map(Vec::as_slice).collect();
        Ok(self.heads.retrieval_select_best(&query, &refs)?)
    }
}

/// Deterministic trainer for one explicitly shared multi-task representation.
#[derive(Clone, Copy, Debug)]
pub struct SequenceCognitiveTrainer {
    pub config: SequenceCognitiveModelConfig,
}

impl SequenceCognitiveTrainer {
    pub fn try_new(
        config: SequenceCognitiveModelConfig,
    ) -> Result<Self, SequenceCognitiveModelError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        examples: &[SequenceCognitiveExample],
    ) -> Result<
        (SequenceCognitiveModel, SequenceCognitiveTrainingReport),
        SequenceCognitiveModelError,
    > {
        validate_examples(examples, self.config)?;
        let tokenizer = ByteTokenizer::try_new(self.config.cognitive.encoder.max_tokens)?;

        // Complete corpus preflight: every single and paired framing is checked
        // before optimizer construction, as are task target/candidate bounds.
        for example in examples {
            let _ = tokenizer.encode(&example.classification_payload)?;
            let _ = tokenizer.encode(&example.preferred)?;
            let _ = tokenizer.encode(&example.dispreferred)?;
            let _ = tokenizer.encode(&example.symbolic_payload)?;
            let _ =
                tokenizer.encode_pair(&example.contradiction_left, &example.contradiction_right)?;
            let _ = tokenizer.encode(&example.retrieval_query)?;
            for candidate in &example.retrieval_candidates {
                let _ = tokenizer.encode(candidate)?;
            }
        }

        let mut heads = SequenceCognitiveHeads::try_new(self.config.cognitive)?;
        let parameters = heads.parameter_count();
        let mut optimizer = SequenceCognitiveAdamW::try_new(self.config.learning_rate, &heads)?;
        let mut final_sum = SequenceCognitiveLossSums::default();

        for epoch in 0..self.config.epochs {
            let last_epoch = epoch + 1 == self.config.epochs;
            if last_epoch {
                final_sum = SequenceCognitiveLossSums::default();
            }
            for example in examples {
                let classification = tokenizer.encode(&example.classification_payload)?;
                let preferred = tokenizer.encode(&example.preferred)?;
                let dispreferred = tokenizer.encode(&example.dispreferred)?;
                let symbolic = tokenizer.encode(&example.symbolic_payload)?;
                let symbolic_targets: Vec<f32> = example
                    .rule_satisfied
                    .iter()
                    .map(|&satisfied| if satisfied { 1.0 } else { 0.0 })
                    .collect();
                let contradiction = tokenizer
                    .encode_pair(&example.contradiction_left, &example.contradiction_right)?;
                let retrieval_query = tokenizer.encode(&example.retrieval_query)?;
                let mut retrieval_candidates =
                    Vec::with_capacity(example.retrieval_candidates.len());
                for candidate in &example.retrieval_candidates {
                    retrieval_candidates.push(tokenizer.encode(candidate)?);
                }
                let retrieval_refs: Vec<&[u16]> =
                    retrieval_candidates.iter().map(Vec::as_slice).collect();

                let report = heads.train_joint_step(
                    &mut optimizer,
                    SequenceCognitiveBatch {
                        classification: CognitiveClassification {
                            token_ids: &classification,
                            target_class: example.classification_target,
                        },
                        preference: CognitivePreference {
                            preferred: &preferred,
                            dispreferred: &dispreferred,
                            margin: self.config.preference_margin,
                        },
                        symbolic: CognitiveSymbolic {
                            token_ids: &symbolic,
                            targets: &symbolic_targets,
                        },
                        contradiction: CognitiveContradiction {
                            pair_token_ids: &contradiction,
                            contradicts: example.contradicts,
                        },
                        retrieval: CognitiveRetrieval {
                            query: &retrieval_query,
                            candidates: &retrieval_refs,
                            positive_idx: example.retrieval_positive_idx,
                            temperature: self.config.retrieval_temperature,
                        },
                    },
                    self.config.loss_weights,
                )?;
                if last_epoch {
                    final_sum.add(report)?;
                }
            }
        }

        let divisor = examples.len() as f32;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(SequenceCognitiveModelError::InvalidConfig);
        }
        let model =
            SequenceCognitiveModel::from_heads(heads, self.config.max_retrieval_candidates)?;
        Ok((
            model,
            SequenceCognitiveTrainingReport {
                examples: examples.len(),
                epochs: self.config.epochs,
                parameters,
                final_mean_classification: final_sum.classification / divisor,
                final_mean_preference: final_sum.preference / divisor,
                final_mean_symbolic: final_sum.symbolic / divisor,
                final_mean_contradiction: final_sum.contradiction / divisor,
                final_mean_retrieval: final_sum.retrieval / divisor,
                final_mean_weighted_total: final_sum.weighted_total / divisor,
            },
        ))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SequenceCognitiveLossSums {
    classification: f32,
    preference: f32,
    symbolic: f32,
    contradiction: f32,
    retrieval: f32,
    weighted_total: f32,
}

impl SequenceCognitiveLossSums {
    fn add(
        &mut self,
        report: SequenceCognitiveLossReport,
    ) -> Result<(), SequenceCognitiveModelError> {
        self.classification += report.classification;
        self.preference += report.preference;
        self.symbolic += report.symbolic;
        self.contradiction += report.contradiction;
        self.retrieval += report.retrieval;
        self.weighted_total += report.weighted_total;
        for value in [
            self.classification,
            self.preference,
            self.symbolic,
            self.contradiction,
            self.retrieval,
            self.weighted_total,
        ] {
            if !value.is_finite() {
                return Err(SequenceCognitiveModelError::SciRust(
                    SciRustError::NonFinite,
                ));
            }
        }
        Ok(())
    }
}

/// Frozen read-only multi-signal facade. It exposes classification and ranking
/// only; no proposal, persistence, tool or rule authority is granted.
#[derive(Clone, Debug)]
pub struct SciRustSequenceCognitiveReadOnlyModel {
    pub model: Arc<SequenceCognitiveModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SEQUENCE_COGNITIVE_READ_ONLY_CAPS: &[ReadOnlyCapability] =
    &[ReadOnlyCapability::Classify, ReadOnlyCapability::Rank];

impl SciRustSequenceCognitiveReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequenceCognitiveModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SEQUENCE_COGNITIVE_READ_ONLY_CAPS,
        }
    }

    pub fn classify(&self, payload: &[u8]) -> Result<usize, BackendError> {
        self.model.classify(payload).map_err(map_backend_error)
    }

    pub fn preference_rank(&self, candidates: &[&[u8]]) -> Result<Vec<usize>, BackendError> {
        self.model
            .preference_rank(candidates)
            .map_err(map_backend_error)
    }

    pub fn symbolic_satisfactions(&self, payload: &[u8]) -> Result<Vec<f32>, BackendError> {
        self.model
            .symbolic_satisfactions(payload)
            .map_err(map_backend_error)
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

    pub fn retrieval_select_best(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<usize, BackendError> {
        self.model
            .retrieval_select_best(query, candidates)
            .map_err(map_backend_error)
    }
}

impl ModelBackend for SciRustSequenceCognitiveReadOnlyModel {
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

fn validate_config(
    config: SequenceCognitiveModelConfig,
) -> Result<(), SequenceCognitiveModelError> {
    let weights = [
        config.loss_weights.classification,
        config.loss_weights.preference,
        config.loss_weights.symbolic,
        config.loss_weights.contradiction,
        config.loss_weights.retrieval,
    ];
    let weights_valid = weights
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0)
        && weights.iter().any(|weight| *weight > 0.0);
    if config.cognitive.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(3..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.cognitive.encoder.max_tokens)
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
        || !config.preference_margin.is_finite()
        || config.preference_margin < 0.0
        || !config.retrieval_temperature.is_finite()
        || config.retrieval_temperature <= 0.0
        || !(2..=MAX_SEQUENCE_RETRIEVAL_CANDIDATES).contains(&config.max_retrieval_candidates)
        || !weights_valid
    {
        return Err(SequenceCognitiveModelError::InvalidConfig);
    }
    let _ = ByteTokenizer::try_new(config.cognitive.encoder.max_tokens)?;
    let _ = SequenceCognitiveHeads::try_new(config.cognitive)?;
    Ok(())
}

fn validate_examples(
    examples: &[SequenceCognitiveExample],
    config: SequenceCognitiveModelConfig,
) -> Result<(), SequenceCognitiveModelError> {
    if examples.is_empty() {
        return Err(SequenceCognitiveModelError::EmptyTrainingSet);
    }
    if examples.len() > MAX_SEQUENCE_COGNITIVE_EXAMPLES {
        return Err(SequenceCognitiveModelError::TooManyExamples {
            actual: examples.len(),
            maximum: MAX_SEQUENCE_COGNITIVE_EXAMPLES,
        });
    }
    for example in examples {
        if example.classification_target >= config.cognitive.num_classes {
            return Err(
                SequenceCognitiveModelError::ClassificationTargetOutOfRange {
                    target: example.classification_target,
                    classes: config.cognitive.num_classes,
                },
            );
        }
        if example.rule_satisfied.len() != config.cognitive.num_rules {
            return Err(SequenceCognitiveModelError::RuleTargetCountMismatch {
                actual: example.rule_satisfied.len(),
                expected: config.cognitive.num_rules,
            });
        }
        validate_retrieval_count(
            example.retrieval_candidates.len(),
            config.max_retrieval_candidates,
        )?;
        if example.retrieval_positive_idx >= example.retrieval_candidates.len() {
            return Err(SequenceCognitiveModelError::InvalidRetrievalPositiveIndex {
                idx: example.retrieval_positive_idx,
                len: example.retrieval_candidates.len(),
            });
        }
    }
    Ok(())
}

fn validate_retrieval_count(
    actual: usize,
    maximum: usize,
) -> Result<(), SequenceCognitiveModelError> {
    if actual < 2 {
        return Err(SequenceCognitiveModelError::TooFewRetrievalCandidates { actual });
    }
    if actual > maximum {
        return Err(SequenceCognitiveModelError::TooManyRetrievalCandidates { actual, maximum });
    }
    Ok(())
}

fn map_backend_error(error: SequenceCognitiveModelError) -> BackendError {
    match error {
        SequenceCognitiveModelError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded {
            ..
        })
        | SequenceCognitiveModelError::TooManyExamples { .. }
        | SequenceCognitiveModelError::TooManyRetrievalCandidates { .. }
        | SequenceCognitiveModelError::TooManyRankCandidates { .. } => BackendError::InputTooLarge,
        SequenceCognitiveModelError::InvalidConfig
        | SequenceCognitiveModelError::EmptyTrainingSet
        | SequenceCognitiveModelError::ClassificationTargetOutOfRange { .. }
        | SequenceCognitiveModelError::RuleTargetCountMismatch { .. }
        | SequenceCognitiveModelError::TooFewRetrievalCandidates { .. }
        | SequenceCognitiveModelError::InvalidRetrievalPositiveIndex { .. }
        | SequenceCognitiveModelError::Tokenizer(_)
        | SequenceCognitiveModelError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequenceCognitiveModelConfig {
        SequenceCognitiveModelConfig {
            cognitive: SequenceCognitiveConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 48,
                    embedding_dim: 8,
                    hidden_dim: 8,
                    seed: 1409,
                },
                num_classes: 3,
                num_rules: 2,
                classification_seed: 1423,
                preference_seed: 1427,
                symbolic_seed: 1429,
                contradiction_seed: 1433,
            },
            epochs: 8,
            learning_rate: 0.01,
            loss_weights: SequenceCognitiveLossWeights::default(),
            preference_margin: 2.0,
            retrieval_temperature: 0.5,
            max_retrieval_candidates: 2,
        }
    }

    fn example() -> SequenceCognitiveExample {
        SequenceCognitiveExample {
            classification_payload: b"accepted feedback".to_vec(),
            classification_target: 1,
            preferred: b"preferred answer".to_vec(),
            dispreferred: b"rejected answer".to_vec(),
            symbolic_payload: b"rule evidence".to_vec(),
            rule_satisfied: vec![true, false],
            contradiction_left: b"sky is blue".to_vec(),
            contradiction_right: b"sky is green".to_vec(),
            contradicts: true,
            retrieval_query: b"alpha query".to_vec(),
            retrieval_candidates: vec![b"alpha memory".to_vec(), b"omega memory".to_vec()],
            retrieval_positive_idx: 0,
        }
    }

    #[test]
    fn deterministic_joint_training_repeats_exactly() {
        let trainer = SequenceCognitiveTrainer::try_new(config()).expect("trainer");
        let examples = vec![example()];
        let (left, left_report) = trainer.train(&examples).expect("left");
        let (right, right_report) = trainer.train(&examples).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 1);
        assert!(left_report.final_mean_weighted_total.is_finite());
    }

    #[test]
    fn complete_preflight_rejects_hostile_targets_and_candidates() {
        let trainer = SequenceCognitiveTrainer::try_new(config()).expect("trainer");
        let mut invalid = example();
        invalid.rule_satisfied = vec![true];
        assert!(matches!(
            trainer.train(&[invalid]),
            Err(SequenceCognitiveModelError::RuleTargetCountMismatch { .. })
        ));

        let mut invalid = example();
        invalid.retrieval_positive_idx = 2;
        assert!(matches!(
            trainer.train(&[invalid]),
            Err(SequenceCognitiveModelError::InvalidRetrievalPositiveIndex { .. })
        ));
    }

    #[test]
    fn frozen_joint_backend_exposes_only_classify_and_rank() {
        let trainer = SequenceCognitiveTrainer::try_new(config()).expect("trainer");
        let (model, _) = trainer.train(&[example()]).expect("model");
        let mut readonly = SciRustSequenceCognitiveReadOnlyModel::from_trained(model);
        assert_eq!(
            readonly.capabilities,
            &[ReadOnlyCapability::Classify, ReadOnlyCapability::Rank]
        );
        assert!(readonly.classify(b"accepted feedback").is_ok());
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
