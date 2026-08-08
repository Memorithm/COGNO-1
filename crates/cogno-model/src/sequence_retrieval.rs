//! Byte-tokenized memory/rule retrieval model backed by the connected
//! `cogno-scirust` sequence InfoNCE retriever.
//!
//! This bridge trains only a soft relevance representation. Returned
//! similarities and indices remain non-trust data: rule adoption, evidence
//! provenance and every hard safety/policy decision stay outside the model.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::neural::MAX_NEURAL_EPOCHS;
use crate::readonly::ReadOnlyCapability;
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_scirust::{
    SciRustError, SequenceEncoderConfig, SequenceRetriever, SequenceRetrieverAdamW,
    SequenceRetrieverConfig, MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
};
use std::sync::Arc;

/// Maximum explicit retrieval examples accepted by one deterministic run.
pub const MAX_SEQUENCE_RETRIEVAL_EXAMPLES: usize = 4_096;

/// One supervised retrieval observation: `positive_idx` identifies which
/// candidate should be most relevant to the query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceRetrievalExample {
    pub query: Vec<u8>,
    pub candidates: Vec<Vec<u8>>,
    pub positive_idx: usize,
}

impl SequenceRetrievalExample {
    #[must_use]
    pub fn new(query: Vec<u8>, candidates: Vec<Vec<u8>>, positive_idx: usize) -> Self {
        Self {
            query,
            candidates,
            positive_idx,
        }
    }
}

/// Bounded deterministic InfoNCE training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceRetrievalConfig {
    pub retriever: SequenceRetrieverConfig,
    pub epochs: u32,
    pub learning_rate: f32,
}

impl Default for SequenceRetrievalConfig {
    fn default() -> Self {
        Self {
            retriever: SequenceRetrieverConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                temperature: 0.2,
                max_candidates: 8,
            },
            epochs: 24,
            learning_rate: 0.01,
        }
    }
}

/// Fail-closed model-facing retrieval errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceRetrievalError {
    InvalidConfig,
    EmptyTrainingSet,
    TooManyExamples { actual: usize, maximum: usize },
    TooFewCandidates { actual: usize },
    TooManyCandidates { actual: usize, maximum: usize },
    InvalidPositiveIndex { idx: usize, len: usize },
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
}

impl From<ByteTokenizerError> for SequenceRetrievalError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceRetrievalError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

/// Deterministic summary of one retrieval training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceRetrievalTrainingReport {
    pub examples: usize,
    pub epochs: u32,
    pub parameters: usize,
    pub final_mean_loss: f32,
}

/// Frozen byte-tokenized relevance model.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceRetrievalModel {
    retriever: SequenceRetriever,
    tokenizer: ByteTokenizer,
}

impl SequenceRetrievalModel {
    fn from_retriever(retriever: SequenceRetriever) -> Result<Self, SequenceRetrievalError> {
        if retriever.config().encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE {
            return Err(SequenceRetrievalError::InvalidConfig);
        }
        let tokenizer = ByteTokenizer::try_new(retriever.config().encoder.max_tokens)?;
        Ok(Self {
            retriever,
            tokenizer,
        })
    }

    #[must_use]
    pub const fn retriever(&self) -> &SequenceRetriever {
        &self.retriever
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.tokenizer.max_tokens()
    }

    #[must_use]
    pub const fn max_candidates(&self) -> usize {
        self.retriever.config().max_candidates
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.retriever.parameter_count()
    }

    /// Return non-authoritative relevance similarities for already-provided
    /// candidate payloads.
    pub fn similarities(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<Vec<f32>, SequenceRetrievalError> {
        validate_candidate_count(candidates.len(), self.max_candidates())?;
        let query = self.tokenizer.encode(query)?;
        let mut encoded = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            encoded.push(self.tokenizer.encode(candidate)?);
        }
        let refs: Vec<&[u16]> = encoded.iter().map(Vec::as_slice).collect();
        Ok(self.retriever.similarities(&query, &refs)?)
    }

    /// Return the best candidate index. Exact ties preserve the first index.
    pub fn select_best(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<usize, SequenceRetrievalError> {
        validate_candidate_count(candidates.len(), self.max_candidates())?;
        let query = self.tokenizer.encode(query)?;
        let mut encoded = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            encoded.push(self.tokenizer.encode(candidate)?);
        }
        let refs: Vec<&[u16]> = encoded.iter().map(Vec::as_slice).collect();
        Ok(self.retriever.select_best(&query, &refs)?)
    }
}

/// Deterministic trainer for explicit query/candidate relevance examples.
#[derive(Clone, Copy, Debug)]
pub struct SequenceRetrievalTrainer {
    pub config: SequenceRetrievalConfig,
}

impl SequenceRetrievalTrainer {
    pub fn try_new(config: SequenceRetrievalConfig) -> Result<Self, SequenceRetrievalError> {
        validate_config(config)?;
        Ok(Self { config })
    }

    pub fn train(
        &self,
        examples: &[SequenceRetrievalExample],
    ) -> Result<(SequenceRetrievalModel, SequenceRetrievalTrainingReport), SequenceRetrievalError>
    {
        validate_examples(examples, self.config.retriever.max_candidates)?;
        let tokenizer = ByteTokenizer::try_new(self.config.retriever.encoder.max_tokens)?;

        // Preflight every payload before any optimization work. The model is
        // local either way, but failing before training makes hostile-input
        // behavior deterministic and avoids wasted bounded compute.
        for example in examples {
            let _ = tokenizer.encode(&example.query)?;
            for candidate in &example.candidates {
                let _ = tokenizer.encode(candidate)?;
            }
        }

        let mut retriever = SequenceRetriever::try_new(self.config.retriever)?;
        let parameters = retriever.parameter_count();
        let mut optimizer = SequenceRetrieverAdamW::try_new(self.config.learning_rate, &retriever)?;
        let mut final_loss_sum = 0.0f32;

        for epoch in 0..self.config.epochs {
            let last_epoch = epoch + 1 == self.config.epochs;
            if last_epoch {
                final_loss_sum = 0.0;
            }
            for example in examples {
                let query = tokenizer.encode(&example.query)?;
                let mut candidates = Vec::with_capacity(example.candidates.len());
                for candidate in &example.candidates {
                    candidates.push(tokenizer.encode(candidate)?);
                }
                let refs: Vec<&[u16]> = candidates.iter().map(Vec::as_slice).collect();
                let loss = retriever.train_step(
                    &mut optimizer,
                    &query,
                    &refs,
                    example.positive_idx,
                )?;
                if last_epoch {
                    final_loss_sum += loss;
                    if !final_loss_sum.is_finite() {
                        return Err(SequenceRetrievalError::SciRust(SciRustError::NonFinite));
                    }
                }
            }
        }

        let divisor = examples.len() as f32;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(SequenceRetrievalError::InvalidConfig);
        }
        let final_mean_loss = final_loss_sum / divisor;
        if !final_mean_loss.is_finite() {
            return Err(SequenceRetrievalError::SciRust(SciRustError::NonFinite));
        }
        let model = SequenceRetrievalModel::from_retriever(retriever)?;
        Ok((
            model,
            SequenceRetrievalTrainingReport {
                examples: examples.len(),
                epochs: self.config.epochs,
                parameters,
                final_mean_loss,
            },
        ))
    }
}

/// Frozen read-only facade for relevance ranking. It exposes no proposal or
/// mutation authority.
#[derive(Clone, Debug)]
pub struct SciRustSequenceRetrievalReadOnlyModel {
    pub model: Arc<SequenceRetrievalModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const SEQUENCE_RETRIEVAL_READ_ONLY_CAPS: &[ReadOnlyCapability] = &[ReadOnlyCapability::Rank];

impl SciRustSequenceRetrievalReadOnlyModel {
    #[must_use]
    pub fn from_trained(model: SequenceRetrievalModel) -> Self {
        Self {
            model: Arc::new(model),
            capabilities: SEQUENCE_RETRIEVAL_READ_ONLY_CAPS,
        }
    }

    pub fn similarities(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<Vec<f32>, BackendError> {
        self.model
            .similarities(query, candidates)
            .map_err(map_backend_error)
    }

    pub fn select_best(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<usize, BackendError> {
        self.model
            .select_best(query, candidates)
            .map_err(map_backend_error)
    }
}

impl ModelBackend for SciRustSequenceRetrievalReadOnlyModel {
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

fn validate_config(config: SequenceRetrievalConfig) -> Result<(), SequenceRetrievalError> {
    if config.retriever.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.retriever.encoder.max_tokens)
        || config.retriever.max_candidates > MAX_SEQUENCE_RETRIEVAL_CANDIDATES
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(SequenceRetrievalError::InvalidConfig);
    }
    let _ = ByteTokenizer::try_new(config.retriever.encoder.max_tokens)?;
    let _ = SequenceRetriever::try_new(config.retriever)?;
    Ok(())
}

fn validate_examples(
    examples: &[SequenceRetrievalExample],
    maximum_candidates: usize,
) -> Result<(), SequenceRetrievalError> {
    if examples.is_empty() {
        return Err(SequenceRetrievalError::EmptyTrainingSet);
    }
    if examples.len() > MAX_SEQUENCE_RETRIEVAL_EXAMPLES {
        return Err(SequenceRetrievalError::TooManyExamples {
            actual: examples.len(),
            maximum: MAX_SEQUENCE_RETRIEVAL_EXAMPLES,
        });
    }
    for example in examples {
        validate_candidate_count(example.candidates.len(), maximum_candidates)?;
        if example.positive_idx >= example.candidates.len() {
            return Err(SequenceRetrievalError::InvalidPositiveIndex {
                idx: example.positive_idx,
                len: example.candidates.len(),
            });
        }
    }
    Ok(())
}

fn validate_candidate_count(
    actual: usize,
    maximum: usize,
) -> Result<(), SequenceRetrievalError> {
    if actual < 2 {
        return Err(SequenceRetrievalError::TooFewCandidates { actual });
    }
    if actual > maximum {
        return Err(SequenceRetrievalError::TooManyCandidates { actual, maximum });
    }
    Ok(())
}

fn map_backend_error(error: SequenceRetrievalError) -> BackendError {
    match error {
        SequenceRetrievalError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        | SequenceRetrievalError::TooManyExamples { .. }
        | SequenceRetrievalError::TooManyCandidates { .. } => BackendError::InputTooLarge,
        SequenceRetrievalError::InvalidConfig
        | SequenceRetrievalError::EmptyTrainingSet
        | SequenceRetrievalError::TooFewCandidates { .. }
        | SequenceRetrievalError::InvalidPositiveIndex { .. }
        | SequenceRetrievalError::Tokenizer(_)
        | SequenceRetrievalError::SciRust(_) => BackendError::HostileArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequenceRetrievalConfig {
        SequenceRetrievalConfig {
            retriever: SequenceRetrieverConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 32,
                    embedding_dim: 8,
                    hidden_dim: 8,
                    seed: 701,
                },
                temperature: 0.5,
                max_candidates: 2,
            },
            epochs: 64,
            learning_rate: 0.02,
        }
    }

    fn examples() -> Vec<SequenceRetrievalExample> {
        vec![
            SequenceRetrievalExample::new(
                b"alpha query".to_vec(),
                vec![b"alpha memory".to_vec(), b"omega memory".to_vec()],
                0,
            ),
            SequenceRetrievalExample::new(
                b"omega query".to_vec(),
                vec![b"alpha memory".to_vec(), b"omega memory".to_vec()],
                1,
            ),
        ]
    }

    #[test]
    fn deterministic_training_learns_byte_tokenized_retrieval() {
        let trainer = SequenceRetrievalTrainer::try_new(config()).expect("trainer");
        let (left, left_report) = trainer.train(&examples()).expect("left");
        let (right, right_report) = trainer.train(&examples()).expect("right");
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert_eq!(left_report.examples, 2);
        assert_eq!(
            left.select_best(b"alpha query", &[b"alpha memory", b"omega memory"])
                .expect("alpha selection"),
            0
        );
        assert_eq!(
            left.select_best(b"omega query", &[b"alpha memory", b"omega memory"])
                .expect("omega selection"),
            1
        );
    }

    #[test]
    fn hostile_token_lengths_fail_before_training() {
        let trainer = SequenceRetrievalTrainer::try_new(config()).expect("trainer");
        let oversized = vec![b'x'; 31];
        let error = trainer
            .train(&[SequenceRetrievalExample::new(
                oversized,
                vec![b"left".to_vec(), b"right".to_vec()],
                0,
            )])
            .expect_err("framing exceeds max tokens");
        assert!(matches!(
            error,
            SequenceRetrievalError::Tokenizer(ByteTokenizerError::TokenCapacityExceeded { .. })
        ));
    }

    #[test]
    fn invalid_positive_index_fails_closed_before_training() {
        let trainer = SequenceRetrievalTrainer::try_new(config()).expect("trainer");
        assert!(matches!(
            trainer.train(&[SequenceRetrievalExample::new(
                b"query".to_vec(),
                vec![b"left".to_vec(), b"right".to_vec()],
                2,
            )]),
            Err(SequenceRetrievalError::InvalidPositiveIndex { idx: 2, len: 2 })
        ));
    }

    #[test]
    fn frozen_retrieval_backend_is_rank_only_and_never_proposes() {
        let trainer = SequenceRetrievalTrainer::try_new(config()).expect("trainer");
        let (model, _) = trainer.train(&examples()).expect("model");
        let mut readonly = SciRustSequenceRetrievalReadOnlyModel::from_trained(model);
        assert_eq!(readonly.capabilities, &[ReadOnlyCapability::Rank]);
        assert_eq!(
            readonly.select_best(b"alpha query", &[b"alpha memory", b"omega memory"]),
            Ok(0)
        );
        assert_eq!(readonly.next_proposal(), Err(BackendError::ReadOnlyViolation));
        let info = readonly.info();
        assert!(info.read_only);
        assert!(info.differentiable);
        assert!(!info.tools_enabled);
    }
}
