//! InfoNCE sequence retriever whose query and candidate representations share
//! one trainable [`crate::SequenceEncoder`].
//!
//! Every query/candidate view is appended to one bounded tape from the same
//! frozen parameter values. Dot-product similarities stay connected through
//! [`crate::InfoNCE::loss_similarity_vars`]. Reverse-mode gradients from the
//! query and every candidate view are summed before one AdamW update, giving
//! COGNO a deterministic memory/rule-selection training path without detaching
//! the shared sequence representation.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    AdamW, InfoNCE, Optimizer, SequenceEncoder, SequenceEncoderConfig, SequenceEncoderGradients,
    SequenceEncoderGraph, Tape, MAX_SEQUENCE_PARAMETERS, SEQUENCE_ENCODER_TAPE_NODES,
};

/// Maximum candidate memories/rules in one bounded InfoNCE comparison.
pub const MAX_SEQUENCE_RETRIEVAL_CANDIDATES: usize = 32;
/// Maximum trainable scalars: the retriever owns only the shared encoder.
pub const MAX_SEQUENCE_RETRIEVER_PARAMETERS: usize = MAX_SEQUENCE_PARAMETERS;
/// Tape bound for query + max candidates + dot products + connected InfoNCE.
pub const SEQUENCE_RETRIEVER_TAPE_NODES: usize =
    SEQUENCE_ENCODER_TAPE_NODES * (MAX_SEQUENCE_RETRIEVAL_CANDIDATES + 1)
        + 2 * MAX_SEQUENCE_RETRIEVAL_CANDIDATES
        + 8;

/// Configuration of the shared sequence retriever.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceRetrieverConfig {
    pub encoder: SequenceEncoderConfig,
    pub temperature: f32,
    pub max_candidates: usize,
}

impl SequenceRetrieverConfig {
    fn validate(self) -> SciRustResult<()> {
        if self.max_candidates > MAX_SEQUENCE_RETRIEVAL_CANDIDATES {
            return Err(SciRustError::CapacityExceeded {
                requested: self.max_candidates,
                maximum: MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
            });
        }
        let _ = self.encoder.parameter_count()?;
        InfoNCE::try_new(
            self.temperature,
            self.max_candidates,
            self.max_candidates.max(1),
        )?;
        Ok(())
    }
}

/// Exact shared-encoder gradients after one query/candidate InfoNCE batch.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceRetrieverGradients {
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    mixing_weights: Vec<f32>,
}

impl SequenceRetrieverGradients {
    fn zeros(encoder: &SequenceEncoder) -> Self {
        Self {
            token_embeddings: vec![0.0; encoder.token_embeddings().len()],
            position_embeddings: vec![0.0; encoder.position_embeddings().len()],
            mixing_weights: vec![0.0; encoder.mixing_weights().len()],
        }
    }

    fn accumulate(&mut self, source: &SequenceEncoderGradients) -> SciRustResult<()> {
        add_assign_checked(&mut self.token_embeddings, source.token_embeddings())?;
        add_assign_checked(&mut self.position_embeddings, source.position_embeddings())?;
        add_assign_checked(&mut self.mixing_weights, source.mixing_weights())?;
        Ok(())
    }

    #[must_use]
    pub fn token_embeddings(&self) -> &[f32] {
        &self.token_embeddings
    }

    #[must_use]
    pub fn position_embeddings(&self) -> &[f32] {
        &self.position_embeddings
    }

    #[must_use]
    pub fn mixing_weights(&self) -> &[f32] {
        &self.mixing_weights
    }
}

/// Bounded retriever with one encoder shared by query and candidate views.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceRetriever {
    config: SequenceRetrieverConfig,
    encoder: SequenceEncoder,
}

impl SequenceRetriever {
    /// Deterministically initialize the shared encoder from the explicit seed.
    pub fn try_new(config: SequenceRetrieverConfig) -> SciRustResult<Self> {
        config.validate()?;
        let encoder = SequenceEncoder::try_new(config.encoder)?;
        Self::from_encoder(config, encoder)
    }

    /// Reconstruct around an already-validated encoder with exact config match.
    pub fn from_encoder(
        config: SequenceRetrieverConfig,
        encoder: SequenceEncoder,
    ) -> SciRustResult<Self> {
        config.validate()?;
        if encoder.config() != config.encoder {
            return Err(SciRustError::Shape {
                lhs: vec![encoder.config().hidden_dim],
                rhs: vec![config.encoder.hidden_dim],
            });
        }
        Ok(Self { config, encoder })
    }

    #[must_use]
    pub const fn config(&self) -> SequenceRetrieverConfig {
        self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &SequenceEncoder {
        &self.encoder
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.encoder.parameter_count()
    }

    /// Deterministic dot-product similarities for inference/ranking.
    pub fn similarities(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
    ) -> SciRustResult<Vec<f32>> {
        self.validate_candidates(candidates.len())?;
        let query_embedding = self.encoder.forward(query)?;
        let mut similarities = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let candidate_embedding = self.encoder.forward(candidate)?;
            similarities.push(dot(&query_embedding, &candidate_embedding)?);
        }
        Ok(similarities)
    }

    /// Index of the highest-similarity candidate; ties retain the first index.
    pub fn select_best(&self, query: &[u16], candidates: &[&[u16]]) -> SciRustResult<usize> {
        let similarities = self.similarities(query, candidates)?;
        let mut best_index = 0usize;
        let mut best_value = similarities[0];
        for (index, &value) in similarities.iter().enumerate().skip(1) {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        Ok(best_index)
    }

    /// Connected InfoNCE loss and gradients summed over the shared query/key
    /// encoder parameters.
    pub fn loss_and_gradients(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
        positive_idx: usize,
    ) -> SciRustResult<(f32, SequenceRetrieverGradients)> {
        self.validate_batch(candidates.len(), positive_idx)?;
        let max_elements = self.required_max_elements(query, candidates)?;
        let mut tape = Tape::new(SEQUENCE_RETRIEVER_TAPE_NODES, max_elements);

        let query_graph = self.encoder.append_to_tape(&mut tape, query)?;
        let mut candidate_graphs = Vec::with_capacity(candidates.len());
        let mut similarities = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let candidate_graph = self.encoder.append_to_tape(&mut tape, candidate)?;
            let product = tape.mul(query_graph.pooled(), candidate_graph.pooled())?;
            let similarity = tape.sum(product)?;
            candidate_graphs.push(candidate_graph);
            similarities.push(similarity);
        }

        let loss = InfoNCE::try_new(
            self.config.temperature,
            self.config.max_candidates,
            max_elements,
        )?
        .loss_similarity_vars(&mut tape, &similarities, positive_idx)?;
        tape.backward(loss)?;
        let loss_value = tape
            .value_of(loss)
            .as_slice()
            .first()
            .copied()
            .ok_or(SciRustError::Empty)?;
        ensure_finite(loss_value)?;

        let mut gradients = SequenceRetrieverGradients::zeros(&self.encoder);
        let query_gradients = self.encoder.gradients_from_tape(&tape, query_graph);
        gradients.accumulate(&query_gradients)?;
        for graph in candidate_graphs {
            let candidate_gradients = self.encoder.gradients_from_tape(&tape, graph);
            gradients.accumulate(&candidate_gradients)?;
        }
        Ok((loss_value, gradients))
    }

    /// One deterministic AdamW update for a query and bounded candidate set.
    pub fn train_step(
        &mut self,
        optimizer: &mut SequenceRetrieverAdamW,
        query: &[u16],
        candidates: &[&[u16]],
        positive_idx: usize,
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients(query, candidates, positive_idx)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn validate_candidates(&self, candidate_count: usize) -> SciRustResult<()> {
        if candidate_count < 2 {
            return Err(SciRustError::Empty);
        }
        if candidate_count > self.config.max_candidates {
            return Err(SciRustError::CapacityExceeded {
                requested: candidate_count,
                maximum: self.config.max_candidates,
            });
        }
        Ok(())
    }

    fn validate_batch(&self, candidate_count: usize, positive_idx: usize) -> SciRustResult<()> {
        self.validate_candidates(candidate_count)?;
        if positive_idx >= candidate_count {
            return Err(SciRustError::Index {
                idx: positive_idx,
                len: candidate_count,
            });
        }
        Ok(())
    }

    fn required_max_elements(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
    ) -> SciRustResult<usize> {
        let mut required = self.encoder.required_max_elements(query.len())?;
        for candidate in candidates {
            required = required.max(self.encoder.required_max_elements(candidate.len())?);
        }
        Ok(required.max(candidates.len()))
    }
}

/// AdamW state for the three tensors of the shared sequence encoder.
#[derive(Clone, Debug)]
pub struct SequenceRetrieverAdamW {
    token_embeddings: AdamW,
    position_embeddings: AdamW,
    mixing_weights: AdamW,
}

impl SequenceRetrieverAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceRetriever) -> SciRustResult<Self> {
        Ok(Self {
            token_embeddings: AdamW::try_new(
                learning_rate,
                model.encoder.token_embeddings().len(),
            )?,
            position_embeddings: AdamW::try_new(
                learning_rate,
                model.encoder.position_embeddings().len(),
            )?,
            mixing_weights: AdamW::try_new(learning_rate, model.encoder.mixing_weights().len())?,
        })
    }

    pub fn step(
        &mut self,
        model: &mut SequenceRetriever,
        gradients: &SequenceRetrieverGradients,
    ) -> SciRustResult<()> {
        let mut token_embeddings = model.encoder.token_embeddings().to_vec();
        let mut position_embeddings = model.encoder.position_embeddings().to_vec();
        let mut mixing_weights = model.encoder.mixing_weights().to_vec();
        self.token_embeddings
            .step(&mut token_embeddings, &gradients.token_embeddings)?;
        self.position_embeddings
            .step(&mut position_embeddings, &gradients.position_embeddings)?;
        self.mixing_weights
            .step(&mut mixing_weights, &gradients.mixing_weights)?;
        model.encoder = SequenceEncoder::from_parts(
            model.config.encoder,
            token_embeddings,
            position_embeddings,
            mixing_weights,
        )?;
        Ok(())
    }
}

fn add_assign_checked(target: &mut [f32], source: &[f32]) -> SciRustResult<()> {
    if target.len() != source.len() {
        return Err(SciRustError::Shape {
            lhs: vec![target.len()],
            rhs: vec![source.len()],
        });
    }
    for (target, &source) in target.iter_mut().zip(source) {
        let value = *target + source;
        ensure_finite(value)?;
        *target = value;
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> SciRustResult<f32> {
    if left.len() != right.len() {
        return Err(SciRustError::Shape {
            lhs: vec![left.len()],
            rhs: vec![right.len()],
        });
    }
    let mut value = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        value += left * right;
    }
    ensure_finite(value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controlled_retriever() -> SequenceRetriever {
        let encoder_config = SequenceEncoderConfig {
            vocab_size: 3,
            max_tokens: 1,
            embedding_dim: 2,
            hidden_dim: 2,
            seed: 0,
        };
        let encoder = SequenceEncoder::from_parts(
            encoder_config,
            vec![1.0, 0.0, 0.8, 0.2, 0.1, 0.9],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .expect("controlled encoder");
        SequenceRetriever::from_encoder(
            SequenceRetrieverConfig {
                encoder: encoder_config,
                temperature: 0.5,
                max_candidates: 2,
            },
            encoder,
        )
        .expect("controlled retriever")
    }

    #[test]
    fn deterministic_initialization_repeats_exactly() {
        let config = SequenceRetrieverConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 16,
                max_tokens: 8,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 503,
            },
            temperature: 0.2,
            max_candidates: 4,
        };
        assert_eq!(
            SequenceRetriever::try_new(config).expect("left"),
            SequenceRetriever::try_new(config).expect("right")
        );
    }

    #[test]
    fn connected_infonce_reaches_shared_encoder_parameters() {
        let model = controlled_retriever();
        let positive = [1u16];
        let negative = [2u16];
        let (_, gradients) = model
            .loss_and_gradients(&[0], &[positive.as_slice(), negative.as_slice()], 0)
            .expect("gradients");
        assert!(gradients
            .token_embeddings()
            .iter()
            .any(|gradient| *gradient != 0.0));
        assert!(gradients
            .mixing_weights()
            .iter()
            .any(|gradient| *gradient != 0.0));
    }

    #[test]
    fn deterministic_training_reduces_infonce_and_preserves_positive_selection() {
        let mut left = controlled_retriever();
        let mut right = controlled_retriever();
        let positive = [1u16];
        let negative = [2u16];
        let candidates = [positive.as_slice(), negative.as_slice()];
        let initial = left
            .loss_and_gradients(&[0], &candidates, 0)
            .expect("initial")
            .0;
        let initial_embeddings = left.encoder().token_embeddings().to_vec();
        let mut left_optimizer = SequenceRetrieverAdamW::try_new(0.02, &left).expect("optimizer");
        let mut right_optimizer = SequenceRetrieverAdamW::try_new(0.02, &right).expect("optimizer");
        for _ in 0..64 {
            left.train_step(&mut left_optimizer, &[0], &candidates, 0)
                .expect("left step");
            right
                .train_step(&mut right_optimizer, &[0], &candidates, 0)
                .expect("right step");
        }
        let final_loss = left
            .loss_and_gradients(&[0], &candidates, 0)
            .expect("final")
            .0;
        assert!(final_loss < initial);
        assert_eq!(left.select_best(&[0], &candidates).expect("selection"), 0);
        assert_ne!(left.encoder().token_embeddings(), initial_embeddings);
        assert_eq!(left, right);
    }

    #[test]
    fn hostile_candidate_sets_and_indices_fail_closed() {
        let model = controlled_retriever();
        let positive = [1u16];
        let negative = [2u16];
        assert!(matches!(
            model.similarities(&[0], &[positive.as_slice()]),
            Err(SciRustError::Empty)
        ));
        assert!(matches!(
            model.loss_and_gradients(
                &[0],
                &[positive.as_slice(), negative.as_slice()],
                2,
            ),
            Err(SciRustError::Index { idx: 2, len: 2 })
        ));
    }
}
