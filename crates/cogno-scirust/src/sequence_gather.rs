//! Sparse selector graph. The default dense graph remains the oracle.
use super::*;

impl SequenceEncoder {
    /// Largest tensor in the opt-in gather graph, excluding dense selectors.
    pub fn required_gather_max_elements(&self, token_count: usize) -> SciRustResult<usize> {
        self.validate_token_count(token_count)?;
        let required = [
            self.token_embeddings.len(),
            self.position_embeddings.len(),
            self.mixing_weights.len(),
            token_count * self.config.embedding_dim,
            token_count * self.config.hidden_dim,
        ]
        .into_iter()
        .max()
        .ok_or(SciRustError::Empty)?;
        validate_bound(required, MAX_SEQUENCE_ACTIVATION_ELEMENTS)?;
        Ok(required)
    }

    /// Opt-in graph with row gather instead of dense selector matmuls.
    /// Parameter layout and downstream pooling are unchanged. Ten tape nodes
    /// replace twelve, and selectors of size `tokens * vocabulary` disappear.
    pub fn append_to_tape_gather(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
    ) -> SciRustResult<SequenceEncoderGraph> {
        self.validate_tokens(token_ids)?;
        let scale = (token_ids.len() as f32).recip();
        self.append_gather_pool(tape, token_ids, vec![scale; token_ids.len()])
    }

    /// Differentiable masked pooling. Weights are fixed external controls;
    /// gradients flow through unmasked token/position embeddings and projection.
    pub fn append_to_tape_weighted(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
        weights: &[f32],
    ) -> SciRustResult<SequenceEncoderGraph> {
        self.validate_tokens(token_ids)?;
        let weights = super::features::normalized_pool_weights(weights, token_ids.len())?;
        self.append_gather_pool(tape, token_ids, weights)
    }

    fn append_gather_pool(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
        weights: Vec<f32>,
    ) -> SciRustResult<SequenceEncoderGraph> {
        let required = self.required_gather_max_elements(token_ids.len())?;
        validate_bound(required, tape.max_elements)?;
        let token_embeddings = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.vocab_size, self.config.embedding_dim])?,
            self.token_embeddings.clone(),
            tape.max_elements,
        )?)?;
        let indices: Vec<_> = token_ids.iter().copied().map(usize::from).collect();
        let token_features = tape.gather_rows(token_embeddings, &indices)?;
        let position_embeddings = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.max_tokens, self.config.embedding_dim])?,
            self.position_embeddings.clone(),
            tape.max_elements,
        )?)?;
        let positions: Vec<_> = (0..token_ids.len()).collect();
        let position_features = tape.gather_rows(position_embeddings, &positions)?;
        let combined = tape.add(token_features, position_features)?;
        let mixing_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.embedding_dim, self.config.hidden_dim])?,
            self.mixing_weights.clone(),
            tape.max_elements,
        )?)?;
        let mixed = tape.matmul(combined, mixing_weights)?;
        let hidden = tape.relu(mixed)?;
        let pooling = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, token_ids.len()])?,
            weights,
            tape.max_elements,
        )?)?;
        let pooled = tape.matmul(pooling, hidden)?;
        Ok(SequenceEncoderGraph {
            token_embeddings,
            position_embeddings,
            mixing_weights,
            pooled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gather_encoder_matches_dense_values_and_gradients_across_shapes() {
        for seed in [0, 7, 42] {
            for ids in [&[0][..], &[2, 1, 2, 0][..]] {
                let model = SequenceEncoder::try_new(SequenceEncoderConfig {
                    vocab_size: 5,
                    max_tokens: 8,
                    embedding_dim: 3,
                    hidden_dim: 7,
                    seed,
                })
                .unwrap();
                let mut dense = Tape::new(20, model.required_max_elements(ids.len()).unwrap());
                let a = model.append_to_tape(&mut dense, ids).unwrap();
                let mut gather =
                    Tape::new(20, model.required_gather_max_elements(ids.len()).unwrap());
                let b = model.append_to_tape_gather(&mut gather, ids).unwrap();
                assert_eq!(dense.value_of(a.pooled), gather.value_of(b.pooled));
                assert_eq!(dense.nodes.len(), 12);
                assert_eq!(gather.nodes.len(), 10);
                let al = dense.sum(a.pooled).unwrap();
                let bl = gather.sum(b.pooled).unwrap();
                dense.backward(al).unwrap();
                gather.backward(bl).unwrap();
                assert_eq!(
                    model.gradients_from_tape(&dense, a),
                    model.gradients_from_tape(&gather, b)
                );
            }
        }
    }
    #[test]
    fn gather_reduces_peak_tensor_bound_without_changing_admission() {
        let model = SequenceEncoder::try_new(SequenceEncoderConfig {
            vocab_size: 384,
            max_tokens: 512,
            embedding_dim: 8,
            hidden_dim: 16,
            seed: 1,
        })
        .unwrap();
        assert_eq!(model.required_max_elements(192).unwrap(), 98304);
        assert_eq!(model.required_gather_max_elements(192).unwrap(), 4096);
        assert!(model.required_gather_max_elements(513).is_err());
    }
}
