//! Explicit research path for adjacent-token interactions; not attention.
use super::*;

pub(super) fn validate_strength(strength: f32) -> SciRustResult<()> {
    ensure_finite(strength)?;
    if !(0.0..=1.0).contains(&strength) {
        return Err(SciRustError::NonFinite);
    }
    Ok(())
}

impl SequenceEncoder {
    /// Direct inference matching the experimental contextual tape. Each token
    /// embedding receives `strength * previous_embedding` before position and
    /// projection. The left boundary repeats the first token; strength is [0,1].
    pub fn forward_contextual(&self, token_ids: &[u16], strength: f32) -> SciRustResult<Vec<f32>> {
        self.validate_tokens(token_ids)?;
        validate_strength(strength)?;
        self.required_gather_max_elements(token_ids.len())?;
        let width = self.config.embedding_dim;
        let hidden = self.config.hidden_dim;
        let scale = (token_ids.len() as f32).recip();
        let mut combined = vec![0.; width];
        let mut output = vec![0.; hidden];
        for (position, &token) in token_ids.iter().enumerate() {
            let previous = usize::from(token_ids[position.saturating_sub(1)]);
            for (k, value) in combined.iter_mut().enumerate() {
                *value = ((0. + self.token_embeddings[usize::from(token) * width + k])
                    + strength * (0. + self.token_embeddings[previous * width + k]))
                    + (0. + self.position_embeddings[position * width + k]);
                ensure_finite(*value)?;
            }
            for (j, value) in output.iter_mut().enumerate() {
                let mut mixed = 0.;
                for (k, feature) in combined.iter().enumerate() {
                    mixed += feature * self.mixing_weights[k * hidden + j];
                }
                ensure_finite(mixed)?;
                *value += scale * mixed.max(0.);
            }
        }
        validate_finite(&output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contextual_features_distinguish_order_without_position_embeddings() {
        let config = SequenceEncoderConfig {
            vocab_size: 2,
            max_tokens: 2,
            embedding_dim: 1,
            hidden_dim: 1,
            seed: 1,
        };
        let model =
            SequenceEncoder::from_parts(config, vec![1., -1.], vec![0.; 2], vec![1.]).unwrap();
        assert_eq!(
            model.forward(&[0, 1]).unwrap(),
            model.forward(&[1, 0]).unwrap()
        );
        assert_ne!(
            model.forward_contextual(&[0, 1], 1.).unwrap(),
            model.forward_contextual(&[1, 0], 1.).unwrap()
        );
    }
    #[test]
    fn context_direct_matches_tape_and_embedding_finite_difference() {
        let config = SequenceEncoderConfig {
            vocab_size: 2,
            max_tokens: 3,
            embedding_dim: 1,
            hidden_dim: 1,
            seed: 1,
        };
        let model =
            SequenceEncoder::from_parts(config, vec![0.7, 0.3], vec![0.1; 3], vec![0.8]).unwrap();
        let tokens = &[0, 1, 0];
        let mut tape = Tape::new(20, 16);
        let graph = model
            .append_to_tape_contextual(&mut tape, tokens, 0.5)
            .unwrap();
        assert_eq!(
            tape.value_of(graph.pooled).as_slice(),
            model.forward_contextual(tokens, 0.5).unwrap()
        );
        let loss = tape.sum(graph.pooled).unwrap();
        tape.backward(loss).unwrap();
        let analytic = model.gradients_from_tape(&tape, graph);
        for index in 0..2 {
            let mut a = model.clone();
            let mut b = model.clone();
            a.token_embeddings[index] += 0.001;
            b.token_embeddings[index] -= 0.001;
            let numerical = (a.forward_contextual(tokens, 0.5).unwrap()[0]
                - b.forward_contextual(tokens, 0.5).unwrap()[0])
                / 0.002;
            assert!((analytic.token_embeddings()[index] - numerical).abs() < 1e-4);
        }
        assert_eq!(
            model.forward_contextual(tokens, 0.).unwrap(),
            model.forward(tokens).unwrap()
        );
        for strength in [f32::NAN, -0.1, 1.1] {
            assert!(model.forward_contextual(tokens, strength).is_err());
        }
    }
}
