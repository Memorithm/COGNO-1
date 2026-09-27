//! Token-local representations and explicit pooling masks.
use super::*;

impl SequenceEncoder {
    /// Unpooled `[tokens, hidden_dim]` ReLU features. These are local token plus
    /// position features, not attention, explanations or independent predictions.
    pub fn token_features(&self, token_ids: &[u16]) -> SciRustResult<Tensor> {
        self.validate_tokens(token_ids)?;
        self.required_max_elements(token_ids.len())?;
        let width = self.config.embedding_dim;
        let hidden = self.config.hidden_dim;
        let mut features = vec![0.; token_ids.len() * hidden];
        for (position, &token) in token_ids.iter().enumerate() {
            for j in 0..hidden {
                let mut mixed = 0.;
                for k in 0..width {
                    let combined = (0. + self.token_embeddings[usize::from(token) * width + k])
                        + (0. + self.position_embeddings[position * width + k]);
                    ensure_finite(combined)?;
                    mixed += combined * self.mixing_weights[k * hidden + j];
                }
                ensure_finite(mixed)?;
                features[position * hidden + j] = mixed.max(0.);
            }
        }
        Tensor::try_new(
            Shape::try_new(&[token_ids.len(), hidden])?,
            features,
            MAX_SEQUENCE_ACTIVATION_ELEMENTS,
        )
    }

    /// Normalize nonnegative external weights and pool all original positions.
    /// A zero weight masks a token from pooling without renumbering positions.
    /// Weights are controls, not learned attention scores.
    pub fn forward_weighted(&self, token_ids: &[u16], weights: &[f32]) -> SciRustResult<Vec<f32>> {
        self.validate_tokens(token_ids)?;
        let weights = normalized_pool_weights(weights, token_ids.len())?;
        let features = self.token_features(token_ids)?;
        let mut output = vec![0.; self.config.hidden_dim];
        for (row, weight) in features
            .data
            .chunks_exact(self.config.hidden_dim)
            .zip(weights)
        {
            for (value, feature) in output.iter_mut().zip(row) {
                *value += weight * feature;
            }
        }
        validate_finite(&output)?;
        Ok(output)
    }
}

pub(super) fn normalized_pool_weights(weights: &[f32], count: usize) -> SciRustResult<Vec<f32>> {
    validate_len(weights, count)?;
    if count == 0 {
        return Err(SciRustError::Empty);
    }
    let mut total = 0_f64;
    for &weight in weights {
        ensure_finite(weight)?;
        if weight < 0. {
            return Err(SciRustError::NonFinite);
        }
        total += f64::from(weight);
    }
    if total == 0. {
        return Err(SciRustError::Empty);
    }
    Ok(weights
        .iter()
        .map(|&weight| (f64::from(weight) / total) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_pool_matches_default_and_masks_preserve_positions() {
        let model = SequenceEncoder::try_new(SequenceEncoderConfig {
            vocab_size: 5,
            max_tokens: 4,
            embedding_dim: 3,
            hidden_dim: 7,
            seed: 42,
        })
        .unwrap();
        let tokens = &[1, 2, 1];
        assert_eq!(
            model.forward_weighted(tokens, &[1.; 3]).unwrap(),
            model.forward(tokens).unwrap()
        );
        let features = model.token_features(tokens).unwrap();
        assert_eq!(
            model.forward_weighted(tokens, &[0., 1., 0.]).unwrap(),
            features.data[7..14]
        );
        for weights in [
            &[0., 0., 0.][..],
            &[-1., 1., 0.][..],
            &[f32::NAN, 1., 0.][..],
            &[1.][..],
        ] {
            assert!(model.forward_weighted(tokens, weights).is_err());
        }
        assert!(model.forward_weighted(tokens, &[f32::MAX; 3]).is_ok());
    }
}
