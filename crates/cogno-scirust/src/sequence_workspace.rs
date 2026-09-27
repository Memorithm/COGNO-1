//! Reusable read-only inference scratch, independent of model weights.
use super::*;

/// Fixed-size inference scratch. Reuse across equal-width models or sequences;
/// no cached parameters or outputs from previous calls enter a successful result.
#[derive(Clone, Debug)]
pub struct SequenceWorkspace {
    combined: Vec<f32>,
    pooled: Vec<f32>,
}

impl SequenceWorkspace {
    pub fn try_new(config: SequenceEncoderConfig) -> SciRustResult<Self> {
        config.validate()?;
        Ok(Self {
            combined: vec![0.; config.embedding_dim],
            pooled: vec![0.; config.hidden_dim],
        })
    }
}

impl SequenceEncoder {
    /// Streaming inference into reusable scratch. The returned slice is valid
    /// until the next mutable workspace use. On error, scratch is unspecified.
    /// Success allocates no vectors, and preserves `forward` accumulation order.
    pub fn forward_with_workspace<'a>(
        &self,
        token_ids: &[u16],
        workspace: &'a mut SequenceWorkspace,
    ) -> SciRustResult<&'a [f32]> {
        self.validate_tokens(token_ids)?;
        self.required_max_elements(token_ids.len())?;
        validate_len(&workspace.combined, self.config.embedding_dim)?;
        validate_len(&workspace.pooled, self.config.hidden_dim)?;
        validate_finite(&self.token_embeddings)?;
        validate_finite(&self.position_embeddings)?;
        validate_finite(&self.mixing_weights)?;
        let width = self.config.embedding_dim;
        let hidden = self.config.hidden_dim;
        let scale = (token_ids.len() as f32).recip();
        workspace.pooled.fill(0.);
        for (position, &token) in token_ids.iter().enumerate() {
            for (k, value) in workspace.combined.iter_mut().enumerate() {
                *value = (0.0 + self.token_embeddings[usize::from(token) * width + k])
                    + (0.0 + self.position_embeddings[position * width + k]);
                ensure_finite(*value)?;
            }
            for (j, output) in workspace.pooled.iter_mut().enumerate() {
                let mut mixed = 0.0_f32;
                for (k, value) in workspace.combined.iter().enumerate() {
                    mixed += value * self.mixing_weights[k * hidden + j];
                }
                ensure_finite(mixed)?;
                *output += scale * mixed.max(0.0);
            }
        }
        validate_finite(&workspace.pooled)?;
        Ok(&workspace.pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_reuse_preserves_reference_and_survives_bad_call() {
        let config = SequenceEncoderConfig {
            vocab_size: 8,
            max_tokens: 8,
            embedding_dim: 3,
            hidden_dim: 7,
            seed: 1,
        };
        let model = SequenceEncoder::try_new(config).unwrap();
        let mut workspace = SequenceWorkspace::try_new(config).unwrap();
        let pointer = workspace.pooled.as_ptr();
        for tokens in [&[1, 2, 3][..], &[0][..], &[7, 7][..]] {
            assert_eq!(
                model
                    .forward_with_workspace(tokens, &mut workspace)
                    .unwrap(),
                model.forward(tokens).unwrap()
            );
            assert_eq!(workspace.pooled.as_ptr(), pointer);
            assert!(model.forward_with_workspace(&[8], &mut workspace).is_err());
        }
        let mut wrong = SequenceWorkspace::try_new(SequenceEncoderConfig {
            hidden_dim: 8,
            ..config
        })
        .unwrap();
        assert!(model.forward_with_workspace(&[1], &mut wrong).is_err());
    }
}
