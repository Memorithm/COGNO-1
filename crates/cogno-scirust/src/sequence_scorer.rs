//! Pairwise sequence scorer whose preference loss remains connected to the
//! shared [`crate::SequenceEncoder`].
//!
//! Preferred and dispreferred sequences are encoded on the same tape using two
//! views of the same frozen parameter values. Their scalar score heads feed the
//! connected [`crate::PairwiseLoss`]. Reverse-mode gradients from both views
//! are added before a single AdamW update, so one preference comparison trains
//! token embeddings, positional embeddings, encoder mixing weights and the
//! scalar score head together.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    AdamW, Optimizer, PairwiseLoss, SequenceEncoder, SequenceEncoderConfig, Shape, Tape, Tensor,
    Var, MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_PARAMETERS,
};

/// Maximum total trainable scalars in encoder plus scalar score head.
pub const MAX_SEQUENCE_SCORER_PARAMETERS: usize =
    MAX_SEQUENCE_PARAMETERS + MAX_SEQUENCE_HIDDEN_DIM + 1;
/// Nodes for one encoder plus its scalar head.
pub const SEQUENCE_SCORER_TAPE_NODES: usize = 16;
/// Bound for two scorer views plus the connected pairwise hinge loss.
pub const SEQUENCE_PAIRWISE_TAPE_NODES: usize = 48;

/// Architecture of a shared sequence encoder plus scalar score head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceScorerConfig {
    pub encoder: SequenceEncoderConfig,
    pub head_seed: u64,
}

impl SequenceScorerConfig {
    pub fn parameter_count(self) -> SciRustResult<usize> {
        self.encoder
            .parameter_count()?
            .checked_add(self.encoder.hidden_dim)
            .and_then(|value| value.checked_add(1))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        let parameters = self.parameter_count()?;
        if parameters > MAX_SEQUENCE_SCORER_PARAMETERS {
            return Err(SciRustError::CapacityExceeded {
                requested: parameters,
                maximum: MAX_SEQUENCE_SCORER_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// Exact gradients after one connected preference comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceScorerGradients {
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    mixing_weights: Vec<f32>,
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceScorerGradients {
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

    #[must_use]
    pub fn head_weights(&self) -> &[f32] {
        &self.head_weights
    }

    #[must_use]
    pub fn head_bias(&self) -> &[f32] {
        &self.head_bias
    }
}

/// Trainable scalar scorer over bounded token sequences.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceScorer {
    config: SequenceScorerConfig,
    encoder: SequenceEncoder,
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceScorer {
    /// Deterministically initialize encoder and scalar head from explicit seeds.
    pub fn try_new(config: SequenceScorerConfig) -> SciRustResult<Self> {
        config.validate()?;
        let encoder = SequenceEncoder::try_new(config.encoder)?;
        let mut state = config.head_seed
            ^ 0xE703_7ED1_A0B4_28DB
            ^ (config.encoder.hidden_dim as u64).rotate_left(31);
        let head_weights = deterministic_weights(
            config.encoder.hidden_dim,
            config.encoder.hidden_dim,
            &mut state,
        )?;
        Self::from_parts(config, encoder, head_weights, vec![0.0])
    }

    /// Reconstruct a scorer from already-validated encoder state and explicit
    /// head tensors.
    pub fn from_parts(
        config: SequenceScorerConfig,
        encoder: SequenceEncoder,
        head_weights: Vec<f32>,
        head_bias: Vec<f32>,
    ) -> SciRustResult<Self> {
        config.validate()?;
        if encoder.config() != config.encoder {
            return Err(SciRustError::Shape {
                lhs: vec![encoder.config().hidden_dim],
                rhs: vec![config.encoder.hidden_dim],
            });
        }
        validate_len(&head_weights, config.encoder.hidden_dim)?;
        validate_len(&head_bias, 1)?;
        validate_finite(&head_weights)?;
        validate_finite(&head_bias)?;
        Ok(Self {
            config,
            encoder,
            head_weights,
            head_bias,
        })
    }

    #[must_use]
    pub const fn config(&self) -> SequenceScorerConfig {
        self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &SequenceEncoder {
        &self.encoder
    }

    #[must_use]
    pub fn head_weights(&self) -> &[f32] {
        &self.head_weights
    }

    #[must_use]
    pub fn head_bias(&self) -> &[f32] {
        &self.head_bias
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.encoder.parameter_count() + self.head_weights.len() + self.head_bias.len()
    }

    /// Scalar preference score for one sequence.
    pub fn score(&self, token_ids: &[u16]) -> SciRustResult<f32> {
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_SCORER_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        let score = tape
            .value_of(graph.score)
            .as_slice()
            .first()
            .copied()
            .ok_or(SciRustError::Empty)?;
        ensure_finite(score)?;
        Ok(score)
    }

    /// Connected hinge-ranking loss and exact summed gradients for one
    /// preferred/dispreferred sequence pair.
    pub fn pairwise_loss_and_gradients(
        &self,
        preferred: &[u16],
        dispreferred: &[u16],
        margin: f32,
    ) -> SciRustResult<(f32, SequenceScorerGradients)> {
        ensure_finite(margin)?;
        if margin < 0.0 {
            return Err(SciRustError::Shape {
                lhs: vec![0],
                rhs: vec![1],
            });
        }
        let max_elements = self
            .required_max_elements(preferred.len())?
            .max(self.required_max_elements(dispreferred.len())?);
        let mut tape = Tape::new(SEQUENCE_PAIRWISE_TAPE_NODES, max_elements);
        let preferred_graph = self.append_to_tape(&mut tape, preferred)?;
        let dispreferred_graph = self.append_to_tape(&mut tape, dispreferred)?;
        let loss = PairwiseLoss::try_new(margin, 1, max_elements)?.loss_vars(
            &mut tape,
            preferred_graph.score,
            dispreferred_graph.score,
        )?;
        tape.backward(loss)?;
        let loss_value = tape
            .value_of(loss)
            .as_slice()
            .first()
            .copied()
            .ok_or(SciRustError::Empty)?;
        ensure_finite(loss_value)?;

        let preferred_encoder = self
            .encoder
            .gradients_from_tape(&tape, preferred_graph.encoder);
        let dispreferred_encoder = self
            .encoder
            .gradients_from_tape(&tape, dispreferred_graph.encoder);
        Ok((
            loss_value,
            SequenceScorerGradients {
                token_embeddings: sum_gradients(
                    preferred_encoder.token_embeddings(),
                    dispreferred_encoder.token_embeddings(),
                )?,
                position_embeddings: sum_gradients(
                    preferred_encoder.position_embeddings(),
                    dispreferred_encoder.position_embeddings(),
                )?,
                mixing_weights: sum_gradients(
                    preferred_encoder.mixing_weights(),
                    dispreferred_encoder.mixing_weights(),
                )?,
                head_weights: sum_gradients(
                    tape.grad_of(preferred_graph.head_weights),
                    tape.grad_of(dispreferred_graph.head_weights),
                )?,
                head_bias: sum_gradients(
                    tape.grad_of(preferred_graph.head_bias),
                    tape.grad_of(dispreferred_graph.head_bias),
                )?,
            },
        ))
    }

    /// One deterministic AdamW update for a preference comparison.
    pub fn train_pairwise_step(
        &mut self,
        optimizer: &mut SequenceScorerAdamW,
        preferred: &[u16],
        dispreferred: &[u16],
        margin: f32,
    ) -> SciRustResult<f32> {
        let (loss, gradients) =
            self.pairwise_loss_and_gradients(preferred, dispreferred, margin)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn append_to_tape(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
    ) -> SciRustResult<SequenceScorerGraph> {
        let encoder = self.encoder.append_to_tape(tape, token_ids)?;
        let head_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.encoder.hidden_dim, 1])?,
            self.head_weights.clone(),
            tape.max_elements,
        )?)?;
        let head_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, 1])?,
            self.head_bias.clone(),
            tape.max_elements,
        )?)?;
        let score = tape.matmul(encoder.pooled(), head_weights)?;
        let score = tape.add(score, head_bias)?;
        Ok(SequenceScorerGraph {
            encoder,
            head_weights,
            head_bias,
            score,
        })
    }

    fn required_max_elements(&self, token_count: usize) -> SciRustResult<usize> {
        Ok(self
            .encoder
            .required_max_elements(token_count)?
            .max(self.config.encoder.hidden_dim))
    }
}

#[derive(Clone, Copy, Debug)]
struct SequenceScorerGraph {
    encoder: crate::SequenceEncoderGraph,
    head_weights: Var,
    head_bias: Var,
    score: Var,
}

/// AdamW state for all shared encoder and scalar-head parameters.
#[derive(Clone, Debug)]
pub struct SequenceScorerAdamW {
    token_embeddings: AdamW,
    position_embeddings: AdamW,
    mixing_weights: AdamW,
    head_weights: AdamW,
    head_bias: AdamW,
}

impl SequenceScorerAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceScorer) -> SciRustResult<Self> {
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
            head_weights: AdamW::try_new(learning_rate, model.head_weights.len())?,
            head_bias: AdamW::try_new(learning_rate, model.head_bias.len())?,
        })
    }

    pub fn step(
        &mut self,
        model: &mut SequenceScorer,
        gradients: &SequenceScorerGradients,
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
        self.head_weights
            .step(&mut model.head_weights, &gradients.head_weights)?;
        self.head_bias
            .step(&mut model.head_bias, &gradients.head_bias)?;
        model.encoder = SequenceEncoder::from_parts(
            model.config.encoder,
            token_embeddings,
            position_embeddings,
            mixing_weights,
        )?;
        validate_finite(&model.head_weights)?;
        validate_finite(&model.head_bias)?;
        Ok(())
    }
}

fn sum_gradients(left: &[f32], right: &[f32]) -> SciRustResult<Vec<f32>> {
    if left.len() != right.len() {
        return Err(SciRustError::Shape {
            lhs: vec![left.len()],
            rhs: vec![right.len()],
        });
    }
    let mut sum = Vec::with_capacity(left.len());
    for (&left, &right) in left.iter().zip(right) {
        let value = left + right;
        ensure_finite(value)?;
        sum.push(value);
    }
    Ok(sum)
}

fn deterministic_weights(len: usize, fan_in: usize, state: &mut u64) -> SciRustResult<Vec<f32>> {
    if fan_in == 0 {
        return Err(SciRustError::Empty);
    }
    let fan_in = fan_in as f32;
    ensure_finite(fan_in)?;
    let scale = fan_in.sqrt().recip();
    ensure_finite(scale)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let sample = ((*state >> 40) & 0x00ff_ffff) as u32;
        let unit = sample as f32 / 16_777_215.0;
        let value = (unit * 2.0 - 1.0) * scale;
        ensure_finite(value)?;
        values.push(value);
    }
    Ok(values)
}

fn validate_len(values: &[f32], expected: usize) -> SciRustResult<()> {
    if values.len() != expected {
        return Err(SciRustError::Shape {
            lhs: vec![values.len()],
            rhs: vec![expected],
        });
    }
    Ok(())
}

fn validate_finite(values: &[f32]) -> SciRustResult<()> {
    for &value in values {
        ensure_finite(value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controlled_scorer() -> SequenceScorer {
        let encoder_config = SequenceEncoderConfig {
            vocab_size: 4,
            max_tokens: 4,
            embedding_dim: 2,
            hidden_dim: 2,
            seed: 0,
        };
        let encoder = SequenceEncoder::from_parts(
            encoder_config,
            vec![0.1, 0.1, 0.7, 0.2, 0.2, 0.7, 0.4, 0.4],
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.8, 0.2, 0.2, 0.8],
        )
        .expect("controlled encoder");
        SequenceScorer::from_parts(
            SequenceScorerConfig {
                encoder: encoder_config,
                head_seed: 0,
            },
            encoder,
            vec![0.2, -0.1],
            vec![0.0],
        )
        .expect("controlled scorer")
    }

    #[test]
    fn deterministic_initialization_repeats_exactly() {
        let config = SequenceScorerConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 16,
                max_tokens: 8,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 431,
            },
            head_seed: 433,
        };
        assert_eq!(
            SequenceScorer::try_new(config).expect("left"),
            SequenceScorer::try_new(config).expect("right")
        );
    }

    #[test]
    fn pairwise_loss_reaches_both_encoder_and_score_head() {
        let model = controlled_scorer();
        let (_, gradients) = model
            .pairwise_loss_and_gradients(&[1, 1], &[2, 2], 1.0)
            .expect("pairwise gradients");
        assert!(gradients
            .token_embeddings()
            .iter()
            .any(|gradient| *gradient != 0.0));
        assert!(gradients
            .mixing_weights()
            .iter()
            .any(|gradient| *gradient != 0.0));
        assert!(gradients
            .head_weights()
            .iter()
            .any(|gradient| *gradient != 0.0));
    }

    #[test]
    fn pairwise_training_is_deterministic_and_improves_preference_margin() {
        let mut left = controlled_scorer();
        let mut right = controlled_scorer();
        let initial_encoder = left.encoder().token_embeddings().to_vec();
        let preferred = [1, 1];
        let dispreferred = [2, 2];
        let initial_gap = left.score(&preferred).expect("preferred")
            - left.score(&dispreferred).expect("dispreferred");
        let mut left_optimizer = SequenceScorerAdamW::try_new(0.02, &left).expect("optimizer");
        let mut right_optimizer = SequenceScorerAdamW::try_new(0.02, &right).expect("optimizer");
        for _ in 0..64 {
            left.train_pairwise_step(&mut left_optimizer, &preferred, &dispreferred, 1.0)
                .expect("left step");
            right
                .train_pairwise_step(&mut right_optimizer, &preferred, &dispreferred, 1.0)
                .expect("right step");
        }
        let final_gap = left.score(&preferred).expect("preferred")
            - left.score(&dispreferred).expect("dispreferred");
        assert!(final_gap > initial_gap);
        assert!(final_gap > 0.0);
        assert_ne!(left.encoder().token_embeddings(), initial_encoder);
        assert_eq!(left, right);
    }

    #[test]
    fn hostile_sequences_still_fail_through_encoder_bounds() {
        let model = controlled_scorer();
        assert!(matches!(model.score(&[]), Err(SciRustError::Empty)));
        assert!(matches!(
            model.score(&[0, 1, 2, 3, 0]),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }
}
