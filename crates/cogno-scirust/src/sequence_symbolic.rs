//! Connected sequence symbolic-satisfaction head over a shared bounded encoder.
//!
//! Each configured rule owns a deterministic scalar sigmoid head over one
//! shared [`crate::SequenceEncoder`] representation. The resulting satisfaction
//! scalars remain on the same [`crate::Tape`] and feed
//! [`crate::SymbolicSatisfaction::loss_vars`], so the soft conjunction trains
//! both the rule heads and upstream sequence representation.
//!
//! This module is deliberately non-authoritative: predicted satisfactions are
//! soft numerical signals only. Hard rule evaluation and admission remain in
//! `cogno-core` and cannot be overridden by this loss.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    AdamW, Optimizer, SequenceEncoder, SequenceEncoderAdamW, SequenceEncoderConfig,
    SequenceEncoderGradients, SequenceEncoderGraph, Shape, SymbolicSatisfaction, Tape, Tensor, Var,
    MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_PARAMETERS, SEQUENCE_ENCODER_TAPE_NODES,
};

/// Maximum number of soft symbolic rule heads sharing one sequence encoder.
pub const MAX_SEQUENCE_SYMBOLIC_RULES: usize = 32;
/// Maximum total trainable scalars in encoder plus all symbolic heads.
pub const MAX_SEQUENCE_SYMBOLIC_PARAMETERS: usize = MAX_SEQUENCE_PARAMETERS
    + MAX_SEQUENCE_HIDDEN_DIM * MAX_SEQUENCE_SYMBOLIC_RULES
    + MAX_SEQUENCE_SYMBOLIC_RULES;
/// Encoder nodes + six nodes per rule head + the connected conjunction loss.
pub const SEQUENCE_SYMBOLIC_TAPE_NODES: usize =
    SEQUENCE_ENCODER_TAPE_NODES + 7 * MAX_SEQUENCE_SYMBOLIC_RULES + 1;

/// Architecture of a shared sequence encoder plus bounded symbolic rule heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceSymbolicConfig {
    pub encoder: SequenceEncoderConfig,
    pub num_rules: usize,
    pub head_seed: u64,
}

impl SequenceSymbolicConfig {
    /// Checked total trainable parameter count.
    pub fn parameter_count(self) -> SciRustResult<usize> {
        let head_weights = self
            .encoder
            .hidden_dim
            .checked_mul(self.num_rules)
            .ok_or(SciRustError::Overflow)?;
        self.encoder
            .parameter_count()?
            .checked_add(head_weights)
            .and_then(|value| value.checked_add(self.num_rules))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        if self.num_rules == 0 {
            return Err(SciRustError::Empty);
        }
        if self.num_rules > MAX_SEQUENCE_SYMBOLIC_RULES {
            return Err(SciRustError::CapacityExceeded {
                requested: self.num_rules,
                maximum: MAX_SEQUENCE_SYMBOLIC_RULES,
            });
        }
        let parameters = self.parameter_count()?;
        if parameters > MAX_SEQUENCE_SYMBOLIC_PARAMETERS {
            return Err(SciRustError::CapacityExceeded {
                requested: parameters,
                maximum: MAX_SEQUENCE_SYMBOLIC_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// Exact gradients for the shared encoder and every symbolic head.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSymbolicGradients {
    encoder: SequenceEncoderGradients,
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceSymbolicGradients {
    #[must_use]
    pub const fn encoder(&self) -> &SequenceEncoderGradients {
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
}

/// Trainable non-authoritative per-rule satisfaction model.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSymbolicHead {
    config: SequenceSymbolicConfig,
    encoder: SequenceEncoder,
    // Rule-major: each rule owns `hidden_dim` contiguous weights.
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceSymbolicHead {
    /// Deterministically initialize encoder and rule heads from explicit seeds.
    pub fn try_new(config: SequenceSymbolicConfig) -> SciRustResult<Self> {
        config.validate()?;
        let encoder = SequenceEncoder::try_new(config.encoder)?;
        let head_len = config
            .encoder
            .hidden_dim
            .checked_mul(config.num_rules)
            .ok_or(SciRustError::Overflow)?;
        let mut state = config.head_seed
            ^ 0x8EBC_6AF0_9C88_C6E3
            ^ (config.encoder.hidden_dim as u64).rotate_left(23)
            ^ (config.num_rules as u64).rotate_left(47);
        let head_weights = deterministic_weights(head_len, config.encoder.hidden_dim, &mut state)?;
        Self::from_parts(config, encoder, head_weights, vec![0.0; config.num_rules])
    }

    /// Reconstruct from explicit bounded tensors after validating shapes and
    /// finiteness.
    pub fn from_parts(
        config: SequenceSymbolicConfig,
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
        let expected_weights = config
            .encoder
            .hidden_dim
            .checked_mul(config.num_rules)
            .ok_or(SciRustError::Overflow)?;
        validate_len(&head_weights, expected_weights)?;
        validate_len(&head_bias, config.num_rules)?;
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
    pub const fn config(&self) -> SequenceSymbolicConfig {
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

    /// Return one bounded sigmoid satisfaction in `[0, 1]` per configured rule.
    pub fn satisfactions(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_SYMBOLIC_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        let mut values = Vec::with_capacity(self.config.num_rules);
        for satisfaction in graph.satisfactions {
            let value = tape
                .value_of(satisfaction)
                .as_slice()
                .first()
                .copied()
                .ok_or(SciRustError::Empty)?;
            ensure_finite(value)?;
            if !(0.0..=1.0).contains(&value) {
                return Err(SciRustError::NonFinite);
            }
            values.push(value);
        }
        Ok(values)
    }

    /// Product of all soft rule satisfactions. This is an inference diagnostic,
    /// not a hard-rule decision.
    pub fn soft_conjunction(&self, token_ids: &[u16]) -> SciRustResult<f32> {
        let mut product = 1.0f32;
        for satisfaction in self.satisfactions(token_ids)? {
            product *= satisfaction;
            ensure_finite(product)?;
        }
        Ok(product)
    }

    /// Connected soft-conjunction loss and exact gradients for one sequence.
    pub fn loss_and_gradients(
        &self,
        token_ids: &[u16],
    ) -> SciRustResult<(f32, SequenceSymbolicGradients)> {
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_SYMBOLIC_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        let loss = SymbolicSatisfaction::try_new(self.config.num_rules, max_elements)?
            .loss_vars(&mut tape, &graph.satisfactions)?;
        tape.backward(loss)?;
        let loss_value = tape
            .value_of(loss)
            .as_slice()
            .first()
            .copied()
            .ok_or(SciRustError::Empty)?;
        ensure_finite(loss_value)?;

        let mut head_weights = Vec::with_capacity(self.head_weights.len());
        for weights in &graph.head_weights {
            head_weights.extend_from_slice(tape.grad_of(*weights));
        }
        let mut head_bias = Vec::with_capacity(self.head_bias.len());
        for bias in &graph.head_bias {
            head_bias.extend_from_slice(tape.grad_of(*bias));
        }
        validate_finite(&head_weights)?;
        validate_finite(&head_bias)?;

        Ok((
            loss_value,
            SequenceSymbolicGradients {
                encoder: self.encoder.gradients_from_tape(&tape, graph.encoder),
                head_weights,
                head_bias,
            },
        ))
    }

    /// Apply one deterministic AdamW update through all symbolic heads and the
    /// shared sequence encoder.
    pub fn train_step(
        &mut self,
        optimizer: &mut SequenceSymbolicAdamW,
        token_ids: &[u16],
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients(token_ids)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn append_to_tape(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
    ) -> SciRustResult<SequenceSymbolicGraph> {
        let encoder = self.encoder.append_to_tape(tape, token_ids)?;
        let hidden_dim = self.config.encoder.hidden_dim;
        let mut head_weights = Vec::with_capacity(self.config.num_rules);
        let mut head_bias = Vec::with_capacity(self.config.num_rules);
        let mut satisfactions = Vec::with_capacity(self.config.num_rules);

        for (rule, &bias_value) in self.head_bias.iter().enumerate() {
            let start = rule.checked_mul(hidden_dim).ok_or(SciRustError::Overflow)?;
            let end = start.checked_add(hidden_dim).ok_or(SciRustError::Overflow)?;
            let weights = tape.variable(Tensor::try_new(
                Shape::try_new(&[1, hidden_dim])?,
                self.head_weights[start..end].to_vec(),
                tape.max_elements,
            )?)?;
            let weighted = tape.mul(encoder.pooled(), weights)?;
            let score = tape.sum(weighted)?;
            let bias = tape.variable(Tensor::try_scalar(bias_value)?)?;
            let score = tape.add(score, bias)?;
            let satisfaction = tape.sigmoid(score)?;
            head_weights.push(weights);
            head_bias.push(bias);
            satisfactions.push(satisfaction);
        }

        Ok(SequenceSymbolicGraph {
            encoder,
            head_weights,
            head_bias,
            satisfactions,
        })
    }

    fn required_max_elements(&self, token_count: usize) -> SciRustResult<usize> {
        Ok(self
            .encoder
            .required_max_elements(token_count)?
            .max(self.config.encoder.hidden_dim)
            .max(self.config.num_rules))
    }
}

#[derive(Clone, Debug)]
struct SequenceSymbolicGraph {
    encoder: SequenceEncoderGraph,
    head_weights: Vec<Var>,
    head_bias: Vec<Var>,
    satisfactions: Vec<Var>,
}

/// AdamW states for the shared encoder and all symbolic heads.
#[derive(Clone, Debug)]
pub struct SequenceSymbolicAdamW {
    encoder: SequenceEncoderAdamW,
    head_weights: AdamW,
    head_bias: AdamW,
}

impl SequenceSymbolicAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceSymbolicHead) -> SciRustResult<Self> {
        Ok(Self {
            encoder: SequenceEncoderAdamW::try_new(learning_rate, &model.encoder)?,
            head_weights: AdamW::try_new(learning_rate, model.head_weights.len())?,
            head_bias: AdamW::try_new(learning_rate, model.head_bias.len())?,
        })
    }

    pub fn step(
        &mut self,
        model: &mut SequenceSymbolicHead,
        gradients: &SequenceSymbolicGradients,
    ) -> SciRustResult<()> {
        self.encoder.step(&mut model.encoder, &gradients.encoder)?;
        self.head_weights
            .step(&mut model.head_weights, &gradients.head_weights)?;
        self.head_bias
            .step(&mut model.head_bias, &gradients.head_bias)?;
        validate_finite(&model.head_weights)?;
        validate_finite(&model.head_bias)?;
        Ok(())
    }
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

    fn controlled_head() -> SequenceSymbolicHead {
        let encoder_config = SequenceEncoderConfig {
            vocab_size: 4,
            max_tokens: 4,
            embedding_dim: 2,
            hidden_dim: 2,
            seed: 0,
        };
        let encoder = SequenceEncoder::from_parts(
            encoder_config,
            vec![0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4],
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.5, 0.1, 0.1, 0.5],
        )
        .expect("controlled encoder");
        SequenceSymbolicHead::from_parts(
            SequenceSymbolicConfig {
                encoder: encoder_config,
                num_rules: 2,
                head_seed: 0,
            },
            encoder,
            vec![0.4, 0.3, 0.2, 0.5],
            vec![0.0, 0.0],
        )
        .expect("controlled symbolic head")
    }

    #[test]
    fn deterministic_initialization_repeats_exactly() {
        let config = SequenceSymbolicConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 16,
                max_tokens: 8,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 811,
            },
            num_rules: 3,
            head_seed: 813,
        };
        assert_eq!(
            SequenceSymbolicHead::try_new(config).expect("left"),
            SequenceSymbolicHead::try_new(config).expect("right")
        );
    }

    #[test]
    fn satisfactions_are_bounded_and_deterministic() {
        let model = controlled_head();
        let left = model.satisfactions(&[1, 2]).expect("left");
        let right = model.satisfactions(&[1, 2]).expect("right");
        assert_eq!(left, right);
        assert_eq!(left.len(), 2);
        assert!(left.iter().all(|value| (0.0..=1.0).contains(value)));
        let conjunction = model.soft_conjunction(&[1, 2]).expect("conjunction");
        assert!((0.0..=1.0).contains(&conjunction));
    }

    #[test]
    fn connected_symbolic_loss_reaches_heads_and_shared_encoder() {
        let model = controlled_head();
        let (loss, gradients) = model.loss_and_gradients(&[1, 2]).expect("loss");
        assert!(loss > 0.0 && loss < 1.0);
        assert!(gradients
            .encoder()
            .token_embeddings()
            .iter()
            .any(|value| *value != 0.0));
        assert!(gradients.head_weights().iter().any(|value| *value != 0.0));
        assert!(gradients.head_bias().iter().any(|value| *value != 0.0));
    }

    #[test]
    fn deterministic_training_reduces_soft_conjunction_loss() {
        let mut model = controlled_head();
        let before = model.loss_and_gradients(&[1, 2]).expect("before").0;
        let mut optimizer = SequenceSymbolicAdamW::try_new(0.01, &model).expect("optimizer");
        for _ in 0..32 {
            model
                .train_step(&mut optimizer, &[1, 2])
                .expect("training step");
        }
        let after = model.loss_and_gradients(&[1, 2]).expect("after").0;
        assert!(after < before, "before={before}, after={after}");
    }

    #[test]
    fn hostile_rule_counts_fail_closed() {
        let encoder = SequenceEncoderConfig {
            vocab_size: 4,
            max_tokens: 4,
            embedding_dim: 2,
            hidden_dim: 2,
            seed: 0,
        };
        assert_eq!(
            SequenceSymbolicHead::try_new(SequenceSymbolicConfig {
                encoder,
                num_rules: 0,
                head_seed: 0,
            }),
            Err(SciRustError::Empty)
        );
        assert!(matches!(
            SequenceSymbolicHead::try_new(SequenceSymbolicConfig {
                encoder,
                num_rules: MAX_SEQUENCE_SYMBOLIC_RULES + 1,
                head_seed: 0,
            }),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }
}
