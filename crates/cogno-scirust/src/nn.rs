//! Bounded deterministic neural-network primitives for COGNO-1.
//!
//! This module deliberately implements a very small neural substrate rather
//! than a general deep-learning framework. It provides one-hidden-layer MLPs
//! with deterministic initialization, bounded tensor sizes, differentiable
//! forward passes through [`crate::Tape`], and AdamW updates without granting
//! model outputs any authority over `cogno-core`.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{AdamW, Optimizer, Shape, Tape, Tensor, Var};

/// Maximum width accepted for any MLP dimension.
pub const MAX_MLP_DIM: usize = 4_096;
/// Maximum number of trainable scalar parameters in one bounded MLP.
pub const MAX_MLP_PARAMETERS: usize = 262_144;
/// Tape bound for one forward/backward pass through the fixed two-layer graph.
pub const MAX_MLP_TAPE_NODES: usize = 16;

/// Architecture of the bounded one-hidden-layer MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub seed: u64,
}

impl MlpConfig {
    /// Checked parameter count for `W1 + b1 + W2 + b2`.
    pub fn parameter_count(self) -> SciRustResult<usize> {
        let input_hidden = self
            .input_dim
            .checked_mul(self.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let hidden_output = self
            .hidden_dim
            .checked_mul(self.output_dim)
            .ok_or(SciRustError::Overflow)?;
        input_hidden
            .checked_add(self.hidden_dim)
            .and_then(|value| value.checked_add(hidden_output))
            .and_then(|value| value.checked_add(self.output_dim))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        let dims = [self.input_dim, self.hidden_dim, self.output_dim];
        if dims.contains(&0) {
            return Err(SciRustError::Empty);
        }
        if let Some(&requested) = dims.iter().find(|&&dim| dim > MAX_MLP_DIM) {
            return Err(SciRustError::CapacityExceeded {
                requested,
                maximum: MAX_MLP_DIM,
            });
        }
        let parameters = self.parameter_count()?;
        if parameters > MAX_MLP_PARAMETERS {
            return Err(SciRustError::CapacityExceeded {
                requested: parameters,
                maximum: MAX_MLP_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// Gradients matching the four trainable MLP tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct MlpGradients {
    input_hidden: Vec<f32>,
    hidden_bias: Vec<f32>,
    hidden_output: Vec<f32>,
    output_bias: Vec<f32>,
}

impl MlpGradients {
    #[must_use]
    pub fn input_hidden(&self) -> &[f32] {
        &self.input_hidden
    }

    #[must_use]
    pub fn hidden_bias(&self) -> &[f32] {
        &self.hidden_bias
    }

    #[must_use]
    pub fn hidden_output(&self) -> &[f32] {
        &self.hidden_output
    }

    #[must_use]
    pub fn output_bias(&self) -> &[f32] {
        &self.output_bias
    }
}

/// Deterministically initialized, one-hidden-layer ReLU MLP.
#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    config: MlpConfig,
    input_hidden: Vec<f32>,
    hidden_bias: Vec<f32>,
    hidden_output: Vec<f32>,
    output_bias: Vec<f32>,
}

impl Mlp {
    /// Construct a bounded MLP using deterministic fan-in-scaled weights.
    pub fn try_new(config: MlpConfig) -> SciRustResult<Self> {
        config.validate()?;
        let input_hidden_len = config
            .input_dim
            .checked_mul(config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let hidden_output_len = config
            .hidden_dim
            .checked_mul(config.output_dim)
            .ok_or(SciRustError::Overflow)?;
        let mut state = config.seed
            ^ 0x9E37_79B9_7F4A_7C15
            ^ (config.input_dim as u64).rotate_left(11)
            ^ (config.hidden_dim as u64).rotate_left(29)
            ^ (config.output_dim as u64).rotate_left(47);
        let input_hidden = deterministic_weights(input_hidden_len, config.input_dim, &mut state)?;
        let hidden_output =
            deterministic_weights(hidden_output_len, config.hidden_dim, &mut state)?;
        Self::from_parts(
            config,
            input_hidden,
            vec![0.0; config.hidden_dim],
            hidden_output,
            vec![0.0; config.output_dim],
        )
    }

    /// Reconstruct a model from explicit tensors after validating every bound.
    pub fn from_parts(
        config: MlpConfig,
        input_hidden: Vec<f32>,
        hidden_bias: Vec<f32>,
        hidden_output: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> SciRustResult<Self> {
        config.validate()?;
        let expected_input_hidden = config
            .input_dim
            .checked_mul(config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let expected_hidden_output = config
            .hidden_dim
            .checked_mul(config.output_dim)
            .ok_or(SciRustError::Overflow)?;
        validate_len(&input_hidden, expected_input_hidden)?;
        validate_len(&hidden_bias, config.hidden_dim)?;
        validate_len(&hidden_output, expected_hidden_output)?;
        validate_len(&output_bias, config.output_dim)?;
        validate_finite(&input_hidden)?;
        validate_finite(&hidden_bias)?;
        validate_finite(&hidden_output)?;
        validate_finite(&output_bias)?;
        Ok(Self {
            config,
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        })
    }

    #[must_use]
    pub const fn config(&self) -> MlpConfig {
        self.config
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.input_hidden.len()
            + self.hidden_bias.len()
            + self.hidden_output.len()
            + self.output_bias.len()
    }

    #[must_use]
    pub fn input_hidden(&self) -> &[f32] {
        &self.input_hidden
    }

    #[must_use]
    pub fn hidden_bias(&self) -> &[f32] {
        &self.hidden_bias
    }

    #[must_use]
    pub fn hidden_output(&self) -> &[f32] {
        &self.hidden_output
    }

    #[must_use]
    pub fn output_bias(&self) -> &[f32] {
        &self.output_bias
    }

    /// Execute a bounded differentiable forward pass and return raw logits.
    pub fn forward(&self, input: &[f32]) -> SciRustResult<Vec<f32>> {
        validate_len(input, self.config.input_dim)?;
        validate_finite(input)?;
        let max_elements = self.max_tensor_elements()?;
        let mut tape = Tape::new(MAX_MLP_TAPE_NODES, max_elements);
        let graph = self.build_forward_graph(&mut tape, input, max_elements)?;
        Ok(tape.value_of(graph.output).as_slice().to_vec())
    }

    /// Mean squared error and exact reverse-mode gradients for one example.
    pub fn loss_and_gradients_squared(
        &self,
        input: &[f32],
        target: &[f32],
    ) -> SciRustResult<(f32, MlpGradients)> {
        validate_len(input, self.config.input_dim)?;
        validate_len(target, self.config.output_dim)?;
        validate_finite(input)?;
        validate_finite(target)?;
        let max_elements = self.max_tensor_elements()?;
        let mut tape = Tape::new(MAX_MLP_TAPE_NODES, max_elements);
        let graph = self.build_forward_graph(&mut tape, input, max_elements)?;
        let target_var = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.output_dim])?,
            target.to_vec(),
            max_elements,
        )?)?;
        let difference = tape.sub(graph.output, target_var)?;
        let squared = tape.mul(difference, difference)?;
        let summed = tape.sum(squared)?;
        let divisor = self.config.output_dim as f32;
        ensure_finite(divisor)?;
        let loss = tape.scale(summed, divisor.recip())?;
        tape.backward(loss)?;
        let loss_value = tape
            .value_of(loss)
            .as_slice()
            .first()
            .copied()
            .ok_or(SciRustError::Empty)?;
        ensure_finite(loss_value)?;
        Ok((
            loss_value,
            MlpGradients {
                input_hidden: tape.grad_of(graph.input_hidden).to_vec(),
                hidden_bias: tape.grad_of(graph.hidden_bias).to_vec(),
                hidden_output: tape.grad_of(graph.hidden_output).to_vec(),
                output_bias: tape.grad_of(graph.output_bias).to_vec(),
            },
        ))
    }

    /// Train on one example using a bounded AdamW optimizer bundle.
    pub fn train_step_squared(
        &mut self,
        optimizer: &mut MlpAdamW,
        input: &[f32],
        target: &[f32],
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients_squared(input, target)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn build_forward_graph(
        &self,
        tape: &mut Tape,
        input: &[f32],
        max_elements: usize,
    ) -> SciRustResult<ForwardGraph> {
        let input_var = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.input_dim])?,
            input.to_vec(),
            max_elements,
        )?)?;
        let input_hidden = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.input_dim, self.config.hidden_dim])?,
            self.input_hidden.clone(),
            max_elements,
        )?)?;
        let hidden_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.hidden_dim])?,
            self.hidden_bias.clone(),
            max_elements,
        )?)?;
        let hidden_linear = tape.matmul(input_var, input_hidden)?;
        let hidden_shifted = tape.add(hidden_linear, hidden_bias)?;
        let hidden = tape.relu(hidden_shifted)?;
        let hidden_output = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.hidden_dim, self.config.output_dim])?,
            self.hidden_output.clone(),
            max_elements,
        )?)?;
        let output_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.output_dim])?,
            self.output_bias.clone(),
            max_elements,
        )?)?;
        let output_linear = tape.matmul(hidden, hidden_output)?;
        let output = tape.add(output_linear, output_bias)?;
        Ok(ForwardGraph {
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
            output,
        })
    }

    fn max_tensor_elements(&self) -> SciRustResult<usize> {
        let input_hidden = self
            .config
            .input_dim
            .checked_mul(self.config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let hidden_output = self
            .config
            .hidden_dim
            .checked_mul(self.config.output_dim)
            .ok_or(SciRustError::Overflow)?;
        Ok([
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim,
            input_hidden,
            hidden_output,
        ]
        .into_iter()
        .max()
        .ok_or(SciRustError::Empty)?)
    }
}

/// Four AdamW states, one per trainable MLP tensor.
#[derive(Clone, Debug)]
pub struct MlpAdamW {
    input_hidden: AdamW,
    hidden_bias: AdamW,
    hidden_output: AdamW,
    output_bias: AdamW,
}

impl MlpAdamW {
    pub fn try_new(learning_rate: f32, model: &Mlp) -> SciRustResult<Self> {
        Ok(Self {
            input_hidden: AdamW::try_new(learning_rate, model.input_hidden.len())?,
            hidden_bias: AdamW::try_new(learning_rate, model.hidden_bias.len())?,
            hidden_output: AdamW::try_new(learning_rate, model.hidden_output.len())?,
            output_bias: AdamW::try_new(learning_rate, model.output_bias.len())?,
        })
    }

    /// Apply one checked optimizer step to all MLP tensors.
    pub fn step(&mut self, model: &mut Mlp, gradients: &MlpGradients) -> SciRustResult<()> {
        self.input_hidden
            .step(&mut model.input_hidden, &gradients.input_hidden)?;
        self.hidden_bias
            .step(&mut model.hidden_bias, &gradients.hidden_bias)?;
        self.hidden_output
            .step(&mut model.hidden_output, &gradients.hidden_output)?;
        self.output_bias
            .step(&mut model.output_bias, &gradients.output_bias)?;
        validate_finite(&model.input_hidden)?;
        validate_finite(&model.hidden_bias)?;
        validate_finite(&model.hidden_output)?;
        validate_finite(&model.output_bias)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct ForwardGraph {
    input_hidden: Var,
    hidden_bias: Var,
    hidden_output: Var,
    output_bias: Var,
    output: Var,
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

    fn xor_model() -> Mlp {
        Mlp::try_new(MlpConfig {
            input_dim: 2,
            hidden_dim: 8,
            output_dim: 1,
            seed: 7,
        })
        .expect("bounded MLP")
    }

    fn xor_examples() -> [([f32; 2], [f32; 1]); 4] {
        [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]
    }

    fn average_xor_loss(model: &Mlp) -> f32 {
        let mut total = 0.0f32;
        for (input, target) in xor_examples() {
            total += model
                .loss_and_gradients_squared(&input, &target)
                .expect("bounded loss")
                .0;
        }
        total / 4.0
    }

    #[test]
    fn deterministic_initialization_is_repeatable_and_seeded() {
        let config = MlpConfig {
            input_dim: 5,
            hidden_dim: 7,
            output_dim: 3,
            seed: 19,
        };
        let left = Mlp::try_new(config).expect("left");
        let right = Mlp::try_new(config).expect("right");
        assert_eq!(left, right);

        let different = Mlp::try_new(MlpConfig { seed: 20, ..config }).expect("different");
        assert_ne!(left.input_hidden(), different.input_hidden());
    }

    #[test]
    fn forward_and_gradients_are_shape_checked() {
        let model = xor_model();
        let output = model.forward(&[0.0, 1.0]).expect("forward");
        assert_eq!(output.len(), 1);
        let (_, gradients) = model
            .loss_and_gradients_squared(&[0.0, 1.0], &[1.0])
            .expect("gradients");
        assert_eq!(gradients.input_hidden().len(), 16);
        assert_eq!(gradients.hidden_bias().len(), 8);
        assert_eq!(gradients.hidden_output().len(), 8);
        assert_eq!(gradients.output_bias().len(), 1);
        assert!(matches!(
            model.forward(&[0.0]),
            Err(SciRustError::Shape { .. })
        ));
    }

    #[test]
    fn bounded_mlp_learns_xor_deterministically() {
        let mut left = xor_model();
        let mut right = xor_model();
        let mut left_optimizer = MlpAdamW::try_new(0.02, &left).expect("left optimizer");
        let mut right_optimizer = MlpAdamW::try_new(0.02, &right).expect("right optimizer");
        let initial = average_xor_loss(&left);

        for _ in 0..128 {
            for (input, target) in xor_examples() {
                left.train_step_squared(&mut left_optimizer, &input, &target)
                    .expect("left step");
                right
                    .train_step_squared(&mut right_optimizer, &input, &target)
                    .expect("right step");
            }
        }

        let final_loss = average_xor_loss(&left);
        assert!(final_loss < initial / 100.0);
        assert_eq!(left, right);
        for (input, target) in xor_examples() {
            let prediction = left.forward(&input).expect("prediction")[0];
            if target[0] > 0.5 {
                assert!(prediction > 0.9);
            } else {
                assert!(prediction < 0.1);
            }
        }
    }

    #[test]
    fn hostile_dimensions_fail_before_parameter_allocation() {
        assert!(matches!(
            Mlp::try_new(MlpConfig {
                input_dim: MAX_MLP_DIM,
                hidden_dim: MAX_MLP_DIM,
                output_dim: MAX_MLP_DIM,
                seed: 0,
            }),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }
}
