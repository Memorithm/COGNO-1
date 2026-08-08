//! Bounded differentiable sequence encoder for the COGNO neural substrate.
//!
//! The encoder deliberately uses only the existing safe `Tape` primitives:
//! bounded one-hot token/position matrices select trainable embeddings through
//! matrix multiplication, token and position embeddings are combined, a
//! trainable projection plus ReLU creates contextualized token features, and a
//! fixed averaging row performs differentiable pooling. This gives COGNO an
//! order-sensitive trainable sequence representation without FFI, unsafe code,
//! hidden allocation growth or a second autodiff implementation.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{AdamW, Optimizer, Shape, Tape, Tensor, Var};

/// Maximum discrete vocabulary accepted by the bounded encoder.
pub const MAX_SEQUENCE_VOCAB: usize = 512;
/// Maximum configured sequence length.
pub const MAX_SEQUENCE_TOKENS: usize = 512;
/// Maximum token embedding width.
pub const MAX_SEQUENCE_EMBEDDING_DIM: usize = 256;
/// Maximum pooled hidden width.
pub const MAX_SEQUENCE_HIDDEN_DIM: usize = 512;
/// Maximum trainable scalar parameters in one sequence encoder.
pub const MAX_SEQUENCE_PARAMETERS: usize = 262_144;
/// Maximum elements in any one activation/constant tensor built by the encoder.
pub const MAX_SEQUENCE_ACTIVATION_ELEMENTS: usize = 262_144;
/// Tape nodes required by the encoder graph itself.
pub const SEQUENCE_ENCODER_TAPE_NODES: usize = 12;
/// Tape bound used by the built-in squared-error training helper.
pub const SEQUENCE_TRAINING_TAPE_NODES: usize = 20;

/// Architecture of the bounded token/position sequence encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceEncoderConfig {
    pub vocab_size: usize,
    pub max_tokens: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub seed: u64,
}

impl SequenceEncoderConfig {
    /// Checked parameter count for token embeddings + position embeddings +
    /// embedding-to-hidden projection.
    pub fn parameter_count(self) -> SciRustResult<usize> {
        let token_embeddings = self
            .vocab_size
            .checked_mul(self.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let position_embeddings = self
            .max_tokens
            .checked_mul(self.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let mixing_weights = self
            .embedding_dim
            .checked_mul(self.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        token_embeddings
            .checked_add(position_embeddings)
            .and_then(|value| value.checked_add(mixing_weights))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        if self.vocab_size == 0
            || self.max_tokens == 0
            || self.embedding_dim == 0
            || self.hidden_dim == 0
        {
            return Err(SciRustError::Empty);
        }
        validate_bound(self.vocab_size, MAX_SEQUENCE_VOCAB)?;
        validate_bound(self.max_tokens, MAX_SEQUENCE_TOKENS)?;
        validate_bound(self.embedding_dim, MAX_SEQUENCE_EMBEDDING_DIM)?;
        validate_bound(self.hidden_dim, MAX_SEQUENCE_HIDDEN_DIM)?;
        let parameters = self.parameter_count()?;
        validate_bound(parameters, MAX_SEQUENCE_PARAMETERS)?;
        Ok(())
    }
}

/// Trainable bounded sequence encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceEncoder {
    config: SequenceEncoderConfig,
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    mixing_weights: Vec<f32>,
}

impl SequenceEncoder {
    /// Deterministically initialize all trainable tensors from an explicit seed.
    pub fn try_new(config: SequenceEncoderConfig) -> SciRustResult<Self> {
        config.validate()?;
        let token_len = config
            .vocab_size
            .checked_mul(config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let position_len = config
            .max_tokens
            .checked_mul(config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let mixing_len = config
            .embedding_dim
            .checked_mul(config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let mut state = config.seed
            ^ 0xD6E8_FEB8_6659_FD93
            ^ (config.vocab_size as u64).rotate_left(7)
            ^ (config.max_tokens as u64).rotate_left(19)
            ^ (config.embedding_dim as u64).rotate_left(31)
            ^ (config.hidden_dim as u64).rotate_left(43);
        let token_embeddings = deterministic_weights(token_len, config.embedding_dim, &mut state)?;
        let position_embeddings =
            deterministic_weights(position_len, config.embedding_dim, &mut state)?;
        let mixing_weights = deterministic_weights(mixing_len, config.embedding_dim, &mut state)?;
        Self::from_parts(
            config,
            token_embeddings,
            position_embeddings,
            mixing_weights,
        )
    }

    /// Reconstruct from explicit trainable tensors after checking shape, bounds
    /// and finiteness. This is the safe boundary future hostile loaders use.
    pub fn from_parts(
        config: SequenceEncoderConfig,
        token_embeddings: Vec<f32>,
        position_embeddings: Vec<f32>,
        mixing_weights: Vec<f32>,
    ) -> SciRustResult<Self> {
        config.validate()?;
        let expected_token = config
            .vocab_size
            .checked_mul(config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let expected_position = config
            .max_tokens
            .checked_mul(config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let expected_mixing = config
            .embedding_dim
            .checked_mul(config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        validate_len(&token_embeddings, expected_token)?;
        validate_len(&position_embeddings, expected_position)?;
        validate_len(&mixing_weights, expected_mixing)?;
        validate_finite(&token_embeddings)?;
        validate_finite(&position_embeddings)?;
        validate_finite(&mixing_weights)?;
        Ok(Self {
            config,
            token_embeddings,
            position_embeddings,
            mixing_weights,
        })
    }

    #[must_use]
    pub const fn config(&self) -> SequenceEncoderConfig {
        self.config
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.token_embeddings.len() + self.position_embeddings.len() + self.mixing_weights.len()
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

    /// Largest tensor element count required for this token sequence.
    pub fn required_max_elements(&self, token_count: usize) -> SciRustResult<usize> {
        self.validate_token_count(token_count)?;
        let token_selector = token_count
            .checked_mul(self.config.vocab_size)
            .ok_or(SciRustError::Overflow)?;
        let position_selector = token_count
            .checked_mul(self.config.max_tokens)
            .ok_or(SciRustError::Overflow)?;
        let token_parameters = self
            .config
            .vocab_size
            .checked_mul(self.config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let position_parameters = self
            .config
            .max_tokens
            .checked_mul(self.config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let token_features = token_count
            .checked_mul(self.config.embedding_dim)
            .ok_or(SciRustError::Overflow)?;
        let mixing_parameters = self
            .config
            .embedding_dim
            .checked_mul(self.config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let hidden_features = token_count
            .checked_mul(self.config.hidden_dim)
            .ok_or(SciRustError::Overflow)?;
        let required = [
            token_selector,
            position_selector,
            token_parameters,
            position_parameters,
            token_features,
            mixing_parameters,
            hidden_features,
            self.config.hidden_dim,
        ]
        .into_iter()
        .max()
        .ok_or(SciRustError::Empty)?;
        validate_bound(required, MAX_SEQUENCE_ACTIVATION_ELEMENTS)?;
        Ok(required)
    }

    /// Append the differentiable encoder graph to an existing tape.
    ///
    /// Callers can attach typed task heads to [`SequenceEncoderGraph::pooled`],
    /// call `Tape::backward` on their scalar loss, then recover encoder
    /// gradients through [`Self::gradients_from_tape`].
    pub fn append_to_tape(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
    ) -> SciRustResult<SequenceEncoderGraph> {
        self.validate_tokens(token_ids)?;
        let token_count = token_ids.len();
        let required = self.required_max_elements(token_count)?;
        if tape.max_elements < required {
            return Err(SciRustError::CapacityExceeded {
                requested: required,
                maximum: tape.max_elements,
            });
        }

        let token_selector = selector_tensor(
            token_count,
            self.config.vocab_size,
            token_ids.iter().map(|&token| usize::from(token)),
            tape.max_elements,
        )?;
        let token_selector = tape.variable(token_selector)?;
        let token_embeddings = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.vocab_size, self.config.embedding_dim])?,
            self.token_embeddings.clone(),
            tape.max_elements,
        )?)?;
        let token_features = tape.matmul(token_selector, token_embeddings)?;

        let position_selector = selector_tensor(
            token_count,
            self.config.max_tokens,
            0..token_count,
            tape.max_elements,
        )?;
        let position_selector = tape.variable(position_selector)?;
        let position_embeddings = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.max_tokens, self.config.embedding_dim])?,
            self.position_embeddings.clone(),
            tape.max_elements,
        )?)?;
        let position_features = tape.matmul(position_selector, position_embeddings)?;
        let combined = tape.add(token_features, position_features)?;

        let mixing_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.embedding_dim, self.config.hidden_dim])?,
            self.mixing_weights.clone(),
            tape.max_elements,
        )?)?;
        let mixed = tape.matmul(combined, mixing_weights)?;
        let hidden = tape.relu(mixed)?;

        let scale = (token_count as f32).recip();
        ensure_finite(scale)?;
        let pooling = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, token_count])?,
            vec![scale; token_count],
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

    /// Read the pooled hidden representation without mutating model state.
    pub fn forward(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        self.validate_tokens(token_ids)?;
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_ENCODER_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        Ok(tape.value_of(graph.pooled).as_slice().to_vec())
    }

    /// Mean-squared representation loss and exact encoder gradients.
    pub fn loss_and_gradients_squared(
        &self,
        token_ids: &[u16],
        target: &[f32],
    ) -> SciRustResult<(f32, SequenceEncoderGradients)> {
        self.validate_tokens(token_ids)?;
        validate_len(target, self.config.hidden_dim)?;
        validate_finite(target)?;
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_TRAINING_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        let target_var = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.hidden_dim])?,
            target.to_vec(),
            max_elements,
        )?)?;
        let difference = tape.sub(graph.pooled, target_var)?;
        let squared = tape.mul(difference, difference)?;
        let summed = tape.sum(squared)?;
        let divisor = self.config.hidden_dim as f32;
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
        Ok((loss_value, self.gradients_from_tape(&tape, graph)))
    }

    /// One bounded AdamW training step toward a target representation.
    pub fn train_step_squared(
        &mut self,
        optimizer: &mut SequenceEncoderAdamW,
        token_ids: &[u16],
        target: &[f32],
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients_squared(token_ids, target)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    /// Recover parameter gradients after a caller-owned loss has been
    /// backpropagated through a graph returned by [`Self::append_to_tape`].
    #[must_use]
    pub fn gradients_from_tape(
        &self,
        tape: &Tape,
        graph: SequenceEncoderGraph,
    ) -> SequenceEncoderGradients {
        SequenceEncoderGradients {
            token_embeddings: tape.grad_of(graph.token_embeddings).to_vec(),
            position_embeddings: tape.grad_of(graph.position_embeddings).to_vec(),
            mixing_weights: tape.grad_of(graph.mixing_weights).to_vec(),
        }
    }

    fn validate_token_count(&self, token_count: usize) -> SciRustResult<()> {
        if token_count == 0 {
            return Err(SciRustError::Empty);
        }
        if token_count > self.config.max_tokens {
            return Err(SciRustError::CapacityExceeded {
                requested: token_count,
                maximum: self.config.max_tokens,
            });
        }
        Ok(())
    }

    fn validate_tokens(&self, token_ids: &[u16]) -> SciRustResult<()> {
        self.validate_token_count(token_ids.len())?;
        for &token in token_ids {
            let token = usize::from(token);
            if token >= self.config.vocab_size {
                return Err(SciRustError::Index {
                    idx: token,
                    len: self.config.vocab_size,
                });
            }
        }
        Ok(())
    }
}

/// Tape handles required to recover encoder parameter gradients.
#[derive(Clone, Copy, Debug)]
pub struct SequenceEncoderGraph {
    token_embeddings: Var,
    position_embeddings: Var,
    mixing_weights: Var,
    pooled: Var,
}

impl SequenceEncoderGraph {
    /// Pooled `[1, hidden_dim]` representation to which task heads attach.
    #[must_use]
    pub const fn pooled(self) -> Var {
        self.pooled
    }
}

/// Gradients matching the encoder's three trainable tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceEncoderGradients {
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    mixing_weights: Vec<f32>,
}

impl SequenceEncoderGradients {
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

/// AdamW states for the three trainable encoder tensors.
#[derive(Clone, Debug)]
pub struct SequenceEncoderAdamW {
    token_embeddings: AdamW,
    position_embeddings: AdamW,
    mixing_weights: AdamW,
}

impl SequenceEncoderAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceEncoder) -> SciRustResult<Self> {
        Ok(Self {
            token_embeddings: AdamW::try_new(learning_rate, model.token_embeddings.len())?,
            position_embeddings: AdamW::try_new(learning_rate, model.position_embeddings.len())?,
            mixing_weights: AdamW::try_new(learning_rate, model.mixing_weights.len())?,
        })
    }

    /// Apply one checked optimizer step to every encoder tensor.
    pub fn step(
        &mut self,
        model: &mut SequenceEncoder,
        gradients: &SequenceEncoderGradients,
    ) -> SciRustResult<()> {
        self.token_embeddings
            .step(&mut model.token_embeddings, &gradients.token_embeddings)?;
        self.position_embeddings.step(
            &mut model.position_embeddings,
            &gradients.position_embeddings,
        )?;
        self.mixing_weights
            .step(&mut model.mixing_weights, &gradients.mixing_weights)?;
        validate_finite(&model.token_embeddings)?;
        validate_finite(&model.position_embeddings)?;
        validate_finite(&model.mixing_weights)?;
        Ok(())
    }
}

fn selector_tensor<I>(
    rows: usize,
    columns: usize,
    selected_columns: I,
    max_elements: usize,
) -> SciRustResult<Tensor>
where
    I: IntoIterator<Item = usize>,
{
    let len = rows.checked_mul(columns).ok_or(SciRustError::Overflow)?;
    if len > max_elements {
        return Err(SciRustError::CapacityExceeded {
            requested: len,
            maximum: max_elements,
        });
    }
    let mut data = vec![0.0f32; len];
    let mut seen = 0usize;
    for (row, selected) in selected_columns.into_iter().enumerate() {
        if row >= rows {
            return Err(SciRustError::CapacityExceeded {
                requested: row + 1,
                maximum: rows,
            });
        }
        if selected >= columns {
            return Err(SciRustError::Index {
                idx: selected,
                len: columns,
            });
        }
        let index = row
            .checked_mul(columns)
            .and_then(|base| base.checked_add(selected))
            .ok_or(SciRustError::Overflow)?;
        data[index] = 1.0;
        seen = seen.checked_add(1).ok_or(SciRustError::Overflow)?;
    }
    if seen != rows {
        return Err(SciRustError::Shape {
            lhs: vec![seen],
            rhs: vec![rows],
        });
    }
    Tensor::try_new(Shape::try_new(&[rows, columns])?, data, max_elements)
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

fn validate_bound(requested: usize, maximum: usize) -> SciRustResult<()> {
    if requested > maximum {
        Err(SciRustError::CapacityExceeded { requested, maximum })
    } else {
        Ok(())
    }
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

    #[test]
    fn deterministic_initialization_is_repeatable() {
        let config = SequenceEncoderConfig {
            vocab_size: 16,
            max_tokens: 8,
            embedding_dim: 6,
            hidden_dim: 5,
            seed: 71,
        };
        let left = SequenceEncoder::try_new(config).expect("left");
        let right = SequenceEncoder::try_new(config).expect("right");
        assert_eq!(left, right);
        let different = SequenceEncoder::try_new(SequenceEncoderConfig { seed: 72, ..config })
            .expect("different seed");
        assert_ne!(left.token_embeddings(), different.token_embeddings());
    }

    #[test]
    fn positional_nonlinearity_makes_token_order_observable() {
        let model = SequenceEncoder::from_parts(
            SequenceEncoderConfig {
                vocab_size: 2,
                max_tokens: 2,
                embedding_dim: 1,
                hidden_dim: 1,
                seed: 0,
            },
            vec![1.0, -1.0],
            vec![1.0, -1.0],
            vec![1.0],
        )
        .expect("explicit encoder");
        assert_eq!(model.forward(&[0, 1]).expect("forward ab"), vec![1.0]);
        assert_eq!(model.forward(&[1, 0]).expect("forward ba"), vec![0.0]);
    }

    #[test]
    fn representation_training_reduces_loss_deterministically() {
        let config = SequenceEncoderConfig {
            vocab_size: 8,
            max_tokens: 4,
            embedding_dim: 4,
            hidden_dim: 3,
            seed: 73,
        };
        let mut left = SequenceEncoder::try_new(config).expect("left");
        let mut right = SequenceEncoder::try_new(config).expect("right");
        let mut left_optimizer = SequenceEncoderAdamW::try_new(0.02, &left).expect("optimizer");
        let mut right_optimizer = SequenceEncoderAdamW::try_new(0.02, &right).expect("optimizer");
        let tokens = [1, 2, 3];
        let target = [0.75, 0.25, 0.5];
        let initial = left
            .loss_and_gradients_squared(&tokens, &target)
            .expect("initial")
            .0;
        for _ in 0..128 {
            left.train_step_squared(&mut left_optimizer, &tokens, &target)
                .expect("left step");
            right
                .train_step_squared(&mut right_optimizer, &tokens, &target)
                .expect("right step");
        }
        let final_loss = left
            .loss_and_gradients_squared(&tokens, &target)
            .expect("final")
            .0;
        assert!(final_loss.is_finite());
        assert!(final_loss < initial);
        assert_eq!(left, right);
    }

    #[test]
    fn caller_can_attach_a_loss_to_the_shared_pooled_state() {
        let model = SequenceEncoder::try_new(SequenceEncoderConfig {
            vocab_size: 8,
            max_tokens: 4,
            embedding_dim: 4,
            hidden_dim: 3,
            seed: 79,
        })
        .expect("encoder");
        let tokens = [2, 1];
        let max_elements = model.required_max_elements(tokens.len()).expect("bound");
        let mut tape = Tape::new(SEQUENCE_TRAINING_TAPE_NODES, max_elements);
        let graph = model.append_to_tape(&mut tape, &tokens).expect("graph");
        let loss = tape.sum(graph.pooled()).expect("loss");
        tape.backward(loss).expect("backward");
        let gradients = model.gradients_from_tape(&tape, graph);
        assert!(gradients
            .token_embeddings()
            .iter()
            .any(|gradient| *gradient != 0.0));
        assert!(gradients
            .position_embeddings()
            .iter()
            .any(|gradient| *gradient != 0.0));
        assert!(gradients
            .mixing_weights()
            .iter()
            .any(|gradient| *gradient != 0.0));
    }

    #[test]
    fn hostile_tokens_and_activation_sizes_fail_closed() {
        let model = SequenceEncoder::try_new(SequenceEncoderConfig {
            vocab_size: 8,
            max_tokens: 4,
            embedding_dim: 4,
            hidden_dim: 3,
            seed: 83,
        })
        .expect("encoder");
        assert!(matches!(model.forward(&[]), Err(SciRustError::Empty)));
        assert!(matches!(
            model.forward(&[0, 1, 2, 3, 4]),
            Err(SciRustError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            model.forward(&[0, 8]),
            Err(SciRustError::Index { idx: 8, len: 8 })
        ));
        assert!(matches!(
            SequenceEncoder::try_new(SequenceEncoderConfig {
                vocab_size: MAX_SEQUENCE_VOCAB,
                max_tokens: MAX_SEQUENCE_TOKENS,
                embedding_dim: MAX_SEQUENCE_EMBEDDING_DIM,
                hidden_dim: MAX_SEQUENCE_HIDDEN_DIM,
                seed: 0,
            }),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }
}
