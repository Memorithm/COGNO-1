//! End-to-end trainable classifier built on [`crate::SequenceEncoder`].
//!
//! This is the first bounded SciRust component whose supervised class loss is
//! connected all the way from logits through a task head into the shared
//! sequence encoder. It remains a numerical substrate only: class predictions
//! acquire no policy or tool authority here.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    AdamW, Optimizer, SequenceEncoder, SequenceEncoderAdamW, SequenceEncoderConfig,
    SequenceEncoderGradients, SequenceEncoderGraph, Shape, Tape, Tensor, Var,
    MAX_SEQUENCE_PARAMETERS,
};

/// Maximum number of supervised output classes.
pub const MAX_SEQUENCE_CLASSES: usize = 64;
/// Maximum total trainable parameters in encoder plus classification head.
pub const MAX_SEQUENCE_CLASSIFIER_PARAMETERS: usize =
    MAX_SEQUENCE_PARAMETERS + 512 * MAX_SEQUENCE_CLASSES + MAX_SEQUENCE_CLASSES;
/// Forward graph bound: sequence encoder (12) + head weights/matmul/bias/add.
pub const SEQUENCE_CLASSIFIER_TAPE_NODES: usize = 16;
/// Training graph bound including log-softmax, target selection and NLL.
pub const SEQUENCE_CLASSIFIER_TRAINING_TAPE_NODES: usize = 24;

/// Architecture of an encoder plus dense classification head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceClassifierConfig {
    pub encoder: SequenceEncoderConfig,
    pub num_classes: usize,
    pub head_seed: u64,
}

impl SequenceClassifierConfig {
    /// Checked total trainable parameter count.
    pub fn parameter_count(self) -> SciRustResult<usize> {
        let encoder = self.encoder.parameter_count()?;
        let head_weights = self
            .encoder
            .hidden_dim
            .checked_mul(self.num_classes)
            .ok_or(SciRustError::Overflow)?;
        encoder
            .checked_add(head_weights)
            .and_then(|value| value.checked_add(self.num_classes))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        if self.num_classes < 2 {
            return Err(SciRustError::Empty);
        }
        if self.num_classes > MAX_SEQUENCE_CLASSES {
            return Err(SciRustError::CapacityExceeded {
                requested: self.num_classes,
                maximum: MAX_SEQUENCE_CLASSES,
            });
        }
        let parameters = self.parameter_count()?;
        if parameters > MAX_SEQUENCE_CLASSIFIER_PARAMETERS {
            return Err(SciRustError::CapacityExceeded {
                requested: parameters,
                maximum: MAX_SEQUENCE_CLASSIFIER_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// Exact gradients for encoder and dense classification head.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceClassifierGradients {
    encoder: SequenceEncoderGradients,
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceClassifierGradients {
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

/// Frozen/trainable numerical classifier state.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceClassifier {
    config: SequenceClassifierConfig,
    encoder: SequenceEncoder,
    head_weights: Vec<f32>,
    head_bias: Vec<f32>,
}

impl SequenceClassifier {
    /// Deterministically initialize encoder and head from explicit seeds.
    pub fn try_new(config: SequenceClassifierConfig) -> SciRustResult<Self> {
        config.validate()?;
        let encoder = SequenceEncoder::try_new(config.encoder)?;
        let head_len = config
            .encoder
            .hidden_dim
            .checked_mul(config.num_classes)
            .ok_or(SciRustError::Overflow)?;
        let mut state = config.head_seed
            ^ 0xA076_1D64_78BD_642F
            ^ (config.encoder.hidden_dim as u64).rotate_left(17)
            ^ (config.num_classes as u64).rotate_left(41);
        let head_weights = deterministic_weights(head_len, config.encoder.hidden_dim, &mut state)?;
        Self::from_parts(config, encoder, head_weights, vec![0.0; config.num_classes])
    }

    /// Reconstruct explicit bounded tensors, suitable for a future hostile
    /// artifact loader after it has validated its external manifest.
    pub fn from_parts(
        config: SequenceClassifierConfig,
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
            .checked_mul(config.num_classes)
            .ok_or(SciRustError::Overflow)?;
        validate_len(&head_weights, expected_weights)?;
        validate_len(&head_bias, config.num_classes)?;
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
    pub const fn config(&self) -> SequenceClassifierConfig {
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

    /// Raw class logits for one bounded token sequence.
    pub fn logits(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_CLASSIFIER_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        Ok(tape.value_of(graph.logits).as_slice().to_vec())
    }

    /// Stable normalized class probabilities.
    pub fn probabilities(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        stable_softmax(&self.logits(token_ids)?)
    }

    /// Deterministic argmax; exact ties select the lowest class id.
    pub fn classify(&self, token_ids: &[u16]) -> SciRustResult<usize> {
        let logits = self.logits(token_ids)?;
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in logits.into_iter().enumerate() {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        ensure_finite(best_value)?;
        Ok(best_index)
    }

    /// Cross-entropy NLL and gradients for one labeled sequence.
    pub fn loss_and_gradients(
        &self,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<(f32, SequenceClassifierGradients)> {
        self.validate_target(target_class)?;
        let max_elements = self.required_max_elements(token_ids.len())?;
        let mut tape = Tape::new(SEQUENCE_CLASSIFIER_TRAINING_TAPE_NODES, max_elements);
        let graph = self.append_to_tape(&mut tape, token_ids)?;
        self.loss_from_graph(tape, graph, target_class)
    }

    /// Opt-in sparse embedding training with the same parameter and optimizer layout.
    pub fn loss_and_gradients_gather(
        &self,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<(f32, SequenceClassifierGradients)> {
        self.validate_target(target_class)?;
        let max_elements = self
            .encoder
            .required_gather_max_elements(token_ids.len())?
            .max(self.head_weights.len())
            .max(self.config.num_classes);
        let mut tape = Tape::new(SEQUENCE_CLASSIFIER_TRAINING_TAPE_NODES, max_elements);
        let encoder = self.encoder.append_to_tape_gather(&mut tape, token_ids)?;
        let graph = self.append_head(&mut tape, encoder)?;
        self.loss_from_graph(tape, graph, target_class)
    }

    /// One atomic AdamW step using the opt-in gather graph.
    pub fn train_step_gather(
        &mut self,
        optimizer: &mut SequenceClassifierAdamW,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients_gather(token_ids, target_class)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    /// Class logits using an explicit normalized pooling mask.
    pub fn logits_weighted(&self, token_ids: &[u16], weights: &[f32]) -> SciRustResult<Vec<f32>> {
        let max_elements = self
            .encoder
            .required_gather_max_elements(token_ids.len())?
            .max(self.head_weights.len())
            .max(self.config.num_classes);
        let mut tape = Tape::new(SEQUENCE_CLASSIFIER_TAPE_NODES, max_elements);
        let encoder = self
            .encoder
            .append_to_tape_weighted(&mut tape, token_ids, weights)?;
        let graph = self.append_head(&mut tape, encoder)?;
        Ok(tape.value_of(graph.logits).as_slice().to_vec())
    }

    /// End-to-end masked training; masks must be supplied consistently at evaluation.
    pub fn loss_and_gradients_weighted(
        &self,
        token_ids: &[u16],
        weights: &[f32],
        target_class: usize,
    ) -> SciRustResult<(f32, SequenceClassifierGradients)> {
        self.validate_target(target_class)?;
        let max_elements = self
            .encoder
            .required_gather_max_elements(token_ids.len())?
            .max(self.head_weights.len())
            .max(self.config.num_classes);
        let mut tape = Tape::new(SEQUENCE_CLASSIFIER_TRAINING_TAPE_NODES, max_elements);
        let encoder = self
            .encoder
            .append_to_tape_weighted(&mut tape, token_ids, weights)?;
        let graph = self.append_head(&mut tape, encoder)?;
        self.loss_from_graph(tape, graph, target_class)
    }

    /// Atomic masked AdamW update; no parameters change if mask admission fails.
    pub fn train_step_weighted(
        &mut self,
        optimizer: &mut SequenceClassifierAdamW,
        token_ids: &[u16],
        weights: &[f32],
        target_class: usize,
    ) -> SciRustResult<f32> {
        let (loss, gradients) =
            self.loss_and_gradients_weighted(token_ids, weights, target_class)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn loss_from_graph(
        &self,
        mut tape: Tape,
        graph: SequenceClassifierGraph,
        target_class: usize,
    ) -> SciRustResult<(f32, SequenceClassifierGradients)> {
        let max_elements = tape.max_elements;
        let log_probabilities = tape.log_softmax(graph.logits)?;
        let mut target = vec![0.0f32; self.config.num_classes];
        target[target_class] = 1.0;
        let target = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.num_classes])?,
            target,
            max_elements,
        )?)?;
        let selected = tape.mul(log_probabilities, target)?;
        let selected = tape.sum(selected)?;
        let loss = tape.neg(selected)?;
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
            SequenceClassifierGradients {
                encoder: self.encoder.gradients_from_tape(&tape, graph.encoder),
                head_weights: tape.grad_of(graph.head_weights).to_vec(),
                head_bias: tape.grad_of(graph.head_bias).to_vec(),
            },
        ))
    }

    /// Head-only NLL and gradients with the encoder frozen.
    ///
    /// This is an opt-in representation diagnostic: it measures whether a
    /// dense head can learn from existing encoder features. Features are
    /// detached intentionally; no encoder gradients or optimizer states exist
    /// in this path. It does not establish generalization to unseen code.
    pub fn loss_and_head_gradients(
        &self,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<(f32, FrozenClassifierHeadGradients)> {
        self.validate_target(target_class)?;
        let features = self.encoder.forward(token_ids)?;
        let max_elements = self
            .head_weights
            .len()
            .max(self.config.num_classes)
            .max(self.config.encoder.hidden_dim);
        let mut tape = Tape::new(16, max_elements);
        let features = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.encoder.hidden_dim])?,
            features,
            max_elements,
        )?)?;
        let weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.encoder.hidden_dim, self.config.num_classes])?,
            self.head_weights.clone(),
            max_elements,
        )?)?;
        let bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.num_classes])?,
            self.head_bias.clone(),
            max_elements,
        )?)?;
        let logits = tape.matmul(features, weights)?;
        let logits = tape.add(logits, bias)?;
        let log_probabilities = tape.log_softmax(logits)?;
        let mut target = vec![0.0; self.config.num_classes];
        target[target_class] = 1.0;
        let target = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.num_classes])?,
            target,
            max_elements,
        )?)?;
        let selected = tape.mul(log_probabilities, target)?;
        let selected = tape.sum(selected)?;
        let loss = tape.neg(selected)?;
        tape.backward(loss)?;
        let value = tape.value_of(loss).as_slice()[0];
        ensure_finite(value)?;
        let gradients = FrozenClassifierHeadGradients {
            weights: tape.grad_of(weights).to_vec(),
            bias: tape.grad_of(bias).to_vec(),
        };
        validate_finite(&gradients.weights)?;
        validate_finite(&gradients.bias)?;
        Ok((value, gradients))
    }

    /// Train only the dense head using a separate optimizer; encoder parameters
    /// remain bit-for-bit unchanged. Existing `train_step` remains end-to-end.
    pub fn train_head_step(
        &mut self,
        optimizer: &mut FrozenClassifierHeadAdamW,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_head_gradients(token_ids, target_class)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    /// One checked AdamW step through head and shared encoder.
    pub fn train_step(
        &mut self,
        optimizer: &mut SequenceClassifierAdamW,
        token_ids: &[u16],
        target_class: usize,
    ) -> SciRustResult<f32> {
        let (loss, gradients) = self.loss_and_gradients(token_ids, target_class)?;
        optimizer.step(self, &gradients)?;
        Ok(loss)
    }

    fn append_to_tape(
        &self,
        tape: &mut Tape,
        token_ids: &[u16],
    ) -> SciRustResult<SequenceClassifierGraph> {
        let encoder = self.encoder.append_to_tape(tape, token_ids)?;
        self.append_head(tape, encoder)
    }

    fn append_head(
        &self,
        tape: &mut Tape,
        encoder: SequenceEncoderGraph,
    ) -> SciRustResult<SequenceClassifierGraph> {
        let head_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[self.config.encoder.hidden_dim, self.config.num_classes])?,
            self.head_weights.clone(),
            tape.max_elements,
        )?)?;
        let head_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config.num_classes])?,
            self.head_bias.clone(),
            tape.max_elements,
        )?)?;
        let logits = tape.matmul(encoder.pooled(), head_weights)?;
        let logits = tape.add(logits, head_bias)?;
        Ok(SequenceClassifierGraph {
            encoder,
            head_weights,
            head_bias,
            logits,
        })
    }

    fn required_max_elements(&self, token_count: usize) -> SciRustResult<usize> {
        let encoder = self.encoder.required_max_elements(token_count)?;
        let head = self
            .config
            .encoder
            .hidden_dim
            .checked_mul(self.config.num_classes)
            .ok_or(SciRustError::Overflow)?;
        Ok(encoder.max(head).max(self.config.num_classes))
    }

    fn validate_target(&self, target_class: usize) -> SciRustResult<()> {
        if target_class >= self.config.num_classes {
            return Err(SciRustError::Index {
                idx: target_class,
                len: self.config.num_classes,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct SequenceClassifierGraph {
    encoder: SequenceEncoderGraph,
    head_weights: Var,
    head_bias: Var,
    logits: Var,
}

/// AdamW states for encoder plus classification head.
#[derive(Clone, Debug)]
pub struct SequenceClassifierAdamW {
    encoder: SequenceEncoderAdamW,
    head_weights: AdamW,
    head_bias: AdamW,
}

impl SequenceClassifierAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceClassifier) -> SciRustResult<Self> {
        Ok(Self {
            encoder: SequenceEncoderAdamW::try_new(learning_rate, &model.encoder)?,
            head_weights: AdamW::try_new(learning_rate, model.head_weights.len())?,
            head_bias: AdamW::try_new(learning_rate, model.head_bias.len())?,
        })
    }

    /// Apply one exact optimizer step to all trainable tensors.
    pub fn step(
        &mut self,
        model: &mut SequenceClassifier,
        gradients: &SequenceClassifierGradients,
    ) -> SciRustResult<()> {
        // Numerical optimizers are atomic per tensor. Stage the enclosing model
        // and all moments as well, so late tensor errors cannot partially train it.
        let mut next_optimizer = self.clone();
        let mut next_model = model.clone();
        next_optimizer.step_candidate(&mut next_model, gradients)?;
        *model = next_model;
        *self = next_optimizer;
        Ok(())
    }

    fn step_candidate(
        &mut self,
        model: &mut SequenceClassifier,
        gradients: &SequenceClassifierGradients,
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

/// Gradients of the dense classifier head with detached encoder features.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenClassifierHeadGradients {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl FrozenClassifierHeadGradients {
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

/// Head-only optimizer, intentionally containing no encoder optimizer state.
#[derive(Clone, Debug)]
pub struct FrozenClassifierHeadAdamW {
    weights: AdamW,
    bias: AdamW,
}

impl FrozenClassifierHeadAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceClassifier) -> SciRustResult<Self> {
        Ok(Self {
            weights: AdamW::try_new(learning_rate, model.head_weights.len())?,
            bias: AdamW::try_new(learning_rate, model.head_bias.len())?,
        })
    }

    /// Stage both tensor updates before committing model and optimizer state.
    pub fn step(
        &mut self,
        model: &mut SequenceClassifier,
        gradients: &FrozenClassifierHeadGradients,
    ) -> SciRustResult<()> {
        validate_len(&gradients.weights, model.head_weights.len())?;
        validate_len(&gradients.bias, model.head_bias.len())?;
        // Verify association by shape before calling the numerical optimizers,
        // whose public state representation need not be trusted by this API.
        validate_len(&self.weights.state.m, model.head_weights.len())?;
        validate_len(&self.bias.state.m, model.head_bias.len())?;
        let mut next = self.clone();
        let mut weights = model.head_weights.clone();
        let mut bias = model.head_bias.clone();
        next.weights.step(&mut weights, &gradients.weights)?;
        next.bias.step(&mut bias, &gradients.bias)?;
        validate_finite(&weights)?;
        validate_finite(&bias)?;
        model.head_weights = weights;
        model.head_bias = bias;
        *self = next;
        Ok(())
    }
}

fn stable_softmax(logits: &[f32]) -> SciRustResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(SciRustError::Empty);
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ensure_finite(maximum)?;
    let mut values = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let value = (logit - maximum).exp();
        ensure_finite(value)?;
        values.push(value);
        sum += value;
        ensure_finite(sum)?;
    }
    if sum <= 0.0 {
        return Err(SciRustError::NonFinite);
    }
    for value in &mut values {
        *value /= sum;
        ensure_finite(*value)?;
    }
    Ok(values)
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
    #[test]
    fn late_tensor_failure_is_atomic_and_retry_matches_clean_step() {
        let mut model = controlled_classifier();
        let mut optimizer = SequenceClassifierAdamW::try_new(0.01, &model).unwrap();
        let (_, good) = model.loss_and_gradients(&[1, 2], 1).unwrap();
        let mut bad = good.clone();
        bad.head_bias[0] = f32::NAN;
        let before_model = model.clone();
        let before_optimizer = format!("{optimizer:?}");
        assert!(optimizer.step(&mut model, &bad).is_err());
        assert_eq!(model, before_model);
        assert_eq!(format!("{optimizer:?}"), before_optimizer);
        let mut clean_model = model.clone();
        let mut clean_optimizer = optimizer.clone();
        optimizer.step(&mut model, &good).unwrap();
        clean_optimizer.step(&mut clean_model, &good).unwrap();
        assert_eq!(model, clean_model);
        assert_eq!(format!("{optimizer:?}"), format!("{clean_optimizer:?}"));
    }

    #[test]
    fn late_counter_overflow_preserves_all_tensors_and_moments() {
        let mut model = controlled_classifier();
        let mut optimizer = SequenceClassifierAdamW::try_new(0.01, &model).unwrap();
        let (_, gradients) = model.loss_and_gradients(&[1, 2], 1).unwrap();
        optimizer.head_bias.state.step = u64::MAX;
        let before_model = model.clone();
        let before_optimizer = format!("{optimizer:?}");
        assert_eq!(
            optimizer.step(&mut model, &gradients),
            Err(SciRustError::Overflow)
        );
        assert_eq!(model, before_model);
        assert_eq!(format!("{optimizer:?}"), before_optimizer);
    }

    fn controlled_classifier() -> SequenceClassifier {
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
            vec![0.5, 0.25, 0.25, 0.5],
        )
        .expect("controlled encoder");
        SequenceClassifier::from_parts(
            SequenceClassifierConfig {
                encoder: encoder_config,
                num_classes: 2,
                head_seed: 0,
            },
            encoder,
            vec![0.1, -0.1, 0.1, -0.1],
            vec![0.0, 0.0],
        )
        .expect("controlled classifier")
    }

    #[test]
    fn deterministic_initialization_repeats_exactly() {
        let config = SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 16,
                max_tokens: 8,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 91,
            },
            num_classes: 3,
            head_seed: 97,
        };
        let left = SequenceClassifier::try_new(config).expect("left");
        let right = SequenceClassifier::try_new(config).expect("right");
        assert_eq!(left, right);
    }

    #[test]
    fn supervised_loss_updates_head_and_encoder_deterministically() {
        let mut left = controlled_classifier();
        let mut right = controlled_classifier();
        let initial_encoder = left.encoder().token_embeddings().to_vec();
        let initial_head = left.head_weights().to_vec();
        let tokens = [1, 2];
        let initial_loss = left.loss_and_gradients(&tokens, 1).expect("initial loss").0;
        let mut left_optimizer =
            SequenceClassifierAdamW::try_new(0.02, &left).expect("left optimizer");
        let mut right_optimizer =
            SequenceClassifierAdamW::try_new(0.02, &right).expect("right optimizer");
        for _ in 0..64 {
            left.train_step(&mut left_optimizer, &tokens, 1)
                .expect("left step");
            right
                .train_step(&mut right_optimizer, &tokens, 1)
                .expect("right step");
        }
        let final_loss = left.loss_and_gradients(&tokens, 1).expect("final loss").0;
        assert!(final_loss < initial_loss);
        assert_ne!(left.encoder().token_embeddings(), initial_encoder);
        assert_ne!(left.head_weights(), initial_head);
        assert_eq!(left, right);
        assert_eq!(left.classify(&tokens).expect("classify"), 1);
    }

    #[test]
    fn probabilities_are_finite_and_normalized() {
        let model = controlled_classifier();
        let probabilities = model.probabilities(&[1, 2]).expect("probabilities");
        assert_eq!(probabilities.len(), 2);
        assert!(probabilities.iter().all(|value| value.is_finite()));
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn hostile_class_dimensions_and_targets_fail_closed() {
        assert!(matches!(
            SequenceClassifier::try_new(SequenceClassifierConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: 8,
                    max_tokens: 4,
                    embedding_dim: 4,
                    hidden_dim: 3,
                    seed: 101,
                },
                num_classes: MAX_SEQUENCE_CLASSES + 1,
                head_seed: 103,
            }),
            Err(SciRustError::CapacityExceeded { .. })
        ));
        let model = controlled_classifier();
        assert!(matches!(
            model.loss_and_gradients(&[1, 2], 2),
            Err(SciRustError::Index { idx: 2, len: 2 })
        ));
    }
}

#[cfg(test)]
mod gather_training_tests {
    use super::*;
    #[test]
    fn gather_classifier_training_matches_dense_across_updates() {
        let config = SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 8,
                max_tokens: 12,
                embedding_dim: 4,
                hidden_dim: 7,
                seed: 42,
            },
            num_classes: 2,
            head_seed: 7,
        };
        let mut dense = SequenceClassifier::try_new(config).unwrap();
        let mut gather = dense.clone();
        let mut a = SequenceClassifierAdamW::try_new(0.01, &dense).unwrap();
        let mut b = SequenceClassifierAdamW::try_new(0.01, &gather).unwrap();
        for step in 0..12 {
            let tokens = &[1, 2, 1, 3];
            let label = step % 2;
            assert_eq!(
                dense.loss_and_gradients(tokens, label).unwrap(),
                gather.loss_and_gradients_gather(tokens, label).unwrap()
            );
            assert_eq!(
                dense.train_step(&mut a, tokens, label).unwrap(),
                gather.train_step_gather(&mut b, tokens, label).unwrap()
            );
            assert_eq!(dense, gather);
        }
    }
}

#[cfg(test)]
mod weighted_training_tests {
    use super::*;
    #[test]
    fn weighted_training_masks_gradients_and_reduces_nll() {
        let config = SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 8,
                max_tokens: 12,
                embedding_dim: 4,
                hidden_dim: 7,
                seed: 42,
            },
            num_classes: 2,
            head_seed: 7,
        };
        let mut model = SequenceClassifier::try_new(config).unwrap();
        let tokens = &[1, 2, 3];
        assert_eq!(
            model.loss_and_gradients(tokens, 1).unwrap(),
            model
                .loss_and_gradients_weighted(tokens, &[1.; 3], 1)
                .unwrap()
        );
        let weights = &[0., 1., 0.];
        let (before, gradients) = model
            .loss_and_gradients_weighted(tokens, weights, 1)
            .unwrap();
        assert_eq!(&gradients.encoder.token_embeddings()[4..8], &[0.; 4]);
        assert_eq!(&gradients.encoder.token_embeddings()[12..16], &[0.; 4]);
        let mut optimizer = SequenceClassifierAdamW::try_new(0.01, &model).unwrap();
        for _ in 0..20 {
            model
                .train_step_weighted(&mut optimizer, tokens, weights, 1)
                .unwrap();
        }
        assert!(
            model
                .loss_and_gradients_weighted(tokens, weights, 1)
                .unwrap()
                .0
                < before
        );
        let saved = model.clone();
        assert!(model
            .train_step_weighted(&mut optimizer, tokens, &[0.; 3], 1)
            .is_err());
        assert_eq!(model, saved);
    }
}
