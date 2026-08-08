//! Shared sequence cognitive parameterization.
//!
//! One bounded [`crate::SequenceEncoder`] is owned by classification,
//! preference, symbolic-satisfaction and contradiction heads, while retrieval
//! uses the same pooled representation directly. This removes the architectural
//! split where each cognitive signal owned an unrelated encoder copy.
//!
//! This module only parameterizes and evaluates soft model signals. Hard rules,
//! evidence admission, policy and tool authority remain outside the model.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    SequenceEncoder, SequenceEncoderConfig, MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_HIDDEN_DIM,
    MAX_SEQUENCE_PARAMETERS, MAX_SEQUENCE_RETRIEVAL_CANDIDATES, MAX_SEQUENCE_SYMBOLIC_RULES,
};

/// Contradiction remains an explicitly binary head.
pub const COGNITIVE_CONTRADICTION_CLASSES: usize = 2;
/// Conservative cap across encoder plus all bounded cognitive heads.
pub const MAX_SEQUENCE_COGNITIVE_PARAMETERS: usize = MAX_SEQUENCE_PARAMETERS
    + MAX_SEQUENCE_HIDDEN_DIM * MAX_SEQUENCE_CLASSES
    + MAX_SEQUENCE_CLASSES
    + MAX_SEQUENCE_HIDDEN_DIM
    + 1
    + MAX_SEQUENCE_HIDDEN_DIM * MAX_SEQUENCE_SYMBOLIC_RULES
    + MAX_SEQUENCE_SYMBOLIC_RULES
    + MAX_SEQUENCE_HIDDEN_DIM * COGNITIVE_CONTRADICTION_CLASSES
    + COGNITIVE_CONTRADICTION_CLASSES;

/// Architecture of one shared encoder plus bounded cognitive heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveConfig {
    pub encoder: SequenceEncoderConfig,
    pub num_classes: usize,
    pub num_rules: usize,
    pub classification_seed: u64,
    pub preference_seed: u64,
    pub symbolic_seed: u64,
    pub contradiction_seed: u64,
}

impl SequenceCognitiveConfig {
    /// Checked total number of trainable scalars.
    pub fn parameter_count(self) -> SciRustResult<usize> {
        let hidden = self.encoder.hidden_dim;
        let classification = hidden
            .checked_mul(self.num_classes)
            .and_then(|value| value.checked_add(self.num_classes))
            .ok_or(SciRustError::Overflow)?;
        let preference = hidden.checked_add(1).ok_or(SciRustError::Overflow)?;
        let symbolic = hidden
            .checked_mul(self.num_rules)
            .and_then(|value| value.checked_add(self.num_rules))
            .ok_or(SciRustError::Overflow)?;
        let contradiction = hidden
            .checked_mul(COGNITIVE_CONTRADICTION_CLASSES)
            .and_then(|value| value.checked_add(COGNITIVE_CONTRADICTION_CLASSES))
            .ok_or(SciRustError::Overflow)?;
        self.encoder
            .parameter_count()?
            .checked_add(classification)
            .and_then(|value| value.checked_add(preference))
            .and_then(|value| value.checked_add(symbolic))
            .and_then(|value| value.checked_add(contradiction))
            .ok_or(SciRustError::Overflow)
    }

    fn validate(self) -> SciRustResult<()> {
        if self.num_classes < 2 || self.num_rules == 0 {
            return Err(SciRustError::Empty);
        }
        if self.num_classes > MAX_SEQUENCE_CLASSES {
            return Err(SciRustError::CapacityExceeded {
                requested: self.num_classes,
                maximum: MAX_SEQUENCE_CLASSES,
            });
        }
        if self.num_rules > MAX_SEQUENCE_SYMBOLIC_RULES {
            return Err(SciRustError::CapacityExceeded {
                requested: self.num_rules,
                maximum: MAX_SEQUENCE_SYMBOLIC_RULES,
            });
        }
        let parameters = self.parameter_count()?;
        if parameters > MAX_SEQUENCE_COGNITIVE_PARAMETERS {
            return Err(SciRustError::CapacityExceeded {
                requested: parameters,
                maximum: MAX_SEQUENCE_COGNITIVE_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// One encoder with task-specific linear heads.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceCognitiveHeads {
    config: SequenceCognitiveConfig,
    encoder: SequenceEncoder,
    classification_weights: Vec<f32>,
    classification_bias: Vec<f32>,
    preference_weights: Vec<f32>,
    preference_bias: Vec<f32>,
    symbolic_weights: Vec<f32>,
    symbolic_bias: Vec<f32>,
    contradiction_weights: Vec<f32>,
    contradiction_bias: Vec<f32>,
}

impl SequenceCognitiveHeads {
    /// Deterministic initialization from explicit encoder/head seeds.
    pub fn try_new(config: SequenceCognitiveConfig) -> SciRustResult<Self> {
        config.validate()?;
        let encoder = SequenceEncoder::try_new(config.encoder)?;
        let hidden = config.encoder.hidden_dim;
        let mut classification_state = config.classification_seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut preference_state = config.preference_seed ^ 0xD1B5_4A32_D192_ED03;
        let mut symbolic_state = config.symbolic_seed ^ 0x94D0_49BB_1331_11EB;
        let mut contradiction_state = config.contradiction_seed ^ 0x8538_EC4A_19AB_4C2D;
        let classification_weights = deterministic_weights(
            hidden
                .checked_mul(config.num_classes)
                .ok_or(SciRustError::Overflow)?,
            hidden,
            &mut classification_state,
        )?;
        let preference_weights =
            deterministic_weights(hidden, hidden, &mut preference_state)?;
        let symbolic_weights = deterministic_weights(
            hidden
                .checked_mul(config.num_rules)
                .ok_or(SciRustError::Overflow)?,
            hidden,
            &mut symbolic_state,
        )?;
        let contradiction_weights = deterministic_weights(
            hidden
                .checked_mul(COGNITIVE_CONTRADICTION_CLASSES)
                .ok_or(SciRustError::Overflow)?,
            hidden,
            &mut contradiction_state,
        )?;
        Self::from_parts(
            config,
            encoder,
            classification_weights,
            vec![0.0; config.num_classes],
            preference_weights,
            vec![0.0],
            symbolic_weights,
            vec![0.0; config.num_rules],
            contradiction_weights,
            vec![0.0; COGNITIVE_CONTRADICTION_CLASSES],
        )
    }

    /// Reconstruct explicit bounded tensors after hostile-input validation.
    #[allow(
        clippy::too_many_arguments,
        reason = "the explicit reconstruction boundary names every persisted cognitive tensor"
    )]
    pub fn from_parts(
        config: SequenceCognitiveConfig,
        encoder: SequenceEncoder,
        classification_weights: Vec<f32>,
        classification_bias: Vec<f32>,
        preference_weights: Vec<f32>,
        preference_bias: Vec<f32>,
        symbolic_weights: Vec<f32>,
        symbolic_bias: Vec<f32>,
        contradiction_weights: Vec<f32>,
        contradiction_bias: Vec<f32>,
    ) -> SciRustResult<Self> {
        config.validate()?;
        if encoder.config() != config.encoder {
            return Err(SciRustError::Shape {
                lhs: vec![encoder.config().hidden_dim],
                rhs: vec![config.encoder.hidden_dim],
            });
        }
        let hidden = config.encoder.hidden_dim;
        validate_tensor(
            &classification_weights,
            hidden
                .checked_mul(config.num_classes)
                .ok_or(SciRustError::Overflow)?,
        )?;
        validate_tensor(&classification_bias, config.num_classes)?;
        validate_tensor(&preference_weights, hidden)?;
        validate_tensor(&preference_bias, 1)?;
        validate_tensor(
            &symbolic_weights,
            hidden
                .checked_mul(config.num_rules)
                .ok_or(SciRustError::Overflow)?,
        )?;
        validate_tensor(&symbolic_bias, config.num_rules)?;
        validate_tensor(
            &contradiction_weights,
            hidden
                .checked_mul(COGNITIVE_CONTRADICTION_CLASSES)
                .ok_or(SciRustError::Overflow)?,
        )?;
        validate_tensor(&contradiction_bias, COGNITIVE_CONTRADICTION_CLASSES)?;
        Ok(Self {
            config,
            encoder,
            classification_weights,
            classification_bias,
            preference_weights,
            preference_bias,
            symbolic_weights,
            symbolic_bias,
            contradiction_weights,
            contradiction_bias,
        })
    }

    #[must_use]
    pub const fn config(&self) -> SequenceCognitiveConfig {
        self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &SequenceEncoder {
        &self.encoder
    }

    #[must_use]
    pub fn classification_weights(&self) -> &[f32] {
        &self.classification_weights
    }

    #[must_use]
    pub fn classification_bias(&self) -> &[f32] {
        &self.classification_bias
    }

    #[must_use]
    pub fn preference_weights(&self) -> &[f32] {
        &self.preference_weights
    }

    #[must_use]
    pub fn preference_bias(&self) -> &[f32] {
        &self.preference_bias
    }

    #[must_use]
    pub fn symbolic_weights(&self) -> &[f32] {
        &self.symbolic_weights
    }

    #[must_use]
    pub fn symbolic_bias(&self) -> &[f32] {
        &self.symbolic_bias
    }

    #[must_use]
    pub fn contradiction_weights(&self) -> &[f32] {
        &self.contradiction_weights
    }

    #[must_use]
    pub fn contradiction_bias(&self) -> &[f32] {
        &self.contradiction_bias
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.encoder.parameter_count()
            + self.classification_weights.len()
            + self.classification_bias.len()
            + self.preference_weights.len()
            + self.preference_bias.len()
            + self.symbolic_weights.len()
            + self.symbolic_bias.len()
            + self.contradiction_weights.len()
            + self.contradiction_bias.len()
    }

    /// Feedback/class probabilities over the shared representation.
    pub fn classification_probabilities(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        let pooled = self.encoder.forward(token_ids)?;
        let logits = dense_logits(
            &pooled,
            &self.classification_weights,
            &self.classification_bias,
            self.config.num_classes,
        )?;
        stable_softmax(&logits)
    }

    /// Deterministic preference scalar over the shared representation.
    pub fn preference_score(&self, token_ids: &[u16]) -> SciRustResult<f32> {
        let pooled = self.encoder.forward(token_ids)?;
        dense_scalar(&pooled, &self.preference_weights, self.preference_bias[0])
    }

    /// One bounded sigmoid satisfaction per configured soft rule.
    pub fn symbolic_satisfactions(&self, token_ids: &[u16]) -> SciRustResult<Vec<f32>> {
        let pooled = self.encoder.forward(token_ids)?;
        let logits = dense_logits(
            &pooled,
            &self.symbolic_weights,
            &self.symbolic_bias,
            self.config.num_rules,
        )?;
        let mut satisfactions = Vec::with_capacity(logits.len());
        for logit in logits {
            satisfactions.push(stable_sigmoid(logit));
        }
        Ok(satisfactions)
    }

    /// Binary `[clear, contradiction]` probabilities for an already-framed
    /// pair sequence.
    pub fn contradiction_probabilities(
        &self,
        pair_token_ids: &[u16],
    ) -> SciRustResult<[f32; COGNITIVE_CONTRADICTION_CLASSES]> {
        let pooled = self.encoder.forward(pair_token_ids)?;
        let logits = dense_logits(
            &pooled,
            &self.contradiction_weights,
            &self.contradiction_bias,
            COGNITIVE_CONTRADICTION_CLASSES,
        )?;
        let probabilities = stable_softmax(&logits)?;
        probabilities.try_into().map_err(|_| SciRustError::Shape {
            lhs: vec![probabilities.len()],
            rhs: vec![COGNITIVE_CONTRADICTION_CLASSES],
        })
    }

    /// Dot-product similarities for retrieval using the same encoder and no
    /// separate retrieval parameter copy.
    pub fn retrieval_similarities(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
    ) -> SciRustResult<Vec<f32>> {
        if candidates.len() < 2 {
            return Err(SciRustError::Empty);
        }
        if candidates.len() > MAX_SEQUENCE_RETRIEVAL_CANDIDATES {
            return Err(SciRustError::CapacityExceeded {
                requested: candidates.len(),
                maximum: MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
            });
        }
        let query = self.encoder.forward(query)?;
        let mut similarities = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let candidate = self.encoder.forward(candidate)?;
            similarities.push(dot(&query, &candidate)?);
        }
        Ok(similarities)
    }

    /// Highest retrieval similarity. Exact ties preserve the first candidate.
    pub fn retrieval_select_best(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
    ) -> SciRustResult<usize> {
        let similarities = self.retrieval_similarities(query, candidates)?;
        let mut best_index = 0usize;
        let mut best_score = similarities[0];
        for (index, &score) in similarities.iter().enumerate().skip(1) {
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        Ok(best_index)
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

fn validate_tensor(values: &[f32], expected: usize) -> SciRustResult<()> {
    if values.len() != expected {
        return Err(SciRustError::Shape {
            lhs: vec![values.len()],
            rhs: vec![expected],
        });
    }
    for &value in values {
        ensure_finite(value)?;
    }
    Ok(())
}

fn dense_logits(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    outputs: usize,
) -> SciRustResult<Vec<f32>> {
    if outputs == 0 || input.is_empty() {
        return Err(SciRustError::Empty);
    }
    let expected = input
        .len()
        .checked_mul(outputs)
        .ok_or(SciRustError::Overflow)?;
    validate_tensor(weights, expected)?;
    validate_tensor(bias, outputs)?;
    let mut logits = bias.to_vec();
    for (feature, &value) in input.iter().enumerate() {
        for output in 0..outputs {
            logits[output] += value * weights[feature * outputs + output];
            ensure_finite(logits[output])?;
        }
    }
    Ok(logits)
}

fn dense_scalar(input: &[f32], weights: &[f32], bias: f32) -> SciRustResult<f32> {
    validate_tensor(weights, input.len())?;
    ensure_finite(bias)?;
    let mut value = bias;
    for (&feature, &weight) in input.iter().zip(weights) {
        value += feature * weight;
        ensure_finite(value)?;
    }
    Ok(value)
}

fn stable_softmax(logits: &[f32]) -> SciRustResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(SciRustError::Empty);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ensure_finite(max)?;
    let mut probabilities = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let value = (logit - max).exp();
        ensure_finite(value)?;
        probabilities.push(value);
        sum += value;
        ensure_finite(sum)?;
    }
    if sum <= 0.0 {
        return Err(SciRustError::NonFinite);
    }
    for probability in &mut probabilities {
        *probability /= sum;
        ensure_finite(*probability)?;
    }
    Ok(probabilities)
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
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
        ensure_finite(value)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SequenceCognitiveConfig {
        SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 16,
                max_tokens: 8,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 1201,
            },
            num_classes: 3,
            num_rules: 2,
            classification_seed: 1207,
            preference_seed: 1213,
            symbolic_seed: 1217,
            contradiction_seed: 1223,
        }
    }

    #[test]
    fn deterministic_initialization_and_inference_repeat_exactly() {
        let left = SequenceCognitiveHeads::try_new(config()).expect("left");
        let right = SequenceCognitiveHeads::try_new(config()).expect("right");
        assert_eq!(left, right);
        assert_eq!(
            left.classification_probabilities(&[1, 2, 3])
                .expect("left probabilities"),
            right
                .classification_probabilities(&[1, 2, 3])
                .expect("right probabilities")
        );
        assert_eq!(
            left.preference_score(&[1, 2, 3]).expect("left score"),
            right.preference_score(&[1, 2, 3]).expect("right score")
        );
    }

    #[test]
    fn all_heads_share_one_bounded_encoder_parameterization() {
        let model = SequenceCognitiveHeads::try_new(config()).expect("model");
        assert_eq!(model.parameter_count(), config().parameter_count().expect("count"));
        let classes = model
            .classification_probabilities(&[1, 2, 3])
            .expect("classes");
        assert_eq!(classes.len(), 3);
        assert!((classes.iter().copied().sum::<f32>() - 1.0).abs() < 1.0e-5);
        let rules = model
            .symbolic_satisfactions(&[1, 2, 3])
            .expect("rules");
        assert_eq!(rules.len(), 2);
        assert!(rules.iter().all(|value| (0.0..=1.0).contains(value)));
        let contradiction = model
            .contradiction_probabilities(&[1, 2, 3])
            .expect("contradiction");
        assert!((contradiction.iter().copied().sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn retrieval_reuses_shared_encoder_and_preserves_first_tie() {
        let model = SequenceCognitiveHeads::try_new(config()).expect("model");
        let candidate = [4, 5, 6];
        let best = model
            .retrieval_select_best(&[1, 2, 3], &[&candidate, &candidate])
            .expect("best");
        assert_eq!(best, 0);
    }

    #[test]
    fn hostile_head_counts_fail_closed() {
        let mut invalid = config();
        invalid.num_classes = 1;
        assert_eq!(
            SequenceCognitiveHeads::try_new(invalid),
            Err(SciRustError::Empty)
        );
        let mut invalid = config();
        invalid.num_rules = MAX_SEQUENCE_SYMBOLIC_RULES + 1;
        assert!(matches!(
            SequenceCognitiveHeads::try_new(invalid),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }
}
