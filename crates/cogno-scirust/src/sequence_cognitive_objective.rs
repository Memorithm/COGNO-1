//! Joint differentiable objective for the shared sequence cognitive heads.
//!
//! Classification, pairwise preference, symbolic satisfaction, contradiction
//! and retrieval are assembled on one bounded tape and contribute to one
//! weighted scalar loss. Every sequence view starts from the same owned
//! [`crate::SequenceEncoder`] parameter values; after one reverse sweep their
//! encoder gradients are summed before a single AdamW update.
//!
//! This remains a soft model-training objective. Deterministic hard constraints,
//! evidence admission, measured runtime cost, persistence and tools stay outside
//! the differentiable compensation path.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::{
    AdamW, InfoNCE, Optimizer, PairwiseLoss, SequenceCognitiveHeads, SequenceEncoder,
    SequenceEncoderGraph, Shape, Tape, Tensor, Var, COGNITIVE_CONTRADICTION_CLASSES,
    MAX_SEQUENCE_RETRIEVAL_CANDIDATES, SEQUENCE_ENCODER_TAPE_NODES,
};

/// Worst-case tape nodes for all five cognitive tasks in one joint batch.
pub const SEQUENCE_COGNITIVE_JOINT_TAPE_NODES: usize =
    SEQUENCE_ENCODER_TAPE_NODES * (MAX_SEQUENCE_RETRIEVAL_CANDIDATES + 6) + 192;

/// Classification observation over one already-tokenized sequence.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveClassification<'a> {
    pub token_ids: &'a [u16],
    pub target_class: usize,
}

/// Pairwise preference observation.
#[derive(Clone, Copy, Debug)]
pub struct CognitivePreference<'a> {
    pub preferred: &'a [u16],
    pub dispreferred: &'a [u16],
    pub margin: f32,
}

/// Host-provided per-rule satisfaction targets in `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveSymbolic<'a> {
    pub token_ids: &'a [u16],
    pub targets: &'a [f32],
}

/// Binary contradiction observation over an already-framed pair sequence.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveContradiction<'a> {
    pub pair_token_ids: &'a [u16],
    pub contradicts: bool,
}

/// Multi-candidate retrieval observation.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveRetrieval<'a> {
    pub query: &'a [u16],
    pub candidates: &'a [&'a [u16]],
    pub positive_idx: usize,
    pub temperature: f32,
}

/// One explicit multi-task training batch.
#[derive(Clone, Copy, Debug)]
pub struct SequenceCognitiveBatch<'a> {
    pub classification: CognitiveClassification<'a>,
    pub preference: CognitivePreference<'a>,
    pub symbolic: CognitiveSymbolic<'a>,
    pub contradiction: CognitiveContradiction<'a>,
    pub retrieval: CognitiveRetrieval<'a>,
}

/// Non-negative finite weights for the five soft cognitive losses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceCognitiveLossWeights {
    pub classification: f32,
    pub preference: f32,
    pub symbolic: f32,
    pub contradiction: f32,
    pub retrieval: f32,
}

impl Default for SequenceCognitiveLossWeights {
    fn default() -> Self {
        Self {
            classification: 1.0,
            preference: 1.0,
            symbolic: 1.0,
            contradiction: 1.0,
            retrieval: 1.0,
        }
    }
}

impl SequenceCognitiveLossWeights {
    fn validate(self) -> SciRustResult<()> {
        let weights = [
            self.classification,
            self.preference,
            self.symbolic,
            self.contradiction,
            self.retrieval,
        ];
        let mut any_positive = false;
        for weight in weights {
            ensure_finite(weight)?;
            if weight < 0.0 {
                return Err(SciRustError::Shape {
                    lhs: vec![0],
                    rhs: vec![1],
                });
            }
            any_positive |= weight > 0.0;
        }
        if !any_positive {
            return Err(SciRustError::Empty);
        }
        Ok(())
    }
}

/// Raw component losses plus the weighted total optimized by one backward pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceCognitiveLossReport {
    pub classification: f32,
    pub preference: f32,
    pub symbolic: f32,
    pub contradiction: f32,
    pub retrieval: f32,
    pub weighted_total: f32,
}

/// Exact gradients for one shared encoder and all trainable cognitive heads.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceCognitiveGradients {
    token_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    mixing_weights: Vec<f32>,
    classification_weights: Vec<f32>,
    classification_bias: Vec<f32>,
    preference_weights: Vec<f32>,
    preference_bias: Vec<f32>,
    symbolic_weights: Vec<f32>,
    symbolic_bias: Vec<f32>,
    contradiction_weights: Vec<f32>,
    contradiction_bias: Vec<f32>,
}

impl SequenceCognitiveGradients {
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
}

impl SequenceCognitiveHeads {
    /// Compute all five connected losses and their exact shared gradients with
    /// one reverse-mode sweep.
    pub fn joint_loss_and_gradients(
        &self,
        batch: SequenceCognitiveBatch<'_>,
        weights: SequenceCognitiveLossWeights,
    ) -> SciRustResult<(SequenceCognitiveLossReport, SequenceCognitiveGradients)> {
        weights.validate()?;
        self.validate_joint_batch(batch)?;
        let max_elements = self.required_joint_max_elements(batch)?;
        let mut tape = Tape::new(SEQUENCE_COGNITIVE_JOINT_TAPE_NODES, max_elements);
        let hidden = self.config().encoder.hidden_dim;

        let classification_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[hidden, self.config().num_classes])?,
            self.classification_weights().to_vec(),
            max_elements,
        )?)?;
        let classification_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config().num_classes])?,
            self.classification_bias().to_vec(),
            max_elements,
        )?)?;
        let preference_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[hidden, 1])?,
            self.preference_weights().to_vec(),
            max_elements,
        )?)?;
        let preference_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, 1])?,
            self.preference_bias().to_vec(),
            max_elements,
        )?)?;
        let symbolic_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[hidden, self.config().num_rules])?,
            self.symbolic_weights().to_vec(),
            max_elements,
        )?)?;
        let symbolic_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config().num_rules])?,
            self.symbolic_bias().to_vec(),
            max_elements,
        )?)?;
        let contradiction_weights = tape.variable(Tensor::try_new(
            Shape::try_new(&[hidden, COGNITIVE_CONTRADICTION_CLASSES])?,
            self.contradiction_weights().to_vec(),
            max_elements,
        )?)?;
        let contradiction_bias = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, COGNITIVE_CONTRADICTION_CLASSES])?,
            self.contradiction_bias().to_vec(),
            max_elements,
        )?)?;

        let mut encoder_graphs = Vec::with_capacity(batch.retrieval.candidates.len() + 6);

        let classification_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.classification.token_ids)?;
        let classification_logits = tape.matmul(classification_graph.pooled(), classification_weights)?;
        let classification_logits = tape.add(classification_logits, classification_bias)?;
        let classification_loss = nll_loss(
            &mut tape,
            classification_logits,
            batch.classification.target_class,
            self.config().num_classes,
            max_elements,
        )?;
        encoder_graphs.push(classification_graph);

        let preferred_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.preference.preferred)?;
        let dispreferred_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.preference.dispreferred)?;
        let preferred_score = dense_scalar_var(
            &mut tape,
            preferred_graph.pooled(),
            preference_weights,
            preference_bias,
        )?;
        let dispreferred_score = dense_scalar_var(
            &mut tape,
            dispreferred_graph.pooled(),
            preference_weights,
            preference_bias,
        )?;
        let preference_loss = PairwiseLoss::try_new(batch.preference.margin, 1, max_elements)?
            .loss_vars(&mut tape, preferred_score, dispreferred_score)?;
        encoder_graphs.push(preferred_graph);
        encoder_graphs.push(dispreferred_graph);

        let symbolic_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.symbolic.token_ids)?;
        let symbolic_logits = tape.matmul(symbolic_graph.pooled(), symbolic_weights)?;
        let symbolic_logits = tape.add(symbolic_logits, symbolic_bias)?;
        let symbolic_predictions = tape.sigmoid(symbolic_logits)?;
        let symbolic_targets = tape.variable(Tensor::try_new(
            Shape::try_new(&[1, self.config().num_rules])?,
            batch.symbolic.targets.to_vec(),
            max_elements,
        )?)?;
        let symbolic_difference = tape.sub(symbolic_predictions, symbolic_targets)?;
        let symbolic_squared = tape.mul(symbolic_difference, symbolic_difference)?;
        let symbolic_loss = tape.sum(symbolic_squared)?;
        let symbolic_loss = tape.scale(symbolic_loss, (self.config().num_rules as f32).recip())?;
        encoder_graphs.push(symbolic_graph);

        let contradiction_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.contradiction.pair_token_ids)?;
        let contradiction_logits = tape.matmul(contradiction_graph.pooled(), contradiction_weights)?;
        let contradiction_logits = tape.add(contradiction_logits, contradiction_bias)?;
        let contradiction_target = if batch.contradiction.contradicts { 1 } else { 0 };
        let contradiction_loss = nll_loss(
            &mut tape,
            contradiction_logits,
            contradiction_target,
            COGNITIVE_CONTRADICTION_CLASSES,
            max_elements,
        )?;
        encoder_graphs.push(contradiction_graph);

        let query_graph = self
            .encoder()
            .append_to_tape(&mut tape, batch.retrieval.query)?;
        let mut similarities = Vec::with_capacity(batch.retrieval.candidates.len());
        let mut candidate_graphs = Vec::with_capacity(batch.retrieval.candidates.len());
        for candidate in batch.retrieval.candidates {
            let candidate_graph = self.encoder().append_to_tape(&mut tape, candidate)?;
            let product = tape.mul(query_graph.pooled(), candidate_graph.pooled())?;
            similarities.push(tape.sum(product)?);
            candidate_graphs.push(candidate_graph);
        }
        let retrieval_loss = InfoNCE::try_new(
            batch.retrieval.temperature,
            MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
            max_elements,
        )?
        .loss_similarity_vars(&mut tape, &similarities, batch.retrieval.positive_idx)?;
        encoder_graphs.push(query_graph);
        encoder_graphs.extend(candidate_graphs);

        let weighted = [
            tape.scale(classification_loss, weights.classification)?,
            tape.scale(preference_loss, weights.preference)?,
            tape.scale(symbolic_loss, weights.symbolic)?,
            tape.scale(contradiction_loss, weights.contradiction)?,
            tape.scale(retrieval_loss, weights.retrieval)?,
        ];
        let mut total = weighted[0];
        for &component in &weighted[1..] {
            total = tape.add(total, component)?;
        }
        tape.backward(total)?;

        let report = SequenceCognitiveLossReport {
            classification: scalar_value(&tape, classification_loss)?,
            preference: scalar_value(&tape, preference_loss)?,
            symbolic: scalar_value(&tape, symbolic_loss)?,
            contradiction: scalar_value(&tape, contradiction_loss)?,
            retrieval: scalar_value(&tape, retrieval_loss)?,
            weighted_total: scalar_value(&tape, total)?,
        };

        let mut gradients = SequenceCognitiveGradients::zeros(self);
        for graph in encoder_graphs {
            let encoder_gradients = self.encoder().gradients_from_tape(&tape, graph);
            add_assign_checked(
                &mut gradients.token_embeddings,
                encoder_gradients.token_embeddings(),
            )?;
            add_assign_checked(
                &mut gradients.position_embeddings,
                encoder_gradients.position_embeddings(),
            )?;
            add_assign_checked(
                &mut gradients.mixing_weights,
                encoder_gradients.mixing_weights(),
            )?;
        }
        gradients.classification_weights = tape.grad_of(classification_weights).to_vec();
        gradients.classification_bias = tape.grad_of(classification_bias).to_vec();
        gradients.preference_weights = tape.grad_of(preference_weights).to_vec();
        gradients.preference_bias = tape.grad_of(preference_bias).to_vec();
        gradients.symbolic_weights = tape.grad_of(symbolic_weights).to_vec();
        gradients.symbolic_bias = tape.grad_of(symbolic_bias).to_vec();
        gradients.contradiction_weights = tape.grad_of(contradiction_weights).to_vec();
        gradients.contradiction_bias = tape.grad_of(contradiction_bias).to_vec();
        gradients.validate_finite()?;
        Ok((report, gradients))
    }

    /// Apply one checked AdamW update for the complete joint cognitive batch.
    pub fn train_joint_step(
        &mut self,
        optimizer: &mut SequenceCognitiveAdamW,
        batch: SequenceCognitiveBatch<'_>,
        weights: SequenceCognitiveLossWeights,
    ) -> SciRustResult<SequenceCognitiveLossReport> {
        let (report, gradients) = self.joint_loss_and_gradients(batch, weights)?;
        optimizer.step(self, &gradients)?;
        Ok(report)
    }

    fn validate_joint_batch(&self, batch: SequenceCognitiveBatch<'_>) -> SciRustResult<()> {
        if batch.classification.target_class >= self.config().num_classes {
            return Err(SciRustError::Index {
                idx: batch.classification.target_class,
                len: self.config().num_classes,
            });
        }
        ensure_finite(batch.preference.margin)?;
        if batch.preference.margin < 0.0 {
            return Err(SciRustError::Shape {
                lhs: vec![0],
                rhs: vec![1],
            });
        }
        if batch.symbolic.targets.len() != self.config().num_rules {
            return Err(SciRustError::Shape {
                lhs: vec![batch.symbolic.targets.len()],
                rhs: vec![self.config().num_rules],
            });
        }
        for &target in batch.symbolic.targets {
            ensure_finite(target)?;
            if !(0.0..=1.0).contains(&target) {
                return Err(SciRustError::Shape {
                    lhs: vec![0],
                    rhs: vec![1],
                });
            }
        }
        if batch.retrieval.candidates.len() < 2 {
            return Err(SciRustError::Empty);
        }
        if batch.retrieval.candidates.len() > MAX_SEQUENCE_RETRIEVAL_CANDIDATES {
            return Err(SciRustError::CapacityExceeded {
                requested: batch.retrieval.candidates.len(),
                maximum: MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
            });
        }
        if batch.retrieval.positive_idx >= batch.retrieval.candidates.len() {
            return Err(SciRustError::Index {
                idx: batch.retrieval.positive_idx,
                len: batch.retrieval.candidates.len(),
            });
        }
        if batch.retrieval.temperature <= 0.0 || !batch.retrieval.temperature.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        Ok(())
    }

    fn required_joint_max_elements(&self, batch: SequenceCognitiveBatch<'_>) -> SciRustResult<usize> {
        let encoder = self.encoder();
        let mut required = encoder.required_max_elements(batch.classification.token_ids.len())?;
        required = required.max(encoder.required_max_elements(batch.preference.preferred.len())?);
        required = required.max(encoder.required_max_elements(batch.preference.dispreferred.len())?);
        required = required.max(encoder.required_max_elements(batch.symbolic.token_ids.len())?);
        required = required.max(
            encoder.required_max_elements(batch.contradiction.pair_token_ids.len())?,
        );
        required = required.max(encoder.required_max_elements(batch.retrieval.query.len())?);
        for candidate in batch.retrieval.candidates {
            required = required.max(encoder.required_max_elements(candidate.len())?);
        }
        let hidden = self.config().encoder.hidden_dim;
        required = required.max(
            hidden
                .checked_mul(self.config().num_classes)
                .ok_or(SciRustError::Overflow)?,
        );
        required = required.max(
            hidden
                .checked_mul(self.config().num_rules)
                .ok_or(SciRustError::Overflow)?,
        );
        required = required.max(hidden * COGNITIVE_CONTRADICTION_CLASSES);
        required = required.max(self.config().num_classes);
        required = required.max(self.config().num_rules);
        required = required.max(batch.retrieval.candidates.len());
        Ok(required)
    }
}

impl SequenceCognitiveGradients {
    fn zeros(model: &SequenceCognitiveHeads) -> Self {
        Self {
            token_embeddings: vec![0.0; model.encoder().token_embeddings().len()],
            position_embeddings: vec![0.0; model.encoder().position_embeddings().len()],
            mixing_weights: vec![0.0; model.encoder().mixing_weights().len()],
            classification_weights: vec![0.0; model.classification_weights().len()],
            classification_bias: vec![0.0; model.classification_bias().len()],
            preference_weights: vec![0.0; model.preference_weights().len()],
            preference_bias: vec![0.0; model.preference_bias().len()],
            symbolic_weights: vec![0.0; model.symbolic_weights().len()],
            symbolic_bias: vec![0.0; model.symbolic_bias().len()],
            contradiction_weights: vec![0.0; model.contradiction_weights().len()],
            contradiction_bias: vec![0.0; model.contradiction_bias().len()],
        }
    }

    fn validate_finite(&self) -> SciRustResult<()> {
        for values in [
            self.token_embeddings.as_slice(),
            self.position_embeddings.as_slice(),
            self.mixing_weights.as_slice(),
            self.classification_weights.as_slice(),
            self.classification_bias.as_slice(),
            self.preference_weights.as_slice(),
            self.preference_bias.as_slice(),
            self.symbolic_weights.as_slice(),
            self.symbolic_bias.as_slice(),
            self.contradiction_weights.as_slice(),
            self.contradiction_bias.as_slice(),
        ] {
            for &value in values {
                ensure_finite(value)?;
            }
        }
        Ok(())
    }
}

/// AdamW state for the single encoder and every trainable cognitive head.
#[derive(Clone, Debug)]
pub struct SequenceCognitiveAdamW {
    token_embeddings: AdamW,
    position_embeddings: AdamW,
    mixing_weights: AdamW,
    classification_weights: AdamW,
    classification_bias: AdamW,
    preference_weights: AdamW,
    preference_bias: AdamW,
    symbolic_weights: AdamW,
    symbolic_bias: AdamW,
    contradiction_weights: AdamW,
    contradiction_bias: AdamW,
}

impl SequenceCognitiveAdamW {
    pub fn try_new(learning_rate: f32, model: &SequenceCognitiveHeads) -> SciRustResult<Self> {
        Ok(Self {
            token_embeddings: AdamW::try_new(learning_rate, model.encoder().token_embeddings().len())?,
            position_embeddings: AdamW::try_new(
                learning_rate,
                model.encoder().position_embeddings().len(),
            )?,
            mixing_weights: AdamW::try_new(learning_rate, model.encoder().mixing_weights().len())?,
            classification_weights: AdamW::try_new(
                learning_rate,
                model.classification_weights().len(),
            )?,
            classification_bias: AdamW::try_new(learning_rate, model.classification_bias().len())?,
            preference_weights: AdamW::try_new(learning_rate, model.preference_weights().len())?,
            preference_bias: AdamW::try_new(learning_rate, model.preference_bias().len())?,
            symbolic_weights: AdamW::try_new(learning_rate, model.symbolic_weights().len())?,
            symbolic_bias: AdamW::try_new(learning_rate, model.symbolic_bias().len())?,
            contradiction_weights: AdamW::try_new(
                learning_rate,
                model.contradiction_weights().len(),
            )?,
            contradiction_bias: AdamW::try_new(learning_rate, model.contradiction_bias().len())?,
        })
    }

    pub fn step(
        &mut self,
        model: &mut SequenceCognitiveHeads,
        gradients: &SequenceCognitiveGradients,
    ) -> SciRustResult<()> {
        let config = model.config();
        let mut token_embeddings = model.encoder().token_embeddings().to_vec();
        let mut position_embeddings = model.encoder().position_embeddings().to_vec();
        let mut mixing_weights = model.encoder().mixing_weights().to_vec();
        let mut classification_weights = model.classification_weights().to_vec();
        let mut classification_bias = model.classification_bias().to_vec();
        let mut preference_weights = model.preference_weights().to_vec();
        let mut preference_bias = model.preference_bias().to_vec();
        let mut symbolic_weights = model.symbolic_weights().to_vec();
        let mut symbolic_bias = model.symbolic_bias().to_vec();
        let mut contradiction_weights = model.contradiction_weights().to_vec();
        let mut contradiction_bias = model.contradiction_bias().to_vec();

        self.token_embeddings
            .step(&mut token_embeddings, &gradients.token_embeddings)?;
        self.position_embeddings
            .step(&mut position_embeddings, &gradients.position_embeddings)?;
        self.mixing_weights
            .step(&mut mixing_weights, &gradients.mixing_weights)?;
        self.classification_weights
            .step(&mut classification_weights, &gradients.classification_weights)?;
        self.classification_bias
            .step(&mut classification_bias, &gradients.classification_bias)?;
        self.preference_weights
            .step(&mut preference_weights, &gradients.preference_weights)?;
        self.preference_bias
            .step(&mut preference_bias, &gradients.preference_bias)?;
        self.symbolic_weights
            .step(&mut symbolic_weights, &gradients.symbolic_weights)?;
        self.symbolic_bias
            .step(&mut symbolic_bias, &gradients.symbolic_bias)?;
        self.contradiction_weights
            .step(&mut contradiction_weights, &gradients.contradiction_weights)?;
        self.contradiction_bias
            .step(&mut contradiction_bias, &gradients.contradiction_bias)?;

        let encoder = SequenceEncoder::from_parts(
            config.encoder,
            token_embeddings,
            position_embeddings,
            mixing_weights,
        )?;
        *model = SequenceCognitiveHeads::from_parts(
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
        )?;
        Ok(())
    }
}

fn dense_scalar_var(tape: &mut Tape, pooled: Var, weights: Var, bias: Var) -> SciRustResult<Var> {
    let score = tape.matmul(pooled, weights)?;
    tape.add(score, bias)
}

fn nll_loss(
    tape: &mut Tape,
    logits: Var,
    target_class: usize,
    classes: usize,
    max_elements: usize,
) -> SciRustResult<Var> {
    if target_class >= classes {
        return Err(SciRustError::Index {
            idx: target_class,
            len: classes,
        });
    }
    let log_probabilities = tape.log_softmax(logits)?;
    let mut target = vec![0.0f32; classes];
    target[target_class] = 1.0;
    let target = tape.variable(Tensor::try_new(
        Shape::try_new(&[1, classes])?,
        target,
        max_elements,
    )?)?;
    let selected = tape.mul(log_probabilities, target)?;
    let selected = tape.sum(selected)?;
    tape.neg(selected)
}

fn scalar_value(tape: &Tape, value: Var) -> SciRustResult<f32> {
    let value = tape
        .value_of(value)
        .as_slice()
        .first()
        .copied()
        .ok_or(SciRustError::Empty)?;
    ensure_finite(value)?;
    Ok(value)
}

fn add_assign_checked(target: &mut [f32], source: &[f32]) -> SciRustResult<()> {
    if target.len() != source.len() {
        return Err(SciRustError::Shape {
            lhs: vec![target.len()],
            rhs: vec![source.len()],
        });
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += *source;
        ensure_finite(*target)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SequenceCognitiveConfig, SequenceEncoderConfig};

    fn model() -> SequenceCognitiveHeads {
        SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: 24,
                max_tokens: 12,
                embedding_dim: 6,
                hidden_dim: 5,
                seed: 1301,
            },
            num_classes: 3,
            num_rules: 2,
            classification_seed: 1303,
            preference_seed: 1307,
            symbolic_seed: 1319,
            contradiction_seed: 1321,
        })
        .expect("model")
    }

    fn batch<'a>(
        class_tokens: &'a [u16],
        preferred: &'a [u16],
        dispreferred: &'a [u16],
        symbolic_tokens: &'a [u16],
        symbolic_targets: &'a [f32],
        contradiction_tokens: &'a [u16],
        query: &'a [u16],
        candidates: &'a [&'a [u16]],
    ) -> SequenceCognitiveBatch<'a> {
        SequenceCognitiveBatch {
            classification: CognitiveClassification {
                token_ids: class_tokens,
                target_class: 1,
            },
            preference: CognitivePreference {
                preferred,
                dispreferred,
                margin: 5.0,
            },
            symbolic: CognitiveSymbolic {
                token_ids: symbolic_tokens,
                targets: symbolic_targets,
            },
            contradiction: CognitiveContradiction {
                pair_token_ids: contradiction_tokens,
                contradicts: true,
            },
            retrieval: CognitiveRetrieval {
                query,
                candidates,
                positive_idx: 0,
                temperature: 0.5,
            },
        }
    }

    #[test]
    fn full_joint_backward_reaches_shared_encoder_and_every_head() {
        let model = model();
        let c0 = [2, 3, 4];
        let c1 = [8, 9, 10];
        let candidates: [&[u16]; 2] = [&c0, &c1];
        let (report, gradients) = model
            .joint_loss_and_gradients(
                batch(
                    &[1, 2, 3],
                    &[2, 4, 6],
                    &[7, 8, 9],
                    &[3, 5, 7],
                    &[1.0, 0.0],
                    &[4, 5, 6, 7],
                    &[2, 3, 4],
                    &candidates,
                ),
                SequenceCognitiveLossWeights::default(),
            )
            .expect("joint gradients");
        for loss in [
            report.classification,
            report.preference,
            report.symbolic,
            report.contradiction,
            report.retrieval,
            report.weighted_total,
        ] {
            assert!(loss.is_finite());
        }
        assert!(gradients.token_embeddings().iter().any(|value| *value != 0.0));
        assert!(gradients
            .classification_weights()
            .iter()
            .any(|value| *value != 0.0));
        assert!(gradients
            .preference_weights()
            .iter()
            .any(|value| *value != 0.0));
        assert!(gradients
            .symbolic_weights()
            .iter()
            .any(|value| *value != 0.0));
        assert!(gradients
            .contradiction_weights()
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn seeded_joint_training_step_is_exactly_deterministic() {
        let mut left = model();
        let mut right = model();
        let mut left_optimizer = SequenceCognitiveAdamW::try_new(0.01, &left).expect("left opt");
        let mut right_optimizer = SequenceCognitiveAdamW::try_new(0.01, &right).expect("right opt");
        let c0 = [2, 3, 4];
        let c1 = [8, 9, 10];
        let candidates: [&[u16]; 2] = [&c0, &c1];
        let make_batch = || {
            batch(
                &[1, 2, 3],
                &[2, 4, 6],
                &[7, 8, 9],
                &[3, 5, 7],
                &[1.0, 0.0],
                &[4, 5, 6, 7],
                &[2, 3, 4],
                &candidates,
            )
        };
        let left_report = left
            .train_joint_step(
                &mut left_optimizer,
                make_batch(),
                SequenceCognitiveLossWeights::default(),
            )
            .expect("left step");
        let right_report = right
            .train_joint_step(
                &mut right_optimizer,
                make_batch(),
                SequenceCognitiveLossWeights::default(),
            )
            .expect("right step");
        assert_eq!(left_report, right_report);
        assert_eq!(left, right);
    }

    #[test]
    fn hostile_joint_inputs_fail_closed() {
        let model = model();
        let c0 = [2, 3, 4];
        let c1 = [8, 9, 10];
        let candidates: [&[u16]; 2] = [&c0, &c1];
        let mut weights = SequenceCognitiveLossWeights::default();
        weights.retrieval = f32::NAN;
        assert_eq!(
            model.joint_loss_and_gradients(
                batch(
                    &[1],
                    &[2],
                    &[3],
                    &[4],
                    &[1.0, 0.0],
                    &[5],
                    &[6],
                    &candidates,
                ),
                weights,
            ),
            Err(SciRustError::NonFinite)
        );
        assert!(matches!(
            model.joint_loss_and_gradients(
                batch(
                    &[1],
                    &[2],
                    &[3],
                    &[4],
                    &[1.0],
                    &[5],
                    &[6],
                    &candidates,
                ),
                SequenceCognitiveLossWeights::default(),
            ),
            Err(SciRustError::Shape { .. })
        ));
    }
}
