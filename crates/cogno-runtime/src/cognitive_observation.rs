//! Soft-only observations from the installed shared-cognitive V4 model.
//!
//! This surface deliberately stops before reward or policy integration. It
//! quantizes model outputs into bounded deterministic audit data and binds every
//! observation to the exact persisted model generation and artifact digest.
//! Hard validators and symbolic authority remain entirely outside this module.

use crate::runtime::Runtime;
use cogno_model::SequenceCognitiveModelError;
use std::cmp::Ordering;

/// Borrowed multi-view input for one non-authoritative cognitive observation.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveObservationInput<'a> {
    pub classification_payload: &'a [u8],
    pub preference_left: &'a [u8],
    pub preference_right: &'a [u8],
    pub symbolic_payload: &'a [u8],
    pub contradiction_left: &'a [u8],
    pub contradiction_right: &'a [u8],
    pub retrieval_query: &'a [u8],
    pub retrieval_candidates: &'a [&'a [u8]],
}

/// Quantized pairwise preference relation from the soft model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitivePreferenceRelation {
    Left,
    Tie,
    Right,
}

/// One generation-bound, integer-only cognitive snapshot suitable for audit.
///
/// `authoritative` is permanently false: these values may later contribute to
/// soft scoring only after the deterministic hard gates have already passed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CognitiveObservation {
    pub model_generation: u64,
    pub model_artifact_sha256: [u8; 32],
    pub classification_class: usize,
    pub classification_confidence_bps: u16,
    pub classification_probabilities_bps: Vec<u16>,
    pub preference: CognitivePreferenceRelation,
    pub symbolic_satisfaction_bps: Vec<u16>,
    pub contradiction_bps: u16,
    pub retrieval_selected_index: usize,
    pub retrieval_candidate_count: usize,
    pub authoritative: bool,
}

/// Fail-closed observation errors. No partial observation is audited.
#[derive(Clone, Debug, PartialEq)]
pub enum CognitiveObservationError {
    NoModelInstalled,
    MissingGenerationBinding,
    MissingArtifactBinding,
    InvalidProbability,
    InvalidRetrievalSelection,
    Model(SequenceCognitiveModelError),
}

impl From<SequenceCognitiveModelError> for CognitiveObservationError {
    fn from(error: SequenceCognitiveModelError) -> Self {
        Self::Model(error)
    }
}

impl Runtime {
    /// Evaluate every V4 cognitive head, quantize probabilistic outputs to basis
    /// points, bind the result to the installed generation, and audit it.
    ///
    /// This method does not run, skip or modify the proposal pipeline. It does
    /// not change `Reward`, activate Meta, execute tools or create hard rules.
    pub fn observe_cognitive(
        &mut self,
        input: CognitiveObservationInput<'_>,
    ) -> Result<CognitiveObservation, CognitiveObservationError> {
        if self.cognitive_model().is_none() {
            return Err(CognitiveObservationError::NoModelInstalled);
        }
        let generation = self
            .cognitive_model_generation()
            .ok_or(CognitiveObservationError::MissingGenerationBinding)?;
        let artifact_sha256 = self
            .cognitive_model_artifact_sha256()
            .ok_or(CognitiveObservationError::MissingArtifactBinding)?;
        let retrieval_candidate_count = input.retrieval_candidates.len();

        let (
            classification_class,
            classification_confidence_bps,
            probabilities_bps,
            preference,
            symbolic_satisfaction_bps,
            contradiction_bps,
            retrieval_selected_index,
        ) = {
            let model = self
                .cognitive_model()
                .ok_or(CognitiveObservationError::NoModelInstalled)?;
            let probabilities = model
                .model
                .classification_probabilities(input.classification_payload)?;
            let first_probability = probabilities
                .first()
                .copied()
                .ok_or(CognitiveObservationError::InvalidProbability)?;
            let mut classification_class = 0usize;
            let mut best_probability = first_probability;
            for (index, &probability) in probabilities.iter().enumerate().skip(1) {
                if probability > best_probability {
                    classification_class = index;
                    best_probability = probability;
                }
            }
            let classification_confidence_bps = probability_to_bps(best_probability)?;
            let mut probabilities_bps = Vec::with_capacity(probabilities.len());
            for probability in probabilities {
                probabilities_bps.push(probability_to_bps(probability)?);
            }

            let preference = match model
                .model
                .preference_compare(input.preference_left, input.preference_right)?
            {
                Ordering::Greater => CognitivePreferenceRelation::Left,
                Ordering::Equal => CognitivePreferenceRelation::Tie,
                Ordering::Less => CognitivePreferenceRelation::Right,
            };
            let symbolic = model.model.symbolic_satisfactions(input.symbolic_payload)?;
            let mut symbolic_satisfaction_bps = Vec::with_capacity(symbolic.len());
            for probability in symbolic {
                symbolic_satisfaction_bps.push(probability_to_bps(probability)?);
            }
            let contradiction_bps =
                probability_to_bps(model.model.contradiction_probability(
                    input.contradiction_left,
                    input.contradiction_right,
                )?)?;
            let retrieval_selected_index = model
                .model
                .retrieval_select_best(input.retrieval_query, input.retrieval_candidates)?;
            if retrieval_selected_index >= retrieval_candidate_count {
                return Err(CognitiveObservationError::InvalidRetrievalSelection);
            }
            (
                classification_class,
                classification_confidence_bps,
                probabilities_bps,
                preference,
                symbolic_satisfaction_bps,
                contradiction_bps,
                retrieval_selected_index,
            )
        };

        let observation = CognitiveObservation {
            model_generation: generation,
            model_artifact_sha256: artifact_sha256,
            classification_class,
            classification_confidence_bps,
            classification_probabilities_bps: probabilities_bps,
            preference,
            symbolic_satisfaction_bps,
            contradiction_bps,
            retrieval_selected_index,
            retrieval_candidate_count,
            authoritative: false,
        };
        self.audit.cognitive_observation(&observation);
        Ok(observation)
    }
}

fn probability_to_bps(probability: f32) -> Result<u16, CognitiveObservationError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(CognitiveObservationError::InvalidProbability);
    }
    let scaled = (probability * 10_000.0).round();
    if !scaled.is_finite() || !(0.0..=10_000.0).contains(&scaled) {
        return Err(CognitiveObservationError::InvalidProbability);
    }
    Ok(scaled as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bps_quantization_is_bounded_and_deterministic() {
        assert_eq!(probability_to_bps(0.0), Ok(0));
        assert_eq!(probability_to_bps(0.5), Ok(5_000));
        assert_eq!(probability_to_bps(1.0), Ok(10_000));
        assert_eq!(
            probability_to_bps(f32::NAN),
            Err(CognitiveObservationError::InvalidProbability)
        );
    }
}
