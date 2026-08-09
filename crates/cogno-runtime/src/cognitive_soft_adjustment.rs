//! Deterministic bounded soft adjustment derived from a sealed post-hard V4 context.
//!
//! The host supplies explicit task semantics. The model never decides what a
//! class, rule, contradiction direction or retrieval slot means. Public
//! derivation is available only from [`crate::HardGatedCognitiveContext`], so a
//! raw model observation cannot be converted into a decision influence.

use crate::{
    CognitiveObservation, CognitivePreferenceRelation, HardGatedCognitiveContext,
};
use cogno_core::CandidateScore;

/// Maximum absolute scalar delta one cognitive soft adjustment may request.
pub const MAX_COGNITIVE_SOFT_DELTA: u16 = 100;
/// Maximum relative task weight used only to form a normalized mixture.
pub const MAX_COGNITIVE_SOFT_WEIGHT: u16 = 10_000;

/// Relative non-negative weights. Their absolute scale is irrelevant because
/// the weighted alignment is normalized before the global delta cap is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveSoftWeights {
    pub classification: u16,
    pub preference: u16,
    pub symbolic: u16,
    pub contradiction: u16,
    pub retrieval: u16,
}

impl Default for CognitiveSoftWeights {
    fn default() -> Self {
        Self {
            classification: 1,
            preference: 1,
            symbolic: 1,
            contradiction: 1,
            retrieval: 1,
        }
    }
}

/// Host-owned semantics for the current comparison or decision context.
///
/// Every target is optional, but a task with a non-zero weight must have its
/// corresponding target. Symbolic targets must match the observed rule vector.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveSoftTargets<'a> {
    pub classification_class: Option<usize>,
    pub preference: Option<CognitivePreferenceRelation>,
    pub symbolic_satisfied: Option<&'a [bool]>,
    pub contradiction: Option<bool>,
    pub retrieval_index: Option<usize>,
}

/// Bounded policy for converting task alignment into one scalar soft delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveSoftAdjustmentPolicy {
    pub weights: CognitiveSoftWeights,
    pub max_abs_delta: u16,
}

impl Default for CognitiveSoftAdjustmentPolicy {
    fn default() -> Self {
        Self {
            weights: CognitiveSoftWeights::default(),
            max_abs_delta: 25,
        }
    }
}

/// Auditable integer-only result. It can only be constructed by the sealed
/// post-hard context. Alignment fields are each in `[-10_000, 10_000]` and the
/// delta is bounded by `max_abs_delta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveSoftAdjustment {
    base_score: CandidateScore,
    model_generation: u64,
    model_artifact_sha256: [u8; 32],
    meta_candidate_digest: [u8; 32],
    classification_alignment_bps: Option<i16>,
    preference_alignment_bps: Option<i16>,
    symbolic_alignment_bps: Option<i16>,
    contradiction_alignment_bps: Option<i16>,
    retrieval_alignment_bps: Option<i16>,
    weighted_alignment_bps: i16,
    delta: i64,
    max_abs_delta: u16,
    authoritative: bool,
}

impl CognitiveSoftAdjustment {
    #[must_use]
    pub const fn base_score(&self) -> CandidateScore {
        self.base_score
    }

    #[must_use]
    pub const fn model_generation(&self) -> u64 {
        self.model_generation
    }

    #[must_use]
    pub const fn model_artifact_sha256(&self) -> [u8; 32] {
        self.model_artifact_sha256
    }

    #[must_use]
    pub const fn meta_candidate_digest(&self) -> [u8; 32] {
        self.meta_candidate_digest
    }

    #[must_use]
    pub const fn classification_alignment_bps(&self) -> Option<i16> {
        self.classification_alignment_bps
    }

    #[must_use]
    pub const fn preference_alignment_bps(&self) -> Option<i16> {
        self.preference_alignment_bps
    }

    #[must_use]
    pub const fn symbolic_alignment_bps(&self) -> Option<i16> {
        self.symbolic_alignment_bps
    }

    #[must_use]
    pub const fn contradiction_alignment_bps(&self) -> Option<i16> {
        self.contradiction_alignment_bps
    }

    #[must_use]
    pub const fn retrieval_alignment_bps(&self) -> Option<i16> {
        self.retrieval_alignment_bps
    }

    #[must_use]
    pub const fn weighted_alignment_bps(&self) -> i16 {
        self.weighted_alignment_bps
    }

    #[must_use]
    pub const fn delta(&self) -> i64 {
        self.delta
    }

    #[must_use]
    pub const fn max_abs_delta(&self) -> u16 {
        self.max_abs_delta
    }

    #[must_use]
    pub const fn authoritative(&self) -> bool {
        self.authoritative
    }
}

/// Fail-closed policy or target validation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveSoftAdjustmentError {
    MaxDeltaOutOfRange,
    WeightOutOfRange,
    NoWeightedSignals,
    MissingClassificationTarget,
    MissingPreferenceTarget,
    MissingSymbolicTarget,
    MissingContradictionTarget,
    MissingRetrievalTarget,
    ClassificationTargetOutOfRange,
    SymbolicTargetLengthMismatch,
    RetrievalTargetOutOfRange,
    ArithmeticOverflow,
}

impl HardGatedCognitiveContext {
    /// Derive a deterministic bounded soft delta from the already hard-gated,
    /// Meta/model-bound context. This cannot be called on a raw observation.
    pub fn derive_bounded_soft_adjustment(
        &self,
        policy: CognitiveSoftAdjustmentPolicy,
        targets: CognitiveSoftTargets<'_>,
    ) -> Result<CognitiveSoftAdjustment, CognitiveSoftAdjustmentError> {
        derive_adjustment(
            self.base_score(),
            self.observation(),
            self.meta_candidate_digest(),
            policy,
            targets,
        )
    }
}

fn derive_adjustment(
    base_score: CandidateScore,
    observation: &CognitiveObservation,
    meta_candidate_digest: [u8; 32],
    policy: CognitiveSoftAdjustmentPolicy,
    targets: CognitiveSoftTargets<'_>,
) -> Result<CognitiveSoftAdjustment, CognitiveSoftAdjustmentError> {
    validate_policy(policy, targets)?;

    let classification_alignment_bps = if policy.weights.classification == 0 {
        None
    } else {
        let target = targets
            .classification_class
            .ok_or(CognitiveSoftAdjustmentError::MissingClassificationTarget)?;
        let probability = observation
            .classification_probabilities_bps
            .get(target)
            .copied()
            .ok_or(CognitiveSoftAdjustmentError::ClassificationTargetOutOfRange)?;
        Some(probability_alignment(probability)?)
    };

    let preference_alignment_bps = if policy.weights.preference == 0 {
        None
    } else {
        let target = targets
            .preference
            .ok_or(CognitiveSoftAdjustmentError::MissingPreferenceTarget)?;
        Some(if observation.preference == target {
            10_000
        } else {
            -10_000
        })
    };

    let symbolic_alignment_bps = if policy.weights.symbolic == 0 {
        None
    } else {
        let targets = targets
            .symbolic_satisfied
            .ok_or(CognitiveSoftAdjustmentError::MissingSymbolicTarget)?;
        if targets.len() != observation.symbolic_satisfaction_bps.len() || targets.is_empty() {
            return Err(CognitiveSoftAdjustmentError::SymbolicTargetLengthMismatch);
        }
        let mut aligned_sum = 0i64;
        for (&probability, &target) in observation
            .symbolic_satisfaction_bps
            .iter()
            .zip(targets)
        {
            let aligned_probability = if target {
                probability
            } else {
                10_000u16
                    .checked_sub(probability)
                    .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?
            };
            aligned_sum = aligned_sum
                .checked_add(i64::from(probability_alignment(aligned_probability)?))
                .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
        }
        let divisor = i64::try_from(targets.len())
            .map_err(|_| CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
        Some(to_alignment_i16(aligned_sum / divisor)?)
    };

    let contradiction_alignment_bps = if policy.weights.contradiction == 0 {
        None
    } else {
        let target = targets
            .contradiction
            .ok_or(CognitiveSoftAdjustmentError::MissingContradictionTarget)?;
        let probability = if target {
            observation.contradiction_bps
        } else {
            10_000u16
                .checked_sub(observation.contradiction_bps)
                .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?
        };
        Some(probability_alignment(probability)?)
    };

    let retrieval_alignment_bps = if policy.weights.retrieval == 0 {
        None
    } else {
        let target = targets
            .retrieval_index
            .ok_or(CognitiveSoftAdjustmentError::MissingRetrievalTarget)?;
        if target >= observation.retrieval_candidate_count {
            return Err(CognitiveSoftAdjustmentError::RetrievalTargetOutOfRange);
        }
        Some(if observation.retrieval_selected_index == target {
            10_000
        } else {
            -10_000
        })
    };

    let components = [
        (classification_alignment_bps, policy.weights.classification),
        (preference_alignment_bps, policy.weights.preference),
        (symbolic_alignment_bps, policy.weights.symbolic),
        (contradiction_alignment_bps, policy.weights.contradiction),
        (retrieval_alignment_bps, policy.weights.retrieval),
    ];
    let mut weighted_sum = 0i64;
    let mut total_weight = 0i64;
    for (alignment, weight) in components {
        if let Some(alignment) = alignment {
            let weighted = i64::from(alignment)
                .checked_mul(i64::from(weight))
                .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
            weighted_sum = weighted_sum
                .checked_add(weighted)
                .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
            total_weight = total_weight
                .checked_add(i64::from(weight))
                .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
        }
    }
    if total_weight == 0 {
        return Err(CognitiveSoftAdjustmentError::NoWeightedSignals);
    }
    let weighted_alignment_bps = to_alignment_i16(weighted_sum / total_weight)?;
    let delta = i64::from(weighted_alignment_bps)
        .checked_mul(i64::from(policy.max_abs_delta))
        .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?
        / 10_000;
    if delta.unsigned_abs() > u64::from(policy.max_abs_delta) {
        return Err(CognitiveSoftAdjustmentError::ArithmeticOverflow);
    }

    Ok(CognitiveSoftAdjustment {
        base_score,
        model_generation: observation.model_generation,
        model_artifact_sha256: observation.model_artifact_sha256,
        meta_candidate_digest,
        classification_alignment_bps,
        preference_alignment_bps,
        symbolic_alignment_bps,
        contradiction_alignment_bps,
        retrieval_alignment_bps,
        weighted_alignment_bps,
        delta,
        max_abs_delta: policy.max_abs_delta,
        authoritative: false,
    })
}

fn validate_policy(
    policy: CognitiveSoftAdjustmentPolicy,
    targets: CognitiveSoftTargets<'_>,
) -> Result<(), CognitiveSoftAdjustmentError> {
    if policy.max_abs_delta == 0 || policy.max_abs_delta > MAX_COGNITIVE_SOFT_DELTA {
        return Err(CognitiveSoftAdjustmentError::MaxDeltaOutOfRange);
    }
    let weights = [
        policy.weights.classification,
        policy.weights.preference,
        policy.weights.symbolic,
        policy.weights.contradiction,
        policy.weights.retrieval,
    ];
    if weights
        .iter()
        .any(|weight| *weight > MAX_COGNITIVE_SOFT_WEIGHT)
    {
        return Err(CognitiveSoftAdjustmentError::WeightOutOfRange);
    }
    if weights.iter().all(|weight| *weight == 0) {
        return Err(CognitiveSoftAdjustmentError::NoWeightedSignals);
    }
    if policy.weights.classification > 0 && targets.classification_class.is_none() {
        return Err(CognitiveSoftAdjustmentError::MissingClassificationTarget);
    }
    if policy.weights.preference > 0 && targets.preference.is_none() {
        return Err(CognitiveSoftAdjustmentError::MissingPreferenceTarget);
    }
    if policy.weights.symbolic > 0 && targets.symbolic_satisfied.is_none() {
        return Err(CognitiveSoftAdjustmentError::MissingSymbolicTarget);
    }
    if policy.weights.contradiction > 0 && targets.contradiction.is_none() {
        return Err(CognitiveSoftAdjustmentError::MissingContradictionTarget);
    }
    if policy.weights.retrieval > 0 && targets.retrieval_index.is_none() {
        return Err(CognitiveSoftAdjustmentError::MissingRetrievalTarget);
    }
    Ok(())
}

fn probability_alignment(probability_bps: u16) -> Result<i16, CognitiveSoftAdjustmentError> {
    if probability_bps > 10_000 {
        return Err(CognitiveSoftAdjustmentError::ArithmeticOverflow);
    }
    let alignment = i32::from(probability_bps)
        .checked_mul(2)
        .and_then(|value| value.checked_sub(10_000))
        .ok_or(CognitiveSoftAdjustmentError::ArithmeticOverflow)?;
    i16::try_from(alignment).map_err(|_| CognitiveSoftAdjustmentError::ArithmeticOverflow)
}

fn to_alignment_i16(value: i64) -> Result<i16, CognitiveSoftAdjustmentError> {
    if !(-10_000..=10_000).contains(&value) {
        return Err(CognitiveSoftAdjustmentError::ArithmeticOverflow);
    }
    i16::try_from(value).map_err(|_| CognitiveSoftAdjustmentError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> CognitiveObservation {
        CognitiveObservation {
            model_generation: 4,
            model_artifact_sha256: [7; 32],
            classification_class: 1,
            classification_confidence_bps: 8_000,
            classification_probabilities_bps: vec![2_000, 8_000],
            preference: CognitivePreferenceRelation::Left,
            symbolic_satisfaction_bps: vec![9_000, 2_000],
            contradiction_bps: 1_000,
            retrieval_selected_index: 0,
            retrieval_candidate_count: 2,
            authoritative: false,
        }
    }

    fn base_score() -> CandidateScore {
        CandidateScore::new(42)
    }

    #[test]
    fn aligned_targets_produce_positive_bounded_delta() {
        let adjustment = derive_adjustment(
            base_score(),
            &observation(),
            [7; 32],
            CognitiveSoftAdjustmentPolicy {
                weights: CognitiveSoftWeights::default(),
                max_abs_delta: 25,
            },
            CognitiveSoftTargets {
                classification_class: Some(1),
                preference: Some(CognitivePreferenceRelation::Left),
                symbolic_satisfied: Some(&[true, false]),
                contradiction: Some(false),
                retrieval_index: Some(0),
            },
        )
        .expect("adjustment");
        assert!(adjustment.delta() > 0);
        assert!(adjustment.delta() <= 25);
        assert_eq!(adjustment.model_artifact_sha256(), [7; 32]);
        assert_eq!(adjustment.meta_candidate_digest(), [7; 32]);
        assert!(!adjustment.authoritative());
    }

    #[test]
    fn retrieval_target_must_be_inside_observed_candidate_set() {
        let result = derive_adjustment(
            base_score(),
            &observation(),
            [7; 32],
            CognitiveSoftAdjustmentPolicy {
                weights: CognitiveSoftWeights {
                    classification: 0,
                    preference: 0,
                    symbolic: 0,
                    contradiction: 0,
                    retrieval: 1,
                },
                max_abs_delta: 25,
            },
            CognitiveSoftTargets {
                classification_class: None,
                preference: None,
                symbolic_satisfied: None,
                contradiction: None,
                retrieval_index: Some(2),
            },
        );
        assert_eq!(
            result,
            Err(CognitiveSoftAdjustmentError::RetrievalTargetOutOfRange)
        );
    }

    #[test]
    fn zero_weights_and_oversized_delta_fail_closed() {
        let targets = CognitiveSoftTargets {
            classification_class: None,
            preference: None,
            symbolic_satisfied: None,
            contradiction: None,
            retrieval_index: None,
        };
        assert_eq!(
            derive_adjustment(
                base_score(),
                &observation(),
                [7; 32],
                CognitiveSoftAdjustmentPolicy {
                    weights: CognitiveSoftWeights {
                        classification: 0,
                        preference: 0,
                        symbolic: 0,
                        contradiction: 0,
                        retrieval: 0,
                    },
                    max_abs_delta: 25,
                },
                targets,
            ),
            Err(CognitiveSoftAdjustmentError::NoWeightedSignals)
        );
        assert_eq!(
            validate_policy(
                CognitiveSoftAdjustmentPolicy {
                    weights: CognitiveSoftWeights::default(),
                    max_abs_delta: MAX_COGNITIVE_SOFT_DELTA + 1,
                },
                CognitiveSoftTargets {
                    classification_class: Some(0),
                    preference: Some(CognitivePreferenceRelation::Tie),
                    symbolic_satisfied: Some(&[true, false]),
                    contradiction: Some(false),
                    retrieval_index: Some(0),
                },
            ),
            Err(CognitiveSoftAdjustmentError::MaxDeltaOutOfRange)
        );
    }
}
