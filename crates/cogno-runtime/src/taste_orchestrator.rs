//! Deterministic orchestration for one scientific-taste evaluation cycle.
//!
//! This module deliberately contains no model execution and grants no authority.
//! It accepts already-bounded candidate identifiers plus non-model evaluation
//! evidence, derives deterministic promotion outcomes, and exposes an immutable
//! cycle report suitable for the transactional persistence layer.

use crate::taste_validation_store::{
    StoredTasteValidation, StoredValidationOrigin, StoredValidationVerdict,
};
use std::collections::{BTreeMap, BTreeSet};

/// Maximum number of candidates accepted by one cycle.
pub const MAX_ORCHESTRATED_CANDIDATES: usize = 4_096;
/// Maximum number of validation records accepted by one cycle.
pub const MAX_ORCHESTRATED_VALIDATIONS: usize = 16_384;
/// Number of independent non-model confirmations required for promotion.
pub const MIN_PROMOTION_CONFIRMATIONS: usize = 2;
/// Minimum deterministic confidence required for promotion.
pub const PROMOTION_THRESHOLD_BPS: u16 = 7_000;

/// Candidate entering one autonomous scientific-taste cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrchestratedTasteCandidate {
    pub preference_id: u64,
}

/// Deterministically derived state for one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrchestratedTasteState {
    Quarantined,
    Candidate,
    Active,
    Conflicted,
}

/// One deterministic candidate outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrchestratedTasteOutcome {
    pub preference_id: u64,
    pub state: OrchestratedTasteState,
    pub confidence_bps: u16,
    pub confirmations: u16,
    pub validations: u16,
    pub can_activate: bool,
}

/// Immutable report produced by a complete orchestration pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScientificTasteCycleReport {
    pub candidates_scanned: usize,
    pub validations_scanned: usize,
    pub outcomes: Vec<OrchestratedTasteOutcome>,
}

/// Input or provenance violation detected before a cycle is derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificTasteOrchestratorError {
    TooManyCandidates,
    TooManyValidations,
    DuplicateCandidate(u64),
    UnknownPreference(u64),
    DuplicateValidation(u64),
    ConfidenceOutOfRange(u16),
}

/// Derive one deterministic scientific-taste cycle from bounded evidence.
pub fn orchestrate_scientific_taste_cycle(
    candidates: &[OrchestratedTasteCandidate],
    validations: &[StoredTasteValidation],
) -> Result<ScientificTasteCycleReport, ScientificTasteOrchestratorError> {
    if candidates.len() > MAX_ORCHESTRATED_CANDIDATES {
        return Err(ScientificTasteOrchestratorError::TooManyCandidates);
    }
    if validations.len() > MAX_ORCHESTRATED_VALIDATIONS {
        return Err(ScientificTasteOrchestratorError::TooManyValidations);
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.preference_id) {
            return Err(ScientificTasteOrchestratorError::DuplicateCandidate(
                candidate.preference_id,
            ));
        }
    }

    let mut validation_ids = BTreeSet::new();
    let mut grouped = BTreeMap::<u64, Vec<StoredTasteValidation>>::new();
    for validation in validations {
        if validation.confidence_bps > 10_000 {
            return Err(ScientificTasteOrchestratorError::ConfidenceOutOfRange(
                validation.confidence_bps,
            ));
        }
        if !candidate_ids.contains(&validation.preference_id) {
            return Err(ScientificTasteOrchestratorError::UnknownPreference(
                validation.preference_id,
            ));
        }
        if !validation_ids.insert(validation.validation_id) {
            return Err(ScientificTasteOrchestratorError::DuplicateValidation(
                validation.validation_id,
            ));
        }
        grouped
            .entry(validation.preference_id)
            .or_default()
            .push(*validation);
    }

    let mut outcomes = Vec::with_capacity(candidates.len());
    for candidate in candidate_ids {
        let records = grouped.get(&candidate).map(Vec::as_slice).unwrap_or(&[]);
        outcomes.push(derive_outcome(candidate, records));
    }

    Ok(ScientificTasteCycleReport {
        candidates_scanned: candidates.len(),
        validations_scanned: validations.len(),
        outcomes,
    })
}

fn derive_outcome(
    preference_id: u64,
    records: &[StoredTasteValidation],
) -> OrchestratedTasteOutcome {
    let mut confirmations = 0u16;
    let mut confidence_sum = 0u64;
    let mut contradicted = false;

    for record in records {
        match record.verdict {
            StoredValidationVerdict::Confirmed => {
                confirmations = confirmations.saturating_add(1);
                confidence_sum = confidence_sum.saturating_add(u64::from(record.confidence_bps));
            }
            StoredValidationVerdict::Contradicted => contradicted = true,
        }
    }

    let confidence_bps = if confirmations == 0 {
        0
    } else {
        u16::try_from(confidence_sum / u64::from(confirmations)).unwrap_or(10_000)
    };

    let state = if contradicted {
        OrchestratedTasteState::Conflicted
    } else if usize::from(confirmations) >= MIN_PROMOTION_CONFIRMATIONS
        && confidence_bps >= PROMOTION_THRESHOLD_BPS
    {
        OrchestratedTasteState::Active
    } else if confirmations > 0 {
        OrchestratedTasteState::Candidate
    } else {
        OrchestratedTasteState::Quarantined
    };

    OrchestratedTasteOutcome {
        preference_id,
        state,
        confidence_bps,
        confirmations,
        validations: u16::try_from(records.len()).unwrap_or(u16::MAX),
        can_activate: matches!(state, OrchestratedTasteState::Active),
    }
}

/// Provenance helper: the persistent validation format has no model origin.
#[must_use]
pub const fn validation_origin_is_non_model(origin: StoredValidationOrigin) -> bool {
    matches!(
        origin,
        StoredValidationOrigin::DeterministicEvaluation
            | StoredValidationOrigin::ExplicitUserAction
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validation(
        validation_id: u64,
        preference_id: u64,
        verdict: StoredValidationVerdict,
        confidence_bps: u16,
    ) -> StoredTasteValidation {
        StoredTasteValidation {
            validation_id,
            preference_id,
            evidence_id: validation_id + 100,
            origin: StoredValidationOrigin::DeterministicEvaluation,
            verdict,
            confidence_bps,
        }
    }

    #[test]
    fn two_confirmations_promote_deterministically() {
        let report = orchestrate_scientific_taste_cycle(
            &[OrchestratedTasteCandidate { preference_id: 7 }],
            &[
                validation(1, 7, StoredValidationVerdict::Confirmed, 8_000),
                validation(2, 7, StoredValidationVerdict::Confirmed, 9_000),
            ],
        )
        .expect("cycle");
        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Active);
        assert_eq!(report.outcomes[0].confidence_bps, 8_500);
        assert!(report.outcomes[0].can_activate);
    }

    #[test]
    fn contradiction_always_blocks_activation() {
        let report = orchestrate_scientific_taste_cycle(
            &[OrchestratedTasteCandidate { preference_id: 7 }],
            &[
                validation(1, 7, StoredValidationVerdict::Confirmed, 10_000),
                validation(2, 7, StoredValidationVerdict::Confirmed, 10_000),
                validation(3, 7, StoredValidationVerdict::Contradicted, 1),
            ],
        )
        .expect("cycle");
        assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Conflicted);
        assert!(!report.outcomes[0].can_activate);
    }

    #[test]
    fn unknown_preferences_fail_closed() {
        assert_eq!(
            orchestrate_scientific_taste_cycle(
                &[OrchestratedTasteCandidate { preference_id: 7 }],
                &[validation(1, 9, StoredValidationVerdict::Confirmed, 8_000)],
            ),
            Err(ScientificTasteOrchestratorError::UnknownPreference(9))
        );
    }
}
