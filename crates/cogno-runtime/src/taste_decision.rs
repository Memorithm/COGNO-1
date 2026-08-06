//! Bounded, deterministic use of verified scientific taste.
//!
//! Taste can only break ties or adjust soft ranking. Hard constraints remain
//! lexicographic and non-compensable: an ineligible candidate can never become
//! selectable because of a preference.

use crate::verified_taste_profile::VerifiedTastePreference;
use std::collections::BTreeMap;

/// Maximum absolute soft-score adjustment contributed by one preference.
pub const MAX_TASTE_WEIGHT_BPS: i16 = 1_000;

/// One candidate considered by a runtime decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteDecisionCandidate {
    /// Stable candidate identifier used for deterministic tie-breaking.
    pub candidate_id: u64,
    /// Score before scientific-taste influence.
    pub base_score: i64,
    /// Hard safety and structural constraints have all passed.
    pub hard_constraints_satisfied: bool,
}

/// Explicit, caller-supplied mapping from one verified preference to a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TastePreferenceApplication {
    pub preference_id: u64,
    pub candidate_id: u64,
    /// Signed influence in basis points, clamped to ±1000.
    pub weight_bps: i16,
}

/// One applied influence, suitable for decision audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteInfluence {
    pub preference_id: u64,
    pub candidate_id: u64,
    pub confidence_bps: u16,
    pub requested_weight_bps: i16,
    pub applied_weight_bps: i16,
    pub score_delta: i64,
}

/// Deterministic result of a taste-aware soft decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteDecision {
    pub selected_candidate_id: Option<u64>,
    pub selected_base_score: Option<i64>,
    pub selected_final_score: Option<i64>,
    pub abstained: bool,
    pub influences: Vec<TasteInfluence>,
}

/// Rank candidates using verified active preferences without weakening hard rules.
#[must_use]
pub fn decide_with_verified_taste(
    candidates: &[TasteDecisionCandidate],
    active_preferences: &[VerifiedTastePreference],
    applications: &[TastePreferenceApplication],
) -> TasteDecision {
    let preference_index: BTreeMap<u64, VerifiedTastePreference> = active_preferences
        .iter()
        .copied()
        .map(|preference| (preference.preference_id, preference))
        .collect();

    let mut deltas = BTreeMap::<u64, i64>::new();
    let mut influences = Vec::new();

    for application in applications {
        let Some(preference) = preference_index.get(&application.preference_id) else {
            continue;
        };
        let applied_weight_bps = application
            .weight_bps
            .clamp(-MAX_TASTE_WEIGHT_BPS, MAX_TASTE_WEIGHT_BPS);
        let score_delta = i64::from(preference.confidence_bps)
            .saturating_mul(i64::from(applied_weight_bps))
            / 10_000;
        let entry = deltas.entry(application.candidate_id).or_default();
        *entry = entry.saturating_add(score_delta);
        influences.push(TasteInfluence {
            preference_id: application.preference_id,
            candidate_id: application.candidate_id,
            confidence_bps: preference.confidence_bps,
            requested_weight_bps: application.weight_bps,
            applied_weight_bps,
            score_delta,
        });
    }

    influences.sort_by_key(|item| (item.candidate_id, item.preference_id));

    let selected = candidates
        .iter()
        .filter(|candidate| candidate.hard_constraints_satisfied)
        .map(|candidate| {
            let final_score = candidate
                .base_score
                .saturating_add(*deltas.get(&candidate.candidate_id).unwrap_or(&0));
            (candidate, final_score)
        })
        .max_by(|(left, left_score), (right, right_score)| {
            left_score
                .cmp(right_score)
                .then_with(|| right.candidate_id.cmp(&left.candidate_id))
        });

    match selected {
        Some((candidate, final_score)) => TasteDecision {
            selected_candidate_id: Some(candidate.candidate_id),
            selected_base_score: Some(candidate.base_score),
            selected_final_score: Some(final_score),
            abstained: false,
            influences,
        },
        None => TasteDecision {
            selected_candidate_id: None,
            selected_base_score: None,
            selected_final_score: None,
            abstained: true,
            influences,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFERENCE: VerifiedTastePreference = VerifiedTastePreference {
        preference_id: 7,
        confidence_bps: 8_000,
        non_model_confirmations: 2,
        validations: 2,
    };

    #[test]
    fn taste_can_break_a_soft_tie() {
        let decision = decide_with_verified_taste(
            &[
                TasteDecisionCandidate {
                    candidate_id: 1,
                    base_score: 100,
                    hard_constraints_satisfied: true,
                },
                TasteDecisionCandidate {
                    candidate_id: 2,
                    base_score: 100,
                    hard_constraints_satisfied: true,
                },
            ],
            &[PREFERENCE],
            &[TastePreferenceApplication {
                preference_id: 7,
                candidate_id: 2,
                weight_bps: 500,
            }],
        );
        assert_eq!(decision.selected_candidate_id, Some(2));
        assert_eq!(decision.selected_final_score, Some(500));
    }

    #[test]
    fn taste_never_revives_a_hard_rejected_candidate() {
        let decision = decide_with_verified_taste(
            &[
                TasteDecisionCandidate {
                    candidate_id: 1,
                    base_score: 1,
                    hard_constraints_satisfied: true,
                },
                TasteDecisionCandidate {
                    candidate_id: 2,
                    base_score: i64::MAX,
                    hard_constraints_satisfied: false,
                },
            ],
            &[PREFERENCE],
            &[TastePreferenceApplication {
                preference_id: 7,
                candidate_id: 2,
                weight_bps: i16::MAX,
            }],
        );
        assert_eq!(decision.selected_candidate_id, Some(1));
    }

    #[test]
    fn unknown_preferences_have_no_effect() {
        let decision = decide_with_verified_taste(
            &[TasteDecisionCandidate {
                candidate_id: 1,
                base_score: 42,
                hard_constraints_satisfied: true,
            }],
            &[PREFERENCE],
            &[TastePreferenceApplication {
                preference_id: 99,
                candidate_id: 1,
                weight_bps: 1_000,
            }],
        );
        assert!(decision.influences.is_empty());
        assert_eq!(decision.selected_final_score, Some(42));
    }
}
