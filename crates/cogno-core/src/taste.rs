//! Deterministic scientific-taste profile for COGNO-1.
//!
//! The assistance model may propose preferences, but this module alone derives
//! the active profile. Model output starts quarantined, evidence is bounded,
//! contradictions are preserved, and promotion is deterministic.

use std::collections::{BTreeMap, BTreeSet};

pub const MAX_TASTE_CONFIDENCE_BPS: u16 = 10_000;
pub const MAX_TASTE_EVIDENCE_IDS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TasteScope {
    User,
    Project,
    Domain,
    Team,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteEventKind {
    Proposed,
    Accepted,
    Rejected,
    Edited,
    Confirmed,
    Contradicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteOrigin {
    ModelInference,
    ExplicitUserAction,
    DeterministicEvaluation,
    ImportedProfile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteState {
    Quarantined,
    Candidate,
    Active,
    Conflicted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteEvent {
    pub event_id: u64,
    pub preference_id: u64,
    pub scope: TasteScope,
    pub kind: TasteEventKind,
    pub origin: TasteOrigin,
    pub confidence_bps: u16,
    pub evidence_ids: Vec<u64>,
}

impl TasteEvent {
    pub fn validate(&self) -> Result<(), TasteError> {
        if self.confidence_bps > MAX_TASTE_CONFIDENCE_BPS {
            return Err(TasteError::ConfidenceOutOfRange);
        }
        if self.evidence_ids.len() > MAX_TASTE_EVIDENCE_IDS {
            return Err(TasteError::EvidenceLimitExceeded);
        }
        let mut unique = BTreeSet::new();
        for evidence_id in &self.evidence_ids {
            if !unique.insert(*evidence_id) {
                return Err(TasteError::DuplicateEvidence);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteError {
    ConfidenceOutOfRange,
    EvidenceLimitExceeded,
    DuplicateEvidence,
    DuplicateEvent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TastePolicy {
    pub activation_threshold_bps: u16,
    pub minimum_non_model_confirmations: u16,
    pub acceptance_gain_bps: u16,
    pub confirmation_gain_bps: u16,
    pub edit_gain_bps: u16,
    pub rejection_loss_bps: u16,
    pub contradiction_loss_bps: u16,
}

impl TastePolicy {
    pub const DEFAULT: Self = Self {
        activation_threshold_bps: 7_000,
        minimum_non_model_confirmations: 2,
        acceptance_gain_bps: 1_000,
        confirmation_gain_bps: 1_500,
        edit_gain_bps: 500,
        rejection_loss_bps: 2_000,
        contradiction_loss_bps: 3_000,
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScientificTaste {
    pub preference_id: u64,
    pub scope: TasteScope,
    pub confidence_bps: u16,
    pub state: TasteState,
    pub non_model_confirmations: u16,
    pub accepted_count: u32,
    pub rejected_count: u32,
    pub edited_count: u32,
    pub contradicted_count: u32,
}

impl ScientificTaste {
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, TasteState::Active)
    }
}

#[derive(Debug, Default)]
pub struct ScientificTasteProfile {
    pub preferences: BTreeMap<u64, ScientificTaste>,
}

impl ScientificTasteProfile {
    pub fn derive(events: &[TasteEvent], policy: TastePolicy) -> Result<Self, TasteError> {
        let mut seen_events = BTreeSet::new();
        let mut ordered = events.to_vec();
        ordered.sort_by_key(|event| event.event_id);

        let mut profile = Self::default();
        for event in ordered {
            event.validate()?;
            if !seen_events.insert(event.event_id) {
                return Err(TasteError::DuplicateEvent);
            }

            let entry = profile
                .preferences
                .entry(event.preference_id)
                .or_insert(ScientificTaste {
                    preference_id: event.preference_id,
                    scope: event.scope,
                    confidence_bps: event.confidence_bps,
                    state: TasteState::Quarantined,
                    non_model_confirmations: 0,
                    accepted_count: 0,
                    rejected_count: 0,
                    edited_count: 0,
                    contradicted_count: 0,
                });

            entry.scope = event.scope;
            match event.kind {
                TasteEventKind::Proposed => {
                    entry.confidence_bps = entry.confidence_bps.max(event.confidence_bps);
                }
                TasteEventKind::Accepted => {
                    entry.accepted_count = entry.accepted_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.acceptance_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Edited => {
                    entry.edited_count = entry.edited_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.edit_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Confirmed => {
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.confirmation_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Rejected => {
                    entry.rejected_count = entry.rejected_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_sub(policy.rejection_loss_bps);
                    if matches!(event.origin, TasteOrigin::ExplicitUserAction) {
                        entry.state = TasteState::Rejected;
                    }
                }
                TasteEventKind::Contradicted => {
                    entry.contradicted_count = entry.contradicted_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_sub(policy.contradiction_loss_bps);
                    entry.state = TasteState::Conflicted;
                }
            }
        }

        for entry in profile.preferences.values_mut() {
            if matches!(entry.state, TasteState::Conflicted | TasteState::Rejected) {
                continue;
            }
            entry.state = if entry.non_model_confirmations
                >= policy.minimum_non_model_confirmations
                && entry.confidence_bps >= policy.activation_threshold_bps
            {
                TasteState::Active
            } else if entry.non_model_confirmations > 0 {
                TasteState::Candidate
            } else {
                TasteState::Quarantined
            };
        }

        Ok(profile)
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.preferences.values().filter(|item| item.is_active()).count()
    }
}

fn register_confirmation(entry: &mut ScientificTaste, origin: TasteOrigin) {
    if matches!(
        origin,
        TasteOrigin::ExplicitUserAction | TasteOrigin::DeterministicEvaluation
    ) {
        entry.non_model_confirmations = entry.non_model_confirmations.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(event_id: u64, kind: TasteEventKind, origin: TasteOrigin) -> TasteEvent {
        TasteEvent {
            event_id,
            preference_id: 7,
            scope: TasteScope::Project,
            kind,
            origin,
            confidence_bps: 5_000,
            evidence_ids: vec![event_id],
        }
    }

    #[test]
    fn model_only_preference_remains_quarantined() {
        let profile = ScientificTasteProfile::derive(
            &[event(
                1,
                TasteEventKind::Proposed,
                TasteOrigin::ModelInference,
            )],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Quarantined);
    }

    #[test]
    fn deterministic_evidence_can_activate_preference() {
        let profile = ScientificTasteProfile::derive(
            &[
                event(
                    1,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                ),
                event(
                    2,
                    TasteEventKind::Confirmed,
                    TasteOrigin::ExplicitUserAction,
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Active);
    }

    #[test]
    fn contradiction_is_preserved() {
        let profile = ScientificTasteProfile::derive(
            &[
                event(
                    1,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                ),
                event(
                    2,
                    TasteEventKind::Contradicted,
                    TasteOrigin::ExplicitUserAction,
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);
    }
}
