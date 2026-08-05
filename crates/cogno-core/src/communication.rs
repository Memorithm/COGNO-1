//! Deterministic communication maturity gate for COGNO-1.
//!
//! COGNO begins as an internal observer. It may record preference observations
//! and contradictions, but it cannot address humans or external agents until a
//! deterministic maturity policy explicitly promotes it.

use crate::taste::{ScientificTasteProfile, TasteState};

pub const COMMUNICATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_COMMUNICATION_CONFIDENCE_BPS: u16 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationRole {
    Observer,
    Communicator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationKind {
    PreferenceObservation,
    Contradiction,
    Explanation,
    Critique,
    Hypothesis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationRecipient {
    DeterministicKernel,
    SciAgent,
    ExternalAgent,
    Human,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommunicationMaturityPolicy {
    pub minimum_active_preferences: usize,
    pub maximum_conflicted_preferences: usize,
    pub minimum_mean_confidence_bps: u16,
    pub require_explicit_human_authorization: bool,
}

impl Default for CommunicationMaturityPolicy {
    fn default() -> Self {
        Self {
            minimum_active_preferences: 1_000,
            maximum_conflicted_preferences: 10,
            minimum_mean_confidence_bps: 8_000,
            require_explicit_human_authorization: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommunicationAuthorization {
    pub human_authorized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommunicationMaturity {
    pub role: CommunicationRole,
    pub active_preferences: usize,
    pub conflicted_preferences: usize,
    pub mean_confidence_bps: u16,
}

impl CommunicationMaturity {
    #[must_use]
    pub fn derive(
        profile: &ScientificTasteProfile,
        policy: CommunicationMaturityPolicy,
        authorization: CommunicationAuthorization,
    ) -> Self {
        let mut active_preferences = 0usize;
        let mut conflicted_preferences = 0usize;
        let mut confidence_sum = 0u64;

        for preference in profile.preferences.values() {
            match preference.state {
                TasteState::Active => {
                    active_preferences = active_preferences.saturating_add(1);
                    confidence_sum =
                        confidence_sum.saturating_add(u64::from(preference.confidence_bps));
                }
                TasteState::Conflicted => {
                    conflicted_preferences = conflicted_preferences.saturating_add(1);
                }
                TasteState::Quarantined | TasteState::Candidate | TasteState::Rejected => {}
            }
        }

        let mean_confidence_bps = if active_preferences == 0 {
            0
        } else {
            (confidence_sum / active_preferences as u64) as u16
        };

        let human_gate =
            !policy.require_explicit_human_authorization || authorization.human_authorized;
        let mature = active_preferences >= policy.minimum_active_preferences
            && conflicted_preferences <= policy.maximum_conflicted_preferences
            && mean_confidence_bps >= policy.minimum_mean_confidence_bps
            && human_gate;

        Self {
            role: if mature {
                CommunicationRole::Communicator
            } else {
                CommunicationRole::Observer
            },
            active_preferences,
            conflicted_preferences,
            mean_confidence_bps,
        }
    }

    #[must_use]
    pub const fn permits(
        &self,
        kind: CommunicationKind,
        recipient: CommunicationRecipient,
    ) -> bool {
        match self.role {
            CommunicationRole::Observer => matches!(
                (kind, recipient),
                (
                    CommunicationKind::PreferenceObservation | CommunicationKind::Contradiction,
                    CommunicationRecipient::DeterministicKernel
                )
            ),
            CommunicationRole::Communicator => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taste::{TasteError, TastePolicy};

    fn empty_profile() -> Result<ScientificTasteProfile, TasteError> {
        ScientificTasteProfile::derive(&[], TastePolicy::DEFAULT)
    }

    #[test]
    fn immature_cogno_remains_silent_observer() {
        let profile = empty_profile().unwrap();
        let maturity = CommunicationMaturity::derive(
            &profile,
            CommunicationMaturityPolicy::default(),
            CommunicationAuthorization {
                human_authorized: false,
            },
        );
        assert_eq!(maturity.role, CommunicationRole::Observer);
        assert!(maturity.permits(
            CommunicationKind::Contradiction,
            CommunicationRecipient::DeterministicKernel
        ));
        assert!(!maturity.permits(
            CommunicationKind::Explanation,
            CommunicationRecipient::Human
        ));
    }
}
