//! Phase 4 — Meta-NeuroSymbolic objective, gated (COGNO-1 V2 §27 Phase 4).
//!
//! The meta-objective is activated **only** when every precondition holds:
//!
//!  - scalar reward engine validated;
//!  - reference policy frozen;
//!  - log-probabilities available;
//!  - backend actually differentiable;
//!  - held-out tests in place;
//!  - anti-poisoning guards working.
//!
//! COGNO-1 does not ship a real differentiable tensor backend. The objective
//! is therefore **disabled by default** and `activate` returns
//! `Err(MetaObjectiveError::PreconditionMissing)` until every precondition is
//! explicit. This is the fail-closed behavior mandated by S10; a missing
//! precondition can never be silently skipped.

use crate::RewardEngine;

/// Preconditions for activating the meta-objective. Each flag is a host
/// attestation: the runtime sets it only after the corresponding guard has
/// actually been validated, not because a phase declared it "done".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MetaPreconditions {
    pub scalar_engine_validated: bool,
    pub reference_policy_frozen: bool,
    pub log_probabilities_available: bool,
    pub backend_differentiable: bool,
    pub held_out_tests_in_place: bool,
    pub anti_poisoning_working: bool,
}

impl MetaPreconditions {
    /// All preconditions satisfied? Used by [`MetaObjective::activate`].
    #[must_use]
    pub const fn satisfied(&self) -> bool {
        self.scalar_engine_validated
            && self.reference_policy_frozen
            && self.log_probabilities_available
            && self.backend_differentiable
            && self.held_out_tests_in_place
            && self.anti_poisoning_working
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaObjectiveError {
    /// One or more preconditions missing; fail closed (S10).
    PreconditionMissing,
    /// Integer overflow in scalar reward machinery.
    ScalarOverflow,
}

/// State of the meta-objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetaState {
    #[default]
    Disabled,
    Quarantined,
    Active,
}

/// The meta-objective. Phase 4 keeps it disabled until the host attests every
/// precondition. Even when active, the scalar reward engine — already
/// validated in Phase 0 — is the one computing scores; the meta layer merely
/// combines them with the (external, when present) log-probability signal.
#[derive(Clone, Debug, Default)]
pub struct MetaObjective {
    pub state: MetaState,
    pub preconditions: MetaPreconditions,
    pub reward_engine: RewardEngine,
}

impl MetaObjective {
    /// Construct disabled, with no precondition attested. This is the only
    /// safe initial state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to activate. Returns `Err(PreconditionMissing)` until every
    /// precondition is set; transitions to `Active` otherwise. While
    /// preconditions are partial, the state moves to `Quarantined` (so the
    /// host can observe progress) but never to `Active`.
    pub fn activate(&mut self, pre: MetaPreconditions) -> Result<(), MetaObjectiveError> {
        self.preconditions = pre;
        if pre.satisfied() {
            self.state = MetaState::Active;
            Ok(())
        } else {
            self.state = MetaState::Quarantined;
            Err(MetaObjectiveError::PreconditionMissing)
        }
    }

    /// Force-disable (e.g., on detected poisoning). Quiescent fail-closed.
    pub fn disable(&mut self) {
        self.state = MetaState::Disabled;
    }

    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, MetaState::Active)
    }

    #[must_use]
    pub const fn is_quarantined(&self) -> bool {
        matches!(self.state, MetaState::Quarantined)
    }
}
