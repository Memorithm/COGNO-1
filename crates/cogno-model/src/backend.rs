//! Backend trait and an owning proposal wrapper (COGNO-1 V2 §2, §21).
//!
//! The strict, versioned schema (`CognoProposalView`) lives in `cogno-core`
//! and is borrowed, lifetime-bounded. Model backends own their bytes; this
//! module defines [`OwnedProposal`] which borrows into a
//! [`CognoProposalView`](cogno_core::CognoProposalView) via [`OwnedProposal::as_view`].

use cogno_core::{
    CognoProposalView, EvidenceId, ProposalAction, RuleCategory, RuleScope, SCHEMA_VERSION,
};

/// Errors raised by a model backend. The runtime maps these to
/// `RejectReason::Malformed` (or `Overflow` for size overruns) before the
/// lexicographic decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// No more scripted outputs available (simulator exhausted).
    Exhausted,
    /// Hostile artifact refused at load time (§21).
    HostileArtifact,
    /// Input too large for the backend's reserved budget.
    InputTooLarge,
    /// Backend is read-only and was asked to mutate state.
    ReadOnlyViolation,
    /// Capability not granted by the host.
    CapabilityRefused,
}

/// Static info about a backend (no model weights introspection; used for the
/// CLI `doctor`/`phase` reports).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendInfo {
    pub phase: u8,
    pub read_only: bool,
    pub tools_enabled: bool,
    pub differentiable: bool,
}

/// Owned proposal. Owns evidence ids and payload; borrows into the strict
/// `CognoProposalView` for validation by the core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedProposal {
    pub action: ProposalAction,
    pub category: RuleCategory,
    pub scope: RuleScope,
    pub confidence_bps: u16,
    pub evidence_ids: Vec<EvidenceId>,
    pub payload: Vec<u8>,
}

impl OwnedProposal {
    /// Borrow into the strict, validate-able view. No copies, no allocation.
    #[must_use]
    pub fn as_view(&self) -> CognoProposalView<'_> {
        CognoProposalView {
            schema_version: SCHEMA_VERSION,
            action: self.action,
            category: self.category,
            scope: self.scope,
            confidence_bps: self.confidence_bps,
            evidence_ids: &self.evidence_ids,
            payload: &self.payload,
        }
    }
}

/// Model backend trait. The runtime never trusts outputs (S1); it always runs
/// them through the deterministic pipeline (`cogno_runtime`).
pub trait ModelBackend {
    /// Static info; stable for the backend's lifetime.
    fn info(&self) -> BackendInfo;

    /// Produce the next proposal. Deterministic for a given input state.
    /// Backends may return `Exhausted` when their script is spent (Phase 1).
    fn next_proposal(&mut self) -> Result<OwnedProposal, BackendError>;
}
