//! Strict, versioned proposal schema and validator (COGNO-1 V2 §2).
//!
//! The small model emits only typed, versioned proposals. `confidence_bps` is
//! in basis points: `0` (no confidence) ..= `10_000` (max). Any value above
//! `10_000` is rejected. Unknown, duplicated, or out-of-range fields cause an
//! explicit rejection. The validator is intentionally total: it never panics
//! on hostile input and always returns `Ok`/`Err(RejectReason)`.

use crate::evidence::EvidenceId;
use crate::RejectReason;

/// Proposal schema version. Bumped on incompatible changes to
/// `CognoProposalView`.
pub const SCHEMA_VERSION: u16 = 1;

/// Maximum confidence in basis points. Values above this are rejected.
pub const MAX_CONFIDENCE_BPS: u16 = 10_000;

/// Hard caps on slice lengths in a proposal. External input can never push the
/// validator into unbounded work.
pub const MAX_EVIDENCE_IDS: usize = 64;
pub const MAX_PAYLOAD_BYTES: usize = 4096;

/// Action a proposal may request. The model cannot mint new actions on its
/// own; unknown discriminants are `Unknown` and rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProposalAction {
    ExtractPreference,
    ClassifyFeedback,
    CompareProposal,
    AssociateCategory,
    EstimateRelevance,
    RankCandidates,
    Explain,
    FlagContradiction,
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleCategory {
    Style,
    Format,
    Behavior,
    Safety,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleScope {
    Session,
    Project,
    UserGlobal,
    Unknown,
}

/// Borrowed, lifetime-bounded view of a proposal. No ownership, no copy: the
/// runtime owns the byte storage and hands a view to the core for validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognoProposalView<'a> {
    pub schema_version: u16,
    pub action: ProposalAction,
    pub category: RuleCategory,
    pub scope: RuleScope,
    pub confidence_bps: u16,
    pub evidence_ids: &'a [EvidenceId],
    pub payload: &'a [u8],
}

/// Errors raised while parsing raw bytes into a `CognoProposalView`-shaped
/// structure. Kept separate from `RejectReason` because parsing failures are a
/// lower layer than the policy decision; the runtime maps a `ParseError` to
/// `RejectReason::Malformed` when surfacing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    Truncated,
    UnknownAction,
    UnknownCategory,
    UnknownScope,
    DuplicateEvidence,
    TooManyEvidenceIds,
    PayloadTooLarge,
    VersionMismatch,
}

impl ParseError {
    #[must_use]
    pub const fn to_reject(self) -> RejectReason {
        match self {
            ParseError::Truncated
            | ParseError::UnknownAction
            | ParseError::UnknownCategory
            | ParseError::UnknownScope
            | ParseError::DuplicateEvidence
            | ParseError::TooManyEvidenceIds
            | ParseError::PayloadTooLarge
            | ParseError::VersionMismatch => RejectReason::Malformed,
        }
    }
}

/// Validate a parsed proposal view against the §2 rules:
///  - schema version must equal `SCHEMA_VERSION` (unknown versions rejected);
///  - `confidence_bps` must be `<= MAX_CONFIDENCE_BPS`;
///  - `evidence_ids` must not exceed `MAX_EVIDENCE_IDS`, must not be empty for
///    actions that require evidence, and must not contain duplicates;
///  - `payload` must not exceed `MAX_PAYLOAD_BYTES`;
///  - `Unsupported`/`Unknown` discriminants are rejected.
pub fn validate_proposal(p: &CognoProposalView) -> Result<(), RejectReason> {
    if p.schema_version != SCHEMA_VERSION {
        return Err(RejectReason::OutOfBounds);
    }
    if p.confidence_bps > MAX_CONFIDENCE_BPS {
        return Err(RejectReason::OutOfBounds);
    }
    if p.evidence_ids.len() > MAX_EVIDENCE_IDS {
        return Err(RejectReason::Oversized);
    }
    if p.payload.len() > MAX_PAYLOAD_BYTES {
        return Err(RejectReason::Oversized);
    }
    if matches!(p.action, ProposalAction::Unsupported) {
        return Err(RejectReason::Unauthorized);
    }
    if matches!(p.category, RuleCategory::Unknown) {
        return Err(RejectReason::Malformed);
    }
    if matches!(p.scope, RuleScope::Unknown) {
        return Err(RejectReason::Malformed);
    }

    // Duplicate detection is pairwise over the (bounded, ≤ MAX_EVIDENCE_IDS)
    // id list: adjacency-only checks miss `[a, b, a]`. Allocation-free so the
    // validation hot path stays bounded.
    for i in 0..p.evidence_ids.len() {
        if p.evidence_ids[i].0 == 0 {
            return Err(RejectReason::Malformed);
        }
        for prior in &p.evidence_ids[..i] {
            if prior == &p.evidence_ids[i] {
                return Err(RejectReason::Malformed);
            }
        }
    }

    if requires_evidence(p.action) && p.evidence_ids.is_empty() {
        return Err(RejectReason::Malformed);
    }
    Ok(())
}

const fn requires_evidence(action: ProposalAction) -> bool {
    matches!(
        action,
        ProposalAction::ExtractPreference
            | ProposalAction::CompareProposal
            | ProposalAction::FlagContradiction
    )
}
