//! Primitive identifier and value-carrier types shared across COGNO-1.
//!
//! These are deliberately thin newtypes so that token ids, rule references,
//! candidate scores and validation results cannot be confused with each other.

use crate::RejectReason;

/// Token identifier produced by a trusted tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(pub u32);

/// Reference to a rule held by the core. Opaque index; its lifecycle
/// (`RuleState`) is stored alongside the rule, not in the reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleRef(pub u32);

/// Deterministic integer score for an admissible candidate.
///
/// Deliberately an integer (not floating point) so the lexicographic tie-break
/// is fully deterministic across platforms. Float reward machinery
/// (log-probabilities) belongs to Phase 4 and lives behind the model/backend
/// boundary, never in `cogno-core`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateScore {
    pub points: i64,
}

impl CandidateScore {
    #[must_use]
    pub const fn new(points: i64) -> Self {
        Self { points }
    }

    pub const ZERO: Self = Self { points: 0 };
}

/// Outcome of one validator on one candidate.
///
/// `passed == true` imposes no constraint; `passed == false` carries the reason
/// consumed by the lexicographic decider (§8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidationResult {
    pub passed: bool,
    pub reason: Option<RejectReason>,
}

impl ValidationResult {
    #[must_use]
    pub const fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
        }
    }

    #[must_use]
    pub const fn fail(reason: RejectReason) -> Self {
        Self {
            passed: false,
            reason: Some(reason),
        }
    }
}
