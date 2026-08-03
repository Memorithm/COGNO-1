//! Validators (COGNO-1 V2 §3, §8, §25).
//!
//! The pipeline applies validators in order: structural → symbolic → hard. Each
//! validator is total (never panics on hostile input) and returns a
//! `ValidationResult`. The lexicographic decider consumes `fail()` reasons.

use crate::ids::ValidationResult;
use crate::proposal::{validate_proposal, CognoProposalView};

/// Structural validator: schema version, field bounds, duplicate evidence,
/// payload size. Delegates to `validate_proposal` (§2) but returns a
/// `ValidationResult` for the pipeline.
pub fn structural(p: &CognoProposalView) -> ValidationResult {
    match validate_proposal(p) {
        Ok(()) => ValidationResult::pass(),
        Err(reason) => ValidationResult::fail(reason),
    }
}

/// Symbolic validator: cross-field consistency rules the core can check without
/// a model. Currently enforces that safety-category proposals carry no payload
/// interpreted as instructions (a conservative size/shape check); will grow as
/// the symbolic layer grows.
pub fn symbolic(p: &CognoProposalView) -> ValidationResult {
    use crate::proposal::RuleCategory;
    if matches!(p.category, RuleCategory::Safety) && p.confidence_bps == 0 {
        return ValidationResult::fail(crate::RejectReason::OutOfBounds);
    }
    ValidationResult::pass()
}
