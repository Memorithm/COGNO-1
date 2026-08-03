//! Deterministic decision and lexicographic ordering (COGNO-1 V2 §8).
//!
//! Safety is **not** a numeric component of the score. The decision is
//! lexicographic:
//!
//! 1. structural admissibility;
//! 2. hard constraints;
//! 3. capabilities;
//! 4. privacy;
//! 5. neuro-symbolic score (only reached when 1–4 pass);
//! 6. ranking of admissible candidates;
//! 7. deterministic tie-break.
//!
//! A safety penalty can never be cancelled by style, user score, performance,
//! a model reward, or a speed gain (S4).
//!
//! ```text
//! if !candidate.is_structurally_valid()       { return Reject(Malformed); }
//! if !candidate.hard_constraint_ok()          { return Reject(HardConstraint); }
//! if !candidate.capability_allowed()         { return Reject(Unauthorized); }
//! if let Some(r) = candidate.privacy_rejection() { return Reject(r); }
//! Decision::Eligible(score)
//! ```

/// Reason a candidate was rejected. Enums are `Copy` so validators may carry
/// them by value without allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RejectReason {
    Malformed,
    HardConstraint,
    Unauthorized,
    OutOfBounds,
    Confidential,
    Secret,
    Oversized,
    Overflow,
    DeadlineExceeded,
    Unknown,
}

/// A deterministic decision: either rejected for a reason, or eligible with
/// its score. The score is only present when every higher-priority gate passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision<S> {
    Reject(RejectReason),
    Eligible(S),
}

/// A candidate for the lexicographic decision. The runtime supplies concrete
/// implementations; the trait keeps the ordering explicit and testable.
pub trait Candidate {
    /// Admissibility of structure (schema, bounds on lengths, etc.).
    fn is_structurally_valid(&self) -> bool;
    /// `true` if no hard constraint is violated (§8 step 2).
    fn hard_constraint_ok(&self) -> bool;
    /// `true` if the capability policy authorizes the candidate (§8 step 3).
    fn capability_allowed(&self) -> bool;
    /// `Some(reason)` if a privacy gate rejects (§8 step 4); `None` to pass.
    fn privacy_rejection(&self) -> Option<RejectReason>;
}

/// Apply the lexicographic decision to a candidate whose score has already been
/// computed by the reward engine. Hard failures short-circuit before the score
/// is considered, which is what makes S4 (a hard violation cannot be
/// compensated) structural rather than numeric.
///
/// The reward engine itself is invoked by the caller *before* `decide`, per
/// the §8 ordering; its arithmetic failures surface as `MemoryError` there.
pub fn decide<C, S>(c: &C, score: S) -> Decision<S>
where
    C: Candidate,
{
    if !c.is_structurally_valid() {
        return Decision::Reject(RejectReason::Malformed);
    }
    if !c.hard_constraint_ok() {
        return Decision::Reject(RejectReason::HardConstraint);
    }
    if !c.capability_allowed() {
        return Decision::Reject(RejectReason::Unauthorized);
    }
    if let Some(r) = c.privacy_rejection() {
        return Decision::Reject(r);
    }
    Decision::Eligible(score)
}
