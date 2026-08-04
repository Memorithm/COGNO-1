//! Errors for the SciRust backend. Everything returns [`SciRustResult`]; no
//! `panic!`, no `unwrap`, no `expect` on hostile-shaped inputs.

/// Errors raised by `cogno-scirust`. All are recoverable and structured so the
/// runtime can surface them as `RejectReason::Malformed`/`Overflow` at the
/// lexicographic decision boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum SciRustError {
    /// Shape mismatch in a tensor op. Carries the two offending shapes for the
    /// audit trail.
    Shape { lhs: Vec<usize>, rhs: Vec<usize> },
    /// Integer overflow in a size/stride computation (§11 checked arithmetic).
    Overflow,
    /// A buffer exceeded its bound (§12 admission, §17 KV limit).
    CapacityExceeded { requested: usize, maximum: usize },
    /// Admission refused the request before allocation.
    BudgetExceeded { requested: usize, available: usize },
    /// The backend's objective was invoked while `MetaObjective` is gated
    /// (§27 Phase 4 preconditions not all attested). Fail-closed (S10).
    Gated,
    /// Empty argument where a non-empty one was required.
    Empty,
    /// Index out of range (kept as a structured error instead of panicking).
    Index { idx: usize, len: usize },
    /// A numeric operation produced a non-finite value (NaN/Inf). Surfaced
    /// rather than propagated silently into a reward.
    NonFinite,
}

impl std::fmt::Display for SciRustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SciRustError::Shape { lhs, rhs } => {
                write!(f, "shape mismatch {:?} vs {:?}", lhs, rhs)
            }
            SciRustError::Overflow => write!(f, "arithmetic overflow"),
            SciRustError::CapacityExceeded { requested, maximum } => {
                write!(f, "capacity exceeded: {requested} > {maximum}")
            }
            SciRustError::BudgetExceeded {
                requested,
                available,
            } => {
                write!(f, "budget exceeded: {requested} > {available}")
            }
            SciRustError::Gated => write!(f, "meta-objective gated (§27 preconditions)"),
            SciRustError::Empty => write!(f, "empty argument"),
            SciRustError::Index { idx, len } => write!(f, "index {idx} out of range {len}"),
            SciRustError::NonFinite => write!(f, "non-finite value (NaN/Inf)"),
        }
    }
}

impl std::error::Error for SciRustError {}

/// Convenience alias.
pub type SciRustResult<T> = Result<T, SciRustError>;

/// Check a float is finite, returning a structured error otherwise.
pub fn ensure_finite(v: f32) -> SciRustResult<()> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(SciRustError::NonFinite)
    }
}
