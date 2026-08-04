//! Memory / context / latency cost measure (COGNO-1 §SciRust #6).
//!
//! A deterministic scalar cost combining three contributions:
//!
//! - **memory** bytes (model weights + KV cache + scratch);
//! - **context** tokens;
//! - **latency** estimate (computed tokens * unit cost + fixed overhead).
//!
//! The cost is a **soft** term: the reward engine may subtract it from a
//! score, but it can never override a hard gate (§8/S4). Overflow is saturated
//! to a defined ceiling rather than panicking.

/// Cost breakdown. Every field is bounded by `u64::MAX`; saturating arithmetic
/// on construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostBreakdown {
    pub memory_bytes: u64,
    pub context_tokens: u64,
    pub latency_micros: u64,
}

/// Cost scalar in arbitrary units, derived from a breakdown. Kept as `u64`
/// (no float) so additions cannot drift across platforms and the tie-break
/// stays deterministic (§8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cost(pub u64);

impl Cost {
    pub const ZERO: Self = Self(0);

    /// Combine a breakdown into a scalar with saturating arithmetic so a
    /// hostile or oversized request cannot overflow into a malformed score.
    /// Weights are fixed and documented (§SciRust #6).
    #[must_use]
    pub fn from_breakdown(b: CostBreakdown) -> Self {
        // memory: 1 unit per 1 KiB; context: 4 units per token; latency: 1 unit
        // per 10 µs. Saturating.
        let mem = b.memory_bytes / 1024;
        let ctx = b.context_tokens.saturating_mul(4);
        let lat = b.latency_micros / 10;
        Self(mem.saturating_add(ctx).saturating_add(lat))
    }

    /// Saturating addition.
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl CostBreakdown {
    #[must_use]
    pub const fn new(memory_bytes: u64, context_tokens: u64, latency_micros: u64) -> Self {
        Self {
            memory_bytes,
            context_tokens,
            latency_micros,
        }
    }

    #[must_use]
    pub fn cost(self) -> Cost {
        Cost::from_breakdown(self)
    }
}
