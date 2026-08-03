//! Semantic (persistent) memory budget (COGNO-1 V2 §18).
//!
//! The persistent memory uses an append-only event journal as the source of
//! truth; the active profile is a derived view. When a limit is reached, the
//! runtime stops automatic writes, builds a verified snapshot, compacts under
//! a deterministic policy, keeps active rules and their minimal provenance, and
//! never silently drops audit-relevant evidence.

/// Budget for the semantic, persistent memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticMemoryBudget {
    pub max_log_bytes: u64,
    pub max_events: u64,
    pub max_rules: usize,
    pub max_evidence_per_rule: usize,
    pub max_payload_bytes: usize,
    pub max_snapshot_bytes: u64,
}

impl SemanticMemoryBudget {
    /// Conservative MVP defaults. Small and easy to audit; every field is
    /// non-zero so the runtime can detect misconfiguration.
    pub const MVP_SAFE: Self = Self {
        max_log_bytes: 16 * 1024 * 1024,
        max_events: 100_000,
        max_rules: 512,
        max_evidence_per_rule: 32,
        max_payload_bytes: 4096,
        max_snapshot_bytes: 4 * 1024 * 1024,
    };

    #[must_use]
    pub const fn allows_payload(self, bytes: usize) -> bool {
        bytes <= self.max_payload_bytes
    }
}
