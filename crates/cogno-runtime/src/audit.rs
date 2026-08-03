//! Audit log for the runtime (COGNO-1 V2 §3 last step, S6).
//!
//! Every decision (admit/reject) and every truncation is recorded. The audit
//! log is append-only in spirit; the persistent copy is `events.log` /
//! `manifest.json` (§18), managed by the runtime's journal layer backed by
//! `cogno-core::Journal`.

use cogno_core::{ContextReport, RejectReason};

/// One audit record. Pure data; no allocation on hot paths beyond pushing
/// the record itself (audit happens after the decision, never in the critical
/// pre-decision path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditRecord {
    pub decision: &'static str,
    pub reason: Option<RejectReason>,
    pub note: Option<String>,
}

/// In-memory audit buffer (bounded by the Phase 0 budget; a real runtime uses
/// the on-disk journal).
#[derive(Debug, Default)]
pub struct Audit {
    pub records: Vec<AuditRecord>,
}

impl Audit {
    /// Record a rejection with its reason.
    pub fn reject(&mut self, reason: RejectReason, note: Option<String>) {
        self.records.push(AuditRecord {
            decision: "reject",
            reason: Some(reason),
            note,
        });
    }

    /// Record an acceptance. Score/decision tracked by the caller; the audit
    /// carries the verdict, not the reward value, so secrets never leak (S8).
    pub fn accept(&mut self, note: Option<String>) {
        self.records.push(AuditRecord {
            decision: "accept",
            reason: None,
            note,
        });
    }

    /// Record a context truncation (never silent, §17).
    pub fn truncation(&mut self, report: ContextReport) {
        self.records.push(AuditRecord {
            decision: "truncate_context",
            reason: None,
            note: Some(format!(
                "dropped {} of {} tokens (policy: {:?})",
                report.dropped_tokens, report.requested_tokens, report.policy
            )),
        });
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
