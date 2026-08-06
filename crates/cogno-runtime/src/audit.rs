//! Audit log for the runtime (COGNO-1 V2 §3 last step, S6).
//!
//! Every decision (admit/reject), truncation and taste-influenced soft choice
//! is recorded. Scientific taste never carries authority: the audit records
//! which verified preferences influenced a decision and whether the runtime
//! abstained, while hard safety remains lexicographically prior.

use crate::taste_decision::TasteDecision;
use cogno_core::{ContextReport, RejectReason};

/// One verified-preference influence recorded after a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteInfluenceAudit {
    pub preference_id: u64,
    pub candidate_id: u64,
    pub confidence_bps: u16,
    pub requested_weight_bps: i16,
    pub applied_weight_bps: i16,
    pub score_delta: i64,
}

/// Complete trace of a decision influenced by scientific taste.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteDecisionAudit {
    pub selected_candidate_id: Option<u64>,
    pub selected_base_score: Option<i64>,
    pub selected_final_score: Option<i64>,
    pub abstained: bool,
    pub hard_constraints_preserved: bool,
    pub influences: Vec<TasteInfluenceAudit>,
}

impl From<&TasteDecision> for TasteDecisionAudit {
    fn from(decision: &TasteDecision) -> Self {
        Self {
            selected_candidate_id: decision.selected_candidate_id,
            selected_base_score: decision.selected_base_score,
            selected_final_score: decision.selected_final_score,
            abstained: decision.abstained,
            hard_constraints_preserved: true,
            influences: decision
                .influences
                .iter()
                .map(|influence| TasteInfluenceAudit {
                    preference_id: influence.preference_id,
                    candidate_id: influence.candidate_id,
                    confidence_bps: influence.confidence_bps,
                    requested_weight_bps: influence.requested_weight_bps,
                    applied_weight_bps: influence.applied_weight_bps,
                    score_delta: influence.score_delta,
                })
                .collect(),
        }
    }
}

/// One audit record. Pure data; no allocation on hot paths beyond pushing
/// the record itself (audit happens after the decision, never in the critical
/// pre-decision path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditRecord {
    pub decision: &'static str,
    pub reason: Option<RejectReason>,
    pub note: Option<String>,
    pub taste: Option<TasteDecisionAudit>,
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
            taste: None,
        });
    }

    /// Record an acceptance. Score/decision tracked by the caller; the audit
    /// carries the verdict, not the reward value, so secrets never leak (S8).
    pub fn accept(&mut self, note: Option<String>) {
        self.records.push(AuditRecord {
            decision: "accept",
            reason: None,
            note,
            taste: None,
        });
    }

    /// Record a deterministic soft decision and all verified taste influences.
    pub fn taste_decision(&mut self, decision: &TasteDecision) {
        self.records.push(AuditRecord {
            decision: if decision.abstained {
                "taste_abstain"
            } else {
                "taste_select"
            },
            reason: None,
            note: None,
            taste: Some(TasteDecisionAudit::from(decision)),
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
            taste: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taste_trace_preserves_before_after_and_provenance() {
        let decision = TasteDecision {
            selected_candidate_id: Some(4),
            selected_base_score: Some(10),
            selected_final_score: Some(18),
            abstained: false,
            influences: vec![crate::TasteInfluence {
                preference_id: 7,
                candidate_id: 4,
                confidence_bps: 8_000,
                requested_weight_bps: 10,
                applied_weight_bps: 10,
                score_delta: 8,
            }],
        };
        let mut audit = Audit::default();
        audit.taste_decision(&decision);
        let trace = audit.records[0].taste.as_ref().expect("taste trace");
        assert_eq!(trace.selected_base_score, Some(10));
        assert_eq!(trace.selected_final_score, Some(18));
        assert!(trace.hard_constraints_preserved);
        assert_eq!(trace.influences[0].preference_id, 7);
    }
}
