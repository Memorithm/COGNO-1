//! Audit log for the runtime (COGNO-1 V2 §3 last step, S6).
//!
//! Every decision (admit/reject), truncation, taste-influenced soft choice and
//! explicit cognitive observation is recorded. Scientific taste and neural
//! observations never carry authority; hard safety remains lexicographically
//! prior.

use crate::cognitive_decision::CognitiveRewardDecision;
use crate::cognitive_observation::CognitiveObservation;
use crate::cognitive_reward::AppliedCognitiveReward;
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

/// Integer-only trace of one bounded V4 reward application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveRewardAudit {
    pub base_score: i64,
    pub final_score: i64,
    pub delta: i64,
    pub model_generation: u64,
    pub model_artifact_sha256: [u8; 32],
    pub meta_candidate_digest: [u8; 32],
    pub hard_constraints_preserved: bool,
}

impl From<&AppliedCognitiveReward> for CognitiveRewardAudit {
    fn from(applied: &AppliedCognitiveReward) -> Self {
        Self {
            base_score: applied.base_score().points,
            final_score: applied.final_score().points,
            delta: applied.delta(),
            model_generation: applied.model_generation(),
            model_artifact_sha256: applied.model_artifact_sha256(),
            meta_candidate_digest: applied.meta_candidate_digest(),
            hard_constraints_preserved: applied.hard_constraints_preserved(),
        }
    }
}

/// One audit record. Pure data; no allocation on hot paths beyond pushing
/// the record itself (audit happens after the decision/observation, never in
/// the critical hard-gate path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditRecord {
    pub decision: &'static str,
    pub reason: Option<RejectReason>,
    pub note: Option<String>,
    pub taste: Option<TasteDecisionAudit>,
    pub cognitive: Option<CognitiveObservation>,
    pub cognitive_reward: Option<CognitiveRewardAudit>,
    pub cognitive_decision: Option<CognitiveRewardDecision>,
}

/// Hard cap on the in-memory audit buffer (T16): an attacker generating
/// rejections must not be able to grow the process without bound. Oldest
/// records are evicted first and counted, never silently discarded.
pub const MAX_AUDIT_RECORDS: usize = 4096;

/// In-memory audit buffer, bounded by [`MAX_AUDIT_RECORDS`]; a real runtime
/// uses the on-disk journal. Evictions of the oldest records are surfaced in
/// `dropped_records`.
#[derive(Debug, Default)]
pub struct Audit {
    pub records: Vec<AuditRecord>,
    pub dropped_records: usize,
}

impl Audit {
    /// Bounded push. When the buffer is full the **oldest** record is evicted
    /// (FIFO) and `dropped_records` is incremented — bounded memory, no
    /// silent loss.
    fn push_record(&mut self, record: AuditRecord) {
        if self.records.len() >= MAX_AUDIT_RECORDS {
            self.records.drain(0..1);
            self.dropped_records = self.dropped_records.saturating_add(1);
        }
        self.records.push(record);
    }

    /// Record a rejection with its reason.
    pub fn reject(&mut self, reason: RejectReason, note: Option<String>) {
        self.push_record(AuditRecord {
            decision: "reject",
            reason: Some(reason),
            note,
            taste: None,
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: None,
        });
    }

    /// Record an acceptance. Score/decision tracked by the caller; the audit
    /// carries the verdict, not the reward value, so secrets never leak (S8).
    pub fn accept(&mut self, note: Option<String>) {
        self.push_record(AuditRecord {
            decision: "accept",
            reason: None,
            note,
            taste: None,
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: None,
        });
    }

    /// Record a tool authorization decision. The authorization step is the
    /// most security-sensitive decision in the pipeline; it is audited like
    /// any other state-changing decision (§3, S6).
    pub fn tool_authorize(&mut self, note: Option<String>) {
        self.push_record(AuditRecord {
            decision: "tool_authorize",
            reason: None,
            note,
            taste: None,
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: None,
        });
    }

    /// Record a deterministic soft decision and all verified taste influences.
    pub fn taste_decision(&mut self, decision: &TasteDecision) {
        self.push_record(AuditRecord {
            decision: if decision.abstained {
                "taste_abstain"
            } else {
                "taste_select"
            },
            reason: None,
            note: None,
            taste: Some(TasteDecisionAudit::from(decision)),
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: None,
        });
    }

    /// Record an integer-only V4 observation. The snapshot is explicitly
    /// non-authoritative and carries its exact persisted model provenance.
    pub fn cognitive_observation(&mut self, observation: &CognitiveObservation) {
        self.push_record(AuditRecord {
            decision: "cognitive_observation",
            reason: None,
            note: None,
            taste: None,
            cognitive: Some(observation.clone()),
            cognitive_reward: None,
            cognitive_decision: None,
        });
    }

    /// Record the checked bounded V4 score delta after the hard-gated context
    /// has been sealed. No raw model output is interpreted here.
    pub fn cognitive_reward(&mut self, applied: &AppliedCognitiveReward) {
        self.push_record(AuditRecord {
            decision: "cognitive_reward",
            reason: None,
            note: None,
            taste: None,
            cognitive: None,
            cognitive_reward: Some(CognitiveRewardAudit::from(applied)),
            cognitive_decision: None,
        });
    }

    /// Record a deterministic comparison among sealed post-hard V4 rewards.
    pub fn cognitive_decision(&mut self, decision: &CognitiveRewardDecision) {
        self.push_record(AuditRecord {
            decision: "cognitive_decision",
            reason: None,
            note: None,
            taste: None,
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: Some(decision.clone()),
        });
    }

    /// Record a context truncation (never silent, §17).
    pub fn truncation(&mut self, report: ContextReport) {
        self.push_record(AuditRecord {
            decision: "truncate_context",
            reason: None,
            note: Some(format!(
                "dropped {} of {} tokens (policy: {:?})",
                report.dropped_tokens, report.requested_tokens, report.policy
            )),
            taste: None,
            cognitive: None,
            cognitive_reward: None,
            cognitive_decision: None,
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
    use crate::CognitivePreferenceRelation;

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
        assert!(audit.records[0].cognitive.is_none());
        assert!(audit.records[0].cognitive_reward.is_none());
        assert!(audit.records[0].cognitive_decision.is_none());
    }

    #[test]
    fn cognitive_trace_preserves_generation_digest_and_non_authority() {
        let observation = CognitiveObservation {
            model_generation: 7,
            model_artifact_sha256: [9; 32],
            classification_class: 2,
            classification_confidence_bps: 7_500,
            classification_probabilities_bps: vec![1_000, 1_500, 7_500],
            preference: CognitivePreferenceRelation::Left,
            symbolic_satisfaction_bps: vec![9_000, 2_000],
            contradiction_bps: 8_000,
            retrieval_selected_index: 1,
            retrieval_candidate_count: 2,
            authoritative: false,
        };
        let mut audit = Audit::default();
        audit.cognitive_observation(&observation);
        let trace = audit.records[0]
            .cognitive
            .as_ref()
            .expect("cognitive trace");
        assert_eq!(trace.model_generation, 7);
        assert_eq!(trace.model_artifact_sha256, [9; 32]);
        assert_eq!(trace.retrieval_candidate_count, 2);
        assert!(!trace.authoritative);
        assert!(audit.records[0].taste.is_none());
        assert!(audit.records[0].cognitive_reward.is_none());
        assert!(audit.records[0].cognitive_decision.is_none());
    }
}

#[cfg(test)]
mod bounded_tests {
    use super::*;

    #[test]
    fn audit_buffer_stays_bounded_under_flood_and_counts_evictions() {
        // T16: an attacker generating rejections must not grow the process
        // without bound; oldest records are evicted and counted, never lost
        // silently.
        let mut audit = Audit::default();
        for i in 0..(MAX_AUDIT_RECORDS * 3) {
            audit.reject(RejectReason::Oversized, Some(format!("r{i}")));
            assert!(audit.records.len() <= MAX_AUDIT_RECORDS);
        }
        assert_eq!(audit.records.len(), MAX_AUDIT_RECORDS);
        assert_eq!(
            audit.dropped_records,
            MAX_AUDIT_RECORDS * 2,
            "every evicted record is accounted for"
        );
        // FIFO: the oldest surviving record is the first one not yet dropped.
        assert_eq!(audit.records[0].note.as_deref(), Some("r8192"));
    }

    #[test]
    fn tool_authorization_is_audited() {
        // §3/S6: an authorization is a state-relevant decision and must be
        // traceable like any rejection.
        let mut audit = Audit::default();
        audit.tool_authorize(Some("dry-run authorized".to_string()));
        assert_eq!(audit.len(), 1);
        assert_eq!(audit.records[0].decision, "tool_authorize");
    }
}
