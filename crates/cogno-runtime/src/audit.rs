//! Audit log for the runtime (COGNO-1 V2 §3 last step, S6).
//!
//! Every decision (admit/reject), truncation, taste-influenced soft choice and
//! explicit cognitive observation is recorded. Scientific taste and neural
//! observations never carry authority; hard safety remains lexicographically
//! prior.

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
            cognitive: None,
            cognitive_reward: None,
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
            cognitive: None,
            cognitive_reward: None,
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
            cognitive: None,
            cognitive_reward: None,
        });
    }

    /// Record an integer-only V4 observation. The snapshot is explicitly
    /// non-authoritative and carries its exact persisted model provenance.
    pub fn cognitive_observation(&mut self, observation: &CognitiveObservation) {
        self.records.push(AuditRecord {
            decision: "cognitive_observation",
            reason: None,
            note: None,
            taste: None,
            cognitive: Some(observation.clone()),
            cognitive_reward: None,
        });
    }

    /// Record the checked bounded V4 score delta after the hard-gated context
    /// has been sealed. No raw model output is interpreted here.
    pub fn cognitive_reward(&mut self, applied: &AppliedCognitiveReward) {
        self.records.push(AuditRecord {
            decision: "cognitive_reward",
            reason: None,
            note: None,
            taste: None,
            cognitive: None,
            cognitive_reward: Some(CognitiveRewardAudit::from(applied)),
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
            cognitive: None,
            cognitive_reward: None,
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
    }
}
