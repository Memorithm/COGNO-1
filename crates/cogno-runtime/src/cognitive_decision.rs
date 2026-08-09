//! Deterministic comparison of already hard-gated cognitive V4 rewards.
//!
//! Inputs are references to [`crate::AppliedCognitiveReward`], whose private
//! seal proves the ordinary pipeline already returned `Eligible` before any V4
//! delta was applied. This layer therefore never accepts a caller-supplied
//! `hard_ok` flag. It additionally requires all compared rewards to come from
//! the same persisted model generation, artifact digest and active Meta digest.

use crate::{AppliedCognitiveReward, Runtime};
use cogno_core::CandidateScore;

/// Maximum number of already-admissible candidates in one cognitive decision.
pub const MAX_COGNITIVE_DECISION_CANDIDATES: usize = 256;

/// One stable candidate identifier paired with a sealed applied V4 reward.
#[derive(Clone, Copy, Debug)]
pub struct CognitiveDecisionCandidate<'a> {
    pub candidate_id: u64,
    pub applied: &'a AppliedCognitiveReward,
}

/// Per-candidate integer trace retained by the deterministic decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveDecisionTrace {
    pub candidate_id: u64,
    pub base_score: CandidateScore,
    pub final_score: CandidateScore,
    pub delta: i64,
}

/// Deterministic selection among already hard-admissible V4-adjusted candidates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CognitiveRewardDecision {
    pub selected_candidate_id: u64,
    pub selected_base_score: CandidateScore,
    pub selected_final_score: CandidateScore,
    pub model_generation: u64,
    pub model_artifact_sha256: [u8; 32],
    pub meta_candidate_digest: [u8; 32],
    pub hard_constraints_preserved: bool,
    pub candidates: Vec<CognitiveDecisionTrace>,
}

/// Fail-closed comparison errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveDecisionError {
    Empty,
    TooManyCandidates { actual: usize, maximum: usize },
    DuplicateCandidateId { candidate_id: u64 },
    InvalidHardGateSeal { candidate_id: u64 },
    ProvenanceMismatch { candidate_id: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DecisionView {
    candidate_id: u64,
    base_score: CandidateScore,
    final_score: CandidateScore,
    delta: i64,
    model_generation: u64,
    model_artifact_sha256: [u8; 32],
    meta_candidate_digest: [u8; 32],
    hard_constraints_preserved: bool,
}

/// Rank sealed applied rewards by final integer score. Exact ties are broken by
/// the smallest stable candidate id, independent of input ordering.
pub fn decide_with_applied_cognitive_rewards(
    candidates: &[CognitiveDecisionCandidate<'_>],
) -> Result<CognitiveRewardDecision, CognitiveDecisionError> {
    if candidates.is_empty() {
        return Err(CognitiveDecisionError::Empty);
    }
    if candidates.len() > MAX_COGNITIVE_DECISION_CANDIDATES {
        return Err(CognitiveDecisionError::TooManyCandidates {
            actual: candidates.len(),
            maximum: MAX_COGNITIVE_DECISION_CANDIDATES,
        });
    }

    let mut views = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|previous| previous.candidate_id == candidate.candidate_id)
        {
            return Err(CognitiveDecisionError::DuplicateCandidateId {
                candidate_id: candidate.candidate_id,
            });
        }
        views.push(DecisionView {
            candidate_id: candidate.candidate_id,
            base_score: candidate.applied.base_score(),
            final_score: candidate.applied.final_score(),
            delta: candidate.applied.delta(),
            model_generation: candidate.applied.model_generation(),
            model_artifact_sha256: candidate.applied.model_artifact_sha256(),
            meta_candidate_digest: candidate.applied.meta_candidate_digest(),
            hard_constraints_preserved: candidate.applied.hard_constraints_preserved(),
        });
    }
    decide_views(&views)
}

impl Runtime {
    /// Make and audit one deterministic comparison among already hard-gated V4
    /// rewards. The method does not invoke the model or rerun any hard gate.
    pub fn decide_with_cognitive_rewards(
        &mut self,
        candidates: &[CognitiveDecisionCandidate<'_>],
    ) -> Result<CognitiveRewardDecision, CognitiveDecisionError> {
        let decision = decide_with_applied_cognitive_rewards(candidates)?;
        self.audit.cognitive_decision(&decision);
        Ok(decision)
    }
}

fn decide_views(views: &[DecisionView]) -> Result<CognitiveRewardDecision, CognitiveDecisionError> {
    let first = *views.first().ok_or(CognitiveDecisionError::Empty)?;
    if !first.hard_constraints_preserved {
        return Err(CognitiveDecisionError::InvalidHardGateSeal {
            candidate_id: first.candidate_id,
        });
    }

    let mut best = first;
    let mut traces = Vec::with_capacity(views.len());
    for &view in views {
        if !view.hard_constraints_preserved {
            return Err(CognitiveDecisionError::InvalidHardGateSeal {
                candidate_id: view.candidate_id,
            });
        }
        if view.model_generation != first.model_generation
            || view.model_artifact_sha256 != first.model_artifact_sha256
            || view.meta_candidate_digest != first.meta_candidate_digest
        {
            return Err(CognitiveDecisionError::ProvenanceMismatch {
                candidate_id: view.candidate_id,
            });
        }
        if view.final_score > best.final_score
            || (view.final_score == best.final_score && view.candidate_id < best.candidate_id)
        {
            best = view;
        }
        traces.push(CognitiveDecisionTrace {
            candidate_id: view.candidate_id,
            base_score: view.base_score,
            final_score: view.final_score,
            delta: view.delta,
        });
    }
    traces.sort_by_key(|trace| trace.candidate_id);

    Ok(CognitiveRewardDecision {
        selected_candidate_id: best.candidate_id,
        selected_base_score: best.base_score,
        selected_final_score: best.final_score,
        model_generation: first.model_generation,
        model_artifact_sha256: first.model_artifact_sha256,
        meta_candidate_digest: first.meta_candidate_digest,
        hard_constraints_preserved: true,
        candidates: traces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(candidate_id: u64, final_score: i64) -> DecisionView {
        DecisionView {
            candidate_id,
            base_score: CandidateScore::new(100),
            final_score: CandidateScore::new(final_score),
            delta: final_score - 100,
            model_generation: 7,
            model_artifact_sha256: [3; 32],
            meta_candidate_digest: [3; 32],
            hard_constraints_preserved: true,
        }
    }

    #[test]
    fn highest_final_score_wins_and_trace_is_stable() {
        let decision = decide_views(&[view(9, 110), view(2, 125), view(4, 120)]).expect("decision");
        assert_eq!(decision.selected_candidate_id, 2);
        assert_eq!(decision.selected_final_score, CandidateScore::new(125));
        assert_eq!(
            decision
                .candidates
                .iter()
                .map(|trace| trace.candidate_id)
                .collect::<Vec<_>>(),
            vec![2, 4, 9]
        );
        assert!(decision.hard_constraints_preserved);
    }

    #[test]
    fn exact_score_tie_uses_smallest_candidate_id() {
        let decision = decide_views(&[view(10, 125), view(3, 125)]).expect("decision");
        assert_eq!(decision.selected_candidate_id, 3);
    }

    #[test]
    fn mixed_model_provenance_fails_closed() {
        let first = view(1, 100);
        let mut second = view(2, 101);
        second.model_generation = 8;
        assert_eq!(
            decide_views(&[first, second]),
            Err(CognitiveDecisionError::ProvenanceMismatch { candidate_id: 2 })
        );
    }

    #[test]
    fn invalid_hard_gate_seal_fails_closed() {
        let mut hostile = view(1, i64::MAX);
        hostile.hard_constraints_preserved = false;
        assert_eq!(
            decide_views(&[hostile]),
            Err(CognitiveDecisionError::InvalidHardGateSeal { candidate_id: 1 })
        );
    }
}
