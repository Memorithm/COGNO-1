//! Checked reward application for the sealed cognitive V4 soft path.
//!
//! This is the only runtime surface that turns a V4 soft adjustment into an
//! actual score change. The ordinary deterministic pipeline must already have
//! returned `Eligible`, controlled Meta must be bound to the exact installed
//! V4 artifact, and the adjustment itself is capped before this module sees it.
//! Hard rejection therefore remains non-compensable.

use crate::{
    CognitiveObservationInput, CognitivePostHardGateError, CognitiveSoftAdjustment,
    CognitiveSoftAdjustmentError, CognitiveSoftAdjustmentPolicy, CognitiveSoftTargets,
    HardGatedCognitiveContext, PipelineParams, Runtime,
};
use cogno_core::{CandidateScore, Reward};

/// One immutable result of applying a bounded cognitive soft delta.
///
/// Construction is private so callers cannot synthesize a score that appears
/// to have passed the post-hard V4 path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedCognitiveReward {
    base_score: CandidateScore,
    final_score: CandidateScore,
    adjustment: CognitiveSoftAdjustment,
    hard_constraints_preserved: bool,
    _seal: (),
}

impl AppliedCognitiveReward {
    #[must_use]
    pub const fn base_score(&self) -> CandidateScore {
        self.base_score
    }

    #[must_use]
    pub const fn final_score(&self) -> CandidateScore {
        self.final_score
    }

    #[must_use]
    pub const fn delta(&self) -> i64 {
        self.adjustment.delta()
    }

    #[must_use]
    pub const fn adjustment(&self) -> &CognitiveSoftAdjustment {
        &self.adjustment
    }

    #[must_use]
    pub const fn model_generation(&self) -> u64 {
        self.adjustment.model_generation()
    }

    #[must_use]
    pub const fn model_artifact_sha256(&self) -> [u8; 32] {
        self.adjustment.model_artifact_sha256()
    }

    #[must_use]
    pub const fn meta_candidate_digest(&self) -> [u8; 32] {
        self.adjustment.meta_candidate_digest()
    }

    #[must_use]
    pub const fn hard_constraints_preserved(&self) -> bool {
        self.hard_constraints_preserved
    }
}

/// Fail-closed errors for the complete hard-gated cognitive reward path.
#[derive(Clone, Debug, PartialEq)]
pub enum CognitiveRewardError {
    PostHard(CognitivePostHardGateError),
    Adjustment(CognitiveSoftAdjustmentError),
    ArithmeticOverflow,
}

impl From<CognitivePostHardGateError> for CognitiveRewardError {
    fn from(error: CognitivePostHardGateError) -> Self {
        Self::PostHard(error)
    }
}

impl From<CognitiveSoftAdjustmentError> for CognitiveRewardError {
    fn from(error: CognitiveSoftAdjustmentError) -> Self {
        Self::Adjustment(error)
    }
}

impl HardGatedCognitiveContext {
    /// Apply the already-bounded adjustment through checked integer reward
    /// arithmetic. This method is unavailable without the private post-hard
    /// context seal.
    pub fn apply_bounded_soft_reward(
        &self,
        policy: CognitiveSoftAdjustmentPolicy,
        targets: CognitiveSoftTargets<'_>,
    ) -> Result<AppliedCognitiveReward, CognitiveRewardError> {
        let adjustment = self.derive_bounded_soft_adjustment(policy, targets)?;
        let base_score = adjustment.base_score();
        let final_score = checked_score_add(base_score, adjustment.delta())?;
        Ok(AppliedCognitiveReward {
            base_score,
            final_score,
            adjustment,
            hard_constraints_preserved: true,
            _seal: (),
        })
    }
}

impl Runtime {
    /// Execute the complete controlled path:
    ///
    /// `hard pipeline -> Meta/artifact binding -> V4 observation -> bounded
    /// adjustment -> checked score addition -> audit`.
    ///
    /// The legacy [`Runtime::run_pipeline`] path is unchanged. Any hard
    /// rejection or Meta/model mismatch returns before a cognitive delta exists.
    pub fn run_pipeline_with_cognitive_soft_reward(
        &mut self,
        params: PipelineParams<'_>,
        cognitive: CognitiveObservationInput<'_>,
        policy: CognitiveSoftAdjustmentPolicy,
        targets: CognitiveSoftTargets<'_>,
    ) -> Result<AppliedCognitiveReward, CognitiveRewardError> {
        let context = self.run_pipeline_for_cognitive_soft_context(params, cognitive)?;
        let applied = context.apply_bounded_soft_reward(policy, targets)?;
        self.audit.cognitive_reward(&applied);
        Ok(applied)
    }
}

fn checked_score_add(
    base_score: CandidateScore,
    delta: i64,
) -> Result<CandidateScore, CognitiveRewardError> {
    let base_reward = Reward(base_score.points);
    let delta_reward = Reward(delta);
    let final_reward = base_reward
        .checked_add(delta_reward)
        .ok_or(CognitiveRewardError::ArithmeticOverflow)?;
    Ok(CandidateScore::new(final_reward.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_reward_application_preserves_exact_integer_delta() {
        assert_eq!(
            checked_score_add(CandidateScore::new(40), 25),
            Ok(CandidateScore::new(65))
        );
        assert_eq!(
            checked_score_add(CandidateScore::new(40), -25),
            Ok(CandidateScore::new(15))
        );
    }

    #[test]
    fn checked_reward_application_fails_closed_on_overflow() {
        assert_eq!(
            checked_score_add(CandidateScore::new(i64::MAX), 1),
            Err(CognitiveRewardError::ArithmeticOverflow)
        );
        assert_eq!(
            checked_score_add(CandidateScore::new(i64::MIN), -1),
            Err(CognitiveRewardError::ArithmeticOverflow)
        );
    }
}
