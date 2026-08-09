//! Sealed post-hard-gate context for cognitive soft scoring.
//!
//! The ordinary deterministic pipeline must first return `Eligible`, controlled
//! Meta must already be active, and the Meta candidate digest must equal the
//! exact persisted V4 artifact installed in the runtime. Only then is a
//! non-authoritative cognitive observation produced.

use crate::cognitive_observation::{
    CognitiveObservation, CognitiveObservationError, CognitiveObservationInput,
};
use crate::pipeline::{PipelineOutcome, PipelineParams};
use crate::runtime::Runtime;
use cogno_core::{CandidateScore, RejectReason};

/// Non-forgeable runtime context proving hard-gate success and model/Meta
/// identity before any neural soft factor may be derived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardGatedCognitiveContext {
    base_score: CandidateScore,
    observation: CognitiveObservation,
    meta_candidate_digest: [u8; 32],
    _seal: (),
}

impl HardGatedCognitiveContext {
    #[must_use]
    pub const fn base_score(&self) -> CandidateScore {
        self.base_score
    }

    #[must_use]
    pub const fn observation(&self) -> &CognitiveObservation {
        &self.observation
    }

    #[must_use]
    pub const fn meta_candidate_digest(&self) -> [u8; 32] {
        self.meta_candidate_digest
    }
}

/// Fail-closed reasons why a post-hard cognitive context cannot be minted.
#[derive(Clone, Debug, PartialEq)]
pub enum CognitivePostHardGateError {
    PipelineRejected {
        stage: &'static str,
        reason: RejectReason,
    },
    MetaInactive,
    MissingMetaCandidateDigest,
    NoInstalledCognitiveModel,
    MissingInstalledArtifactDigest,
    MetaModelDigestMismatch {
        meta_candidate_digest: [u8; 32],
        installed_artifact_digest: [u8; 32],
    },
    Observation(CognitiveObservationError),
}

impl From<CognitiveObservationError> for CognitivePostHardGateError {
    fn from(error: CognitiveObservationError) -> Self {
        Self::Observation(error)
    }
}

impl Runtime {
    /// Run the ordinary hard-gated pipeline first, then mint one sealed V4
    /// context only when Meta is active for the exact installed artifact.
    ///
    /// A rejected proposal never invokes the model. An eligible proposal with
    /// inactive or mismatched Meta also never invokes the model.
    pub fn run_pipeline_for_cognitive_soft_context(
        &mut self,
        params: PipelineParams<'_>,
        cognitive: CognitiveObservationInput<'_>,
    ) -> Result<HardGatedCognitiveContext, CognitivePostHardGateError> {
        let base_score = match self.run_pipeline(params) {
            PipelineOutcome::Rejected { stage, reason } => {
                return Err(CognitivePostHardGateError::PipelineRejected { stage, reason });
            }
            PipelineOutcome::Eligible { score } => score,
        };

        if !self.meta_is_active() {
            return Err(CognitivePostHardGateError::MetaInactive);
        }
        let meta_candidate_digest = self
            .meta_candidate_digest()
            .ok_or(CognitivePostHardGateError::MissingMetaCandidateDigest)?;
        if self.cognitive_model().is_none() {
            return Err(CognitivePostHardGateError::NoInstalledCognitiveModel);
        }
        let installed_artifact_digest = self
            .cognitive_model_artifact_sha256()
            .ok_or(CognitivePostHardGateError::MissingInstalledArtifactDigest)?;
        if meta_candidate_digest != installed_artifact_digest {
            return Err(CognitivePostHardGateError::MetaModelDigestMismatch {
                meta_candidate_digest,
                installed_artifact_digest,
            });
        }

        let observation = self.observe_cognitive(cognitive)?;
        if observation.authoritative || observation.model_artifact_sha256 != meta_candidate_digest {
            return Err(CognitivePostHardGateError::MetaModelDigestMismatch {
                meta_candidate_digest,
                installed_artifact_digest: observation.model_artifact_sha256,
            });
        }
        Ok(HardGatedCognitiveContext {
            base_score,
            observation,
            meta_candidate_digest,
            _seal: (),
        })
    }
}
