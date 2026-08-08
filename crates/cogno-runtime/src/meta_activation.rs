//! Controlled runtime activation of the Phase-4 Meta-NeuroSymbolic objective.
//!
//! Model-owned evidence arrives only as an `EligibleMetaModelReview`, whose
//! private digest seal was minted by the held-out review gate in `cogno-model`.
//! The host contributes the two facts only it can attest: scalar-engine
//! validation and a frozen reference policy. The candidate artifact is
//! re-verified through the architecture-selected hostile loader before the six
//! preconditions are combined.

use cogno_core::{MetaObjectiveError, MetaPreconditions};
use cogno_model::{
    load_versioned_neural_artifact, EligibleMetaModelReview, MetaModelEvidence,
    VersionedNeuralArtifactError,
};

/// Explicit host attestation for the two Phase-4 facts that model code cannot
/// self-prove.
///
/// Construction is intentionally verbose: obtaining this token is the host's
/// positive assertion that both the scalar engine has been validated and the
/// reference policy has been frozen for the activation being performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMetaAttestation {
    _private: (),
}

impl HostMetaAttestation {
    #[must_use]
    pub const fn attest_validated_scalar_engine_and_frozen_reference_policy() -> Self {
        Self { _private: () }
    }
}

/// Authority granted by a successful activation receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaActivationAuthority {
    /// Only the in-memory Meta objective was enabled. No model installation,
    /// persistence or tool authority is implied.
    ObjectiveOnly,
}

/// Immutable audit receipt binding activation to the reviewed candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetaActivationReceipt {
    pub candidate_artifact_sha256: [u8; 32],
    pub validation_accuracy_bps: u16,
    pub test_accuracy_bps: u16,
    pub authority: MetaActivationAuthority,
}

/// Fail-closed controlled-activation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlledMetaActivationError {
    AlreadyActive,
    CandidateArtifactInvalid(VersionedNeuralArtifactError),
    Precondition(MetaObjectiveError),
}

pub(crate) fn prepare_controlled_meta_activation(
    review: &EligibleMetaModelReview,
    _host: HostMetaAttestation,
) -> Result<(MetaPreconditions, MetaActivationReceipt), ControlledMetaActivationError> {
    let artifact = review.artifact();
    load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes)
        .map_err(ControlledMetaActivationError::CandidateArtifactInvalid)?;

    let evidence = review.evidence();
    let preconditions = combined_preconditions(evidence);
    if !preconditions.satisfied() {
        return Err(ControlledMetaActivationError::Precondition(
            MetaObjectiveError::PreconditionMissing,
        ));
    }
    Ok((
        preconditions,
        MetaActivationReceipt {
            candidate_artifact_sha256: artifact.manifest.weights_hash,
            validation_accuracy_bps: review.validation_accuracy_bps(),
            test_accuracy_bps: review.test_accuracy_bps(),
            authority: MetaActivationAuthority::ObjectiveOnly,
        },
    ))
}

const fn combined_preconditions(evidence: MetaModelEvidence) -> MetaPreconditions {
    MetaPreconditions {
        scalar_engine_validated: true,
        reference_policy_frozen: true,
        log_probabilities_available: evidence.log_probabilities_available,
        backend_differentiable: evidence.backend_differentiable,
        held_out_tests_in_place: evidence.held_out_tests_in_place,
        anti_poisoning_working: evidence.anti_poisoning_working,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_attestation_alone_cannot_synthesize_model_evidence() {
        let _host =
            HostMetaAttestation::attest_validated_scalar_engine_and_frozen_reference_policy();
        let missing_model = combined_preconditions(MetaModelEvidence {
            log_probabilities_available: false,
            backend_differentiable: false,
            held_out_tests_in_place: false,
            anti_poisoning_working: false,
        });
        assert!(!missing_model.satisfied());
    }
}
