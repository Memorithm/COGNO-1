//! Controlled Meta activation for explicitly versioned reviewed model artifacts.
//!
//! This is additive to the historical linear-v1 activation path. The sealed
//! review variant selects the exact hostile artifact decoder, then the runtime
//! combines model-owned evidence with the same two explicit host attestations.
//! No model installation or persistence occurs here.

use crate::meta_activation::{HostMetaAttestation, MetaActivationAuthority, MetaActivationReceipt};
use cogno_core::{MetaObjectiveError, MetaPreconditions};
use cogno_model::{
    load_versioned_neural_artifact, EligibleVersionedMetaModelReview, MetaModelEvidence,
    VersionedNeuralArtifactError,
};

/// Fail-closed errors for versioned controlled Meta activation.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlledVersionedMetaActivationError {
    AlreadyActive,
    CandidateArtifactInvalid(VersionedNeuralArtifactError),
    Precondition(MetaObjectiveError),
}

pub(crate) fn prepare_controlled_versioned_meta_activation(
    review: &EligibleVersionedMetaModelReview,
    _host: HostMetaAttestation,
) -> Result<(MetaPreconditions, MetaActivationReceipt), ControlledVersionedMetaActivationError> {
    let artifact = review.artifact();
    load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes)
        .map_err(ControlledVersionedMetaActivationError::CandidateArtifactInvalid)?;

    let preconditions = combined_preconditions(review.evidence());
    if !preconditions.satisfied() {
        return Err(ControlledVersionedMetaActivationError::Precondition(
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
    fn host_attestation_still_cannot_supply_missing_model_evidence() {
        let _host =
            HostMetaAttestation::attest_validated_scalar_engine_and_frozen_reference_policy();
        let preconditions = combined_preconditions(MetaModelEvidence {
            log_probabilities_available: false,
            backend_differentiable: false,
            held_out_tests_in_place: false,
            anti_poisoning_working: false,
        });
        assert!(!preconditions.satisfied());
    }
}
