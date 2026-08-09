//! Shared §20 gate for data entering model training corpora.
//!
//! This module owns the classification policy only. It grants no model,
//! persistence, Meta, tool or runtime authority.

use crate::training::Corpus;
use cogno_core::DataClassification;

/// Explicit host authority required before `Confidential` data may enter a
/// model training corpus. This token never authorizes `Secret`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostConfidentialTrainingAttestation {
    _private: (),
}

impl HostConfidentialTrainingAttestation {
    /// Explicitly authorize `Confidential` data for a bounded host-owned
    /// training operation. `Secret` remains forbidden.
    #[must_use]
    pub const fn authorize_confidential_training_data() -> Self {
        Self { _private: () }
    }
}

/// Fail-closed errors for model-training data governance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingDataGovernanceError {
    /// `Secret` is forbidden in model training sets.
    SecretTrainingData { index: usize },
    /// `Confidential` requires an explicit host attestation.
    ConfidentialTrainingDataRequiresAuthorization { index: usize },
    /// Internal example/classification storage lost its one-to-one invariant.
    ClassificationStateMismatch {
        examples: usize,
        classifications: usize,
    },
}

/// Defensively revalidate every stored training-data classification.
///
/// Public corpus admission already enforces §20 before storage. This function
/// exists for review/trainer boundaries that want to reassert that invariant
/// before allocating tokenizer/optimizer/training state.
pub fn validate_training_corpus(corpus: &Corpus) -> Result<(), TrainingDataGovernanceError> {
    corpus.validate_data_classifications()
}

/// Crate-owned admission policy shared by every model training corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrainingDataAdmissionPolicy {
    allow_confidential: bool,
}

impl TrainingDataAdmissionPolicy {
    pub(crate) const DEFAULT: Self = Self {
        allow_confidential: false,
    };

    pub(crate) const fn with_confidential_authorization(
        _attestation: HostConfidentialTrainingAttestation,
    ) -> Self {
        Self {
            allow_confidential: true,
        }
    }

    pub(crate) fn validate(
        self,
        data_class: DataClassification,
        index: usize,
    ) -> Result<(), TrainingDataGovernanceError> {
        match data_class {
            DataClassification::Public | DataClassification::Internal => Ok(()),
            DataClassification::Confidential if self.allow_confidential => Ok(()),
            DataClassification::Confidential => Err(
                TrainingDataGovernanceError::ConfidentialTrainingDataRequiresAuthorization {
                    index,
                },
            ),
            DataClassification::Secret => {
                Err(TrainingDataGovernanceError::SecretTrainingData { index })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_and_internal_are_allowed_without_attestation() {
        assert_eq!(
            TrainingDataAdmissionPolicy::DEFAULT.validate(DataClassification::Public, 0),
            Ok(())
        );
        assert_eq!(
            TrainingDataAdmissionPolicy::DEFAULT.validate(DataClassification::Internal, 1),
            Ok(())
        );
    }

    #[test]
    fn confidential_requires_explicit_host_authorization() {
        assert_eq!(
            TrainingDataAdmissionPolicy::DEFAULT.validate(DataClassification::Confidential, 7),
            Err(
                TrainingDataGovernanceError::ConfidentialTrainingDataRequiresAuthorization {
                    index: 7,
                }
            )
        );
        let authorized = TrainingDataAdmissionPolicy::with_confidential_authorization(
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            authorized.validate(DataClassification::Confidential, 7),
            Ok(())
        );
    }

    #[test]
    fn secret_is_forbidden_even_when_confidential_is_authorized() {
        let authorized = TrainingDataAdmissionPolicy::with_confidential_authorization(
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            authorized.validate(DataClassification::Secret, 11),
            Err(TrainingDataGovernanceError::SecretTrainingData { index: 11 })
        );
    }
}
