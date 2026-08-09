//! Data-classified public entry point for the V4 cognitive Meta review.
//!
//! The underlying weakest-link review engine remains internal. Public callers
//! must attach a [`DataClassification`] to every review example before any
//! training can occur. `Secret` is unconditionally forbidden; `Confidential`
//! requires an explicit host attestation. `Public` and `Internal` are admitted
//! by default, mirroring `cogno-core` §20.

use crate::sequence_cognitive_meta_review as inner;
use crate::training::CorpusSplit;
use cogno_core::DataClassification;

pub use inner::{
    EligibleSequenceCognitiveMetaModelReview, SequenceCognitiveHeldOutMetrics,
    SequenceCognitiveMetaEligibilityError, SequenceCognitiveMetaReviewConfig,
    SequenceCognitiveMetaReviewError, SequenceCognitiveMetaReviewPolicy,
    SequenceCognitiveMetaReviewReport, SequenceCognitiveReviewExample,
};

/// Explicit host authority required before `Confidential` review data may be
/// supplied to the neural training/review path. This token never authorizes
/// `Secret`, which is forbidden unconditionally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostConfidentialTrainingAttestation {
    _private: (),
}

impl HostConfidentialTrainingAttestation {
    #[must_use]
    pub const fn authorize_confidential_training_data() -> Self {
        Self { _private: () }
    }
}

/// Public V4 review corpus with one explicit data classification per example.
///
/// The canonical anti-poisoning fingerprint remains based on example content
/// and targets rather than classification metadata, so the same payload cannot
/// bypass duplicate detection by being relabelled with a different class.
#[derive(Clone, Debug)]
pub struct SequenceCognitiveReviewCorpus {
    inner: inner::SequenceCognitiveReviewCorpus,
    data_classes: Vec<DataClassification>,
    allow_confidential: bool,
}

impl SequenceCognitiveReviewCorpus {
    /// Construct a corpus admitting only `Public` and `Internal` examples.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            inner: inner::SequenceCognitiveReviewCorpus::with_seed(seed),
            data_classes: Vec::new(),
            allow_confidential: false,
        }
    }

    /// Construct a corpus for which the host explicitly authorizes
    /// `Confidential` examples. `Secret` remains forbidden.
    #[must_use]
    pub fn with_seed_and_confidential_authorization(
        seed: u64,
        _attestation: HostConfidentialTrainingAttestation,
    ) -> Self {
        Self {
            inner: inner::SequenceCognitiveReviewCorpus::with_seed(seed),
            data_classes: Vec::new(),
            allow_confidential: true,
        }
    }

    /// Add one review example together with its host-owned classification.
    /// Duplicate canonical examples are rejected by the underlying corpus and
    /// do not advance the classification vector.
    pub fn add(
        &mut self,
        example: SequenceCognitiveReviewExample,
        data_class: DataClassification,
    ) -> bool {
        if !self.inner.add(example) {
            return false;
        }
        self.data_classes.push(data_class);
        true
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data_classes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data_classes.is_empty()
    }
}

/// Fail-closed errors added by the public data-classification boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceCognitiveDataReviewError {
    /// `Secret` may never enter prompts, traces, reports or training sets.
    SecretTrainingData { index: usize },
    /// `Confidential` requires the explicit host attestation constructor.
    ConfidentialTrainingDataRequiresAuthorization { index: usize },
    /// Existing weakest-link review validation/training error.
    Review(SequenceCognitiveMetaReviewError),
}

impl From<SequenceCognitiveMetaReviewError> for SequenceCognitiveDataReviewError {
    fn from(error: SequenceCognitiveMetaReviewError) -> Self {
        Self::Review(error)
    }
}

/// Run the V4 weakest-link Meta review only after data governance has passed.
///
/// Classification is checked before delegating to the internal review engine,
/// therefore before tokenizer preflight, optimizer construction or training.
pub fn review_sequence_cognitive_model_for_meta(
    corpus: &SequenceCognitiveReviewCorpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: SequenceCognitiveMetaReviewConfig,
    policy: SequenceCognitiveMetaReviewPolicy,
) -> Result<SequenceCognitiveMetaReviewReport, SequenceCognitiveDataReviewError> {
    validate_data_classifications(corpus)?;
    Ok(inner::review_sequence_cognitive_model_for_meta(
        &corpus.inner,
        train,
        validation,
        test,
        config,
        policy,
    )?)
}

fn validate_data_classifications(
    corpus: &SequenceCognitiveReviewCorpus,
) -> Result<(), SequenceCognitiveDataReviewError> {
    for (index, data_class) in corpus.data_classes.iter().copied().enumerate() {
        match data_class {
            DataClassification::Public | DataClassification::Internal => {}
            DataClassification::Confidential if corpus.allow_confidential => {}
            DataClassification::Confidential => {
                return Err(
                    SequenceCognitiveDataReviewError::ConfidentialTrainingDataRequiresAuthorization {
                        index,
                    },
                );
            }
            DataClassification::Secret => {
                return Err(SequenceCognitiveDataReviewError::SecretTrainingData { index });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceCognitiveExample;
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn example(index: u8) -> SequenceCognitiveReviewExample {
        SequenceCognitiveReviewExample::new(
            SequenceCognitiveExample {
                classification_payload: vec![index, 1],
                classification_target: 0,
                preferred: vec![index, 2],
                dispreferred: vec![index, 3],
                symbolic_payload: vec![index, 4],
                rule_satisfied: vec![true],
                contradiction_left: vec![index, 5],
                contradiction_right: vec![index, 6],
                contradicts: false,
                retrieval_query: vec![index, 7],
                retrieval_candidates: vec![vec![index, 8], vec![index, 9]],
                retrieval_positive_idx: 0,
            },
            InputOrigin::ExplicitUserInstruction,
            EvidenceOrigin::ExplicitUserApproval,
        )
    }

    #[test]
    fn public_and_internal_are_admitted_by_default() {
        let mut corpus = SequenceCognitiveReviewCorpus::with_seed(1);
        assert!(corpus.add(example(1), DataClassification::Public));
        assert!(corpus.add(example(2), DataClassification::Internal));
        assert_eq!(validate_data_classifications(&corpus), Ok(()));
    }

    #[test]
    fn secret_is_forbidden_even_with_confidential_authorization() {
        let mut corpus = SequenceCognitiveReviewCorpus::with_seed_and_confidential_authorization(
            2,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert!(corpus.add(example(1), DataClassification::Secret));
        assert_eq!(
            validate_data_classifications(&corpus),
            Err(SequenceCognitiveDataReviewError::SecretTrainingData { index: 0 })
        );
    }

    #[test]
    fn confidential_requires_explicit_host_attestation() {
        let mut denied = SequenceCognitiveReviewCorpus::with_seed(3);
        assert!(denied.add(example(1), DataClassification::Confidential));
        assert_eq!(
            validate_data_classifications(&denied),
            Err(
                SequenceCognitiveDataReviewError::ConfidentialTrainingDataRequiresAuthorization {
                    index: 0,
                }
            )
        );

        let mut allowed = SequenceCognitiveReviewCorpus::with_seed_and_confidential_authorization(
            4,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert!(allowed.add(example(1), DataClassification::Confidential));
        assert_eq!(validate_data_classifications(&allowed), Ok(()));
    }
}
