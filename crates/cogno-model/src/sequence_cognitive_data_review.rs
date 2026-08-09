//! Data-classified public entry point for the V4 cognitive Meta review.
//!
//! The underlying weakest-link review engine remains internal. Public callers
//! must attach a [`DataClassification`] to every review example. `Secret` is
//! rejected before corpus storage; `Confidential` requires an explicit host
//! attestation before storage. `Public` and `Internal` are admitted by default,
//! mirroring `cogno-core` §20.

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
/// Classification is validated before the example is stored. The canonical
/// anti-poisoning fingerprint remains based on example content and targets
/// rather than classification metadata, so the same payload cannot bypass
/// duplicate detection by being relabelled with a different class.
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

    /// Validate classification and then add one review example.
    ///
    /// `Secret` and unauthorized `Confidential` data fail before the inner
    /// corpus sees the example. Canonical duplicates return `Ok(false)` and do
    /// not advance the classification vector.
    pub fn add(
        &mut self,
        example: SequenceCognitiveReviewExample,
        data_class: DataClassification,
    ) -> Result<bool, SequenceCognitiveDataReviewError> {
        let index = self.data_classes.len();
        validate_data_classification(data_class, self.allow_confidential, index)?;
        if !self.inner.add(example) {
            return Ok(false);
        }
        self.data_classes.push(data_class);
        Ok(true)
    }

    /// Produce the deterministic train/validation/test split while preserving
    /// the exact index order used by the internal weakest-link review engine.
    #[must_use]
    pub fn split(
        &self,
        train_ratio: f32,
        validation_ratio: f32,
    ) -> (CorpusSplit, CorpusSplit, CorpusSplit) {
        self.inner.split(train_ratio, validation_ratio)
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
/// The corpus is revalidated defensively before delegating to the internal
/// review engine, therefore before tokenizer preflight, optimizer construction
/// or training.
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

fn validate_data_classification(
    data_class: DataClassification,
    allow_confidential: bool,
    index: usize,
) -> Result<(), SequenceCognitiveDataReviewError> {
    match data_class {
        DataClassification::Public | DataClassification::Internal => Ok(()),
        DataClassification::Confidential if allow_confidential => Ok(()),
        DataClassification::Confidential => Err(
            SequenceCognitiveDataReviewError::ConfidentialTrainingDataRequiresAuthorization {
                index,
            },
        ),
        DataClassification::Secret => {
            Err(SequenceCognitiveDataReviewError::SecretTrainingData { index })
        }
    }
}

fn validate_data_classifications(
    corpus: &SequenceCognitiveReviewCorpus,
) -> Result<(), SequenceCognitiveDataReviewError> {
    for (index, data_class) in corpus.data_classes.iter().copied().enumerate() {
        validate_data_classification(data_class, corpus.allow_confidential, index)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SequenceCognitiveExample, SplitKind};
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
        assert_eq!(corpus.add(example(1), DataClassification::Public), Ok(true));
        assert_eq!(
            corpus.add(example(2), DataClassification::Internal),
            Ok(true)
        );
        assert_eq!(corpus.len(), 2);
        assert_eq!(validate_data_classifications(&corpus), Ok(()));
    }

    #[test]
    fn secret_is_rejected_before_storage_even_with_confidential_authorization() {
        let mut corpus = SequenceCognitiveReviewCorpus::with_seed_and_confidential_authorization(
            2,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            corpus.add(example(1), DataClassification::Secret),
            Err(SequenceCognitiveDataReviewError::SecretTrainingData { index: 0 })
        );
        assert!(corpus.is_empty());
    }

    #[test]
    fn defensive_review_rejects_secret_before_inner_validation() {
        let mut inner = inner::SequenceCognitiveReviewCorpus::with_seed(5);
        assert!(inner.add(example(1)));
        let corpus = SequenceCognitiveReviewCorpus {
            inner,
            data_classes: vec![DataClassification::Secret],
            allow_confidential: false,
        };
        let train = CorpusSplit {
            kind: SplitKind::Train,
            indices: Vec::new(),
        };
        let validation = CorpusSplit {
            kind: SplitKind::Validation,
            indices: Vec::new(),
        };
        let test = CorpusSplit {
            kind: SplitKind::Test,
            indices: Vec::new(),
        };
        assert_eq!(
            review_sequence_cognitive_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                SequenceCognitiveMetaReviewConfig::default(),
                SequenceCognitiveMetaReviewPolicy::default(),
            ),
            Err(SequenceCognitiveDataReviewError::SecretTrainingData { index: 0 })
        );
    }

    #[test]
    fn confidential_requires_explicit_host_attestation_before_storage() {
        let mut denied = SequenceCognitiveReviewCorpus::with_seed(3);
        assert_eq!(
            denied.add(example(1), DataClassification::Confidential),
            Err(
                SequenceCognitiveDataReviewError::ConfidentialTrainingDataRequiresAuthorization {
                    index: 0,
                }
            )
        );
        assert!(denied.is_empty());

        let mut allowed = SequenceCognitiveReviewCorpus::with_seed_and_confidential_authorization(
            4,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            allowed.add(example(1), DataClassification::Confidential),
            Ok(true)
        );
        assert_eq!(allowed.len(), 1);
        assert_eq!(validate_data_classifications(&allowed), Ok(()));
    }
}
