//! Data-governed public entry point for the byte-tokenized V3 Meta review.
//!
//! The original sequence held-out review engine remains crate-private. Public
//! callers reach it only after the shared §20 training-corpus invariant is
//! revalidated.

use crate::sequence_meta_review_engine as inner;
use crate::training::{Corpus, CorpusSplit};
use crate::training_data_policy::{validate_training_corpus, TrainingDataGovernanceError};

pub use inner::{
    EligibleSequenceMetaModelReview, SequenceMetaEligibilityError, SequenceMetaReviewConfig,
    SequenceMetaReviewError, SequenceMetaReviewPolicy, SequenceMetaReviewReport,
};

/// Fail-closed error at the public V3 review boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceDataReviewError {
    /// Stored training-data classification no longer satisfies §20.
    TrainingData(TrainingDataGovernanceError),
    /// The same input appears in two partitions, irrespective of target label.
    PayloadOverlap { first: usize, second: usize },
    /// Existing V3 structural, provenance, numerical or artifact review error.
    Review(SequenceMetaReviewError),
}

impl From<TrainingDataGovernanceError> for SequenceDataReviewError {
    fn from(error: TrainingDataGovernanceError) -> Self {
        Self::TrainingData(error)
    }
}

impl From<SequenceMetaReviewError> for SequenceDataReviewError {
    fn from(error: SequenceMetaReviewError) -> Self {
        Self::Review(error)
    }
}

/// Revalidate §20 before any V3 split validation, tokenizer, optimizer or
/// training state is created.
pub fn review_sequence_model_for_meta(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: SequenceMetaReviewConfig,
    policy: SequenceMetaReviewPolicy,
) -> Result<SequenceMetaReviewReport, SequenceDataReviewError> {
    validate_training_corpus(corpus)?;
    validate_payload_separation(corpus, train, validation, test)?;
    Ok(inner::review_sequence_model_for_meta(
        corpus, train, validation, test, config, policy,
    )?)
}

// Enforce input isolation, not merely disjoint row IDs. The canonical example
// fingerprint includes the target, so relabeling alone creates a distinct row.
fn validate_payload_separation(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
) -> Result<(), SequenceDataReviewError> {
    let total = train
        .indices
        .len()
        .checked_add(validation.indices.len())
        .and_then(|n| n.checked_add(test.indices.len()))
        .ok_or(SequenceMetaReviewError::ArithmeticOverflow)?;
    if total > crate::MAX_META_REVIEW_EXAMPLES {
        return Err(SequenceMetaReviewError::TooManyExamples.into());
    }
    let mut seen = std::collections::BTreeMap::new();
    for (partition, split) in [train, validation, test].into_iter().enumerate() {
        for &index in &split.indices {
            let example =
                corpus
                    .examples()
                    .get(index)
                    .ok_or(SequenceMetaReviewError::InvalidSplitIndex {
                        index,
                        corpus_len: corpus.len(),
                    })?;
            if let Some(&(first, original_partition)) = seen.get(example.payload.as_slice()) {
                if original_partition != partition {
                    return Err(SequenceDataReviewError::PayloadOverlap {
                        first,
                        second: index,
                    });
                }
            } else {
                seen.insert(example.payload.as_slice(), (index, partition));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Label, LabeledExample, SplitKind};
    use cogno_core::{DataClassification, EvidenceOrigin, InputOrigin};

    fn fixture() -> (Corpus, CorpusSplit, CorpusSplit, CorpusSplit) {
        let mut corpus = Corpus::with_seed(1);
        for (label, payload) in [(0, b"same".as_slice()), (1, b"same"), (0, b"other")] {
            corpus
                .add_classified(
                    LabeledExample::new(
                        Label(label),
                        payload.to_vec(),
                        InputOrigin::ExplicitUserInstruction,
                        EvidenceOrigin::ExplicitUserApproval,
                    ),
                    DataClassification::Public,
                )
                .unwrap();
        }
        (
            corpus,
            CorpusSplit {
                kind: SplitKind::Train,
                indices: vec![0],
            },
            CorpusSplit {
                kind: SplitKind::Validation,
                indices: vec![1],
            },
            CorpusSplit {
                kind: SplitKind::Test,
                indices: vec![2],
            },
        )
    }

    #[test]
    fn relabeling_does_not_hide_input_leakage_before_training() {
        let (corpus, train, validation, test) = fixture();
        assert_eq!(
            review_sequence_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                SequenceMetaReviewConfig::default(),
                SequenceMetaReviewPolicy::default()
            ),
            Err(SequenceDataReviewError::PayloadOverlap {
                first: 0,
                second: 1
            })
        );
        assert_eq!(
            validate_payload_separation(&corpus, &test, &train, &validation),
            Err(SequenceDataReviewError::PayloadOverlap {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn partition_preflight_rejects_indices_without_panicking() {
        let (corpus, train, mut validation, test) = fixture();
        validation.indices = vec![usize::MAX];
        assert!(matches!(
            validate_payload_separation(&corpus, &train, &validation, &test),
            Err(SequenceDataReviewError::Review(
                SequenceMetaReviewError::InvalidSplitIndex { .. }
            ))
        ));
        validation.indices.clear();
        assert!(validate_payload_separation(&corpus, &train, &validation, &test).is_ok());
    }
}
