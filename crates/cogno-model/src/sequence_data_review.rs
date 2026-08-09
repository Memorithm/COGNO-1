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
    Ok(inner::review_sequence_model_for_meta(
        corpus, train, validation, test, config, policy,
    )?)
}
