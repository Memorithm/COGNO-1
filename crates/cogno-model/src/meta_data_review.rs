//! Data-governed public entry point for the historical V1 Meta review.
//!
//! The original held-out review engine remains crate-private. Public callers
//! reach it only after the shared §20 training-corpus invariant is revalidated.

use crate::meta_review_engine as inner;
use crate::neural::NeuralConfig;
use crate::training::{Corpus, CorpusSplit};
use crate::training_data_policy::{validate_training_corpus, TrainingDataGovernanceError};

pub use inner::{
    EligibleMetaModelReview, HeldOutMetrics, MetaEligibilityError, MetaModelEvidence,
    MetaNeuralReviewError, MetaNeuralReviewPolicy, MetaNeuralReviewReport, MetaPromotionAuthority,
    MetaPromotionBlocker, MetaPromotionDisposition, DEFAULT_META_MAX_REGRESSION_BPS,
    DEFAULT_META_MIN_ACCURACY_BPS, MAX_META_REVIEW_EXAMPLES,
};

/// Fail-closed error at the public V1 review boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaDataReviewError {
    /// Stored training-data classification no longer satisfies §20.
    TrainingData(TrainingDataGovernanceError),
    /// Existing V1 structural, provenance, numerical or artifact review error.
    Review(MetaNeuralReviewError),
}

impl From<TrainingDataGovernanceError> for MetaDataReviewError {
    fn from(error: TrainingDataGovernanceError) -> Self {
        Self::TrainingData(error)
    }
}

impl From<MetaNeuralReviewError> for MetaDataReviewError {
    fn from(error: MetaNeuralReviewError) -> Self {
        Self::Review(error)
    }
}

/// Revalidate §20 before any V1 review validation, allocation or training.
pub fn review_neural_model_for_meta(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: NeuralConfig,
    policy: MetaNeuralReviewPolicy,
) -> Result<MetaNeuralReviewReport, MetaDataReviewError> {
    validate_training_corpus(corpus)?;
    Ok(inner::review_neural_model_for_meta(
        corpus, train, validation, test, config, policy,
    )?)
}
