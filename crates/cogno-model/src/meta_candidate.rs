//! Architecture-neutral surface over COGNO-minted Meta review proofs.
//!
//! The public trait is sealed: external crates can consume reviewed candidates
//! generically but cannot implement the trait for arbitrary artifacts. This
//! preserves the non-forgeability of both historical v1 and sequence-v3 review
//! proofs while allowing runtime persistence/activation to share one interface.

use crate::artifact::EncodedNeuralArtifact;
use crate::meta_review::{EligibleMetaModelReview, MetaModelEvidence};
use crate::sequence_meta_review::EligibleSequenceMetaModelReview;

mod sealed {
    pub trait Sealed {}
}

/// Read-only facts exposed by a model review proof minted inside `cogno-model`.
pub trait MetaReviewedCandidate: sealed::Sealed {
    fn artifact(&self) -> &EncodedNeuralArtifact;
    fn evidence(&self) -> MetaModelEvidence;
    fn validation_accuracy_bps(&self) -> u16;
    fn test_accuracy_bps(&self) -> u16;
}

impl sealed::Sealed for EligibleMetaModelReview {}

impl MetaReviewedCandidate for EligibleMetaModelReview {
    fn artifact(&self) -> &EncodedNeuralArtifact {
        EligibleMetaModelReview::artifact(self)
    }

    fn evidence(&self) -> MetaModelEvidence {
        EligibleMetaModelReview::evidence(self)
    }

    fn validation_accuracy_bps(&self) -> u16 {
        EligibleMetaModelReview::validation_accuracy_bps(self)
    }

    fn test_accuracy_bps(&self) -> u16 {
        EligibleMetaModelReview::test_accuracy_bps(self)
    }
}

impl sealed::Sealed for EligibleSequenceMetaModelReview {}

impl MetaReviewedCandidate for EligibleSequenceMetaModelReview {
    fn artifact(&self) -> &EncodedNeuralArtifact {
        EligibleSequenceMetaModelReview::artifact(self)
    }

    fn evidence(&self) -> MetaModelEvidence {
        EligibleSequenceMetaModelReview::evidence(self)
    }

    fn validation_accuracy_bps(&self) -> u16 {
        EligibleSequenceMetaModelReview::validation_accuracy_bps(self)
    }

    fn test_accuracy_bps(&self) -> u16 {
        EligibleSequenceMetaModelReview::test_accuracy_bps(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_candidate_surface<T: MetaReviewedCandidate>() {}

    #[test]
    fn both_cogno_review_proofs_implement_the_sealed_surface() {
        assert_candidate_surface::<EligibleMetaModelReview>();
        assert_candidate_surface::<EligibleSequenceMetaModelReview>();
    }
}
