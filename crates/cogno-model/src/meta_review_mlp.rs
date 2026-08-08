//! Held-out Meta review for the nonlinear MLP v2 candidate.
//!
//! The historical linear-v1 review remains untouched in [`crate::meta_review`].
//! This module mirrors its fail-closed split/provenance/regression policy for
//! the nonlinear MLP, seals the canonical v2 artifact, and exposes an explicit
//! versioned eligible-review wrapper. A model-side review still cannot attest
//! the host-owned scalar-engine or frozen-reference-policy preconditions.

use crate::artifact::EncodedNeuralArtifact;
use crate::meta_review::{
    EligibleMetaModelReview, HeldOutMetrics, MetaModelEvidence, MetaNeuralReviewPolicy,
    MetaPromotionAuthority, MetaPromotionBlocker, MetaPromotionDisposition,
    MAX_META_REVIEW_EXAMPLES,
};
use crate::mlp::{MlpNeuralConfig, MlpNeuralModel, MlpNeuralTrainer};
use crate::mlp_artifact::{
    encode_mlp_neural_artifact, load_mlp_neural_artifact, MlpNeuralArtifactError,
};
use crate::neural::NeuralModelError;
use crate::training::{Corpus, CorpusSplit, Label, SplitKind, ToyTrainer, TrainedModel};
use cogno_core::{ArchitectureId, EvidenceOrigin, InputOrigin};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MlpMetaReviewSeal {
    eligible: bool,
    candidate_digest: Option<[u8; 32]>,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

/// A consuming, non-forgeable proof that the nonlinear MLP v2 review passed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibleMlpMetaModelReview {
    artifact: EncodedNeuralArtifact,
    evidence: MetaModelEvidence,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

impl EligibleMlpMetaModelReview {
    #[must_use]
    pub const fn evidence(&self) -> MetaModelEvidence {
        self.evidence
    }

    #[must_use]
    pub const fn validation_accuracy_bps(&self) -> u16 {
        self.validation_accuracy_bps
    }

    #[must_use]
    pub const fn test_accuracy_bps(&self) -> u16 {
        self.test_accuracy_bps
    }

    #[must_use]
    pub fn artifact(&self) -> &EncodedNeuralArtifact {
        &self.artifact
    }
}

/// Explicit architecture tag for a sealed model-side Meta review.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EligibleVersionedMetaModelReview {
    LinearV1(EligibleMetaModelReview),
    MlpV2(EligibleMlpMetaModelReview),
}

impl EligibleVersionedMetaModelReview {
    #[must_use]
    pub fn artifact(&self) -> &EncodedNeuralArtifact {
        match self {
            Self::LinearV1(review) => review.artifact(),
            Self::MlpV2(review) => review.artifact(),
        }
    }

    #[must_use]
    pub const fn evidence(&self) -> MetaModelEvidence {
        match self {
            Self::LinearV1(review) => review.evidence(),
            Self::MlpV2(review) => review.evidence(),
        }
    }

    #[must_use]
    pub const fn validation_accuracy_bps(&self) -> u16 {
        match self {
            Self::LinearV1(review) => review.validation_accuracy_bps(),
            Self::MlpV2(review) => review.validation_accuracy_bps(),
        }
    }

    #[must_use]
    pub const fn test_accuracy_bps(&self) -> u16 {
        match self {
            Self::LinearV1(review) => review.test_accuracy_bps(),
            Self::MlpV2(review) => review.test_accuracy_bps(),
        }
    }

    #[must_use]
    pub const fn architecture_id(&self) -> ArchitectureId {
        self.artifact().manifest.architecture_id
    }
}

impl From<EligibleMetaModelReview> for EligibleVersionedMetaModelReview {
    fn from(review: EligibleMetaModelReview) -> Self {
        Self::LinearV1(review)
    }
}

impl From<EligibleMlpMetaModelReview> for EligibleVersionedMetaModelReview {
    fn from(review: EligibleMlpMetaModelReview) -> Self {
        Self::MlpV2(review)
    }
}

/// Failure while consuming an MLP v2 review into its stronger eligibility seal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpMetaEligibilityError {
    ReviewHeld,
    IncompleteModelEvidence,
    CandidateMissing,
    CandidateDigestMismatch,
    CandidateArtifactInvalid(MlpNeuralArtifactError),
}

/// Complete non-authoritative MLP v2 held-out review result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlpMetaNeuralReviewReport {
    pub train_examples: usize,
    pub validation_neural: HeldOutMetrics,
    pub validation_baseline: HeldOutMetrics,
    pub test_neural: HeldOutMetrics,
    pub test_baseline: HeldOutMetrics,
    pub model_evidence: MetaModelEvidence,
    pub blockers: Vec<MetaPromotionBlocker>,
    pub disposition: MetaPromotionDisposition,
    pub authority: MetaPromotionAuthority,
    pub candidate_artifact: Option<EncodedNeuralArtifact>,
    seal: MlpMetaReviewSeal,
}

impl MlpMetaNeuralReviewReport {
    /// Consume an eligible MLP report and revalidate its exact v2 artifact.
    pub fn into_eligible(self) -> Result<EligibleMlpMetaModelReview, MlpMetaEligibilityError> {
        if !self.seal.eligible
            || self.disposition != MetaPromotionDisposition::EligibleForControlledPromotionReview
            || self.authority != MetaPromotionAuthority::ReviewOnly
            || !self.blockers.is_empty()
        {
            return Err(MlpMetaEligibilityError::ReviewHeld);
        }
        if !evidence_complete(self.model_evidence) {
            return Err(MlpMetaEligibilityError::IncompleteModelEvidence);
        }
        let artifact = self
            .candidate_artifact
            .ok_or(MlpMetaEligibilityError::CandidateMissing)?;
        let expected_digest = self
            .seal
            .candidate_digest
            .ok_or(MlpMetaEligibilityError::CandidateMissing)?;
        if artifact.manifest.weights_hash != expected_digest {
            return Err(MlpMetaEligibilityError::CandidateDigestMismatch);
        }
        load_mlp_neural_artifact(&artifact.manifest, &artifact.bytes)
            .map_err(MlpMetaEligibilityError::CandidateArtifactInvalid)?;
        Ok(EligibleMlpMetaModelReview {
            artifact,
            evidence: self.model_evidence,
            validation_accuracy_bps: self.seal.validation_accuracy_bps,
            test_accuracy_bps: self.seal.test_accuracy_bps,
        })
    }
}

/// Structural/provenance failures stop MLP review rather than becoming soft blockers.
#[derive(Clone, Debug, PartialEq)]
pub enum MlpMetaNeuralReviewError {
    InvalidPolicy,
    WrongSplitKind {
        expected: SplitKind,
        actual: SplitKind,
    },
    EmptySplit(SplitKind),
    TooManyExamples,
    InvalidSplitIndex {
        index: usize,
        corpus_len: usize,
    },
    SplitOverlap(usize),
    HeldOutLabelAbsentFromTraining(Label),
    UntrustedHeldOutProvenance {
        index: usize,
        input_origin: InputOrigin,
        evidence_origin: EvidenceOrigin,
    },
    ArithmeticOverflow,
    Neural(NeuralModelError),
    Artifact(MlpNeuralArtifactError),
    InvalidLogProbability,
}

impl From<NeuralModelError> for MlpMetaNeuralReviewError {
    fn from(error: NeuralModelError) -> Self {
        Self::Neural(error)
    }
}

impl From<MlpNeuralArtifactError> for MlpMetaNeuralReviewError {
    fn from(error: MlpNeuralArtifactError) -> Self {
        Self::Artifact(error)
    }
}

/// Train and review the bounded nonlinear MLP against the frozen integer baseline.
pub fn review_mlp_model_for_meta(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    mlp_config: MlpNeuralConfig,
    policy: MetaNeuralReviewPolicy,
) -> Result<MlpMetaNeuralReviewReport, MlpMetaNeuralReviewError> {
    validate_policy(policy)?;
    validate_splits(corpus, train, validation, test)?;

    let trainer = MlpNeuralTrainer::try_new(mlp_config)?;
    let (candidate, _) = trainer.train(corpus, train)?;
    verify_log_probabilities(&candidate, corpus, validation)?;
    verify_log_probabilities(&candidate, corpus, test)?;

    let baseline_trainer = ToyTrainer::new(mlp_config.input_dim, mlp_config.epochs);
    let (baseline, _) = baseline_trainer.train(corpus, train);

    let validation_neural = evaluate_mlp(&candidate, corpus, validation)?;
    let validation_baseline = evaluate_baseline(&baseline, corpus, validation)?;
    let test_neural = evaluate_mlp(&candidate, corpus, test)?;
    let test_baseline = evaluate_baseline(&baseline, corpus, test)?;

    let mut blockers = Vec::new();
    if validation_neural.accuracy_bps < policy.minimum_validation_accuracy_bps {
        blockers.push(MetaPromotionBlocker::ValidationAccuracyBelowThreshold);
    }
    if test_neural.accuracy_bps < policy.minimum_test_accuracy_bps {
        blockers.push(MetaPromotionBlocker::TestAccuracyBelowThreshold);
    }
    if regressed_beyond(
        validation_neural.accuracy_bps,
        validation_baseline.accuracy_bps,
        policy.maximum_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::ValidationRegression);
    }
    if regressed_beyond(
        test_neural.accuracy_bps,
        test_baseline.accuracy_bps,
        policy.maximum_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::TestRegression);
    }
    blockers.sort_unstable();
    blockers.dedup();

    let eligible = blockers.is_empty();
    let candidate_artifact = if eligible {
        Some(encode_mlp_neural_artifact(
            &candidate,
            policy.artifact_max_context_tokens,
        )?)
    } else {
        None
    };
    let candidate_digest = candidate_artifact
        .as_ref()
        .map(|artifact| artifact.manifest.weights_hash);
    let model_evidence = MetaModelEvidence {
        log_probabilities_available: true,
        backend_differentiable: true,
        held_out_tests_in_place: true,
        anti_poisoning_working: true,
    };

    Ok(MlpMetaNeuralReviewReport {
        train_examples: train.indices.len(),
        validation_neural,
        validation_baseline,
        test_neural,
        test_baseline,
        model_evidence,
        blockers,
        disposition: if eligible {
            MetaPromotionDisposition::EligibleForControlledPromotionReview
        } else {
            MetaPromotionDisposition::Hold
        },
        authority: MetaPromotionAuthority::ReviewOnly,
        candidate_artifact,
        seal: MlpMetaReviewSeal {
            eligible,
            candidate_digest,
            validation_accuracy_bps: validation_neural.accuracy_bps,
            test_accuracy_bps: test_neural.accuracy_bps,
        },
    })
}

const fn evidence_complete(evidence: MetaModelEvidence) -> bool {
    evidence.log_probabilities_available
        && evidence.backend_differentiable
        && evidence.held_out_tests_in_place
        && evidence.anti_poisoning_working
}

fn validate_policy(policy: MetaNeuralReviewPolicy) -> Result<(), MlpMetaNeuralReviewError> {
    if policy.minimum_validation_accuracy_bps > 10_000
        || policy.minimum_test_accuracy_bps > 10_000
        || policy.maximum_regression_bps > 10_000
        || policy.artifact_max_context_tokens == 0
    {
        return Err(MlpMetaNeuralReviewError::InvalidPolicy);
    }
    Ok(())
}

fn validate_splits(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
) -> Result<(), MlpMetaNeuralReviewError> {
    for (split, expected) in [
        (train, SplitKind::Train),
        (validation, SplitKind::Validation),
        (test, SplitKind::Test),
    ] {
        if split.kind != expected {
            return Err(MlpMetaNeuralReviewError::WrongSplitKind {
                expected,
                actual: split.kind,
            });
        }
        if split.indices.is_empty() {
            return Err(MlpMetaNeuralReviewError::EmptySplit(expected));
        }
    }

    let total = train
        .indices
        .len()
        .checked_add(validation.indices.len())
        .and_then(|value| value.checked_add(test.indices.len()))
        .ok_or(MlpMetaNeuralReviewError::ArithmeticOverflow)?;
    if total > MAX_META_REVIEW_EXAMPLES {
        return Err(MlpMetaNeuralReviewError::TooManyExamples);
    }

    let mut seen = BTreeSet::new();
    for &index in train
        .indices
        .iter()
        .chain(validation.indices.iter())
        .chain(test.indices.iter())
    {
        if index >= corpus.examples.len() {
            return Err(MlpMetaNeuralReviewError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            });
        }
        if !seen.insert(index) {
            return Err(MlpMetaNeuralReviewError::SplitOverlap(index));
        }
        let example = &corpus.examples[index];
        if !trusted_review_provenance(
            example.provenance.origin,
            example.provenance.evidence_origin,
        ) {
            return Err(MlpMetaNeuralReviewError::UntrustedHeldOutProvenance {
                index,
                input_origin: example.provenance.origin,
                evidence_origin: example.provenance.evidence_origin,
            });
        }
    }

    let train_labels: BTreeSet<Label> = train
        .indices
        .iter()
        .map(|&index| corpus.examples[index].label)
        .collect();
    for &index in validation.indices.iter().chain(test.indices.iter()) {
        let example = &corpus.examples[index];
        if !train_labels.contains(&example.label) {
            return Err(MlpMetaNeuralReviewError::HeldOutLabelAbsentFromTraining(
                example.label,
            ));
        }
    }
    Ok(())
}

const fn trusted_review_provenance(input: InputOrigin, evidence: EvidenceOrigin) -> bool {
    !matches!(input, InputOrigin::ModelOutput)
        && !matches!(evidence, EvidenceOrigin::ModelInference)
}

fn evaluate_mlp(
    model: &MlpNeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, MlpMetaNeuralReviewError> {
    let mut correct = 0usize;
    for &index in &split.indices {
        let example = &corpus.examples[index];
        if model.classify(&example.payload)? == example.label {
            correct = correct.saturating_add(1);
        }
    }
    metrics(correct, split.indices.len())
}

fn evaluate_baseline(
    model: &TrainedModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, MlpMetaNeuralReviewError> {
    let correct = split
        .indices
        .iter()
        .filter(|&&index| {
            model.classify(&corpus.examples[index].payload) == corpus.examples[index].label
        })
        .count();
    metrics(correct, split.indices.len())
}

fn metrics(correct: usize, examples: usize) -> Result<HeldOutMetrics, MlpMetaNeuralReviewError> {
    let scaled = correct
        .checked_mul(10_000)
        .ok_or(MlpMetaNeuralReviewError::ArithmeticOverflow)?
        .checked_div(examples)
        .ok_or(MlpMetaNeuralReviewError::ArithmeticOverflow)?;
    let accuracy_bps =
        u16::try_from(scaled).map_err(|_| MlpMetaNeuralReviewError::ArithmeticOverflow)?;
    Ok(HeldOutMetrics {
        examples,
        correct,
        accuracy_bps,
    })
}

fn regressed_beyond(candidate: u16, baseline: u16, tolerance: u16) -> bool {
    candidate < baseline.saturating_sub(tolerance)
}

fn verify_log_probabilities(
    model: &MlpNeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<(), MlpMetaNeuralReviewError> {
    for &index in &split.indices {
        let payload = &corpus.examples[index].payload;
        for label in 0..model.num_labels() {
            let value = model.log_probability(label, payload)?;
            if !value.is_finite() {
                return Err(MlpMetaNeuralReviewError::InvalidLogProbability);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp_artifact::MLP_NEURAL_ARCHITECTURE_ID;
    use crate::training::LabeledExample;

    fn corpus_and_splits() -> (Corpus, CorpusSplit, CorpusSplit, CorpusSplit) {
        let mut corpus = Corpus::with_seed(17);
        for (label, payload) in [
            (Label(0), b"alpha train one".as_slice()),
            (Label(1), b"omega train one"),
            (Label(0), b"alpha train two"),
            (Label(1), b"omega train two"),
            (Label(0), b"alpha validation"),
            (Label(1), b"omega validation"),
            (Label(0), b"alpha test"),
            (Label(1), b"omega test"),
        ] {
            assert!(corpus.add(LabeledExample::new(
                label,
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        (
            corpus,
            CorpusSplit {
                kind: SplitKind::Train,
                indices: vec![0, 1, 2, 3],
            },
            CorpusSplit {
                kind: SplitKind::Validation,
                indices: vec![4, 5],
            },
            CorpusSplit {
                kind: SplitKind::Test,
                indices: vec![6, 7],
            },
        )
    }

    fn config() -> MlpNeuralConfig {
        MlpNeuralConfig {
            input_dim: 64,
            hidden_dim: 16,
            epochs: 64,
            learning_rate: 0.02,
            max_payload_bytes: 128,
            initialization_seed: 29,
        }
    }

    fn permissive_policy() -> MetaNeuralReviewPolicy {
        MetaNeuralReviewPolicy {
            minimum_validation_accuracy_bps: 0,
            minimum_test_accuracy_bps: 0,
            maximum_regression_bps: 10_000,
            artifact_max_context_tokens: 2_048,
        }
    }

    #[test]
    fn eligible_mlp_review_seals_exact_v2_artifact() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let report = review_mlp_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("review");
        assert_eq!(
            report.disposition,
            MetaPromotionDisposition::EligibleForControlledPromotionReview
        );
        let eligible = report.into_eligible().expect("eligible MLP review");
        assert_eq!(
            eligible.artifact().manifest.architecture_id,
            MLP_NEURAL_ARCHITECTURE_ID
        );
        let versioned = EligibleVersionedMetaModelReview::from(eligible);
        assert_eq!(versioned.architecture_id(), MLP_NEURAL_ARCHITECTURE_ID);
        assert!(!versioned.evidence().as_partial_preconditions().satisfied());
    }

    #[test]
    fn post_review_mlp_artifact_replacement_breaks_private_digest_seal() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let mut report = review_mlp_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("review");
        let artifact = report.candidate_artifact.as_mut().expect("candidate");
        artifact.manifest.weights_hash[0] ^= 1;
        assert_eq!(
            report.into_eligible(),
            Err(MlpMetaEligibilityError::CandidateDigestMismatch)
        );
    }

    #[test]
    fn model_origin_mlp_review_evidence_is_rejected_before_training() {
        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].provenance.origin = InputOrigin::ModelOutput;
        assert!(matches!(
            review_mlp_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                permissive_policy(),
            ),
            Err(MlpMetaNeuralReviewError::UntrustedHeldOutProvenance { .. })
        ));
    }
}
