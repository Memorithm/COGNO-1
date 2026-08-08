//! Held-out review gate for the differentiable COGNO model.
//!
//! This module proves the four model-side Phase-4 preconditions that COGNO can
//! establish itself: differentiability, finite log-probabilities, held-out
//! evaluation, and anti-poisoning provenance. It deliberately leaves the two
//! host-owned preconditions (`scalar_engine_validated` and
//! `reference_policy_frozen`) false, so a model review can never activate the
//! meta-objective by itself.

use crate::artifact::{
    encode_neural_artifact, load_neural_artifact, EncodedNeuralArtifact, NeuralArtifactError,
};
use crate::neural::{NeuralConfig, NeuralModel, NeuralModelError, NeuralTrainer};
use crate::training::{Corpus, CorpusSplit, Label, SplitKind, ToyTrainer, TrainedModel};
use cogno_core::{EvidenceOrigin, InputOrigin, MetaPreconditions};
use std::collections::BTreeSet;

/// Maximum number of examples inspected across all splits by one review.
pub const MAX_META_REVIEW_EXAMPLES: usize = 65_536;
/// Default minimum held-out accuracy required on validation and test splits.
pub const DEFAULT_META_MIN_ACCURACY_BPS: u16 = 7_000;
/// Default tolerated regression against the frozen integer baseline.
pub const DEFAULT_META_MAX_REGRESSION_BPS: u16 = 0;

/// Deterministic policy for held-out promotion review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetaNeuralReviewPolicy {
    pub minimum_validation_accuracy_bps: u16,
    pub minimum_test_accuracy_bps: u16,
    pub maximum_regression_bps: u16,
    pub artifact_max_context_tokens: u32,
}

impl Default for MetaNeuralReviewPolicy {
    fn default() -> Self {
        Self {
            minimum_validation_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            minimum_test_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            maximum_regression_bps: DEFAULT_META_MAX_REGRESSION_BPS,
            artifact_max_context_tokens: 2_048,
        }
    }
}

/// The review has no installation or policy authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaPromotionAuthority {
    ReviewOnly,
}

/// Result of the held-out gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaPromotionDisposition {
    Hold,
    EligibleForControlledPromotionReview,
}

/// Explicit reasons an otherwise valid review was held.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetaPromotionBlocker {
    ValidationAccuracyBelowThreshold,
    TestAccuracyBelowThreshold,
    ValidationRegression,
    TestRegression,
}

/// Model-side evidence for the Phase-4 meta preconditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetaModelEvidence {
    pub log_probabilities_available: bool,
    pub backend_differentiable: bool,
    pub held_out_tests_in_place: bool,
    pub anti_poisoning_working: bool,
}

impl MetaModelEvidence {
    /// Convert only model-owned facts into the core precondition type.
    ///
    /// The host-owned scalar-engine and frozen-reference-policy attestations
    /// intentionally remain false.
    #[must_use]
    pub const fn as_partial_preconditions(self) -> MetaPreconditions {
        MetaPreconditions {
            scalar_engine_validated: false,
            reference_policy_frozen: false,
            log_probabilities_available: self.log_probabilities_available,
            backend_differentiable: self.backend_differentiable,
            held_out_tests_in_place: self.held_out_tests_in_place,
            anti_poisoning_working: self.anti_poisoning_working,
        }
    }

    const fn complete(self) -> bool {
        self.log_probabilities_available
            && self.backend_differentiable
            && self.held_out_tests_in_place
            && self.anti_poisoning_working
    }
}

/// Integer held-out metrics for one classifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldOutMetrics {
    pub examples: usize,
    pub correct: usize,
    pub accuracy_bps: u16,
}

/// Private proof minted only by `review_neural_model_for_meta`.
///
/// The digest and authoritative held-out metrics are captured before the
/// report is returned. External callers cannot construct or modify this token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetaReviewSeal {
    eligible: bool,
    candidate_digest: Option<[u8; 32]>,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

/// A consuming, non-forgeable proof that the model-side held-out review passed.
///
/// Its fields are private and there is no public constructor. The only public
/// creation path is [`MetaNeuralReviewReport::into_eligible`], which checks the
/// private review seal and re-verifies the candidate artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibleMetaModelReview {
    artifact: EncodedNeuralArtifact,
    evidence: MetaModelEvidence,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

impl EligibleMetaModelReview {
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

/// Failure while converting a report into the stronger eligible-review proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaEligibilityError {
    ReviewHeld,
    IncompleteModelEvidence,
    CandidateMissing,
    CandidateDigestMismatch,
    CandidateArtifactInvalid(NeuralArtifactError),
}

/// Complete non-authoritative review result.
///
/// The visible report remains useful for audit, but its private `seal` field
/// prevents external construction of a fake review result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaNeuralReviewReport {
    pub train_examples: usize,
    pub validation_neural: HeldOutMetrics,
    pub validation_baseline: HeldOutMetrics,
    pub test_neural: HeldOutMetrics,
    pub test_baseline: HeldOutMetrics,
    pub model_evidence: MetaModelEvidence,
    pub blockers: Vec<MetaPromotionBlocker>,
    pub disposition: MetaPromotionDisposition,
    pub authority: MetaPromotionAuthority,
    /// Present only after every review gate passes. This remains a candidate
    /// artifact and is not installed by this module.
    pub candidate_artifact: Option<EncodedNeuralArtifact>,
    seal: MetaReviewSeal,
}

impl MetaNeuralReviewReport {
    /// Consume an eligible report into the proof required by runtime Meta
    /// activation. The candidate artifact is revalidated to make post-review
    /// mutation fail closed.
    pub fn into_eligible(self) -> Result<EligibleMetaModelReview, MetaEligibilityError> {
        if !self.seal.eligible
            || self.disposition != MetaPromotionDisposition::EligibleForControlledPromotionReview
            || self.authority != MetaPromotionAuthority::ReviewOnly
            || !self.blockers.is_empty()
        {
            return Err(MetaEligibilityError::ReviewHeld);
        }
        if !self.model_evidence.complete() {
            return Err(MetaEligibilityError::IncompleteModelEvidence);
        }
        let artifact = self
            .candidate_artifact
            .ok_or(MetaEligibilityError::CandidateMissing)?;
        let expected_digest = self
            .seal
            .candidate_digest
            .ok_or(MetaEligibilityError::CandidateMissing)?;
        if artifact.manifest.weights_hash != expected_digest {
            return Err(MetaEligibilityError::CandidateDigestMismatch);
        }
        load_neural_artifact(&artifact.manifest, &artifact.bytes)
            .map_err(MetaEligibilityError::CandidateArtifactInvalid)?;
        Ok(EligibleMetaModelReview {
            artifact,
            evidence: self.model_evidence,
            validation_accuracy_bps: self.seal.validation_accuracy_bps,
            test_accuracy_bps: self.seal.test_accuracy_bps,
        })
    }
}

/// Structural or provenance failures stop review entirely rather than being
/// converted into a soft blocker.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaNeuralReviewError {
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
    Artifact(NeuralArtifactError),
    InvalidLogProbability,
}

impl From<NeuralModelError> for MetaNeuralReviewError {
    fn from(error: NeuralModelError) -> Self {
        Self::Neural(error)
    }
}

impl From<NeuralArtifactError> for MetaNeuralReviewError {
    fn from(error: NeuralArtifactError) -> Self {
        Self::Artifact(error)
    }
}

/// Train both the differentiable candidate and the historical deterministic
/// baseline on the training split, then compare them only on untouched
/// validation/test examples.
pub fn review_neural_model_for_meta(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    neural_config: NeuralConfig,
    policy: MetaNeuralReviewPolicy,
) -> Result<MetaNeuralReviewReport, MetaNeuralReviewError> {
    validate_policy(policy)?;
    validate_splits(corpus, train, validation, test)?;

    let trainer = NeuralTrainer::try_new(neural_config)?;
    let (neural, _) = trainer.train(corpus, train)?;
    verify_log_probabilities(&neural, corpus, validation)?;
    verify_log_probabilities(&neural, corpus, test)?;

    // The historical integer model is frozen after training and acts only as
    // a deterministic reference policy for held-out regression comparison.
    let baseline_trainer = ToyTrainer::new(neural_config.input_dim, neural_config.epochs);
    let (baseline, _) = baseline_trainer.train(corpus, train);

    let validation_neural = evaluate_neural(&neural, corpus, validation)?;
    let validation_baseline = evaluate_baseline(&baseline, corpus, validation)?;
    let test_neural = evaluate_neural(&neural, corpus, test)?;
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
        Some(encode_neural_artifact(
            &neural,
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

    Ok(MetaNeuralReviewReport {
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
        seal: MetaReviewSeal {
            eligible,
            candidate_digest,
            validation_accuracy_bps: validation_neural.accuracy_bps,
            test_accuracy_bps: test_neural.accuracy_bps,
        },
    })
}

fn validate_policy(policy: MetaNeuralReviewPolicy) -> Result<(), MetaNeuralReviewError> {
    if policy.minimum_validation_accuracy_bps > 10_000
        || policy.minimum_test_accuracy_bps > 10_000
        || policy.maximum_regression_bps > 10_000
        || policy.artifact_max_context_tokens == 0
    {
        return Err(MetaNeuralReviewError::InvalidPolicy);
    }
    Ok(())
}

fn validate_splits(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
) -> Result<(), MetaNeuralReviewError> {
    for (split, expected) in [
        (train, SplitKind::Train),
        (validation, SplitKind::Validation),
        (test, SplitKind::Test),
    ] {
        if split.kind != expected {
            return Err(MetaNeuralReviewError::WrongSplitKind {
                expected,
                actual: split.kind,
            });
        }
        if split.indices.is_empty() {
            return Err(MetaNeuralReviewError::EmptySplit(expected));
        }
    }

    let total = train
        .indices
        .len()
        .checked_add(validation.indices.len())
        .and_then(|value| value.checked_add(test.indices.len()))
        .ok_or(MetaNeuralReviewError::ArithmeticOverflow)?;
    if total > MAX_META_REVIEW_EXAMPLES {
        return Err(MetaNeuralReviewError::TooManyExamples);
    }

    let mut seen = BTreeSet::new();
    for &index in train
        .indices
        .iter()
        .chain(validation.indices.iter())
        .chain(test.indices.iter())
    {
        if index >= corpus.examples.len() {
            return Err(MetaNeuralReviewError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            });
        }
        if !seen.insert(index) {
            return Err(MetaNeuralReviewError::SplitOverlap(index));
        }
        let example = &corpus.examples[index];
        if !trusted_review_provenance(
            example.provenance.origin,
            example.provenance.evidence_origin,
        ) {
            return Err(MetaNeuralReviewError::UntrustedHeldOutProvenance {
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
            return Err(MetaNeuralReviewError::HeldOutLabelAbsentFromTraining(
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

fn evaluate_neural(
    model: &NeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, MetaNeuralReviewError> {
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
) -> Result<HeldOutMetrics, MetaNeuralReviewError> {
    let correct = split
        .indices
        .iter()
        .filter(|&&index| {
            model.classify(&corpus.examples[index].payload) == corpus.examples[index].label
        })
        .count();
    metrics(correct, split.indices.len())
}

fn metrics(correct: usize, examples: usize) -> Result<HeldOutMetrics, MetaNeuralReviewError> {
    let scaled = correct
        .checked_mul(10_000)
        .ok_or(MetaNeuralReviewError::ArithmeticOverflow)?
        .checked_div(examples)
        .ok_or(MetaNeuralReviewError::ArithmeticOverflow)?;
    let accuracy_bps =
        u16::try_from(scaled).map_err(|_| MetaNeuralReviewError::ArithmeticOverflow)?;
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
    model: &NeuralModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<(), MetaNeuralReviewError> {
    for &index in &split.indices {
        let payload = &corpus.examples[index].payload;
        let mut sum = 0.0f32;
        let mut scores = Vec::with_capacity(usize::from(model.num_labels()));
        for label in 0..model.num_labels() {
            let score = model.score(label, payload)?;
            if !score.is_finite() || score <= 0.0 {
                return Err(MetaNeuralReviewError::InvalidLogProbability);
            }
            sum += score;
            scores.push(score);
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(MetaNeuralReviewError::InvalidLogProbability);
        }
        for score in scores {
            let probability = score / sum;
            let log_probability = probability.ln();
            if !probability.is_finite()
                || probability <= 0.0
                || probability > 1.0
                || !log_probability.is_finite()
            {
                return Err(MetaNeuralReviewError::InvalidLogProbability);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::LabeledExample;

    fn corpus_and_splits() -> (Corpus, CorpusSplit, CorpusSplit, CorpusSplit) {
        let mut corpus = Corpus::with_seed(1);
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

    fn config() -> NeuralConfig {
        NeuralConfig {
            input_dim: 64,
            epochs: 64,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        }
    }

    #[test]
    fn eligible_review_produces_sealed_candidate_but_cannot_activate_meta_alone() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let report = review_neural_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            MetaNeuralReviewPolicy {
                minimum_validation_accuracy_bps: 5_000,
                minimum_test_accuracy_bps: 5_000,
                maximum_regression_bps: 5_000,
                artifact_max_context_tokens: 2_048,
            },
        )
        .expect("review");
        assert_eq!(
            report.disposition,
            MetaPromotionDisposition::EligibleForControlledPromotionReview
        );
        assert_eq!(report.authority, MetaPromotionAuthority::ReviewOnly);
        assert!(report.candidate_artifact.is_some());
        let partial = report.model_evidence.as_partial_preconditions();
        assert!(!partial.scalar_engine_validated);
        assert!(!partial.reference_policy_frozen);
        assert!(partial.log_probabilities_available);
        assert!(partial.backend_differentiable);
        assert!(partial.held_out_tests_in_place);
        assert!(partial.anti_poisoning_working);
        assert!(!partial.satisfied());
        let eligible = report.into_eligible().expect("sealed eligible review");
        assert!(eligible.evidence().complete());
    }

    #[test]
    fn post_review_artifact_replacement_breaks_private_digest_seal() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let mut report = review_neural_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            MetaNeuralReviewPolicy {
                minimum_validation_accuracy_bps: 5_000,
                minimum_test_accuracy_bps: 5_000,
                maximum_regression_bps: 5_000,
                artifact_max_context_tokens: 2_048,
            },
        )
        .expect("review");
        let artifact = report.candidate_artifact.as_mut().expect("candidate");
        artifact.manifest.weights_hash[0] ^= 1;
        assert_eq!(
            report.into_eligible(),
            Err(MetaEligibilityError::CandidateDigestMismatch)
        );
    }

    #[test]
    fn model_origin_examples_are_rejected_before_training() {
        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].provenance.origin = InputOrigin::ModelOutput;
        assert!(matches!(
            review_neural_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                MetaNeuralReviewPolicy::default(),
            ),
            Err(MetaNeuralReviewError::UntrustedHeldOutProvenance { .. })
        ));

        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[train.indices[0]].provenance.evidence_origin =
            EvidenceOrigin::ModelInference;
        assert!(matches!(
            review_neural_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                MetaNeuralReviewPolicy::default(),
            ),
            Err(MetaNeuralReviewError::UntrustedHeldOutProvenance { .. })
        ));
    }

    #[test]
    fn split_overlap_and_unseen_heldout_label_fail_closed() {
        let (mut corpus, train, mut validation, test) = corpus_and_splits();
        validation.indices[0] = train.indices[0];
        assert!(matches!(
            review_neural_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                MetaNeuralReviewPolicy::default(),
            ),
            Err(MetaNeuralReviewError::SplitOverlap(_))
        ));

        let (_, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].label = Label(7);
        assert_eq!(
            review_neural_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                MetaNeuralReviewPolicy::default(),
            ),
            Err(MetaNeuralReviewError::HeldOutLabelAbsentFromTraining(
                Label(7)
            ))
        );
    }

    #[test]
    fn regression_tolerance_is_fail_closed() {
        assert!(regressed_beyond(7_999, 8_000, 0));
        assert!(!regressed_beyond(7_999, 8_000, 1));
        assert!(!regressed_beyond(8_000, 8_000, 0));
    }
}
