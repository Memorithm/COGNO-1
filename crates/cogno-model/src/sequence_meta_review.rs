//! Held-out review and sealed eligibility proof for the byte-tokenized
//! [`cogno_scirust::SequenceClassifier`].
//!
//! This path mirrors the Phase-4 guarantees of the historical v1 review while
//! keeping the two artifact families explicit. A sequence review trains only
//! on the train split, verifies finite class log-probabilities on untouched
//! validation/test splits, compares held-out accuracy against the frozen
//! integer baseline, enforces anti-poisoning provenance, and emits a v3 hostile
//! artifact only when every model-side gate passes.
//!
//! The resulting [`EligibleSequenceMetaModelReview`] remains non-authoritative:
//! it cannot install weights, enable tools, or satisfy the two host-owned Meta
//! preconditions by itself.

use crate::artifact::EncodedNeuralArtifact;
use crate::meta_review::{
    HeldOutMetrics, MetaModelEvidence, MetaPromotionAuthority, MetaPromotionBlocker,
    MetaPromotionDisposition, DEFAULT_META_MAX_REGRESSION_BPS, DEFAULT_META_MIN_ACCURACY_BPS,
    MAX_META_REVIEW_EXAMPLES,
};
use crate::neural::{MAX_NEURAL_EPOCHS, MAX_NEURAL_FEATURES, MIN_NEURAL_FEATURES};
use crate::sequence_artifact::{
    encode_sequence_neural_artifact, load_sequence_neural_artifact, SequenceNeuralArtifactError,
};
use crate::tokenizer::{
    ByteTokenizer, ByteTokenizerError, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use crate::training::{Corpus, CorpusSplit, Label, SplitKind, ToyTrainer, TrainedModel};
use cogno_core::{EvidenceOrigin, InputOrigin};
use cogno_scirust::{
    SciRustError, SequenceClassifier, SequenceClassifierAdamW, SequenceClassifierConfig,
};
use std::collections::BTreeSet;

/// Training configuration for the sequence candidate and frozen baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceMetaReviewConfig {
    pub classifier: SequenceClassifierConfig,
    pub epochs: u32,
    pub learning_rate: f32,
    pub baseline_feature_dim: usize,
}

impl Default for SequenceMetaReviewConfig {
    fn default() -> Self {
        Self {
            classifier: SequenceClassifierConfig {
                encoder: cogno_scirust::SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 128,
                    embedding_dim: 32,
                    hidden_dim: 64,
                    seed: 0,
                },
                num_classes: 2,
                head_seed: 1,
            },
            epochs: 32,
            learning_rate: 0.01,
            baseline_feature_dim: 64,
        }
    }
}

/// Held-out thresholds for sequence-model promotion review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceMetaReviewPolicy {
    pub minimum_validation_accuracy_bps: u16,
    pub minimum_test_accuracy_bps: u16,
    pub maximum_regression_bps: u16,
}

impl Default for SequenceMetaReviewPolicy {
    fn default() -> Self {
        Self {
            minimum_validation_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            minimum_test_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            maximum_regression_bps: DEFAULT_META_MAX_REGRESSION_BPS,
        }
    }
}

/// Structural, provenance, numerical or artifact failure during v3 review.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceMetaReviewError {
    InvalidPolicy,
    InvalidConfig,
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
    LabelOutOfRange {
        label: Label,
        classes: usize,
    },
    UntrustedHeldOutProvenance {
        index: usize,
        input_origin: InputOrigin,
        evidence_origin: EvidenceOrigin,
    },
    ArithmeticOverflow,
    InvalidLogProbability,
    Tokenizer(ByteTokenizerError),
    SciRust(SciRustError),
    Artifact(SequenceNeuralArtifactError),
}

impl From<ByteTokenizerError> for SequenceMetaReviewError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SciRustError> for SequenceMetaReviewError {
    fn from(error: SciRustError) -> Self {
        Self::SciRust(error)
    }
}

impl From<SequenceNeuralArtifactError> for SequenceMetaReviewError {
    fn from(error: SequenceNeuralArtifactError) -> Self {
        Self::Artifact(error)
    }
}

/// Failure while consuming a visible report into its non-forgeable proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceMetaEligibilityError {
    ReviewHeld,
    IncompleteModelEvidence,
    CandidateMissing,
    CandidateDigestMismatch,
    CandidateArtifactInvalid(SequenceNeuralArtifactError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SequenceMetaReviewSeal {
    eligible: bool,
    candidate_digest: Option<[u8; 32]>,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

/// A consuming, non-forgeable proof that a v3 sequence candidate passed the
/// complete model-side held-out review.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibleSequenceMetaModelReview {
    artifact: EncodedNeuralArtifact,
    evidence: MetaModelEvidence,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

impl EligibleSequenceMetaModelReview {
    #[must_use]
    pub fn artifact(&self) -> &EncodedNeuralArtifact {
        &self.artifact
    }

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
}

/// Visible audit report for one sequence-model held-out review.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceMetaReviewReport {
    pub train_examples: usize,
    pub validation_sequence: HeldOutMetrics,
    pub validation_baseline: HeldOutMetrics,
    pub test_sequence: HeldOutMetrics,
    pub test_baseline: HeldOutMetrics,
    pub model_evidence: MetaModelEvidence,
    pub blockers: Vec<MetaPromotionBlocker>,
    pub disposition: MetaPromotionDisposition,
    pub authority: MetaPromotionAuthority,
    pub candidate_artifact: Option<EncodedNeuralArtifact>,
    seal: SequenceMetaReviewSeal,
}

impl SequenceMetaReviewReport {
    /// Consume an eligible report into the proof accepted by later controlled
    /// runtime stages. The v3 artifact is decoded again before the proof is
    /// returned so post-review mutation fails closed.
    pub fn into_eligible(
        self,
    ) -> Result<EligibleSequenceMetaModelReview, SequenceMetaEligibilityError> {
        if !self.seal.eligible
            || self.disposition != MetaPromotionDisposition::EligibleForControlledPromotionReview
            || self.authority != MetaPromotionAuthority::ReviewOnly
            || !self.blockers.is_empty()
        {
            return Err(SequenceMetaEligibilityError::ReviewHeld);
        }
        if !evidence_complete(self.model_evidence) {
            return Err(SequenceMetaEligibilityError::IncompleteModelEvidence);
        }
        let artifact = self
            .candidate_artifact
            .ok_or(SequenceMetaEligibilityError::CandidateMissing)?;
        let expected_digest = self
            .seal
            .candidate_digest
            .ok_or(SequenceMetaEligibilityError::CandidateMissing)?;
        if artifact.manifest.weights_hash != expected_digest {
            return Err(SequenceMetaEligibilityError::CandidateDigestMismatch);
        }
        load_sequence_neural_artifact(&artifact.manifest, &artifact.bytes)
            .map_err(SequenceMetaEligibilityError::CandidateArtifactInvalid)?;
        Ok(EligibleSequenceMetaModelReview {
            artifact,
            evidence: self.model_evidence,
            validation_accuracy_bps: self.seal.validation_accuracy_bps,
            test_accuracy_bps: self.seal.test_accuracy_bps,
        })
    }
}

/// Train and review one byte-tokenized sequence classifier using only the train
/// split for optimization and untouched validation/test splits for gating.
pub fn review_sequence_model_for_meta(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: SequenceMetaReviewConfig,
    policy: SequenceMetaReviewPolicy,
) -> Result<SequenceMetaReviewReport, SequenceMetaReviewError> {
    validate_policy(policy)?;
    validate_config(config)?;
    validate_splits(
        corpus,
        train,
        validation,
        test,
        config.classifier.num_classes,
    )?;

    let tokenizer = ByteTokenizer::try_new(config.classifier.encoder.max_tokens)?;
    let mut sequence = SequenceClassifier::try_new(config.classifier)?;
    let mut optimizer = SequenceClassifierAdamW::try_new(config.learning_rate, &sequence)?;
    for _ in 0..config.epochs {
        for &index in &train.indices {
            let example = &corpus.examples[index];
            let tokens = tokenizer.encode(&example.payload)?;
            sequence.train_step(&mut optimizer, &tokens, usize::from(example.label.0))?;
        }
    }

    verify_log_probabilities(&sequence, &tokenizer, corpus, validation)?;
    verify_log_probabilities(&sequence, &tokenizer, corpus, test)?;

    let baseline_trainer = ToyTrainer::new(config.baseline_feature_dim, config.epochs);
    let (baseline, _) = baseline_trainer.train(corpus, train);

    let validation_sequence = evaluate_sequence(&sequence, &tokenizer, corpus, validation)?;
    let validation_baseline = evaluate_baseline(&baseline, corpus, validation)?;
    let test_sequence = evaluate_sequence(&sequence, &tokenizer, corpus, test)?;
    let test_baseline = evaluate_baseline(&baseline, corpus, test)?;

    let mut blockers = Vec::new();
    if validation_sequence.accuracy_bps < policy.minimum_validation_accuracy_bps {
        blockers.push(MetaPromotionBlocker::ValidationAccuracyBelowThreshold);
    }
    if test_sequence.accuracy_bps < policy.minimum_test_accuracy_bps {
        blockers.push(MetaPromotionBlocker::TestAccuracyBelowThreshold);
    }
    if regressed_beyond(
        validation_sequence.accuracy_bps,
        validation_baseline.accuracy_bps,
        policy.maximum_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::ValidationRegression);
    }
    if regressed_beyond(
        test_sequence.accuracy_bps,
        test_baseline.accuracy_bps,
        policy.maximum_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::TestRegression);
    }
    blockers.sort_unstable();
    blockers.dedup();

    let eligible = blockers.is_empty();
    let candidate_artifact = if eligible {
        Some(encode_sequence_neural_artifact(&sequence)?)
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

    Ok(SequenceMetaReviewReport {
        train_examples: train.indices.len(),
        validation_sequence,
        validation_baseline,
        test_sequence,
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
        seal: SequenceMetaReviewSeal {
            eligible,
            candidate_digest,
            validation_accuracy_bps: validation_sequence.accuracy_bps,
            test_accuracy_bps: test_sequence.accuracy_bps,
        },
    })
}

fn validate_policy(policy: SequenceMetaReviewPolicy) -> Result<(), SequenceMetaReviewError> {
    if policy.minimum_validation_accuracy_bps > 10_000
        || policy.minimum_test_accuracy_bps > 10_000
        || policy.maximum_regression_bps > 10_000
    {
        return Err(SequenceMetaReviewError::InvalidPolicy);
    }
    Ok(())
}

fn validate_config(config: SequenceMetaReviewConfig) -> Result<(), SequenceMetaReviewError> {
    if config.classifier.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.classifier.encoder.max_tokens)
        || config.epochs == 0
        || config.epochs > MAX_NEURAL_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
        || !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&config.baseline_feature_dim)
    {
        return Err(SequenceMetaReviewError::InvalidConfig);
    }
    SequenceClassifier::try_new(config.classifier)?;
    Ok(())
}

fn validate_splits(
    corpus: &Corpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    classes: usize,
) -> Result<(), SequenceMetaReviewError> {
    for (split, expected) in [
        (train, SplitKind::Train),
        (validation, SplitKind::Validation),
        (test, SplitKind::Test),
    ] {
        if split.kind != expected {
            return Err(SequenceMetaReviewError::WrongSplitKind {
                expected,
                actual: split.kind,
            });
        }
        if split.indices.is_empty() {
            return Err(SequenceMetaReviewError::EmptySplit(expected));
        }
    }

    let total = train
        .indices
        .len()
        .checked_add(validation.indices.len())
        .and_then(|value| value.checked_add(test.indices.len()))
        .ok_or(SequenceMetaReviewError::ArithmeticOverflow)?;
    if total > MAX_META_REVIEW_EXAMPLES {
        return Err(SequenceMetaReviewError::TooManyExamples);
    }

    let mut seen = BTreeSet::new();
    for &index in train
        .indices
        .iter()
        .chain(validation.indices.iter())
        .chain(test.indices.iter())
    {
        if index >= corpus.examples.len() {
            return Err(SequenceMetaReviewError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            });
        }
        if !seen.insert(index) {
            return Err(SequenceMetaReviewError::SplitOverlap(index));
        }
        let example = &corpus.examples[index];
        if usize::from(example.label.0) >= classes {
            return Err(SequenceMetaReviewError::LabelOutOfRange {
                label: example.label,
                classes,
            });
        }
        if !trusted_review_provenance(
            example.provenance.origin,
            example.provenance.evidence_origin,
        ) {
            return Err(SequenceMetaReviewError::UntrustedHeldOutProvenance {
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
        let label = corpus.examples[index].label;
        if !train_labels.contains(&label) {
            return Err(SequenceMetaReviewError::HeldOutLabelAbsentFromTraining(
                label,
            ));
        }
    }
    Ok(())
}

const fn trusted_review_provenance(input: InputOrigin, evidence: EvidenceOrigin) -> bool {
    !matches!(input, InputOrigin::ModelOutput)
        && !matches!(evidence, EvidenceOrigin::ModelInference)
}

const fn evidence_complete(evidence: MetaModelEvidence) -> bool {
    evidence.log_probabilities_available
        && evidence.backend_differentiable
        && evidence.held_out_tests_in_place
        && evidence.anti_poisoning_working
}

fn evaluate_sequence(
    model: &SequenceClassifier,
    tokenizer: &ByteTokenizer,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, SequenceMetaReviewError> {
    let mut correct = 0usize;
    for &index in &split.indices {
        let example = &corpus.examples[index];
        let tokens = tokenizer.encode(&example.payload)?;
        let predicted = model.classify(&tokens)?;
        if predicted == usize::from(example.label.0) {
            correct = correct.saturating_add(1);
        }
    }
    metrics(correct, split.indices.len())
}

fn evaluate_baseline(
    model: &TrainedModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, SequenceMetaReviewError> {
    let correct = split
        .indices
        .iter()
        .filter(|&&index| {
            model.classify(&corpus.examples[index].payload) == corpus.examples[index].label
        })
        .count();
    metrics(correct, split.indices.len())
}

fn metrics(correct: usize, examples: usize) -> Result<HeldOutMetrics, SequenceMetaReviewError> {
    let scaled = correct
        .checked_mul(10_000)
        .ok_or(SequenceMetaReviewError::ArithmeticOverflow)?
        .checked_div(examples)
        .ok_or(SequenceMetaReviewError::ArithmeticOverflow)?;
    let accuracy_bps =
        u16::try_from(scaled).map_err(|_| SequenceMetaReviewError::ArithmeticOverflow)?;
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
    model: &SequenceClassifier,
    tokenizer: &ByteTokenizer,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<(), SequenceMetaReviewError> {
    for &index in &split.indices {
        let tokens = tokenizer.encode(&corpus.examples[index].payload)?;
        let probabilities = model.probabilities(&tokens)?;
        if probabilities.len() != model.config().num_classes {
            return Err(SequenceMetaReviewError::InvalidLogProbability);
        }
        let mut sum = 0.0f32;
        for probability in probabilities {
            if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
                return Err(SequenceMetaReviewError::InvalidLogProbability);
            }
            let log_probability = probability.ln();
            if !log_probability.is_finite() {
                return Err(SequenceMetaReviewError::InvalidLogProbability);
            }
            sum += probability;
        }
        if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-4 {
            return Err(SequenceMetaReviewError::InvalidLogProbability);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_artifact::SEQUENCE_NEURAL_ARCHITECTURE_ID;
    use crate::training::LabeledExample;
    use cogno_scirust::SequenceEncoderConfig;

    fn corpus_and_splits() -> (Corpus, CorpusSplit, CorpusSplit, CorpusSplit) {
        let mut corpus = Corpus::with_seed(211);
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

    fn config() -> SequenceMetaReviewConfig {
        SequenceMetaReviewConfig {
            classifier: SequenceClassifierConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                    max_tokens: 64,
                    embedding_dim: 8,
                    hidden_dim: 12,
                    seed: 223,
                },
                num_classes: 2,
                head_seed: 227,
            },
            epochs: 16,
            learning_rate: 0.02,
            baseline_feature_dim: 64,
        }
    }

    fn permissive_policy() -> SequenceMetaReviewPolicy {
        SequenceMetaReviewPolicy {
            minimum_validation_accuracy_bps: 0,
            minimum_test_accuracy_bps: 0,
            maximum_regression_bps: 10_000,
        }
    }

    #[test]
    fn sequence_review_mints_v3_sealed_candidate_without_host_authority() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let report = review_sequence_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("sequence review");
        assert_eq!(report.authority, MetaPromotionAuthority::ReviewOnly);
        assert_eq!(
            report.disposition,
            MetaPromotionDisposition::EligibleForControlledPromotionReview
        );
        assert_eq!(report.validation_sequence.examples, 2);
        assert_eq!(report.test_sequence.examples, 2);
        let eligible = report.into_eligible().expect("sealed sequence review");
        assert_eq!(
            eligible.artifact().manifest.architecture_id,
            SEQUENCE_NEURAL_ARCHITECTURE_ID
        );
        assert!(evidence_complete(eligible.evidence()));
    }

    #[test]
    fn deterministic_sequence_review_repeats_artifact_and_metrics() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let left = review_sequence_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("left review");
        let right = review_sequence_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("right review");
        assert_eq!(left.validation_sequence, right.validation_sequence);
        assert_eq!(left.test_sequence, right.test_sequence);
        assert_eq!(left.candidate_artifact, right.candidate_artifact);
    }

    #[test]
    fn post_review_artifact_replacement_breaks_sequence_digest_seal() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let mut report = review_sequence_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("review");
        report
            .candidate_artifact
            .as_mut()
            .expect("candidate")
            .manifest
            .weights_hash[0] ^= 1;
        assert_eq!(
            report.into_eligible(),
            Err(SequenceMetaEligibilityError::CandidateDigestMismatch)
        );
    }

    #[test]
    fn untrusted_sequence_review_provenance_fails_before_training() {
        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].provenance.origin = InputOrigin::ModelOutput;
        assert!(matches!(
            review_sequence_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                permissive_policy(),
            ),
            Err(SequenceMetaReviewError::UntrustedHeldOutProvenance { .. })
        ));
    }

    #[test]
    fn labels_outside_sequence_head_fail_closed() {
        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].label = Label(2);
        assert_eq!(
            review_sequence_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                permissive_policy(),
            ),
            Err(SequenceMetaReviewError::LabelOutOfRange {
                label: Label(2),
                classes: 2,
            })
        );
    }
}
