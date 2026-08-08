//! Held-out Meta review for the V4 shared sequence cognitive model.
//!
//! Promotion is weakest-link: classification, pairwise preference, symbolic
//! satisfaction, contradiction and retrieval are all evaluated on untouched
//! validation/test splits. A strong classifier cannot hide a weak auxiliary
//! cognitive head. The scalar accuracy carried into persistence is the minimum
//! of the five task accuracies.
//!
//! The review remains non-authoritative. It mints only a sealed model-side proof
//! and leaves host-owned Meta preconditions, persistence, activation and tools
//! outside this module.

use crate::artifact::EncodedNeuralArtifact;
use crate::meta_review::{
    HeldOutMetrics, MetaModelEvidence, MetaPromotionAuthority, MetaPromotionBlocker,
    MetaPromotionDisposition, DEFAULT_META_MAX_REGRESSION_BPS, DEFAULT_META_MIN_ACCURACY_BPS,
    MAX_META_REVIEW_EXAMPLES,
};
use crate::neural::{MAX_NEURAL_FEATURES, MIN_NEURAL_FEATURES};
use crate::sequence_cognitive::{
    SequenceCognitiveExample, SequenceCognitiveModel, SequenceCognitiveModelConfig,
    SequenceCognitiveModelError, SequenceCognitiveTrainer,
};
use crate::sequence_cognitive_artifact::{
    encode_sequence_cognitive_artifact, load_sequence_cognitive_artifact,
    SequenceCognitiveArtifactError,
};
use crate::tokenizer::{ByteTokenizer, ByteTokenizerError};
use crate::training::{Corpus, CorpusSplit, Label, LabeledExample, SplitKind, ToyTrainer, TrainedModel};
use cogno_core::{EvidenceOrigin, Fingerprint, InputOrigin};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// One multi-signal observation plus provenance covering the complete example.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveReviewExample {
    pub example: SequenceCognitiveExample,
    pub origin: InputOrigin,
    pub evidence_origin: EvidenceOrigin,
    pub fingerprint: Fingerprint,
}

impl SequenceCognitiveReviewExample {
    /// Construct an observation with a canonical SHA-256 fingerprint over all
    /// five task inputs and targets plus provenance-independent supervision.
    #[must_use]
    pub fn new(
        example: SequenceCognitiveExample,
        origin: InputOrigin,
        evidence_origin: EvidenceOrigin,
    ) -> Self {
        let fingerprint = cognitive_fingerprint(&example);
        Self {
            example,
            origin,
            evidence_origin,
            fingerprint,
        }
    }
}

/// In-memory V4 review corpus with deterministic duplicate rejection/splitting.
#[derive(Clone, Debug)]
pub struct SequenceCognitiveReviewCorpus {
    pub examples: Vec<SequenceCognitiveReviewExample>,
    seen: BTreeSet<Fingerprint>,
    pub seed: u64,
}

impl SequenceCognitiveReviewCorpus {
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            examples: Vec::new(),
            seen: BTreeSet::new(),
            seed,
        }
    }

    /// Add once; a complete multi-signal fingerprint duplicate is counted once.
    pub fn add(&mut self, example: SequenceCognitiveReviewExample) -> bool {
        if !self.seen.insert(example.fingerprint) {
            return false;
        }
        self.examples.push(example);
        true
    }

    /// Deterministic seeded split matching the historical corpus semantics.
    #[must_use]
    pub fn split(
        &self,
        train_ratio: f32,
        validation_ratio: f32,
    ) -> (CorpusSplit, CorpusSplit, CorpusSplit) {
        let count = self.examples.len();
        let mut order: Vec<usize> = (0..count).collect();
        let mut state = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for index in (1..count).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let swap = (state >> 33) as usize % (index + 1);
            order.swap(index, swap);
        }
        let train_end = (count as f32 * train_ratio).round() as usize;
        let validation_end = train_end + (count as f32 * validation_ratio).round() as usize;
        let train_end = train_end.min(count);
        let validation_end = validation_end.min(count);
        (
            CorpusSplit {
                kind: SplitKind::Train,
                indices: order[..train_end].to_vec(),
            },
            CorpusSplit {
                kind: SplitKind::Validation,
                indices: order[train_end..validation_end].to_vec(),
            },
            CorpusSplit {
                kind: SplitKind::Test,
                indices: order[validation_end..].to_vec(),
            },
        )
    }
}

/// V4 model training plus the historical frozen classification baseline size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceCognitiveMetaReviewConfig {
    pub model: SequenceCognitiveModelConfig,
    pub baseline_feature_dim: usize,
}

impl Default for SequenceCognitiveMetaReviewConfig {
    fn default() -> Self {
        Self {
            model: SequenceCognitiveModelConfig::default(),
            baseline_feature_dim: 64,
        }
    }
}

/// Weakest-link thresholds and classification-only regression tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveMetaReviewPolicy {
    pub minimum_validation_task_accuracy_bps: u16,
    pub minimum_test_task_accuracy_bps: u16,
    pub maximum_classification_regression_bps: u16,
}

impl Default for SequenceCognitiveMetaReviewPolicy {
    fn default() -> Self {
        Self {
            minimum_validation_task_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            minimum_test_task_accuracy_bps: DEFAULT_META_MIN_ACCURACY_BPS,
            maximum_classification_regression_bps: DEFAULT_META_MAX_REGRESSION_BPS,
        }
    }
}

/// Auditable held-out metrics for every V4 cognitive signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveHeldOutMetrics {
    pub examples: usize,
    pub classification_correct: usize,
    pub preference_correct: usize,
    pub symbolic_correct: usize,
    pub symbolic_decisions: usize,
    pub contradiction_correct: usize,
    pub retrieval_correct: usize,
    pub classification_accuracy_bps: u16,
    pub preference_accuracy_bps: u16,
    pub symbolic_accuracy_bps: u16,
    pub contradiction_accuracy_bps: u16,
    pub retrieval_accuracy_bps: u16,
    pub weakest_accuracy_bps: u16,
}

/// Structural, provenance, numerical or artifact failure during V4 review.
#[derive(Clone, Debug, PartialEq)]
pub enum SequenceCognitiveMetaReviewError {
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
    HeldOutClassAbsentFromTraining(usize),
    HeldOutContradictionClassAbsentFromTraining(bool),
    UntrustedReviewProvenance {
        index: usize,
        input_origin: InputOrigin,
        evidence_origin: EvidenceOrigin,
    },
    DuplicateClassificationBaseline,
    ArithmeticOverflow,
    InvalidProbability,
    InvalidScore,
    Tokenizer(ByteTokenizerError),
    Model(SequenceCognitiveModelError),
    Artifact(SequenceCognitiveArtifactError),
}

impl From<ByteTokenizerError> for SequenceCognitiveMetaReviewError {
    fn from(error: ByteTokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<SequenceCognitiveModelError> for SequenceCognitiveMetaReviewError {
    fn from(error: SequenceCognitiveModelError) -> Self {
        Self::Model(error)
    }
}

impl From<SequenceCognitiveArtifactError> for SequenceCognitiveMetaReviewError {
    fn from(error: SequenceCognitiveArtifactError) -> Self {
        Self::Artifact(error)
    }
}

/// Failure while consuming an audit report into the sealed V4 proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceCognitiveMetaEligibilityError {
    ReviewHeld,
    IncompleteModelEvidence,
    CandidateMissing,
    CandidateDigestMismatch,
    CandidateArtifactInvalid(SequenceCognitiveArtifactError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SequenceCognitiveMetaReviewSeal {
    eligible: bool,
    candidate_digest: Option<[u8; 32]>,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

/// Non-forgeable model-side proof that all five V4 held-out gates passed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibleSequenceCognitiveMetaModelReview {
    artifact: EncodedNeuralArtifact,
    evidence: MetaModelEvidence,
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
}

impl EligibleSequenceCognitiveMetaModelReview {
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

/// Visible audit report. The private seal prevents external proof forgery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceCognitiveMetaReviewReport {
    pub train_examples: usize,
    pub validation_cognitive: SequenceCognitiveHeldOutMetrics,
    pub validation_baseline: HeldOutMetrics,
    pub test_cognitive: SequenceCognitiveHeldOutMetrics,
    pub test_baseline: HeldOutMetrics,
    pub model_evidence: MetaModelEvidence,
    pub blockers: Vec<MetaPromotionBlocker>,
    pub disposition: MetaPromotionDisposition,
    pub authority: MetaPromotionAuthority,
    pub candidate_artifact: Option<EncodedNeuralArtifact>,
    seal: SequenceCognitiveMetaReviewSeal,
}

impl SequenceCognitiveMetaReviewReport {
    /// Consume only a complete weakest-link review and re-decode the V4 artifact
    /// so post-review mutation fails closed.
    pub fn into_eligible(
        self,
    ) -> Result<EligibleSequenceCognitiveMetaModelReview, SequenceCognitiveMetaEligibilityError> {
        if !self.seal.eligible
            || self.disposition != MetaPromotionDisposition::EligibleForControlledPromotionReview
            || self.authority != MetaPromotionAuthority::ReviewOnly
            || !self.blockers.is_empty()
        {
            return Err(SequenceCognitiveMetaEligibilityError::ReviewHeld);
        }
        if !evidence_complete(self.model_evidence) {
            return Err(SequenceCognitiveMetaEligibilityError::IncompleteModelEvidence);
        }
        let artifact = self
            .candidate_artifact
            .ok_or(SequenceCognitiveMetaEligibilityError::CandidateMissing)?;
        let expected_digest = self
            .seal
            .candidate_digest
            .ok_or(SequenceCognitiveMetaEligibilityError::CandidateMissing)?;
        if artifact.manifest.weights_hash != expected_digest {
            return Err(SequenceCognitiveMetaEligibilityError::CandidateDigestMismatch);
        }
        load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes)
            .map_err(SequenceCognitiveMetaEligibilityError::CandidateArtifactInvalid)?;
        Ok(EligibleSequenceCognitiveMetaModelReview {
            artifact,
            evidence: self.model_evidence,
            validation_accuracy_bps: self.seal.validation_accuracy_bps,
            test_accuracy_bps: self.seal.test_accuracy_bps,
        })
    }
}

/// Train only on `train`, evaluate all five signals on untouched held-out
/// splits, compare classification against the frozen integer baseline, and mint
/// V4 only if every gate passes.
pub fn review_sequence_cognitive_model_for_meta(
    corpus: &SequenceCognitiveReviewCorpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: SequenceCognitiveMetaReviewConfig,
    policy: SequenceCognitiveMetaReviewPolicy,
) -> Result<SequenceCognitiveMetaReviewReport, SequenceCognitiveMetaReviewError> {
    validate_policy(policy)?;
    validate_config(config)?;
    validate_splits(corpus, train, validation, test, config.model)?;
    let baseline_corpus = classification_baseline_corpus(corpus)?;

    let train_examples: Vec<SequenceCognitiveExample> = train
        .indices
        .iter()
        .map(|&index| corpus.examples[index].example.clone())
        .collect();
    let trainer = SequenceCognitiveTrainer::try_new(config.model)?;
    let (model, _) = trainer.train(&train_examples)?;

    verify_outputs(&model, corpus, validation)?;
    verify_outputs(&model, corpus, test)?;
    let validation_cognitive = evaluate_cognitive(&model, corpus, validation)?;
    let test_cognitive = evaluate_cognitive(&model, corpus, test)?;

    let baseline_trainer = ToyTrainer::new(config.baseline_feature_dim, config.model.epochs);
    let (baseline, _) = baseline_trainer.train(&baseline_corpus, train);
    let validation_baseline = evaluate_baseline(&baseline, &baseline_corpus, validation)?;
    let test_baseline = evaluate_baseline(&baseline, &baseline_corpus, test)?;

    let mut blockers = Vec::new();
    if validation_cognitive.weakest_accuracy_bps < policy.minimum_validation_task_accuracy_bps {
        blockers.push(MetaPromotionBlocker::ValidationAccuracyBelowThreshold);
    }
    if test_cognitive.weakest_accuracy_bps < policy.minimum_test_task_accuracy_bps {
        blockers.push(MetaPromotionBlocker::TestAccuracyBelowThreshold);
    }
    if regressed_beyond(
        validation_cognitive.classification_accuracy_bps,
        validation_baseline.accuracy_bps,
        policy.maximum_classification_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::ValidationRegression);
    }
    if regressed_beyond(
        test_cognitive.classification_accuracy_bps,
        test_baseline.accuracy_bps,
        policy.maximum_classification_regression_bps,
    ) {
        blockers.push(MetaPromotionBlocker::TestRegression);
    }
    blockers.sort_unstable();
    blockers.dedup();

    let eligible = blockers.is_empty();
    let candidate_artifact = if eligible {
        Some(encode_sequence_cognitive_artifact(
            model.heads(),
            model.max_retrieval_candidates(),
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

    Ok(SequenceCognitiveMetaReviewReport {
        train_examples: train.indices.len(),
        validation_cognitive,
        validation_baseline,
        test_cognitive,
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
        seal: SequenceCognitiveMetaReviewSeal {
            eligible,
            candidate_digest,
            validation_accuracy_bps: validation_cognitive.weakest_accuracy_bps,
            test_accuracy_bps: test_cognitive.weakest_accuracy_bps,
        },
    })
}

fn validate_policy(
    policy: SequenceCognitiveMetaReviewPolicy,
) -> Result<(), SequenceCognitiveMetaReviewError> {
    if policy.minimum_validation_task_accuracy_bps > 10_000
        || policy.minimum_test_task_accuracy_bps > 10_000
        || policy.maximum_classification_regression_bps > 10_000
    {
        return Err(SequenceCognitiveMetaReviewError::InvalidPolicy);
    }
    Ok(())
}

fn validate_config(
    config: SequenceCognitiveMetaReviewConfig,
) -> Result<(), SequenceCognitiveMetaReviewError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&config.baseline_feature_dim) {
        return Err(SequenceCognitiveMetaReviewError::InvalidConfig);
    }
    let _ = SequenceCognitiveTrainer::try_new(config.model)?;
    Ok(())
}

fn validate_splits(
    corpus: &SequenceCognitiveReviewCorpus,
    train: &CorpusSplit,
    validation: &CorpusSplit,
    test: &CorpusSplit,
    config: SequenceCognitiveModelConfig,
) -> Result<(), SequenceCognitiveMetaReviewError> {
    for (split, expected) in [
        (train, SplitKind::Train),
        (validation, SplitKind::Validation),
        (test, SplitKind::Test),
    ] {
        if split.kind != expected {
            return Err(SequenceCognitiveMetaReviewError::WrongSplitKind {
                expected,
                actual: split.kind,
            });
        }
        if split.indices.is_empty() {
            return Err(SequenceCognitiveMetaReviewError::EmptySplit(expected));
        }
    }
    let total = train
        .indices
        .len()
        .checked_add(validation.indices.len())
        .and_then(|value| value.checked_add(test.indices.len()))
        .ok_or(SequenceCognitiveMetaReviewError::ArithmeticOverflow)?;
    if total > MAX_META_REVIEW_EXAMPLES {
        return Err(SequenceCognitiveMetaReviewError::TooManyExamples);
    }

    let tokenizer = ByteTokenizer::try_new(config.cognitive.encoder.max_tokens)?;
    let mut seen_indices = BTreeSet::new();
    for &index in train
        .indices
        .iter()
        .chain(validation.indices.iter())
        .chain(test.indices.iter())
    {
        if index >= corpus.examples.len() {
            return Err(SequenceCognitiveMetaReviewError::InvalidSplitIndex {
                index,
                corpus_len: corpus.examples.len(),
            });
        }
        if !seen_indices.insert(index) {
            return Err(SequenceCognitiveMetaReviewError::SplitOverlap(index));
        }
        let review = &corpus.examples[index];
        if !trusted_review_provenance(review.origin, review.evidence_origin) {
            return Err(SequenceCognitiveMetaReviewError::UntrustedReviewProvenance {
                index,
                input_origin: review.origin,
                evidence_origin: review.evidence_origin,
            });
        }
        validate_example(&review.example, config, &tokenizer)?;
    }

    let train_classes: BTreeSet<usize> = train
        .indices
        .iter()
        .map(|&index| corpus.examples[index].example.classification_target)
        .collect();
    let train_contradictions: BTreeSet<bool> = train
        .indices
        .iter()
        .map(|&index| corpus.examples[index].example.contradicts)
        .collect();
    for &index in validation.indices.iter().chain(test.indices.iter()) {
        let example = &corpus.examples[index].example;
        if !train_classes.contains(&example.classification_target) {
            return Err(SequenceCognitiveMetaReviewError::HeldOutClassAbsentFromTraining(
                example.classification_target,
            ));
        }
        if !train_contradictions.contains(&example.contradicts) {
            return Err(
                SequenceCognitiveMetaReviewError::HeldOutContradictionClassAbsentFromTraining(
                    example.contradicts,
                ),
            );
        }
    }
    Ok(())
}

fn validate_example(
    example: &SequenceCognitiveExample,
    config: SequenceCognitiveModelConfig,
    tokenizer: &ByteTokenizer,
) -> Result<(), SequenceCognitiveMetaReviewError> {
    if example.classification_target >= config.cognitive.num_classes
        || example.rule_satisfied.len() != config.cognitive.num_rules
        || example.retrieval_candidates.len() < 2
        || example.retrieval_candidates.len() > config.max_retrieval_candidates
        || example.retrieval_positive_idx >= example.retrieval_candidates.len()
    {
        return Err(SequenceCognitiveMetaReviewError::InvalidConfig);
    }
    let _ = tokenizer.encode(&example.classification_payload)?;
    let _ = tokenizer.encode(&example.preferred)?;
    let _ = tokenizer.encode(&example.dispreferred)?;
    let _ = tokenizer.encode(&example.symbolic_payload)?;
    let _ = tokenizer.encode_pair(&example.contradiction_left, &example.contradiction_right)?;
    let _ = tokenizer.encode(&example.retrieval_query)?;
    for candidate in &example.retrieval_candidates {
        let _ = tokenizer.encode(candidate)?;
    }
    Ok(())
}

fn classification_baseline_corpus(
    corpus: &SequenceCognitiveReviewCorpus,
) -> Result<Corpus, SequenceCognitiveMetaReviewError> {
    let mut baseline = Corpus::with_seed(corpus.seed);
    for review in &corpus.examples {
        let label = u16::try_from(review.example.classification_target)
            .map_err(|_| SequenceCognitiveMetaReviewError::ArithmeticOverflow)?;
        if !baseline.add(LabeledExample::new(
            Label(label),
            review.example.classification_payload.clone(),
            review.origin,
            review.evidence_origin,
        )) {
            return Err(SequenceCognitiveMetaReviewError::DuplicateClassificationBaseline);
        }
    }
    Ok(baseline)
}

fn verify_outputs(
    model: &SequenceCognitiveModel,
    corpus: &SequenceCognitiveReviewCorpus,
    split: &CorpusSplit,
) -> Result<(), SequenceCognitiveMetaReviewError> {
    for &index in &split.indices {
        let example = &corpus.examples[index].example;
        let probabilities = model.classification_probabilities(&example.classification_payload)?;
        verify_probability_vector(&probabilities)?;

        for score in [
            model.preference_score(&example.preferred)?,
            model.preference_score(&example.dispreferred)?,
        ] {
            if !score.is_finite() {
                return Err(SequenceCognitiveMetaReviewError::InvalidScore);
            }
        }

        let symbolic = model.symbolic_satisfactions(&example.symbolic_payload)?;
        if symbolic.len() != example.rule_satisfied.len()
            || symbolic
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
        }

        let contradiction = model.contradiction_probability(
            &example.contradiction_left,
            &example.contradiction_right,
        )?;
        if !contradiction.is_finite() || contradiction <= 0.0 || contradiction >= 1.0 {
            return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
        }
        if !contradiction.ln().is_finite() || !(1.0 - contradiction).ln().is_finite() {
            return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
        }

        let candidate_refs: Vec<&[u8]> = example
            .retrieval_candidates
            .iter()
            .map(Vec::as_slice)
            .collect();
        let similarities =
            model.retrieval_similarities(&example.retrieval_query, &candidate_refs)?;
        if similarities.len() != candidate_refs.len()
            || similarities.iter().any(|score| !score.is_finite())
        {
            return Err(SequenceCognitiveMetaReviewError::InvalidScore);
        }
    }
    Ok(())
}

fn verify_probability_vector(values: &[f32]) -> Result<(), SequenceCognitiveMetaReviewError> {
    if values.len() < 2 {
        return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
    }
    let mut sum = 0.0f32;
    for &probability in values {
        if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
            return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
        }
        if !probability.ln().is_finite() {
            return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
        }
        sum += probability;
    }
    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-4 {
        return Err(SequenceCognitiveMetaReviewError::InvalidProbability);
    }
    Ok(())
}

fn evaluate_cognitive(
    model: &SequenceCognitiveModel,
    corpus: &SequenceCognitiveReviewCorpus,
    split: &CorpusSplit,
) -> Result<SequenceCognitiveHeldOutMetrics, SequenceCognitiveMetaReviewError> {
    let mut classification_correct = 0usize;
    let mut preference_correct = 0usize;
    let mut symbolic_correct = 0usize;
    let mut symbolic_decisions = 0usize;
    let mut contradiction_correct = 0usize;
    let mut retrieval_correct = 0usize;

    for &index in &split.indices {
        let example = &corpus.examples[index].example;
        if model.classify(&example.classification_payload)? == example.classification_target {
            classification_correct = checked_increment(classification_correct)?;
        }
        if model.preference_compare(&example.preferred, &example.dispreferred)? == Ordering::Greater {
            preference_correct = checked_increment(preference_correct)?;
        }
        let satisfactions = model.symbolic_satisfactions(&example.symbolic_payload)?;
        for (&satisfaction, &target) in satisfactions.iter().zip(&example.rule_satisfied) {
            symbolic_decisions = checked_increment(symbolic_decisions)?;
            if (satisfaction >= 0.5) == target {
                symbolic_correct = checked_increment(symbolic_correct)?;
            }
        }
        let contradiction = model.contradiction_probability(
            &example.contradiction_left,
            &example.contradiction_right,
        )?;
        if (contradiction >= 0.5) == example.contradicts {
            contradiction_correct = checked_increment(contradiction_correct)?;
        }
        let candidate_refs: Vec<&[u8]> = example
            .retrieval_candidates
            .iter()
            .map(Vec::as_slice)
            .collect();
        if model.retrieval_select_best(&example.retrieval_query, &candidate_refs)?
            == example.retrieval_positive_idx
        {
            retrieval_correct = checked_increment(retrieval_correct)?;
        }
    }

    let examples = split.indices.len();
    let classification_accuracy_bps = accuracy_bps(classification_correct, examples)?;
    let preference_accuracy_bps = accuracy_bps(preference_correct, examples)?;
    let symbolic_accuracy_bps = accuracy_bps(symbolic_correct, symbolic_decisions)?;
    let contradiction_accuracy_bps = accuracy_bps(contradiction_correct, examples)?;
    let retrieval_accuracy_bps = accuracy_bps(retrieval_correct, examples)?;
    let weakest_accuracy_bps = [
        classification_accuracy_bps,
        preference_accuracy_bps,
        symbolic_accuracy_bps,
        contradiction_accuracy_bps,
        retrieval_accuracy_bps,
    ]
    .into_iter()
    .min()
    .ok_or(SequenceCognitiveMetaReviewError::ArithmeticOverflow)?;

    Ok(SequenceCognitiveHeldOutMetrics {
        examples,
        classification_correct,
        preference_correct,
        symbolic_correct,
        symbolic_decisions,
        contradiction_correct,
        retrieval_correct,
        classification_accuracy_bps,
        preference_accuracy_bps,
        symbolic_accuracy_bps,
        contradiction_accuracy_bps,
        retrieval_accuracy_bps,
        weakest_accuracy_bps,
    })
}

fn evaluate_baseline(
    model: &TrainedModel,
    corpus: &Corpus,
    split: &CorpusSplit,
) -> Result<HeldOutMetrics, SequenceCognitiveMetaReviewError> {
    let correct = split
        .indices
        .iter()
        .filter(|&&index| model.classify(&corpus.examples[index].payload) == corpus.examples[index].label)
        .count();
    Ok(HeldOutMetrics {
        examples: split.indices.len(),
        correct,
        accuracy_bps: accuracy_bps(correct, split.indices.len())?,
    })
}

fn accuracy_bps(
    correct: usize,
    decisions: usize,
) -> Result<u16, SequenceCognitiveMetaReviewError> {
    let scaled = correct
        .checked_mul(10_000)
        .ok_or(SequenceCognitiveMetaReviewError::ArithmeticOverflow)?
        .checked_div(decisions)
        .ok_or(SequenceCognitiveMetaReviewError::ArithmeticOverflow)?;
    u16::try_from(scaled).map_err(|_| SequenceCognitiveMetaReviewError::ArithmeticOverflow)
}

fn checked_increment(value: usize) -> Result<usize, SequenceCognitiveMetaReviewError> {
    value
        .checked_add(1)
        .ok_or(SequenceCognitiveMetaReviewError::ArithmeticOverflow)
}

fn regressed_beyond(candidate: u16, baseline: u16, tolerance: u16) -> bool {
    candidate < baseline.saturating_sub(tolerance)
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

fn cognitive_fingerprint(example: &SequenceCognitiveExample) -> Fingerprint {
    let mut hash = Sha256::new();
    hash.update(b"cogno-sequence-cognitive-review-v4\0");
    hash_bytes(&mut hash, &example.classification_payload);
    hash_usize(&mut hash, example.classification_target);
    hash_bytes(&mut hash, &example.preferred);
    hash_bytes(&mut hash, &example.dispreferred);
    hash_bytes(&mut hash, &example.symbolic_payload);
    hash_usize(&mut hash, example.rule_satisfied.len());
    for &target in &example.rule_satisfied {
        hash.update([u8::from(target)]);
    }
    hash_bytes(&mut hash, &example.contradiction_left);
    hash_bytes(&mut hash, &example.contradiction_right);
    hash.update([u8::from(example.contradicts)]);
    hash_bytes(&mut hash, &example.retrieval_query);
    hash_usize(&mut hash, example.retrieval_candidates.len());
    for candidate in &example.retrieval_candidates {
        hash_bytes(&mut hash, candidate);
    }
    hash_usize(&mut hash, example.retrieval_positive_idx);
    Fingerprint(hash.finalize().into())
}

fn hash_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash_usize(hash, bytes.len());
    hash.update(bytes);
}

fn hash_usize(hash: &mut Sha256, value: usize) {
    hash.update((value as u128).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_cognitive_artifact::SEQUENCE_COGNITIVE_ARCHITECTURE_ID;
    use cogno_scirust::{SequenceCognitiveConfig, SequenceCognitiveLossWeights, SequenceEncoderConfig};

    fn observation(index: u8, class: usize, contradicts: bool) -> SequenceCognitiveReviewExample {
        let alpha = class == 0;
        let positive = if alpha { b"alpha memory" } else { b"omega memory" };
        let negative = if alpha { b"omega memory" } else { b"alpha memory" };
        SequenceCognitiveReviewExample::new(
            SequenceCognitiveExample {
                classification_payload: format!("class-{class}-sample-{index}").into_bytes(),
                classification_target: class,
                preferred: format!("preferred-{class}-{index}").into_bytes(),
                dispreferred: format!("rejected-{class}-{index}").into_bytes(),
                symbolic_payload: format!("rule-{class}-{index}").into_bytes(),
                rule_satisfied: vec![alpha, !alpha],
                contradiction_left: format!("claim-{index}").into_bytes(),
                contradiction_right: format!("counter-{index}").into_bytes(),
                contradicts,
                retrieval_query: if alpha { b"alpha".to_vec() } else { b"omega".to_vec() },
                retrieval_candidates: vec![positive.to_vec(), negative.to_vec()],
                retrieval_positive_idx: 0,
            },
            InputOrigin::ExplicitUserInstruction,
            EvidenceOrigin::ExplicitUserApproval,
        )
    }

    fn corpus_and_splits() -> (
        SequenceCognitiveReviewCorpus,
        CorpusSplit,
        CorpusSplit,
        CorpusSplit,
    ) {
        let mut corpus = SequenceCognitiveReviewCorpus::with_seed(1901);
        for (index, class, contradiction) in [
            (0, 0, false),
            (1, 1, true),
            (2, 0, true),
            (3, 1, false),
            (4, 0, false),
            (5, 1, true),
            (6, 0, true),
            (7, 1, false),
            (8, 0, false),
            (9, 1, true),
            (10, 0, true),
            (11, 1, false),
        ] {
            assert!(corpus.add(observation(index, class, contradiction)));
        }
        (
            corpus,
            CorpusSplit {
                kind: SplitKind::Train,
                indices: vec![0, 1, 2, 3],
            },
            CorpusSplit {
                kind: SplitKind::Validation,
                indices: vec![4, 5, 6, 7],
            },
            CorpusSplit {
                kind: SplitKind::Test,
                indices: vec![8, 9, 10, 11],
            },
        )
    }

    fn config() -> SequenceCognitiveMetaReviewConfig {
        SequenceCognitiveMetaReviewConfig {
            model: SequenceCognitiveModelConfig {
                cognitive: SequenceCognitiveConfig {
                    encoder: SequenceEncoderConfig {
                        vocab_size: crate::BYTE_TOKENIZER_VOCAB_SIZE,
                        max_tokens: 64,
                        embedding_dim: 8,
                        hidden_dim: 8,
                        seed: 1913,
                    },
                    num_classes: 2,
                    num_rules: 2,
                    classification_seed: 1931,
                    preference_seed: 1933,
                    symbolic_seed: 1949,
                    contradiction_seed: 1951,
                },
                epochs: 4,
                learning_rate: 0.01,
                loss_weights: SequenceCognitiveLossWeights::default(),
                preference_margin: 1.0,
                retrieval_temperature: 0.5,
                max_retrieval_candidates: 2,
            },
            baseline_feature_dim: 64,
        }
    }

    fn permissive_policy() -> SequenceCognitiveMetaReviewPolicy {
        SequenceCognitiveMetaReviewPolicy {
            minimum_validation_task_accuracy_bps: 0,
            minimum_test_task_accuracy_bps: 0,
            maximum_classification_regression_bps: 10_000,
        }
    }

    #[test]
    fn v4_review_mints_sealed_weakest_link_candidate() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let report = review_sequence_cognitive_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("review");
        assert_eq!(report.authority, MetaPromotionAuthority::ReviewOnly);
        assert_eq!(
            report.disposition,
            MetaPromotionDisposition::EligibleForControlledPromotionReview
        );
        let expected_weakest = [
            report.validation_cognitive.classification_accuracy_bps,
            report.validation_cognitive.preference_accuracy_bps,
            report.validation_cognitive.symbolic_accuracy_bps,
            report.validation_cognitive.contradiction_accuracy_bps,
            report.validation_cognitive.retrieval_accuracy_bps,
        ]
        .into_iter()
        .min()
        .expect("five metrics");
        assert_eq!(report.validation_cognitive.weakest_accuracy_bps, expected_weakest);
        let eligible = report.into_eligible().expect("eligible");
        assert_eq!(
            eligible.artifact().manifest.architecture_id,
            SEQUENCE_COGNITIVE_ARCHITECTURE_ID
        );
        assert_eq!(eligible.validation_accuracy_bps(), expected_weakest);
    }

    #[test]
    fn cognitive_review_is_exactly_deterministic() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let left = review_sequence_cognitive_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("left");
        let right = review_sequence_cognitive_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            config(),
            permissive_policy(),
        )
        .expect("right");
        assert_eq!(left.validation_cognitive, right.validation_cognitive);
        assert_eq!(left.test_cognitive, right.test_cognitive);
        assert_eq!(left.candidate_artifact, right.candidate_artifact);
    }

    #[test]
    fn post_review_mutation_breaks_v4_digest_seal() {
        let (corpus, train, validation, test) = corpus_and_splits();
        let mut report = review_sequence_cognitive_model_for_meta(
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
            Err(SequenceCognitiveMetaEligibilityError::CandidateDigestMismatch)
        );
    }

    #[test]
    fn untrusted_v4_review_provenance_fails_before_training() {
        let (mut corpus, train, validation, test) = corpus_and_splits();
        corpus.examples[validation.indices[0]].origin = InputOrigin::ModelOutput;
        assert!(matches!(
            review_sequence_cognitive_model_for_meta(
                &corpus,
                &train,
                &validation,
                &test,
                config(),
                permissive_policy(),
            ),
            Err(SequenceCognitiveMetaReviewError::UntrustedReviewProvenance { .. })
        ));
    }

    #[test]
    fn complete_multi_signal_duplicates_are_rejected() {
        let mut corpus = SequenceCognitiveReviewCorpus::with_seed(1);
        let example = observation(1, 0, false);
        assert!(corpus.add(example.clone()));
        assert!(!corpus.add(example));
    }
}
