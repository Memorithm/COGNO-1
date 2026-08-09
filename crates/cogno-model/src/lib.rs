//! # cogno-model — model backend abstraction for COGNO-1.
//!
//! Treats weights and tokenizer as **hostile** (§21). The model is never a
//! trusted source (S1): every output is non-trust data that the runtime parses,
//! bounds, validates and confronts against policy before any use.
//!
//! ## Role of the small model (§2)
//!
//! The model MAY ONLY: classify a feedback event, compare a proposal and the
//! retained version, extract a candidate preference, associate a preference
//! with a category, estimate the contextual relevance of a rule, rank several
//! already-produced outputs, generate an explanation, signal a possible
//! contradiction.
//!
//! The model MAY NEVER, on its own: create a mandatory safety rule, delete an
//! existing rule, execute a command, write to a repository, access the
//! network, open an arbitrary file, modify its own weights, promote a model,
//! modify a memory budget, modify validators, decide a hard constraint may be
//! ignored, or turn retrieved data into a privileged instruction.
//!
//! ## Phases
//!
//! - **Phase 1**: [`simulator::SimBackend`] — deterministic scripted proposals.
//! - **Phase 2**: [`readonly`] — frozen read-only inference over the historical
//!   integer baseline.
//! - **Phase 3**: [`training`] — provenance-aware corpus/splits plus the integer
//!   perceptron baseline retained for regression/oracle comparisons.
//! - **Phase 4 foundation**: [`neural`] retains the canonical v1 bounded linear
//!   differentiable classifier, while [`mlp`] adds a genuinely nonlinear
//!   one-hidden-layer model trained through `cogno-scirust`. Both freeze behind
//!   read-only model surfaces. [`sequence_preference`] adds the byte-tokenized
//!   pairwise ranking bridge to the shared sequence substrate,
//!   [`sequence_retrieval`] adds the byte-tokenized InfoNCE bridge for soft
//!   memory/rule relevance, [`sequence_symbolic`] exposes host-labelled per-rule
//!   soft satisfaction training without granting rule authority,
//!   [`sequence_contradiction`] classifies explicitly framed evidence pairs for
//!   a non-authoritative contradiction signal, and [`sequence_cognitive`] trains
//!   these signals jointly over one shared byte-tokenized representation.
//!   [`artifact`] keeps the canonical v1 hostile format, [`mlp_artifact`] defines
//!   the separate v2 four-tensor format, [`sequence_artifact`] defines the v3
//!   five-tensor sequence-classifier format, and [`sequence_cognitive_artifact`]
//!   defines the V4 eleven-tensor shared-cognitive state. All byte-sequence
//!   formats bind to the deterministic byte tokenizer. [`versioned_artifact`]
//!   dispatches without reinterpreting versions. [`meta_review`] retains the
//!   historical v1 held-out proof, [`sequence_meta_review`] provides the sealed
//!   v3 review, and [`sequence_cognitive_data_review`] exposes the data-classified
//!   V4 weakest-link review boundary across all five cognitive signals.
//!   [`meta_candidate`] exposes all COGNO-minted proofs through one
//!   architecture-neutral but externally non-implementable surface.
//! - **Phase 5**: tools are added only after specific audit and remain behind
//!   an explicit capability gate in `cogno-runtime`.
//!
//! ## Crate lint policy (§23)
//!
//! ```ignore
//! #![forbid(unsafe_code)]
//! #![deny(warnings, missing_debug_implementations, unreachable_pub)]
//! ```
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

pub mod artifact;
pub mod backend;
pub mod meta_candidate;
pub mod meta_review;
#[allow(
    clippy::too_many_arguments,
    reason = "the crate-private verified MLP reconstruction boundary names canonical shape metadata and all four persisted tensors explicitly"
)]
pub mod mlp;
pub mod mlp_artifact;
pub mod neural;
pub mod readonly;
pub mod sequence_artifact;
pub mod sequence_cognitive;
mod sequence_cognitive_activation;
pub mod sequence_cognitive_artifact;
pub mod sequence_cognitive_data_review;
mod sequence_cognitive_meta_review;
pub mod sequence_contradiction;
pub mod sequence_meta_review;
pub mod sequence_preference;
pub mod sequence_retrieval;
pub mod sequence_symbolic;
pub mod simulator;
pub mod tokenizer;
pub mod training;
pub mod versioned_artifact;

pub use artifact::{
    encode_neural_artifact, load_neural_artifact, neural_tokenizer_hash, EncodedNeuralArtifact,
    NeuralArtifactError, MAX_NEURAL_ARTIFACT_BYTES, MAX_NEURAL_CONTEXT_TOKENS,
    NEURAL_ARCHITECTURE_ID, NEURAL_ARTIFACT_HEADER_BYTES, NEURAL_ARTIFACT_MAGIC,
    NEURAL_ARTIFACT_VERSION, NEURAL_TENSOR_COUNT, NEURAL_TOKENIZER_DESCRIPTOR,
};
pub use backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
pub use meta_candidate::MetaReviewedCandidate;
pub use meta_review::{
    review_neural_model_for_meta, EligibleMetaModelReview, HeldOutMetrics, MetaEligibilityError,
    MetaModelEvidence, MetaNeuralReviewError, MetaNeuralReviewPolicy, MetaNeuralReviewReport,
    MetaPromotionAuthority, MetaPromotionBlocker, MetaPromotionDisposition,
    DEFAULT_META_MAX_REGRESSION_BPS, DEFAULT_META_MIN_ACCURACY_BPS, MAX_META_REVIEW_EXAMPLES,
};
pub use mlp::{
    MlpNeuralConfig, MlpNeuralModel, MlpNeuralTrainer, SciRustMlpReadOnlyModel,
    MAX_MLP_HIDDEN_FEATURES, MIN_MLP_HIDDEN_FEATURES,
};
pub use mlp_artifact::{
    encode_mlp_neural_artifact, load_mlp_neural_artifact, MlpNeuralArtifactError,
    MAX_MLP_NEURAL_ARTIFACT_BYTES, MLP_NEURAL_ARCHITECTURE_ID, MLP_NEURAL_ARTIFACT_HEADER_BYTES,
    MLP_NEURAL_ARTIFACT_MAGIC, MLP_NEURAL_ARTIFACT_VERSION, MLP_NEURAL_TENSOR_COUNT,
};
pub use neural::{
    NeuralConfig, NeuralModel, NeuralModelError, NeuralTrainer, NeuralTrainingReport,
    SciRustReadOnlyModel, MAX_NEURAL_EPOCHS, MAX_NEURAL_FEATURES, MAX_NEURAL_LABELS,
    MAX_NEURAL_PARAMETERS, MAX_NEURAL_PAYLOAD_BYTES, MAX_NEURAL_RANK_CANDIDATES,
    MIN_NEURAL_FEATURES,
};
pub use readonly::{ReadOnlyCapability, ReadOnlyModel};
pub use sequence_artifact::{
    encode_sequence_neural_artifact, load_sequence_neural_artifact, SequenceNeuralArtifactError,
    MAX_SEQUENCE_NEURAL_ARTIFACT_BYTES, SEQUENCE_NEURAL_ARCHITECTURE_ID,
    SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES, SEQUENCE_NEURAL_ARTIFACT_MAGIC,
    SEQUENCE_NEURAL_ARTIFACT_VERSION, SEQUENCE_NEURAL_TENSOR_COUNT,
};
pub use sequence_cognitive::{
    SciRustSequenceCognitiveReadOnlyModel, SequenceCognitiveExample, SequenceCognitiveModel,
    SequenceCognitiveModelConfig, SequenceCognitiveModelError, SequenceCognitiveTrainer,
    SequenceCognitiveTrainingReport, MAX_SEQUENCE_COGNITIVE_EXAMPLES,
};
pub use sequence_cognitive_artifact::{
    encode_sequence_cognitive_artifact, load_sequence_cognitive_artifact,
    SequenceCognitiveArtifactError, SequenceCognitiveArtifactState,
    MAX_SEQUENCE_COGNITIVE_ARTIFACT_BYTES, SEQUENCE_COGNITIVE_ARCHITECTURE_ID,
    SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES, SEQUENCE_COGNITIVE_ARTIFACT_MAGIC,
    SEQUENCE_COGNITIVE_ARTIFACT_VERSION, SEQUENCE_COGNITIVE_TENSOR_COUNT,
};
pub use sequence_cognitive_data_review::{
    review_sequence_cognitive_model_for_meta, EligibleSequenceCognitiveMetaModelReview,
    HostConfidentialTrainingAttestation, SequenceCognitiveDataReviewError,
    SequenceCognitiveHeldOutMetrics, SequenceCognitiveMetaEligibilityError,
    SequenceCognitiveMetaReviewConfig, SequenceCognitiveMetaReviewError,
    SequenceCognitiveMetaReviewPolicy, SequenceCognitiveMetaReviewReport,
    SequenceCognitiveReviewCorpus, SequenceCognitiveReviewExample,
};
pub use sequence_contradiction::{
    SciRustSequenceContradictionReadOnlyModel, SequenceContradictionConfig,
    SequenceContradictionError, SequenceContradictionExample, SequenceContradictionModel,
    SequenceContradictionTrainer, SequenceContradictionTrainingReport,
    MAX_SEQUENCE_CONTRADICTION_EXAMPLES, SEQUENCE_CONTRADICTION_CLASS,
    SEQUENCE_CONTRADICTION_CLASSES, SEQUENCE_CONTRADICTION_CLEAR_CLASS,
};
pub use sequence_meta_review::{
    review_sequence_model_for_meta, EligibleSequenceMetaModelReview, SequenceMetaEligibilityError,
    SequenceMetaReviewConfig, SequenceMetaReviewError, SequenceMetaReviewPolicy,
    SequenceMetaReviewReport,
};
pub use sequence_preference::{
    SciRustSequencePreferenceReadOnlyModel, SequencePreferenceConfig, SequencePreferenceError,
    SequencePreferenceModel, SequencePreferencePair, SequencePreferenceTrainer,
    SequencePreferenceTrainingReport, MAX_SEQUENCE_PREFERENCE_PAIRS,
};
pub use sequence_retrieval::{
    SciRustSequenceRetrievalReadOnlyModel, SequenceRetrievalConfig, SequenceRetrievalError,
    SequenceRetrievalExample, SequenceRetrievalModel, SequenceRetrievalTrainer,
    SequenceRetrievalTrainingReport, MAX_SEQUENCE_RETRIEVAL_EXAMPLES,
};
pub use sequence_symbolic::{
    SciRustSequenceSymbolicReadOnlyModel, SequenceSymbolicExample, SequenceSymbolicModel,
    SequenceSymbolicModelConfig, SequenceSymbolicModelError, SequenceSymbolicTrainer,
    SequenceSymbolicTrainingReport, MAX_SEQUENCE_SYMBOLIC_EXAMPLES,
};
pub use simulator::SimBackend;
pub use tokenizer::{
    byte_tokenizer_hash, ByteTokenizer, ByteTokenizerError, BOS_TOKEN, BYTE_TOKENIZER_DESCRIPTOR,
    BYTE_TOKENIZER_VOCAB_SIZE, BYTE_TOKEN_COUNT, EOS_TOKEN, MAX_BYTE_TOKENIZER_TOKENS, SEP_TOKEN,
};
pub use training::{
    Corpus, CorpusSplit, Label, LabeledExample, Provenance, SplitKind, ToyTrainer, TrainedModel,
};
pub use versioned_artifact::{
    load_versioned_neural_artifact, LoadedNeuralModel, VersionedNeuralArtifactError,
};
