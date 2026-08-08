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
//!   read-only model surfaces. [`artifact`] keeps the canonical v1 hostile
//!   format, [`mlp_artifact`] defines the separate v2 four-tensor format, and
//!   [`versioned_artifact`] dispatches without reinterpreting either version.
//!   [`tokenizer`] defines the deterministic byte-token contract for the future
//!   sequence model without changing v1/v2 manifest hashes. [`meta_review`]
//!   retains the historical v1 held-out gate while [`meta_review_mlp`] adds an
//!   explicitly sealed v2 path and a versioned eligibility wrapper. Runtime
//!   persistence remains a separate explicit migration.
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
pub mod meta_review;
pub mod meta_review_mlp;
#[allow(
    clippy::too_many_arguments,
    reason = "the crate-private verified MLP reconstruction boundary names canonical shape metadata and all four persisted tensors explicitly"
)]
pub mod mlp;
pub mod mlp_artifact;
pub mod neural;
pub mod readonly;
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
pub use meta_review::{
    review_neural_model_for_meta, EligibleMetaModelReview, HeldOutMetrics, MetaEligibilityError,
    MetaModelEvidence, MetaNeuralReviewError, MetaNeuralReviewPolicy, MetaNeuralReviewReport,
    MetaPromotionAuthority, MetaPromotionBlocker, MetaPromotionDisposition,
    DEFAULT_META_MAX_REGRESSION_BPS, DEFAULT_META_MIN_ACCURACY_BPS, MAX_META_REVIEW_EXAMPLES,
};
pub use meta_review_mlp::{
    review_mlp_model_for_meta, EligibleMlpMetaModelReview, EligibleVersionedMetaModelReview,
    MlpMetaEligibilityError, MlpMetaNeuralReviewError, MlpMetaNeuralReviewReport,
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
