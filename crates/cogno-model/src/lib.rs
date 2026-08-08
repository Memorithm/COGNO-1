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
//! - **Phase 4 foundation**: [`neural`] — bounded differentiable supervised
//!   classifier trained through `cogno-scirust` autograd and AdamW, then frozen
//!   behind a read-only model surface. [`artifact`] provides the canonical
//!   non-executable, manifest-bound hostile artifact loader for that model.
//!   This does not activate the Meta-NeuroSymbolic objective by itself.
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
pub mod neural;
pub mod readonly;
pub mod simulator;
pub mod training;

pub use artifact::{
    EncodedNeuralArtifact, MAX_NEURAL_ARTIFACT_BYTES, MAX_NEURAL_CONTEXT_TOKENS,
    NEURAL_ARCHITECTURE_ID, NEURAL_ARTIFACT_HEADER_BYTES, NEURAL_ARTIFACT_MAGIC,
    NEURAL_ARTIFACT_VERSION, NEURAL_TENSOR_COUNT, NEURAL_TOKENIZER_DESCRIPTOR, NeuralArtifactError,
    encode_neural_artifact, load_neural_artifact, neural_tokenizer_hash,
};
pub use backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
pub use neural::{
    NeuralConfig, NeuralModel, NeuralModelError, NeuralTrainer, NeuralTrainingReport,
    SciRustReadOnlyModel, MAX_NEURAL_EPOCHS, MAX_NEURAL_FEATURES, MAX_NEURAL_LABELS,
    MAX_NEURAL_PARAMETERS, MAX_NEURAL_PAYLOAD_BYTES, MAX_NEURAL_RANK_CANDIDATES,
    MIN_NEURAL_FEATURES,
};
pub use readonly::{ReadOnlyCapability, ReadOnlyModel};
pub use simulator::SimBackend;
pub use training::{
    Corpus, CorpusSplit, Label, LabeledExample, Provenance, SplitKind, ToyTrainer, TrainedModel,
};
