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
//! - **Phase 1 (current)**: [`simulator::SimBackend`] — a deterministic backend
//!   returning scripted proposals. No real model is loaded.
//! - **Phase 2**: [`readonly`] — a read-only model backend able to classify,
//!   extract, rank and explain; no direct state mutation.
//! - **Phase 3**: [`training`] — a toy trainer with provenance, splits and
//!   adversarial/negative samples. Honest placeholder for a real differentiable
//!   backend (Phase 4 requires one and a true tensor engine, intentionally
//!   absent here).
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

pub mod backend;
pub mod readonly;
pub mod simulator;
pub mod training;

pub use backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
pub use readonly::{ReadOnlyCapability, ReadOnlyModel};
pub use simulator::SimBackend;
pub use training::{
    Corpus, CorpusSplit, Label, LabeledExample, Provenance, SplitKind, ToyTrainer, TrainedModel,
};
