//! # cogno-runtime — execution runtime for COGNO-1.
//!
//! Hosts the deterministic pipeline that turns external input into either a
//! rejected decision or a (possibly side-effecting) action, never executing a
//! side effect before the full chain has completed (§3):
//!
//! ```text
//! external input
//!   -> trust classification
//!   -> size control
//!   -> strict parsing (via ModelBackend / scripts)
//!   -> structural validation
//!   -> symbolic validation
//!   -> hard safety rules
//!   -> reward engine (scalar)
//!   -> deterministic decision (lexicographic)
//!   -> audit
//!   -> possible side effect (tools gated in Phase 5)
//! ```
//!
//! No allocations happen on hot paths; no tool is executed in the MVP;
//! every validator fails closed (S10).
//!
//! ## Crate lint policy (§23)
//!
//! ```ignore
//! #![forbid(unsafe_code)]
//! #![deny(warnings, missing_debug_implementations, unreachable_pub)]
//! ```
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

pub mod admission;
pub mod audit;
pub mod dialogue_candidates;
pub mod dialogue_snapshot;
pub mod dialogue_store;
pub mod executor;
pub mod kv_controller;
pub mod path;
pub mod pipeline;
pub mod queue;
pub mod runtime;
pub mod taste_validation_store;
pub mod verified_taste_profile;

pub use admission::{Admission, AdmissionError};
pub use audit::Audit;
pub use dialogue_candidates::{
    CandidateOrigin, DialogueCandidateError, DialogueCandidateReport,
    QuarantinedPreferenceCandidate,
};
pub use dialogue_snapshot::{DialogueSnapshot, DialogueSnapshotError};
pub use dialogue_store::{DialogueStoreError, PersistentDialogueStore};
pub use executor::{ToolExecutor, ToolOutcome};
pub use kv_controller::{KvController, KvError};
pub use path::{ResolvedPath, RootError, RootPolicy};
pub use pipeline::{Pipeline, PipelineOutcome, PipelineParams};
pub use queue::{BoundedQueue, QueueError, QueueStats};
pub use runtime::{QueueItem, Runtime, RuntimeConfig, RuntimeReport};
pub use taste_validation_store::{
    PersistentTasteValidationStore, StoredTasteValidation, StoredValidationOrigin,
    StoredValidationVerdict, TasteValidationAppendOutcome, TasteValidationStoreError,
};
pub use verified_taste_profile::{
    VerifiedTastePreference, VerifiedTasteProfile, VerifiedTasteProfileError,
};
