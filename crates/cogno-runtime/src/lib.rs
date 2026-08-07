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
pub mod scientific_exchange;
pub mod taste_autonomy;
pub mod taste_benchmark;
pub mod taste_campaign;
pub mod taste_campaign_review;
pub mod taste_controlled_restart;
pub mod taste_cycle;
pub mod taste_decision;
pub mod taste_generation;
pub mod taste_longitudinal;
pub mod taste_orchestrator;
pub mod taste_persisted_chain;
pub mod taste_persisted_restart;
pub mod taste_restart_manifest;
pub mod taste_validation_store;
pub mod verified_taste_profile;

pub use admission::{Admission, AdmissionError};
pub use audit::{Audit, TasteDecisionAudit, TasteInfluenceAudit};
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
pub use runtime::{QueueItem, Runtime, RuntimeConfig, RuntimeReport, RuntimeTasteProfileError};
pub use scientific_exchange::{
    classify_scientific_exchange, ScientificExchangeDisposition, ScientificExchangeError,
    ScientificExchangeOrigin, ScientificExchangeRecord, ScientificExchangeVerdict,
    MAX_SCIENTIFIC_EXCHANGE_PAYLOAD_BYTES,
};
pub use taste_autonomy::{
    run_autonomous_scientific_taste, AutonomousTasteError, AutonomousTasteReport,
    MAX_AUTONOMOUS_EXCHANGE_RECORDS,
};
pub use taste_benchmark::{
    evaluate_taste_benchmark, TasteBenchmarkCase, TasteBenchmarkError, TasteBenchmarkReport,
    MAX_BENCHMARK_CASES,
};
pub use taste_campaign::{
    run_scientific_taste_campaign, TasteCampaignError, TasteCampaignGeneration,
    TasteCampaignReport, MAX_TASTE_CAMPAIGN_GENERATIONS,
};
pub use taste_campaign_review::{
    review_scientific_taste_campaign, TasteCampaignReviewAuthority, TasteCampaignReviewBlocker,
    TasteCampaignReviewDisposition, TasteCampaignReviewError, TasteCampaignReviewPolicy,
    TasteCampaignReviewReport, TastePreferenceReview, MIN_CAMPAIGN_REVIEW_CONFIDENCE_BPS,
    MIN_CAMPAIGN_REVIEW_GENERATIONS,
};
pub use taste_controlled_restart::{
    ControlledRestartTasteAuthority, ControlledRestartTasteError, ControlledRestartTasteProfile,
    GenerationBoundControlledRestartTasteError, GenerationBoundControlledRestartTasteProfile,
};
pub use taste_cycle::{commit_taste_cycle, TasteCycleArtifacts, TasteCycleCommit, TasteCycleError};
pub use taste_decision::{
    decide_with_verified_taste, TasteDecision, TasteDecisionCandidate, TasteInfluence,
    TastePreferenceApplication, MAX_TASTE_WEIGHT_BPS,
};
pub use taste_generation::{
    TasteGenerationChain, TasteGenerationError, TasteGenerationManifest, GENESIS_DIGEST,
};
pub use taste_longitudinal::{
    TasteDriftState, TasteHistoryObservation, TasteLongitudinalError, TasteLongitudinalStatus,
    TasteLongitudinalTracker, MATERIAL_DRIFT_BPS, MAX_HISTORY_PER_PREFERENCE, STALE_GENERATION_GAP,
};
pub use taste_orchestrator::{
    orchestrate_scientific_taste_cycle, validation_origin_is_non_model, OrchestratedTasteCandidate,
    OrchestratedTasteOutcome, OrchestratedTasteState, ScientificTasteCycleReport,
    ScientificTasteOrchestratorError, MAX_ORCHESTRATED_CANDIDATES, MAX_ORCHESTRATED_VALIDATIONS,
    MIN_PROMOTION_CONFIRMATIONS, PROMOTION_THRESHOLD_BPS,
};
pub use taste_persisted_chain::{
    load_persisted_taste_generation_selection, PersistedTasteGenerationError,
    PersistedTasteGenerationSelection,
};
pub use taste_persisted_restart::{
    prepare_persisted_controlled_restart_taste_profile, PersistedControlledRestartTasteError,
};
pub use taste_restart_manifest::{
    build_taste_restart_manifest, verify_taste_restart_manifest, TasteRestartManifest,
    TasteRestartManifestAuthority, TasteRestartManifestError, TasteRestartPreference,
    MAX_RESTART_MANIFEST_PREFERENCES,
};
pub use taste_validation_store::{
    PersistentTasteValidationStore, StoredTasteValidation, StoredValidationOrigin,
    StoredValidationVerdict, TasteValidationAppendOutcome, TasteValidationStoreError,
};
pub use verified_taste_profile::{
    VerifiedTastePreference, VerifiedTasteProfile, VerifiedTasteProfileError,
};
