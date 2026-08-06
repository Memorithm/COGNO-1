//! # cogno-core — deterministic authority core for COGNO-1.
//!
//! COGNO-1 is a neuro-symbolic personalization system split into two strictly
//! separated parts:
//!
//! - a **small assistance model** (untrusted: proposes, classifies, extracts,
//!   ranks, explains, flags contradictions); and
//! - this **deterministic Rust core** (the sole authority for safety rules,
//!   access policies, memory limits, accepted formats, provenance, adoption
//!   decisions, side effects, tools, writes, deletions, migrations, profile
//!   updates and model promotion).
//!
//! The model is never treated as a trusted source (S1). Every model output is
//! data that must be parsed, bounded, validated, confronted against policies,
//! possibly rejected, and audited before use.
//!
//! ## Phase 0 scope (current)
//!
//! This crate carries **no model** and performs **no inference**. Phase 0
//! hosts the deterministic vocabulary the rest of the system rests on: typed
//! input origin and trust class (§6), evidence provenance and rule lifecycle
//! (§9), the strict versioned proposal schema and its validator (§2), the
//! lexicographic decision (§8), data classification (§20), the memory budget
//! with checked arithmetic (§11–§13), bounded containers (§15), KV cache policy
//! (§17), backpressure (§16), retention (§19), the semantic memory budget
//! (§18), the hostile-artifact manifest (§21), tool proposals (§7) and
//! pre-allocated request scratch buffers (§14).
//!
//! No allocations happen on hot paths; no tool is executed; no side effect is
//! possible (S2/S9/S10). Every validator fails closed.
//!
//! ## Crate invariants (COGNO-1 V2 §23)
//!
//! This crate MUST remain, by policy and by the lint directives below:
//!  - no network access;
//!  - no process execution;
//!  - no implicit filesystem access;
//!  - no mandatory neural backend;
//!  - no proprietary `unsafe`;
//!  - deterministic;
//!  - testable offline.
//!
//! `forbid(unsafe_code)` is preferred to `deny`: a `forbid` level cannot be
//! relaxed by a child module. This does not guarantee that external
//! dependencies contain no `unsafe` (this crate intentionally has none).
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

pub mod bounded;
pub mod communication;
pub mod data;
pub mod decision;
pub mod dialogue_observer;
pub mod evidence;
pub mod ids;
pub mod journal;
pub mod kv;
pub mod manifest;
pub mod memory;
pub mod meta;
pub mod origin;
pub mod policy;
pub mod profile;
pub mod proposal;
pub mod queue;
pub mod retention;
pub mod reward;
pub mod reward_engine;
pub mod scratch;
pub mod semantic;
pub mod taste;
pub mod tool;
pub mod validators;

pub use bounded::{BoundedVec, CapacityError};
pub use communication::{
    CommunicationAuthorization, CommunicationKind, CommunicationMaturity,
    CommunicationMaturityPolicy, CommunicationRecipient, CommunicationRole,
    COMMUNICATION_SCHEMA_VERSION, MAX_COMMUNICATION_CONFIDENCE_BPS,
};
pub use data::DataClassification;
pub use decision::{decide, Candidate, Decision, RejectReason};
pub use dialogue_observer::{
    observe_dialogue, DialogueObservation, DialogueObservationError, DialogueObservationOutcome,
    ObservedDialogueKind, ObservedSenderClass, DIALOGUE_OBSERVATION_CATEGORY_TAG,
    MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES,
};
pub use evidence::{Evidence, EvidenceId, EvidenceOrigin, RuleState};
pub use ids::{CandidateScore, RuleRef, TokenId, ValidationResult};
pub use journal::{AppendOutcome, EventId, Fingerprint, Journal, JournalEvent};
pub use kv::{ContextReport, KvCachePolicy};
pub use manifest::{
    ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
};
pub use memory::{checked_kv_cache_bytes, BudgetError, MemoryBudget, MemoryError, RequestEstimate};
pub use meta::{MetaObjective, MetaObjectiveError, MetaPreconditions, MetaState};
pub use origin::{InputOrigin, TrustClass};
pub use policy::{HardVerdict, PathPolicy, PathVerdict, SafetyPolicy};
pub use profile::{DerivationPolicy, DerivedRule, Profile};
pub use proposal::{
    validate_proposal, CognoProposalView, ParseError, ProposalAction, RuleCategory, RuleScope,
    MAX_CONFIDENCE_BPS, MAX_EVIDENCE_IDS, MAX_PAYLOAD_BYTES, SCHEMA_VERSION,
};
pub use queue::QueueFullPolicy;
pub use retention::{MemoryClass, RetentionPolicy};
pub use reward::Reward;
pub use reward_engine::{RewardEngine, RewardParams};
pub use scratch::RequestScratch;
pub use semantic::SemanticMemoryBudget;
pub use taste::{
    ScientificTaste, ScientificTasteProfile, TasteError, TasteEvent, TasteEventKind, TasteOrigin,
    TastePolicy, TasteScope, TasteState, MAX_TASTE_CONFIDENCE_BPS, MAX_TASTE_EVIDENCE_IDS,
};
pub use tool::{
    classify_tool_proposal, looks_like_shell_invocation, CapabilityId, ReasonCode, ToolId,
    ToolProposalView, TypedArgument, MVP_TOOLS_ENABLED,
};
pub use validators::{structural, symbolic};
