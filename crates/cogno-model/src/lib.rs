//! # cogno-model — model backend abstraction for COGNO-1.
//!
//! Treats weights and tokenizer as **hostile**. Before any major allocation,
//! the loader verifies (§21): file size, fingerprint, schema version, tensor
//! count, dimensions, numeric types, dimension multiplications (checked
//! arithmetic), and rejects duplicated names, out-of-range data, unknown
//! schema fields when the schema requires it, and unsupported architectures.
//! The model format MUST NOT allow code execution during loading. The core
//! never downloads weights from the network.
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
//! Outputs MUST use a strict, versioned schema (`CognoProposalView`).
//! `confidence_bps` is in basis points (`0`..=`10_000`); any value above
//! `10_000` is rejected. Unknown, duplicated, or out-of-range fields cause an
//! explicit rejection.
//!
//! ## Phase 0 scope (current)
//!
//! Skeleton. No model is loaded. Phase 1 will introduce a deterministic
//! backend returning scripted proposals; Phase 2 a read-only model.
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]