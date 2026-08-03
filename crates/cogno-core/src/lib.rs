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
//! The model is never treated as a trusted source. Every model output is data
//! that must be parsed, bounded, validated, confronted against policies,
//! possibly rejected, and audited before use.
//!
//! ## Phase 0 scope (current)
//!
//! This crate is the Phase 0 skeleton. At this stage it carries no model and
//! performs no inference. It will progressively host: types, configuration,
//! the mathematical objective, validators, the reward engine, memory budgets,
//! the append-only event journal, the derived profile, security policy, and
//! tests.
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
//! `forbid(unsafe_code)` is preferred over `deny` because a `forbid` level
//! cannot be relaxed by a child module. This does not, however, guarantee that
//! external dependencies contain no `unsafe` (this crate intentionally has no
//! dependencies at Phase 0).
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]