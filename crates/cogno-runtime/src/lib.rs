//! # cogno-runtime — runtime for COGNO-1.
//!
//! Hosts the deterministic execution pipeline that turns external input into
//! either a rejected decision or a (possibly side-effecting) action, never
//! executing a side effect before the full chain has completed
//! (COGNO-1 V2 §3):
//!
//! ```text
//! external input
//!     -> trust classification
//!     -> size control
//!     -> strict parsing
//!     -> structural validation
//!     -> symbolic validation
//!     -> safety rule application
//!     -> neuro-symbolic evaluation
//!     -> deterministic decision
//!     -> audit
//!     -> possible side effect
//! ```
//!
//! ## Phase 0 scope (current)
//!
//! Skeleton. Will progressively host: `MemoryBudget` admission control (§11-12),
//! allocation policy (§13), `RequestScratch` buffers (§14), `BoundedVec`
//! (§15), backpressure/queues (§16), KV cache policy (§17), and pipeline
//! orchestration.
//!
//! In the MVP, COGNO-1 executes no tools (§7). When a tool system is added, the
//! model only produces a typed `ToolProposalView`; the executor enforces a
//! positive tool list, positive arguments, an allowed file root, and
//! duration/memory/output/process/network limits. The forbidden construction
//! `Command::new("sh").arg("-c").arg(model_text)` is a hard rejection.
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]