//! # cogno — command-line entry point for COGNO-1.
//!
//! Phase 0 skeleton. No model is loaded; no tool is executed; no side effect is
//! possible. This binary currently only reports the phase status. Subsequent
//! phases will wire `cogno-core`, `cogno-runtime` and `cogno-model` behind an
//! explicit capability gate controlled by the host component.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

fn main() {
    println!("COGNO-1 Phase 0 — deterministic core skeleton (no model loaded).");
    println!("Repo:  https://github.com/Memorithm/COGNO-1");
}