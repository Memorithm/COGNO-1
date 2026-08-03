//! Phase 1 — deterministic simulator backend (COGNO-1 V2 §27 Phase 1).
//!
//! A deterministic backend returning scripted proposals. No real model is
//! loaded; no inference happens. The simulator owns a FIFO queue of
//! [`OwnedProposal`]s the runtime injects at construction time; each call to
//! [`SimBackend::next_proposal`] pops one. When the queue is empty it returns
//! [`BackendError::Exhausted`], so the runtime fails closed (S10) rather than
//! inventing proposals.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};

/// Deterministic simulator. Constructed from a script of proposals; the
/// runtime (or tests) supplies the script. The simulator never fabricates
/// output: an exhausted script is a hard stop, not a fallback.
#[derive(Debug)]
pub struct SimBackend {
    script: std::collections::VecDeque<OwnedProposal>,
}

impl SimBackend {
    /// Construct from an owned script. The script is consumed in order.
    #[must_use]
    pub fn new(script: Vec<OwnedProposal>) -> Self {
        Self {
            script: script.into(),
        }
    }

    /// Empty simulator. Every `next_proposal` returns `Exhausted`.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Append one scripted proposal. Used by tests to set up a deterministic
    /// sequence without constructing the whole script up-front.
    pub fn push(&mut self, p: OwnedProposal) {
        self.script.push_back(p);
    }
}

impl ModelBackend for SimBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            phase: 1,
            read_only: true,
            tools_enabled: false,
            differentiable: false,
        }
    }

    fn next_proposal(&mut self) -> Result<OwnedProposal, BackendError> {
        match self.script.pop_front() {
            Some(p) => Ok(p),
            None => Err(BackendError::Exhausted),
        }
    }
}
