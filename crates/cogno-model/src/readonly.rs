//! Phase 2 — read-only model backend (COGNO-1 V2 §27 Phase 2).
//!
//! The model may only classify, extract, rank and explain. It modifies no
//! state directly (S2). This module wraps a frozen [`TrainedModel`] produced
//! by Phase 3's [`crate::training::ToyTrainer`] and exposes the read-only
//! operations the runtime invokes. Inference never mutates the model; the
//! frozen weight table is shared by reference.

use crate::backend::{BackendError, BackendInfo, ModelBackend, OwnedProposal};
use crate::training::{Label, TrainedModel};
use cogno_core::{EvidenceId, ProposalAction, RuleCategory, RuleScope};

/// Capabilities a read-only model may exercise (S2: refused by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadOnlyCapability {
    Classify,
    Extract,
    Rank,
    Explain,
}

/// Frozen read-only model. Holds an `Arc`-shared weight table so that the
/// runtime can hand out cheap clones without copying weights.
#[derive(Clone, Debug)]
pub struct ReadOnlyModel {
    pub model: std::sync::Arc<TrainedModel>,
    pub capabilities: &'static [ReadOnlyCapability],
}

const READ_ONLY_CAPS: &[ReadOnlyCapability] = &[
    ReadOnlyCapability::Classify,
    ReadOnlyCapability::Extract,
    ReadOnlyCapability::Rank,
    ReadOnlyCapability::Explain,
];

impl ReadOnlyModel {
    /// Construct a read-only model from a frozen trained model. The weights
    /// are shared, never mutated.
    #[must_use]
    pub fn from_trained(model: TrainedModel) -> Self {
        Self {
            model: std::sync::Arc::new(model),
            capabilities: READ_ONLY_CAPS,
        }
    }

    /// Classify a feedback payload. Read-only: never mutates the model.
    #[must_use]
    pub fn classify(&self, payload: &[u8]) -> Label {
        self.model.classify(payload)
    }

    /// Extract a candidate preference as a proposal. The runtime validates the
    /// proposal through the deterministic pipeline before adopting anything;
    /// the model only *proposes*. `confidence_bps` is derived from the
    /// classifier's margin, clamped to `0..=10_000`.
    #[must_use]
    pub fn extract(&self, payload: &[u8], evidence_id: EvidenceId) -> OwnedProposal {
        let label = self.classify(payload);
        let score = self.model.score(label.0, payload);
        let runner_up = (0..self.model.num_labels)
            .filter(|&l| l != label.0)
            .map(|l| self.model.score(l, payload))
            .max()
            .unwrap_or(i64::MIN);
        let margin = score.saturating_sub(runner_up);
        let conf = (margin.clamp(0, 10_000) as u16).min(10_000);
        OwnedProposal {
            action: ProposalAction::ExtractPreference,
            category: RuleCategory::Behavior,
            scope: RuleScope::Session,
            confidence_bps: conf,
            evidence_ids: vec![evidence_id],
            payload: payload.to_vec(),
        }
    }

    /// Rank a list of candidate payloads deterministically. Returns indices
    /// into the input slice, best first. Ties broken by ascending index so
    /// the ordering is fully deterministic.
    #[must_use]
    pub fn rank(&self, candidates: &[&[u8]]) -> Vec<usize> {
        let mut order: Vec<(usize, i64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let lbl = self.model.classify(p);
                (i, self.model.score(lbl.0, p))
            })
            .collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        order.into_iter().map(|(i, _)| i).collect()
    }

    /// Explain a label by a stable reason code (no free text — the model
    /// supplies a code, never an instruction-shaped string, §7).
    #[must_use]
    pub fn explain(&self, label: Label) -> cogno_core::ReasonCode {
        cogno_core::ReasonCode(label.0.min(u16::MAX - 1))
    }
}

impl ModelBackend for ReadOnlyModel {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            phase: 2,
            read_only: true,
            tools_enabled: false,
            differentiable: false,
        }
    }

    /// The read-only model surfaces a degenerate proposal stream: each call
    /// produces an `Unsupported` proposal that the runtime's validator rejects.
    /// The read-only model's value is in `classify`/`extract`/`rank`/`explain`,
    /// not in being a general proposal generator. The simulator (Phase 1) is
    /// the deterministic proposal generator.
    fn next_proposal(&mut self) -> Result<OwnedProposal, BackendError> {
        Err(BackendError::ReadOnlyViolation)
    }
}
