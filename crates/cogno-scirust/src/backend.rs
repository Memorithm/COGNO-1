//! SciRust backend wiring (COGNO-1 §SciRust #1–#8).
//!
//! Assembles the autograd engine, the loss components, the calibration head,
//! the cost measure, the optimizer and the bounded KV cache, and exposes a
//! single [`SciRustBackend`] the host drives. The backend is **gated** by
//! [`cogno_core::MetaObjective`] (§27 Phase 4): unless all six preconditions
//! are attested, [`SciRustBackend::objective`] returns [`SciRustError::Gated`]
//! — fail-closed (S10). Adding this crate never silently downgrades the MVP.
//!
//! Hard constraints from `cogno-core` apply **before** the differentiable
//! objective and are never compensable (§8/S4).

use crate::calib::Calibration;
use crate::cost::Cost;
use crate::engine::Tape;
use crate::error::{SciRustError, SciRustResult};
use crate::kv::BoundedKvCache;
use crate::losses::{InfoNCE, PairwiseLoss, SymbolicSatisfaction};
use crate::optim::Optimizer;
use cogno_core::{MetaObjective, MetaPreconditions};

/// Backend report — surfaceable by the CLI `doctor`.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendReport {
    pub enabled: bool,
    pub meta_active: bool,
    pub max_nodes: usize,
    pub max_elements: usize,
    pub temperature: f32,
    pub kv_capacity_tokens: usize,
    pub kv_len: usize,
}

/// The SciRust backend. Fields are `pub` for fine-grained use from tests and
/// the CLI; the host typically drives it through [`Self::objective`] which is
/// the gated entry point. Manual `Debug` because `Box<dyn Optimizer>` is not
/// `Debug` (the optimizer state is opaque by design).
pub struct SciRustBackend {
    pub tape: Tape,
    pub pairwise: PairwiseLoss,
    pub symbolic: SymbolicSatisfaction,
    pub infonce: InfoNCE,
    pub calibration: Calibration,
    pub optimizer: Box<dyn Optimizer>,
    pub kv: BoundedKvCache,
    pub meta: MetaObjective,
}

impl std::fmt::Debug for SciRustBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SciRustBackend")
            .field("tape", &self.tape)
            .field("pairwise", &self.pairwise)
            .field("symbolic", &self.symbolic)
            .field("infonce", &self.infonce)
            .field("calibration", &self.calibration)
            .field("optimizer", &"<dyn Optimizer>")
            .field("kv", &self.kv)
            .field("meta", &self.meta)
            .finish()
    }
}

/// Configuration for constructing a [`SciRustBackend`].
#[derive(Clone, Debug)]
pub struct Config {
    pub max_nodes: usize,
    pub max_elements: usize,
    pub kv_capacity: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub temperature: f32,
    pub lr: f32,
}

impl SciRustBackend {
    /// Construct with all components configured. The builder collects caps so
    /// hostile inputs cannot grow the tape, KV cache, or loss buffers.
    pub fn try_new(config: Config) -> SciRustResult<Self> {
        if config.max_nodes == 0 || config.max_elements == 0 || config.kv_capacity == 0 {
            return Err(SciRustError::Empty);
        }
        let tape = Tape::new(config.max_nodes, config.max_elements);
        let pairwise =
            PairwiseLoss::try_new(1.0, config.max_elements.min(1024), config.max_elements)?;
        let symbolic =
            SymbolicSatisfaction::try_new(config.max_elements.min(512), config.max_elements)?;
        let infonce = InfoNCE::try_new(0.07, config.max_elements.min(256), config.max_elements)?;
        let calibration = Calibration::try_new(config.temperature)?;
        // Optimizer over the temperature parameter (1 scalar).
        let optimizer: Box<dyn Optimizer> = Box::new(crate::optim::AdamW::try_new(config.lr, 1)?);
        let kv = BoundedKvCache::try_new(
            config.kv_capacity,
            config.head_dim,
            config.num_heads,
            config.num_layers,
            cogno_core::KvCachePolicy::RejectOnOverflow,
        )
        .map_err(|_| SciRustError::CapacityExceeded {
            requested: config.kv_capacity,
            maximum: config.kv_capacity,
        })?;
        Ok(Self {
            tape,
            pairwise,
            symbolic,
            infonce,
            calibration,
            optimizer,
            kv,
            meta: MetaObjective::new(),
        })
    }

    /// Try to activate the meta-objective. Mirrors the §27 Phase 4 gate: the
    /// backend's [`Self::objective`] refuses to run unless this returns `Ok`.
    pub fn activate(&mut self, pre: MetaPreconditions) -> SciRustResult<()> {
        self.meta.activate(pre).map_err(|_| SciRustError::Gated)
    }

    /// Compute the joint differentiable objective. *Gated*: returns
    /// [`SciRustError::Gated`] if [`MetaObjective`] is not active.
    pub fn objective(&mut self) -> SciRustResult<f64> {
        if !self.meta.is_active() {
            return Err(SciRustError::Gated);
        }
        // The actual objective is configured by the host feeding pairs and
        // rules; here we expose the *current* scalar loss if the tape has one.
        // The host drives the tape directly; `objective` is the gated view.
        if self.tape.nodes.is_empty() {
            return Ok(0.0);
        }
        // The last scalar node is treated as the loss; the host is responsible
        // for building it via the loss components.
        let last = self.tape.nodes.len().saturating_sub(1);
        let v = self.tape.nodes[last]
            .value
            .data
            .first()
            .copied()
            .unwrap_or(0.0);
        if v.is_finite() {
            Ok(v as f64)
        } else {
            Err(SciRustError::NonFinite)
        }
    }

    /// Add a soft cost term. Returned as a `Cost` so the host's reward engine
    /// can subtract it; the host must ensure it cannot override a hard gate
    /// (§8/S4).
    #[must_use]
    pub fn cost_of(&self, memory_bytes: u64, context_tokens: u64, latency_micros: u64) -> Cost {
        crate::cost::CostBreakdown::new(memory_bytes, context_tokens, latency_micros).cost()
    }

    /// Report for the CLI `doctor`.
    pub fn report(&self) -> BackendReport {
        BackendReport {
            enabled: true,
            meta_active: self.meta.is_active(),
            max_nodes: self.tape.max_nodes,
            max_elements: self.tape.max_elements,
            temperature: self.calibration.temperature,
            kv_capacity_tokens: self.kv.capacity_tokens,
            kv_len: self.kv.len,
        }
    }
}
