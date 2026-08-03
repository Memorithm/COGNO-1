//! Top-level runtime assembly (COGNO-1 V2 §3).
//!
//! Ties the admission gate, the KV controller, the bounded queue, the tool
//! executor, the audit buffer, the deterministic pipeline and the meta-
//! objective (§4) into a single object the CLI drives. Reads only; the runtime
//! performs no tool execution, no FS writes, no network calls — the MVP is a
//! pure evaluator.

use crate::admission::{Admission, AdmissionError};
use crate::audit::Audit;
use crate::executor::{ToolExecutor, ToolOutcome};
use crate::kv_controller::{KvController, KvError};
use crate::pipeline::{Pipeline, PipelineOutcome, PipelineParams};
use crate::queue::{BoundedQueue, QueueError};
use cogno_core::{
    ContextReport, MemoryBudget, MetaObjective, QueueFullPolicy, SafetyPolicy, ToolProposalView,
};

/// Runtime configuration (validated by the construction).
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub budget: MemoryBudget,
    pub reserved_bytes: usize,
    pub kv_capacity_tokens: usize,
    pub kv_policy: cogno_core::KvCachePolicy,
    pub queue_capacity: usize,
    pub queue_policy: QueueFullPolicy,
}

/// Report surfaceable by the CLI `doctor`/`phase` commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReport {
    pub phase: u8,
    pub tools_enabled: bool,
    pub meta_active: bool,
    pub admissions: u64,
    pub rejections: u64,
    pub truncations: u64,
}

/// The runtime. Holds the validated budget, KV controller, queue, tool
/// executor, audit buffer, the §3 pipeline and the §4 meta-objective. Every
/// field is borrowed/Copy on the hot path; no allocation happens per request.
#[derive(Debug)]
pub struct Runtime {
    pub budget: MemoryBudget,
    pub reserved_bytes: usize,
    pub kv: KvController,
    pub queue: BoundedQueue<QueueItem>,
    pub tools: ToolExecutor,
    pub audit: Audit,
    pub pipeline: Pipeline,
    pub meta: MetaObjective,
    pub policy: SafetyPolicy,
    pub admissions: u64,
    pub rejections: u64,
    pub truncations: u64,
}

/// A cheap queue payload used by the runtime (here: a placeholder ticket the
/// CLI/view layer interprets). Kept minimal so the queue stays allocation-free
/// on the critical path.
pub mod queue_item {
    /// One enqueued work ticket.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct QueueItem(pub u64);
}
pub use queue_item::QueueItem;

impl Runtime {
    /// Construct a runtime from a validated config. Construction is the
    /// *initialization* phase (§13): allocations are bounded and fallible.
    pub fn try_new(cfg: RuntimeConfig) -> Result<Self, AdmissionError> {
        Ok(Self {
            budget: cfg.budget,
            reserved_bytes: cfg.reserved_bytes,
            kv: KvController::new(cfg.kv_capacity_tokens, cfg.kv_policy),
            queue: BoundedQueue::try_new(cfg.queue_capacity, cfg.queue_policy)
                .map_err(|_| AdmissionError::Memory(cogno_core::MemoryError::QueueFull))?,
            tools: ToolExecutor::mvp(),
            audit: Audit::default(),
            pipeline: Pipeline,
            meta: MetaObjective::new(),
            policy: SafetyPolicy::MVP,
            admissions: 0,
            rejections: 0,
            truncations: 0,
        })
    }

    /// Run admission control for an incoming request. Updates counters.
    pub fn admit(
        &self,
        input_bytes: usize,
        estimated_tokens: usize,
        kv_bytes: usize,
        workspace_bytes: usize,
        output_reserve_bytes: usize,
    ) -> Result<cogno_core::RequestEstimate, AdmissionError> {
        Admission::new(&self.budget, self.reserved_bytes).admit(
            input_bytes,
            estimated_tokens,
            kv_bytes,
            workspace_bytes,
            output_reserve_bytes,
        )
    }

    /// Run the KV admission step, updating the truncation counter when the
    /// context is truncated (never silent, §17).
    pub fn admit_context(&mut self, requested_tokens: usize) -> Result<ContextReport, KvError> {
        let r = self.kv.admit(requested_tokens)?;
        if r.dropped_tokens > 0 {
            self.truncations = self.truncations.saturating_add(1);
            self.audit.truncation(r);
        }
        Ok(r)
    }

    /// Enqueue one work ticket. Backpressure applied via the chosen policy.
    pub fn enqueue(&mut self, ticket: QueueItem) -> Result<(), QueueError> {
        let r = self.queue.try_push(ticket);
        if r.is_err() {
            self.rejections = self.rejections.saturating_add(1);
        }
        r
    }

    /// Run the §3 pipeline for a proposal.
    pub fn run_pipeline(&mut self, p: PipelineParams<'_>) -> PipelineOutcome {
        let out = self.pipeline.run(&p);
        match &out {
            PipelineOutcome::Rejected { .. } => {
                self.rejections = self.rejections.saturating_add(1);
                self.audit.reject(
                    match &out {
                        PipelineOutcome::Rejected { reason, .. } => *reason,
                        _ => cogno_core::RejectReason::Unknown,
                    },
                    None,
                );
            }
            PipelineOutcome::Eligible { .. } => {
                self.admissions = self.admissions.saturating_add(1);
                self.audit.accept(None);
            }
        }
        out
    }

    /// Decide a tool proposal (Phase 5 gate). MVP refuses everything.
    pub fn execute_tool(&mut self, p: &ToolProposalView<'_>) -> ToolOutcome {
        let o = self.tools.execute(p);
        if matches!(o, ToolOutcome::Refused(_)) {
            self.rejections = self.rejections.saturating_add(1);
            self.audit
                .reject(cogno_core::RejectReason::Unauthorized, None);
        }
        o
    }

    /// Try to activate the §4 meta-objective. Fails closed until all
    /// preconditions are attested.
    pub fn activate_meta(
        &mut self,
        pre: cogno_core::MetaPreconditions,
    ) -> Result<(), cogno_core::MetaObjectiveError> {
        self.meta.activate(pre)
    }

    /// CLI `doctor`/`phase` report.
    pub fn report(&self) -> RuntimeReport {
        RuntimeReport {
            phase: 0,
            tools_enabled: self.tools.tools_enabled,
            meta_active: self.meta.is_active(),
            admissions: self.admissions,
            rejections: self.rejections,
            truncations: self.truncations,
        }
    }
}
