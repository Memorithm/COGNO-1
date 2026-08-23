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
use crate::meta_activation::{
    prepare_controlled_meta_activation, ControlledMetaActivationError, HostMetaAttestation,
    MetaActivationReceipt,
};
use crate::model_controlled_restart::GenerationBoundControlledRestartCognitiveModel;
use crate::pipeline::{Pipeline, PipelineOutcome, PipelineParams};
use crate::queue::{BoundedQueue, QueueError};
use crate::taste_controlled_restart::GenerationBoundControlledRestartTasteProfile;
use crate::taste_decision::{
    decide_with_verified_taste, TasteDecision, TasteDecisionCandidate, TastePreferenceApplication,
};
use crate::verified_taste_profile::{VerifiedTastePreference, VerifiedTasteProfile};
use cogno_core::{
    ContextReport, MemoryBudget, MetaObjective, QueueFullPolicy, SafetyPolicy, ToolProposalView,
};
use cogno_model::{MetaReviewedCandidate, SciRustSequenceCognitiveReadOnlyModel};

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
    pub meta_candidate_digest: Option<[u8; 32]>,
    pub cognitive_model_loaded: bool,
    pub cognitive_model_generation: Option<u64>,
    pub cognitive_model_artifact_sha256: Option<[u8; 32]>,
    pub cognitive_model_meta_bound: bool,
    pub taste_profile_loaded: bool,
    pub active_taste_preferences: usize,
    pub admissions: u64,
    pub rejections: u64,
    pub truncations: u64,
}

/// Failure while attaching verified taste state to a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeTasteProfileError {
    /// A verified profile was already attached and cannot be replaced.
    AlreadyInstalled,
}

/// Failure while installing a generation-bound read-only cognitive model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeModelInstallError {
    /// A model was already installed; live replacement is forbidden.
    AlreadyInstalled,
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
    meta: MetaObjective,
    pub policy: SafetyPolicy,
    pub admissions: u64,
    pub rejections: u64,
    pub truncations: u64,
    taste_profile: Option<VerifiedTasteProfile>,
    cognitive_model: Option<GenerationBoundControlledRestartCognitiveModel>,
    meta_candidate_digest: Option<[u8; 32]>,
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
            taste_profile: None,
            cognitive_model: None,
            meta_candidate_digest: None,
        })
    }

    /// Install scientific taste state during a controlled runtime restart.
    ///
    /// The caller must provide a consuming seal that is bound both to the
    /// reviewed restart manifest and to the immutable selected generation
    /// manifest. This prevents a stale but internally valid profile from being
    /// installed under a different selected generation. Installation remains
    /// one-shot and replacement is rejected after initialization.
    pub fn install_controlled_restart_taste_profile(
        &mut self,
        profile: GenerationBoundControlledRestartTasteProfile,
    ) -> Result<(), RuntimeTasteProfileError> {
        self.install_verified_taste_profile(
            profile
                .into_controlled_restart_profile()
                .into_verified_profile(),
        )
    }

    fn install_verified_taste_profile(
        &mut self,
        profile: VerifiedTasteProfile,
    ) -> Result<(), RuntimeTasteProfileError> {
        if self.taste_profile.is_some() {
            return Err(RuntimeTasteProfileError::AlreadyInstalled);
        }
        self.taste_profile = Some(profile);
        Ok(())
    }

    /// Install one generation-bound V4 model during controlled restart.
    ///
    /// The consuming seal preserves the selected generation and artifact digest
    /// inside the runtime. A second installation is always rejected, so this
    /// path cannot be used as a live hot-swap primitive.
    pub fn install_controlled_restart_cognitive_model(
        &mut self,
        model: GenerationBoundControlledRestartCognitiveModel,
    ) -> Result<(), RuntimeModelInstallError> {
        if self.cognitive_model.is_some() {
            return Err(RuntimeModelInstallError::AlreadyInstalled);
        }
        self.cognitive_model = Some(model);
        Ok(())
    }

    /// Return the installed V4 facade as a read-only runtime view.
    #[must_use]
    pub fn cognitive_model(&self) -> Option<&SciRustSequenceCognitiveReadOnlyModel> {
        self.cognitive_model
            .as_ref()
            .map(GenerationBoundControlledRestartCognitiveModel::model)
    }

    /// Selected persisted generation bound to the installed V4 model.
    #[must_use]
    pub fn cognitive_model_generation(&self) -> Option<u64> {
        self.cognitive_model
            .as_ref()
            .map(GenerationBoundControlledRestartCognitiveModel::generation)
    }

    /// Artifact digest bound to the installed V4 model.
    #[must_use]
    pub fn cognitive_model_artifact_sha256(&self) -> Option<[u8; 32]> {
        self.cognitive_model
            .as_ref()
            .map(GenerationBoundControlledRestartCognitiveModel::artifact_sha256)
    }

    /// Return the complete verified profile as a read-only runtime view.
    #[must_use]
    pub fn verified_taste_profile(&self) -> Option<&VerifiedTasteProfile> {
        self.taste_profile.as_ref()
    }

    /// Return all active verified preferences.
    ///
    /// An unconfigured runtime returns an empty slice and therefore fails
    /// closed rather than inferring or synthesizing preferences.
    #[must_use]
    pub fn active_taste_preferences(&self) -> &[VerifiedTastePreference] {
        self.taste_profile
            .as_ref()
            .map_or(&[], VerifiedTasteProfile::active_preferences)
    }

    /// Look up one active preference by its stable identifier.
    #[must_use]
    pub fn active_taste_preference(&self, preference_id: u64) -> Option<&VerifiedTastePreference> {
        self.active_taste_preferences()
            .iter()
            .find(|preference| preference.preference_id == preference_id)
    }

    /// Make and audit one bounded taste-aware soft decision.
    ///
    /// Only active verified preferences are visible here. Candidate hard
    /// constraints are evaluated before score comparison and can never be
    /// compensated by taste influence.
    pub fn decide_with_taste(
        &mut self,
        candidates: &[TasteDecisionCandidate],
        applications: &[TastePreferenceApplication],
    ) -> TasteDecision {
        let decision =
            decide_with_verified_taste(candidates, self.active_taste_preferences(), applications);
        self.audit.taste_decision(&decision);
        decision
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

    /// Decide a tool proposal (Phase 5 gate). MVP refuses everything. Both
    /// outcomes are audited: an authorization is a state-relevant decision
    /// and must be traceable like any rejection (§3, S6).
    pub fn execute_tool(&mut self, p: &ToolProposalView<'_>) -> ToolOutcome {
        let o = self.tools.execute(p);
        match o {
            ToolOutcome::Refused(_) => {
                self.rejections = self.rejections.saturating_add(1);
                self.audit
                    .reject(cogno_core::RejectReason::Unauthorized, None);
            }
            ToolOutcome::DryRunAuthorized => {
                self.audit
                    .tool_authorize(Some("dry-run authorized".to_string()));
            }
        }
        o
    }

    /// Activate the §4 meta-objective only from a sealed, held-out eligible
    /// model review plus the two explicit host attestations.
    ///
    /// The candidate artifact is re-verified immediately before activation.
    /// The runtime stores only its digest; this method does not install model
    /// weights, persist state, enable tools, or grant any additional authority.
    pub fn activate_meta_from_review(
        &mut self,
        review: &impl MetaReviewedCandidate,
        host: HostMetaAttestation,
    ) -> Result<MetaActivationReceipt, ControlledMetaActivationError> {
        if self.meta.is_active() {
            return Err(ControlledMetaActivationError::AlreadyActive);
        }
        let (preconditions, receipt) = prepare_controlled_meta_activation(review, host)?;
        self.meta
            .activate(preconditions)
            .map_err(ControlledMetaActivationError::Precondition)?;
        self.meta_candidate_digest = Some(receipt.candidate_artifact_sha256);
        Ok(receipt)
    }

    /// Whether the controlled Meta objective is active.
    #[must_use]
    pub const fn meta_is_active(&self) -> bool {
        self.meta.is_active()
    }

    /// Whether the Meta objective remains quarantined.
    #[must_use]
    pub const fn meta_is_quarantined(&self) -> bool {
        self.meta.is_quarantined()
    }

    /// Digest of the reviewed candidate to which active Meta state is bound.
    #[must_use]
    pub const fn meta_candidate_digest(&self) -> Option<[u8; 32]> {
        self.meta_candidate_digest
    }

    /// CLI `doctor`/`phase` report, including the exact installed V4/Meta
    /// binding state required by the cognitive soft-reward path.
    pub fn report(&self) -> RuntimeReport {
        let cognitive_model_artifact_sha256 = self.cognitive_model_artifact_sha256();
        let meta_candidate_digest = self.meta_candidate_digest;
        let cognitive_model_meta_bound = self.meta.is_active()
            && cognitive_model_artifact_sha256.is_some()
            && cognitive_model_artifact_sha256 == meta_candidate_digest;
        RuntimeReport {
            phase: 0,
            tools_enabled: self.tools.tools_enabled,
            meta_active: self.meta.is_active(),
            meta_candidate_digest,
            cognitive_model_loaded: self.cognitive_model.is_some(),
            cognitive_model_generation: self.cognitive_model_generation(),
            cognitive_model_artifact_sha256,
            cognitive_model_meta_bound,
            taste_profile_loaded: self.taste_profile.is_some(),
            active_taste_preferences: self.active_taste_preferences().len(),
            admissions: self.admissions,
            rejections: self.rejections,
            truncations: self.truncations,
        }
    }
}
