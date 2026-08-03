//! Phase 5 — tool executor (COGNO-1 V2 §7).
//!
//! The MVP executes **no** tool. When a tool system is added, the executor
//! enforces a positive tool list, positive arguments, an allowed file root,
//! and duration / memory / output / process / network limits, and emits an
//! audit record. The forbidden construction `Command::new("sh").arg("-c")
//! .arg(model_text)` is a hard rejection (§7); arguments are passed as
//! separate elements to the process API.
//!
//! Until Phase 5 is audited and enabled, [`ToolExecutor::execute`] returns
//! `Unauthorized` for every proposal — fail closed (S10).

use cogno_core::{RejectReason, MVP_TOOLS_ENABLED};
use cogno_core::{ToolId, ToolProposalView};

/// Deterministic outcome of a tool proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolOutcome {
    /// MVP gate: no tool may run.
    Refused(RejectReason),
    /// Would be authorized in Phase 5; recorded but never executed here.
    DryRunAuthorized,
}

/// Tool executor. Stateless and deterministic; the positive tool list and
/// argument policy are owned by the runtime, not by arbitrary model proposals.
#[derive(Debug)]
pub struct ToolExecutor {
    pub tools_enabled: bool,
    pub positive_tools: &'static [ToolId],
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::mvp()
    }
}

impl ToolExecutor {
    /// MVP construction: tools disabled, empty positive list.
    #[must_use]
    pub fn mvp() -> Self {
        Self {
            tools_enabled: MVP_TOOLS_ENABLED,
            positive_tools: &[],
        }
    }

    /// Phase 5 construction: tools enabled by the host after audit, with an
    /// explicit positive tool list. Even enabled, the executor enforces
    /// capability, path and shell checks before returning `DryRunAuthorized`.
    pub fn phase5(tools_enabled: bool, positive_tools: &'static [ToolId]) -> Self {
        Self {
            tools_enabled,
            positive_tools,
        }
    }

    /// Decide a tool proposal. The MVP refuses everything. When enabled, the
    /// proposal must reference a tool in the positive list and must not match
    /// the forbidden `sh -c <text>` shape (`looks_like_shell_invocation`).
    pub fn execute(&self, p: &ToolProposalView<'_>) -> ToolOutcome {
        if !self.tools_enabled {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if !self.positive_tools.contains(&p.tool_id) {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if cogno_core::looks_like_shell_invocation(p) {
            return ToolOutcome::Refused(RejectReason::HardConstraint);
        }
        // Phase 5 (not enabled here) would now spawn the tool with separate
        // argv elements, duration/memory/output/process/network limits, root
        // confinement, and emit an audit record. We never execute anything in
        // cogno-runtime itself: this is a library, not a syscall gateway.
        ToolOutcome::DryRunAuthorized
    }
}
