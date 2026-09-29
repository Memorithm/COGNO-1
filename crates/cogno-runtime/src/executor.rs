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

use cogno_core::{
    CapabilityClass, CapabilityClassification, CapabilityId, RejectReason, TaskCapabilityScope,
    ToolId, ToolProposalView, MVP_TOOLS_ENABLED,
};

/// Deterministic outcome of a tool proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolOutcome {
    /// MVP gate: no tool may run.
    Refused(RejectReason),
    /// Would be authorized in Phase 5; recorded but never executed here.
    DryRunAuthorized,
}

/// Tool executor. Stateless and deterministic; the positive tool list, the
/// capability allowlist and the argument policy are owned by the runtime,
/// not by arbitrary model proposals (S2 — refused by default).
#[derive(Debug)]
pub struct ToolExecutor {
    pub tools_enabled: bool,
    pub positive_tools: &'static [ToolId],
    pub allowed_capabilities: &'static [CapabilityId],
    /// Host-owned class registry used by task-bound authorization.
    pub capability_classifications: &'static [CapabilityClassification],
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::mvp()
    }
}

impl ToolExecutor {
    /// MVP construction: tools disabled, empty positive list, no capability.
    #[must_use]
    pub fn mvp() -> Self {
        Self {
            tools_enabled: MVP_TOOLS_ENABLED,
            positive_tools: &[],
            allowed_capabilities: &[],
            capability_classifications: &[],
        }
    }

    /// Phase 5 construction: tools enabled by the host after audit, with an
    /// explicit positive tool list and an explicit capability allowlist. Even
    /// enabled, the executor enforces capability, tool-list and shell checks
    /// before returning `DryRunAuthorized`.
    pub fn phase5(
        tools_enabled: bool,
        positive_tools: &'static [ToolId],
        allowed_capabilities: &'static [CapabilityId],
    ) -> Self {
        Self {
            tools_enabled,
            positive_tools,
            allowed_capabilities,
            capability_classifications: &[],
        }
    }

    /// Phase 5 construction with an explicit process registry of classified
    /// capabilities. Classified policies must be invoked through a task scope;
    /// the unscoped `execute` entry point fails closed.
    pub fn phase5_classified(
        tools_enabled: bool,
        positive_tools: &'static [ToolId],
        capability_classifications: &'static [CapabilityClassification],
    ) -> Self {
        Self {
            tools_enabled,
            positive_tools,
            allowed_capabilities: &[],
            capability_classifications,
        }
    }

    fn capability_classification(&self, capability_id: CapabilityId) -> Option<CapabilityClass> {
        let mut matching = self
            .capability_classifications
            .iter()
            .filter(|entry| entry.capability_id == capability_id);
        let class = matching.next()?.class;
        matching.next().is_none().then_some(class)
    }

    /// Decide a tool proposal. The MVP refuses everything. When enabled, the
    /// proposal must reference a capability on the allowlist (S2), a tool on
    /// the positive list, and must not match the forbidden `sh -c <text>`
    /// shape (`looks_like_shell_invocation`).
    pub fn execute(&self, p: &ToolProposalView<'_>) -> ToolOutcome {
        if !self.tools_enabled || !self.capability_classifications.is_empty() {
            // A classified policy requires the host-provided task binding;
            // this legacy entry point cannot supply it.
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if !self.allowed_capabilities.contains(&p.capability_id) {
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

    /// Decide a tool proposal inside a host-provided task scope.
    ///
    /// Authorization is the intersection of the runtime's process-wide
    /// allowlists and the task-local lists. The opaque binding is validated for
    /// shape only here; SciRust Hub remains responsible for task identity and
    /// admission. A malformed or over-broad scope therefore cannot expand
    /// authority and is refused before the existing hard checks.
    pub fn execute_for_task(
        &self,
        scope: &TaskCapabilityScope<'_>,
        p: &ToolProposalView<'_>,
    ) -> ToolOutcome {
        if scope.validate().is_err() {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if !self.tools_enabled {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        let runtime_class = self.capability_classification(p.capability_id);
        if runtime_class.is_none() || scope.capability_class(p.capability_id) != runtime_class {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if !self.positive_tools.contains(&p.tool_id) || !scope.permits(p) {
            return ToolOutcome::Refused(RejectReason::Unauthorized);
        }
        if cogno_core::looks_like_shell_invocation(p) {
            return ToolOutcome::Refused(RejectReason::HardConstraint);
        }
        ToolOutcome::DryRunAuthorized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{ReasonCode, TaskCapabilityScope, TypedArgument};

    #[test]
    fn task_scope_intersects_runtime_allowlists() {
        static RUNTIME_TOOLS: &[ToolId] = &[ToolId(1), ToolId(2)];
        static RUNTIME_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
            CapabilityClassification::new(CapabilityId(2), CapabilityClass::Effect),
        ];
        static TASK_TOOLS: &[ToolId] = &[ToolId(1)];
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        ];
        let executor = ToolExecutor::phase5_classified(true, RUNTIME_TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new(b"task-1", TASK_TOOLS, TASK_CAPABILITIES);
        let arguments = [TypedArgument::Bytes(b"payload")];
        let allowed = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(1),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        let task_denied = ToolProposalView {
            tool_id: ToolId(2),
            ..allowed
        };
        assert_eq!(
            executor.execute_for_task(&scope, &allowed),
            ToolOutcome::DryRunAuthorized
        );
        assert_eq!(
            executor.execute_for_task(&scope, &task_denied),
            ToolOutcome::Refused(RejectReason::Unauthorized)
        );
        let effect_proposal = ToolProposalView {
            capability_id: CapabilityId(2),
            ..allowed
        };
        assert_eq!(
            executor.execute_for_task(&scope, &effect_proposal),
            ToolOutcome::Refused(RejectReason::Unauthorized)
        );
        assert_eq!(
            executor.execute(&allowed),
            ToolOutcome::Refused(RejectReason::Unauthorized),
            "classified policies require an admitted task context"
        );
    }

    #[test]
    fn task_and_runtime_classification_must_match() {
        static TOOLS: &[ToolId] = &[ToolId(1)];
        static RUNTIME_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        ];
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Effect),
        ];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new(b"task-1", TOOLS, TASK_CAPABILITIES);
        let arguments = [TypedArgument::Bytes(b"payload")];
        let proposal = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(1),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        assert_eq!(
            executor.execute_for_task(&scope, &proposal),
            ToolOutcome::Refused(RejectReason::Unauthorized)
        );
    }

    #[test]
    fn duplicate_runtime_classification_fails_closed() {
        static TOOLS: &[ToolId] = &[ToolId(1)];
        static RUNTIME_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Effect),
        ];
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        ];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new(b"task-1", TOOLS, TASK_CAPABILITIES);
        let arguments = [TypedArgument::Bytes(b"payload")];
        let proposal = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(1),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        assert_eq!(
            executor.execute_for_task(&scope, &proposal),
            ToolOutcome::Refused(RejectReason::Unauthorized)
        );
    }

    #[test]
    fn task_scope_preserves_hard_shell_rejection() {
        static TOOLS: &[ToolId] = &[ToolId(1)];
        static CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        ];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, CAPABILITIES);
        let scope = TaskCapabilityScope::new(b"task-1", TOOLS, CAPABILITIES);
        let arguments = [TypedArgument::Text("read ; delete")];
        let proposal = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(1),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        assert_eq!(
            executor.execute_for_task(&scope, &proposal),
            ToolOutcome::Refused(RejectReason::HardConstraint)
        );
    }

    #[test]
    fn malformed_task_scope_is_refused_before_authorization() {
        static TOOLS: &[ToolId] = &[ToolId(1)];
        static CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        ];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, CAPABILITIES);
        let scope = TaskCapabilityScope::new(b"", TOOLS, CAPABILITIES);
        let arguments = [TypedArgument::Bytes(b"payload")];
        let proposal = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(1),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        assert_eq!(
            executor.execute_for_task(&scope, &proposal),
            ToolOutcome::Refused(RejectReason::Unauthorized)
        );
    }
}
