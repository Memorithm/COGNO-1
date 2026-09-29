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
    tool_proposal_within_limits, CapabilityClass, CapabilityClassification, CapabilityId,
    RejectReason, TaskCapabilityScope, ToolId, ToolProposalView, MVP_TOOLS_ENABLED,
};

/// Deterministic outcome of a tool proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolOutcome {
    /// MVP gate: no tool may run.
    Refused(RejectReason),
    /// Runtime policy permits the proposal for dry-run after provenance and
    /// audit binding; no tool is executed.
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

    /// Legacy unscoped construction. The positive lists are retained for
    /// compatibility, but this executor cannot authorize a proposal without a
    /// task binding; use `phase5_classified` and `execute_for_task` for dry runs.
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

    /// Decide an unscoped tool proposal. This entry point has no host task
    /// binding and therefore never authorizes a proposal, even when the legacy
    /// runtime allowlists match. Shell-shaped inputs retain their hard refusal.
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
        // A proposal without the host task binding and classified scope must
        // never be authorized by this entry point.
        ToolOutcome::Refused(RejectReason::Unauthorized)
    }

    /// Decide a tool proposal inside a host-provided task scope.
    ///
    /// Authorization is the intersection of the runtime's process-wide
    /// allowlists and the task-local lists. The opaque binding is validated for
    /// shape only here; SciRust Hub remains responsible for task identity and
    /// admission. A malformed or over-broad scope therefore cannot expand
    /// authority and is refused before the existing hard checks.
    pub(crate) fn execute_for_task(
        &self,
        scope: &TaskCapabilityScope<'_>,
        p: &ToolProposalView<'_>,
    ) -> ToolOutcome {
        if scope.validate().is_err() || !tool_proposal_within_limits(p) {
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
    use cogno_core::{
        ReasonCode, TaskCapabilityScope, TaskExecutionProvenance, TypedArgument,
        WorkspaceSnapshotSha256,
    };

    fn test_provenance() -> TaskExecutionProvenance {
        TaskExecutionProvenance::new(WorkspaceSnapshotSha256([0x11; 32]), [0x22; 32])
    }

    #[test]
    fn task_scope_intersects_runtime_allowlists() {
        static RUNTIME_TOOLS: &[ToolId] = &[ToolId(1), ToolId(2)];
        static RUNTIME_CAPABILITIES: &[CapabilityClassification] = &[
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
            CapabilityClassification::new(CapabilityId(2), CapabilityClass::Effect),
        ];
        static TASK_TOOLS: &[ToolId] = &[ToolId(1)];
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Read,
        )];
        let executor = ToolExecutor::phase5_classified(true, RUNTIME_TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new_with_provenance(
            b"task-1",
            TASK_TOOLS,
            TASK_CAPABILITIES,
            test_provenance(),
        );
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
        static RUNTIME_CAPABILITIES: &[CapabilityClassification] =
            &[CapabilityClassification::new(
                CapabilityId(1),
                CapabilityClass::Read,
            )];
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Effect,
        )];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new_with_provenance(
            b"task-1",
            TOOLS,
            TASK_CAPABILITIES,
            test_provenance(),
        );
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
        static TASK_CAPABILITIES: &[CapabilityClassification] = &[CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Read,
        )];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, RUNTIME_CAPABILITIES);
        let scope = TaskCapabilityScope::new_with_provenance(
            b"task-1",
            TOOLS,
            TASK_CAPABILITIES,
            test_provenance(),
        );
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
        static CAPABILITIES: &[CapabilityClassification] = &[CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Read,
        )];
        let executor = ToolExecutor::phase5_classified(true, TOOLS, CAPABILITIES);
        let scope = TaskCapabilityScope::new_with_provenance(
            b"task-1",
            TOOLS,
            CAPABILITIES,
            test_provenance(),
        );
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
        static CAPABILITIES: &[CapabilityClassification] = &[CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Read,
        )];
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
