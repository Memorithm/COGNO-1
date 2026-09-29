//! Tool proposals (COGNO-1 V2 §7).
//!
//! In the MVP, COGNO-1 executes **no** tool. When a tool system is added, the
//! model only produces a typed `ToolProposalView`; the executor enforces a
//! positive tool list, positive arguments, an allowed file root, and
//! duration / memory / output / process / network limits. Arguments are passed
//! as separate elements to the process API. The construction
//! `Command::new("sh").arg("-c").arg(model_text)` is a hard rejection.

/// Whether the MVP may execute any tool at all. `false` until Phase 5.
pub const MVP_TOOLS_ENABLED: bool = false;

/// Opaque tool id resolved against a positive tool list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ToolId(pub u32);

/// Capability the tool invocation requires. Resolved against the capability
/// policy (S2 — refused by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityId(pub u32);

/// Host-owned category for a capability. Model proposals carry only a
/// capability ID; they cannot select or upgrade this category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CapabilityClass {
    Read,
    Reason,
    Effect,
}

/// Deterministic registry entry assigning exactly one class to a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityClassification {
    pub capability_id: CapabilityId,
    pub class: CapabilityClass,
}

impl CapabilityClassification {
    #[must_use]
    pub const fn new(capability_id: CapabilityId, class: CapabilityClass) -> Self {
        Self {
            capability_id,
            class,
        }
    }
}

/// Short, opaque justification code. The model supplies a code, not free text
/// that could double as instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReasonCode(pub u16);

/// Maximum size of the opaque task binding supplied by the host.
pub const MAX_TASK_BINDING_BYTES: usize = 128;

/// Host-provided capability context for one admitted task.
///
/// The binding is deliberately opaque to COGNO-1. SciRust Hub owns task
/// identity and admission; this type carries the already-selected tool list
/// and capability classes into the deterministic domain boundary. The model
/// proposal contains only an ID and cannot choose its own class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskCapabilityScope<'a> {
    pub task_binding: &'a [u8],
    pub positive_tools: &'a [ToolId],
    pub capability_classifications: &'a [CapabilityClassification],
}

impl<'a> TaskCapabilityScope<'a> {
    /// Construct a task scope without claiming that the binding is
    /// authenticated. Upstream admission remains responsible for that.
    #[must_use]
    pub const fn new(
        task_binding: &'a [u8],
        positive_tools: &'a [ToolId],
        capability_classifications: &'a [CapabilityClassification],
    ) -> Self {
        Self {
            task_binding,
            positive_tools,
            capability_classifications,
        }
    }

    /// Check the cheap, deterministic shape of a host-provided scope.
    pub fn validate(&self) -> Result<(), TaskScopeError> {
        if self.task_binding.is_empty() {
            return Err(TaskScopeError::EmptyBinding);
        }
        if self.task_binding.len() > MAX_TASK_BINDING_BYTES {
            return Err(TaskScopeError::BindingTooLarge {
                observed: self.task_binding.len(),
                maximum: MAX_TASK_BINDING_BYTES,
            });
        }
        for (index, tool) in self.positive_tools.iter().enumerate() {
            if self.positive_tools[index + 1..].contains(tool) {
                return Err(TaskScopeError::DuplicateTool(*tool));
            }
        }
        for (index, classification) in self.capability_classifications.iter().enumerate() {
            if self.capability_classifications[index + 1..]
                .iter()
                .any(|other| other.capability_id == classification.capability_id)
            {
                return Err(TaskScopeError::DuplicateCapability(
                    classification.capability_id,
                ));
            }
        }
        Ok(())
    }

    /// Return the host-assigned class for one capability, if it is in scope.
    #[must_use]
    pub fn capability_class(&self, capability_id: CapabilityId) -> Option<CapabilityClass> {
        self.capability_classifications
            .iter()
            .find(|entry| entry.capability_id == capability_id)
            .map(|entry| entry.class)
    }

    /// Check the task-local positive tool list and require a classified grant.
    /// The runtime still intersects this scope with its own registry and
    /// applies hard constraints before any future execution.
    #[must_use]
    pub fn permits(&self, proposal: &ToolProposalView<'_>) -> bool {
        self.positive_tools.contains(&proposal.tool_id)
            && self.capability_class(proposal.capability_id).is_some()
    }
}

/// Deterministic shape errors for a task-scoped capability context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskScopeError {
    EmptyBinding,
    BindingTooLarge { observed: usize, maximum: usize },
    DuplicateTool(ToolId),
    DuplicateCapability(CapabilityId),
}

/// Type-tagged argument to a tool. Shell-shaped free text is suspicious: argv
/// should be typed, and any `Text` argument carrying shell metacharacters is
/// treated as the forbidden `sh -c <model_text>` shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypedArgument<'a> {
    Text(&'a str),
    Bytes(&'a [u8]),
    Int(i64),
    Path(&'a str),
}

/// Typed tool proposal produced by the model (§7). Never executed directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolProposalView<'a> {
    pub tool_id: ToolId,
    pub capability_id: CapabilityId,
    pub arguments: &'a [TypedArgument<'a>],
    pub justification_code: ReasonCode,
}

fn contains_shell_meta(s: &str) -> bool {
    s.bytes().any(|b| {
        matches!(
            b,
            b';' | b'|' | b'&' | b'>' | b'<' | b'$' | b'`' | b'\n' | b'\r' | b'\0'
        )
    })
}

/// Conservative detector for the forbidden `sh -c <model_text>` shape: any
/// text-carrying argument (`Text`, `Path`, or `Bytes` interpreted lossily)
/// carrying shell metacharacters. `Path` and `Bytes` are scanned too: a
/// hostile proposal can smuggle a shell string into any text-like slot, and a
/// false positive only over-rejects, which is the safe direction for this
/// system (S10). Conservative because the MVP rejects *all* tool proposals
/// anyway; this flag supports the dedicated adversarial test
/// `shell_command_proposal`.
#[must_use]
pub fn looks_like_shell_invocation(p: &ToolProposalView) -> bool {
    p.arguments.iter().any(|arg| match arg {
        TypedArgument::Text(s) | TypedArgument::Path(s) => contains_shell_meta(s),
        TypedArgument::Bytes(b) => std::str::from_utf8(b).is_ok_and(contains_shell_meta),
        TypedArgument::Int(_) => false,
    })
}

/// MVP gate: returns `Unauthorized` for every proposal because no tool may run
/// (§7). When Phase 5 enables tools, this becomes a capability/argument/path
/// check rather than a blanket refusal.
pub fn classify_tool_proposal(_p: &ToolProposalView) -> Result<(), crate::RejectReason> {
    if !MVP_TOOLS_ENABLED {
        return Err(crate::RejectReason::Unauthorized);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_scope_permits_only_the_intersection_candidate() {
        let tools = [ToolId(1)];
        let capabilities = [CapabilityClassification::new(
            CapabilityId(2),
            CapabilityClass::Read,
        )];
        let scope = TaskCapabilityScope::new(b"task-1", &tools, &capabilities);
        let arguments = [TypedArgument::Bytes(b"payload")];
        let permitted = ToolProposalView {
            tool_id: ToolId(1),
            capability_id: CapabilityId(2),
            arguments: &arguments,
            justification_code: ReasonCode(1),
        };
        let wrong_tool = ToolProposalView {
            tool_id: ToolId(9),
            ..permitted
        };
        assert_eq!(scope.validate(), Ok(()));
        assert!(scope.permits(&permitted));
        assert!(!scope.permits(&wrong_tool));
    }

    #[test]
    fn malformed_scope_fails_closed() {
        let empty = TaskCapabilityScope::new(b"", &[], &[]);
        assert_eq!(empty.validate(), Err(TaskScopeError::EmptyBinding));

        let oversized_binding = vec![0_u8; MAX_TASK_BINDING_BYTES + 1];
        let oversized = TaskCapabilityScope::new(&oversized_binding, &[], &[]);
        assert_eq!(
            oversized.validate(),
            Err(TaskScopeError::BindingTooLarge {
                observed: MAX_TASK_BINDING_BYTES + 1,
                maximum: MAX_TASK_BINDING_BYTES,
            })
        );
    }

    #[test]
    fn duplicate_scope_entries_are_rejected() {
        let duplicate_tool_ids = [ToolId(1), ToolId(1)];
        let one_capability = [CapabilityClassification::new(
            CapabilityId(1),
            CapabilityClass::Read,
        )];
        let duplicate_tool = TaskCapabilityScope::new(
            b"task-1",
            &duplicate_tool_ids,
            &one_capability,
        );
        assert_eq!(
            duplicate_tool.validate(),
            Err(TaskScopeError::DuplicateTool(ToolId(1)))
        );

        let one_tool = [ToolId(1)];
        let duplicate_capabilities = [
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
            CapabilityClassification::new(CapabilityId(1), CapabilityClass::Effect),
        ];
        let duplicate_capability =
            TaskCapabilityScope::new(b"task-1", &one_tool, &duplicate_capabilities);
        assert_eq!(
            duplicate_capability.validate(),
            Err(TaskScopeError::DuplicateCapability(CapabilityId(1)))
        );
    }
}
