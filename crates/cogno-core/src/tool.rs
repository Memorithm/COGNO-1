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

/// Short, opaque justification code. The model supplies a code, not free text
/// that could double as instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReasonCode(pub u16);

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
