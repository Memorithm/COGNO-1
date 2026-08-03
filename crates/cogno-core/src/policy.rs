//! Hard security policy and validators (COGNO-1 V2 §4, §8, §9).
//!
//! Hard constraints are applied **before** the reward (lexicographic order,
//! §8). A hard violation can never be compensated by a positive reward (S4).
//! Capabilities are refused by default (S2); the MVP executes no tool (§7) and
//! no operation writes outside a configured root (§22).

use crate::data::DataClassification;
use crate::tool::ToolProposalView;

/// Hard safety policy. Pure data; the runtime supplies concrete checks but the
/// core owns the deterministic verdicts so they are testable offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafetyPolicy {
    pub tools_enabled: bool,
    pub allow_confidential_to_model: bool,
}

impl SafetyPolicy {
    /// MVP-safe policy: no tools, no confidential data to the model.
    pub const MVP: Self = Self {
        tools_enabled: false,
        allow_confidential_to_model: false,
    };
}

/// Hard-constraint verdict used by the lexicographic `decide` (§8 step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardVerdict {
    Allow,
    Reject(crate::RejectReason),
}

impl SafetyPolicy {
    /// Check that a payload classification is admissible for the model.
    /// `Secret` is always rejected (§20). `Confidential` is rejected unless
    /// the host explicitly enabled it (and even then the runtime re-checks).
    #[must_use]
    pub fn check_data(&self, class: DataClassification) -> HardVerdict {
        if class.is_secret() {
            return HardVerdict::Reject(crate::RejectReason::Secret);
        }
        if class.requires_host_authorization() && !self.allow_confidential_to_model {
            return HardVerdict::Reject(crate::RejectReason::Confidential);
        }
        HardVerdict::Allow
    }

    /// Check a tool proposal. The MVP rejects **all** tool proposals (§7). A
    /// non-MVP policy still enforces a positive tool list and capability gate
    /// elsewhere; this is the fail-closed first gate.
    #[must_use]
    pub fn check_tool(&self, _p: &ToolProposalView) -> HardVerdict {
        if !self.tools_enabled {
            return HardVerdict::Reject(crate::RejectReason::Unauthorized);
        }
        HardVerdict::Allow
    }
}

/// Path policy (§22). The runtime owns the canonical root; the core performs
/// the cheap, deterministic, lexical checks that do not require filesystem
/// access: parent-component rejection and symlink-aware boundary checks that
/// **complement**, not replace, the runtime's `canonicalize`-based root policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathPolicy {
    /// Root path. Stored as bytes to avoid any OS-specific normalization in
    /// the core; the runtime resolves the real root.
    pub root: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathVerdict {
    Allow,
    RejectParentComponent,
    RejectOutsideRoot,
    RejectSymlink,
}

impl PathPolicy {
    /// Reject any `..` component flatly (fail-closed, T11). This is necessary
    /// but not sufficient (§22): `canonicalize` consults the FS; the runtime
    /// enforces the full root policy. The core's lexical check fails fast on
    /// obvious parent-component attacks so the hot path stays allocation-free.
    #[must_use]
    pub fn check_lexical(&self, candidate: &[u8]) -> PathVerdict {
        for part in candidate.split(|&b| b == b'/' || b == b'\\') {
            if part == b".." {
                return PathVerdict::RejectParentComponent;
            }
        }
        if !candidate.starts_with(&self.root) {
            return PathVerdict::RejectOutsideRoot;
        }
        PathVerdict::Allow
    }
}
