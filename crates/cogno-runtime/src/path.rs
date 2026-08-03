//! File path policy & root confinement (COGNO-1 V2 §22).
//!
//! All operations are limited to a configured root. Before any write, the
//! runtime resolves the root, verifies the target, rejects unauthorized
//! parent components, applies the symlink policy, writes to a temp file in
//! the **same root**, fsyncs if the contract requires it, replaces
//! atomically when the system allows, recomputes the fingerprint and journals
//! the result. Lexical normalization does not resolve symlinks; security
//! rests on a complete root policy, not on the textual removal of `..`.
//!
//! This is the runtime layer; `cognocore::PathPolicy` is the pure lexical
//! first gate.

/// Resolved target path expressed as bytes to stay OS-agnostic at the core
/// boundary. The runtime converts to `PathBuf` only when actually touching the
/// FS, and only for Phase 5 tool execution (disabled in the MVP).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPath {
    pub root: Vec<u8>,
    pub relative: Vec<u8>,
}

/// Root policy carrying the canonical root and the lexical gate. The runtime
/// is the authority on canonicalization (§22).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootPolicy {
    pub root: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootError {
    OutsideRoot,
    ParentComponent,
    SymlinkNotResolved,
}

impl RootPolicy {
    /// Resolve a candidate lexically. **This is necessary but not sufficient**
    /// (§22): before touching the filesystem the runtime must additionally
    /// `canonicalize` the root and the target and assert containment on the
    /// resolved forms. The lexical pass fails fast on obvious attacks.
    pub fn resolve(&self, candidate: &[u8]) -> Result<ResolvedPath, RootError> {
        let check = cogno_core::PathPolicy {
            root: self.root.clone(),
        };
        match check.check_lexical(candidate) {
            cogno_core::PathVerdict::Allow => Ok(ResolvedPath {
                root: self.root.clone(),
                relative: candidate[self.root.len()..].to_vec(),
            }),
            cogno_core::PathVerdict::RejectParentComponent => Err(RootError::ParentComponent),
            cogno_core::PathVerdict::RejectOutsideRoot => Err(RootError::OutsideRoot),
            cogno_core::PathVerdict::RejectSymlink => Err(RootError::SymlinkNotResolved),
        }
    }
}
