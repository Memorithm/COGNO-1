//! Data classification (COGNO-1 V2 §20).
//!
//! The model receives `Public` and `Internal` by default. `Confidential`
//! requires explicit host authorization. `Secret` is forbidden in prompts,
//! traces, reports, preference events, training sets, error messages,
//! metrics, and exported filenames.
//!
//! The first line of defense is **not to load the secret**. `Drop` is not a
//! proof that sensitive bytes were wiped from every memory copy (compiler
//! copies, swap, crash dumps).

/// Data classification. Ordered so that `Public < Internal < Confidential <
/// Secret` for convenience; the numeric order is not a privilege escalation
/// mechanism — `allowed_for_model_by_default` is the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Secret,
}

impl DataClassification {
    /// Whether this class may be sent to the model without explicit host
    /// authorization. `Confidential` requires host authorization (the core
    /// conservatively returns `false` by default; the runtime layers the host
    /// gate on top). `Secret` is never allowed (§20).
    #[must_use]
    pub const fn allowed_for_model_by_default(self) -> bool {
        matches!(
            self,
            DataClassification::Public | DataClassification::Internal
        )
    }

    #[must_use]
    pub const fn is_secret(self) -> bool {
        matches!(self, DataClassification::Secret)
    }

    #[must_use]
    pub const fn requires_host_authorization(self) -> bool {
        matches!(
            self,
            DataClassification::Confidential | DataClassification::Secret
        )
    }
}
