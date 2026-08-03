//! Typed input origin and independent trust class (COGNO-1 V2 §6).
//!
//! A character string never implicitly carries its privilege level. The
//! runtime assigns an `InputOrigin` and an independent `TrustClass` to every
//! input. System instructions, retrieved documents, tool results and model
//! outputs stay in distinct fields up to inference; it is forbidden to
//! concatenate all sources into one untyped prompt. A structural delimiter
//! controlled by the runtime precedes external data, but is **not** considered
//! sufficient protection on its own (S3).

/// Typed origin of an input. Drives the default `TrustClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputOrigin {
    SystemPolicy,
    ExplicitUserInstruction,
    UserData,
    RetrievedDocument,
    ToolOutput,
    ModelOutput,
    ImportedProfile,
    TrainingCorpus,
}

/// Independent trust class. Declared separately from `InputOrigin` so that a
/// string cannot smuggle its privilege. Higher `trust_rank()` is more trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrustClass {
    TrustedPolicy,
    AuthenticatedUser,
    ValidatedLocalData,
    UntrustedExternalData,
    UntrustedModelData,
}

impl TrustClass {
    /// Ordering of trust: higher is more trusted. Used only as a comparable
    /// hint; the lexicographic decision (§8) never lets trust raise a
    /// privileged decision past a hard constraint.
    #[must_use]
    pub const fn trust_rank(self) -> u8 {
        match self {
            TrustClass::TrustedPolicy => 100,
            TrustClass::AuthenticatedUser => 80,
            TrustClass::ValidatedLocalData => 60,
            TrustClass::UntrustedExternalData => 20,
            TrustClass::UntrustedModelData => 10,
        }
    }

    #[must_use]
    pub const fn is_trusted(self) -> bool {
        matches!(
            self,
            TrustClass::TrustedPolicy
                | TrustClass::AuthenticatedUser
                | TrustClass::ValidatedLocalData
        )
    }
}

impl InputOrigin {
    /// Default trust for a given origin, per the §6 separation rule.
    ///
    /// `UserData` is considered `ValidatedLocalData` because the host has
    /// authenticated the user and validated the envelope before the core sees
    /// it; the *contents* remain subject to symbolic validation.
    #[must_use]
    pub const fn default_trust(self) -> TrustClass {
        match self {
            InputOrigin::SystemPolicy => TrustClass::TrustedPolicy,
            InputOrigin::ExplicitUserInstruction => TrustClass::AuthenticatedUser,
            InputOrigin::UserData => TrustClass::ValidatedLocalData,
            InputOrigin::RetrievedDocument => TrustClass::UntrustedExternalData,
            InputOrigin::ToolOutput => TrustClass::UntrustedExternalData,
            InputOrigin::ModelOutput => TrustClass::UntrustedModelData,
            InputOrigin::ImportedProfile => TrustClass::UntrustedExternalData,
            InputOrigin::TrainingCorpus => TrustClass::UntrustedExternalData,
        }
    }
}
