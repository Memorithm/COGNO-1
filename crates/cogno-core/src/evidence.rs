//! Evidence provenance, rule lifecycle and a minimal `Evidence` record
//! (COGNO-1 V2 §9).
//!
//! Not all evidence has the same authority. The default trust order is:
//!
//! ```text
//! signed policy
//!   > explicit user instruction
//!   > formal validation
//!   > repeated user edit
//!   > implicit acceptance
//!   > tool observation
//!   > imported profile
//!   > model inference
//! ```
//!
//! Constraints enforced here (partial; the runtime adds the rest):
//! - a model inference never creates a hard rule;
//! - a single acceptance does not make a stable preference;
//! - an inferred rule starts in quarantine;
//! - contradictory preferences are kept (not silently replaced);
//! - every evidence has an id and a fingerprint.

/// Origin of a piece of evidence. Drives `trust_rank()` and `can_create_hard_rule()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceOrigin {
    ExplicitUserStatement,
    ExplicitUserApproval,
    UserEdit,
    UserRejection,
    TestResult,
    FormalValidator,
    ToolObservation,
    ImportedProfile,
    ModelInference,
}

/// Lifecycle of a rule. A rule moves `Candidate -> Active` only by a
/// deterministic policy. Contradictory rules become `Conflicted`, never
/// silently dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleState {
    Quarantined,
    Candidate,
    Active,
    Conflicted,
    Disabled,
    Revoked,
}

/// Opaque, stable evidence identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EvidenceId(pub u64);

impl EvidenceId {
    #[must_use]
    pub const fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

/// A single evidence record. The fingerprint is a 32-byte digest over the
///payload and provenance, computed by the runtime; the core only carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evidence {
    pub id: EvidenceId,
    pub origin: EvidenceOrigin,
    pub fingerprint: [u8; 32],
}

impl EvidenceOrigin {
    /// Higher is more authoritative, per the §9 default trust order.
    #[must_use]
    pub const fn trust_rank(self) -> u8 {
        match self {
            EvidenceOrigin::FormalValidator => 95,
            EvidenceOrigin::ExplicitUserStatement => 85,
            EvidenceOrigin::ExplicitUserApproval => 75,
            EvidenceOrigin::UserEdit => 65,
            EvidenceOrigin::UserRejection => 60,
            EvidenceOrigin::TestResult => 55,
            EvidenceOrigin::ToolObservation => 40,
            EvidenceOrigin::ImportedProfile => 25,
            EvidenceOrigin::ModelInference => 10,
        }
    }

    /// A hard safety rule may never be minted from model inference alone, nor
    /// from imported profiles or tool observations (§9 + §2). Only formal
    /// validators and explicit user statements/approvals can create a hard
    /// rule, and even then through the deterministic core.
    #[must_use]
    pub const fn can_create_hard_rule(self) -> bool {
        matches!(
            self,
            EvidenceOrigin::FormalValidator
                | EvidenceOrigin::ExplicitUserStatement
                | EvidenceOrigin::ExplicitUserApproval
        )
    }

    #[must_use]
    pub const fn is_model_inference(self) -> bool {
        matches!(self, EvidenceOrigin::ModelInference)
    }
}
