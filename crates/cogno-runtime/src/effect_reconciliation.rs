//! Fail-closed reconciliation for an ambiguous side effect before re-proposal.
//!
//! RemoteOps owns the durable host receipt and its authentication. This module
//! binds that receipt to the exact COGNO task, workspace, model, proposal and
//! attempt. A confirmed no-effect receipt yields only a digest-only
//! ReproposalCandidate; it is never execution permission.

use crate::runtime::{tool_result_provenance_digests, ToolResultProvenanceDigests};
use cogno_core::{CapabilityClass, TaskCapabilityScope, ToolProposalView};
use sha2::{Digest, Sha256};

pub const AXCOG4_EFFECT_RECEIPT_SCHEMA_V1: &str = "remoteops.effect-reconciliation/v1";
const MAX_REFERENCE_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideEffectResolution {
    Unknown,
    ConfirmedNoEffect,
    ConfirmedEffect,
}

/// Host-authenticated RemoteOps evidence mapped into the COGNO reconciliation
/// contract. Hash fields are SHA-256 values; raw task and argument values are
/// never copied into this receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteOpsEffectReceipt {
    pub schema_version: String,
    pub attempt_id_sha256: [u8; 32],
    pub workspace_snapshot_sha256: [u8; 32],
    pub model_artifact_sha256: [u8; 32],
    pub task_binding_sha256: [u8; 32],
    pub proposal_sha256: [u8; 32],
    pub remote_execution_id_sha256: [u8; 32],
    pub host_identity_sha256: [u8; 32],
    pub evidence_sha256: [u8; 32],
    pub resolution: SideEffectResolution,
}

impl RemoteOpsEffectReceipt {
    /// Build the COGNO binding fields for an adapter-owned receipt.
    ///
    /// This constructor does not authenticate host evidence. The caller must
    /// obtain and verify a durable RemoteOps receipt before asking COGNO to
    /// reconcile it.
    pub fn new(
        attempt_id: &str,
        scope: &TaskCapabilityScope<'_>,
        proposal: &ToolProposalView<'_>,
        remote_execution_id: &str,
        host_identity: &str,
        evidence_sha256: [u8; 32],
        resolution: SideEffectResolution,
    ) -> Result<Self, ReceiptBuildError> {
        let context = effect_context_digests(scope, proposal)
            .map_err(ReceiptBuildError::InvalidEffectContext)?;
        if evidence_sha256 == [0; 32] {
            return Err(ReceiptBuildError::MissingEvidenceDigest);
        }
        Ok(Self {
            schema_version: AXCOG4_EFFECT_RECEIPT_SCHEMA_V1.to_owned(),
            attempt_id_sha256: hash_reference(b"cogno-1:axcog4:attempt-id:v1\0", attempt_id)
                .ok_or(ReceiptBuildError::InvalidReference {
                    field: "attempt_id",
                })?,
            workspace_snapshot_sha256: context.workspace_snapshot_sha256,
            model_artifact_sha256: context.model_artifact_sha256,
            task_binding_sha256: context.task_binding_sha256,
            proposal_sha256: context.result_sha256,
            remote_execution_id_sha256: hash_reference(
                b"remoteops:execution-id:v1\0",
                remote_execution_id,
            )
            .ok_or(ReceiptBuildError::InvalidReference {
                field: "remote_execution_id",
            })?,
            host_identity_sha256: hash_reference(b"remoteops:host-identity:v1\0", host_identity)
                .ok_or(ReceiptBuildError::InvalidReference {
                    field: "host_identity",
                })?,
            evidence_sha256,
            resolution,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptBuildError {
    InvalidReference { field: &'static str },
    MissingEvidenceDigest,
    InvalidEffectContext(EffectReconciliationReason),
}

impl std::fmt::Display for ReceiptBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid AXCOG4 effect receipt: {self:?}")
    }
}

impl std::error::Error for ReceiptBuildError {}

/// Trusted host adapter boundary. An implementation must authenticate the
/// durable RemoteOps receipt; structural validity alone is not proof.
pub trait RemoteOpsEffectReceiptVerifier {
    fn verify_authenticated_receipt(&self, receipt: &RemoteOpsEffectReceipt) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectReconciliationReason {
    InvalidAttemptId,
    InvalidTaskScope,
    OversizedProposal,
    ProposalOutsideScope,
    NonEffectProposal,
    MissingProvenance,
    UnsupportedReceiptSchema,
    AttemptMismatch,
    WorkspaceMismatch,
    ModelMismatch,
    TaskBindingMismatch,
    ProposalMismatch,
    MissingRemoteExecutionIdentity,
    MissingHostIdentity,
    MissingEvidenceDigest,
    UnauthenticatedReceipt,
    OutcomeUnknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReproposalCandidate {
    attempt_id_sha256: [u8; 32],
    proposal_sha256: [u8; 32],
    evidence_sha256: [u8; 32],
}

impl ReproposalCandidate {
    #[must_use]
    pub const fn attempt_id_sha256(&self) -> [u8; 32] {
        self.attempt_id_sha256
    }

    #[must_use]
    pub const fn proposal_sha256(&self) -> [u8; 32] {
        self.proposal_sha256
    }

    #[must_use]
    pub const fn evidence_sha256(&self) -> [u8; 32] {
        self.evidence_sha256
    }
}

/// Result of reconciliation. A candidate only permits the caller to start a
/// fresh deterministic authorization cycle; it does not authorize execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectReconciliationDecision {
    NeedExternalReconciliation(EffectReconciliationReason),
    RetryBlockedByConfirmedEffect,
    ReproposalCandidate(ReproposalCandidate),
}

/// Bind a durable RemoteOps receipt to the current task and proposal before a
/// caller may consider another authorization attempt.
#[must_use]
pub fn reconcile_ambiguous_effect(
    attempt_id: &str,
    scope: &TaskCapabilityScope<'_>,
    proposal: &ToolProposalView<'_>,
    receipt: &RemoteOpsEffectReceipt,
    verifier: &impl RemoteOpsEffectReceiptVerifier,
) -> EffectReconciliationDecision {
    let Some(attempt_id_sha256) = hash_reference(b"cogno-1:axcog4:attempt-id:v1\0", attempt_id)
    else {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::InvalidAttemptId,
        );
    };
    if receipt.schema_version != AXCOG4_EFFECT_RECEIPT_SCHEMA_V1 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::UnsupportedReceiptSchema,
        );
    }
    if receipt.attempt_id_sha256 != attempt_id_sha256 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::AttemptMismatch,
        );
    }
    let context = match effect_context_digests(scope, proposal) {
        Ok(context) => context,
        Err(reason) => {
            return EffectReconciliationDecision::NeedExternalReconciliation(reason);
        }
    };
    if receipt.workspace_snapshot_sha256 != context.workspace_snapshot_sha256 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::WorkspaceMismatch,
        );
    }
    if receipt.model_artifact_sha256 != context.model_artifact_sha256 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::ModelMismatch,
        );
    }
    if receipt.task_binding_sha256 != context.task_binding_sha256 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::TaskBindingMismatch,
        );
    }
    if receipt.proposal_sha256 != context.result_sha256 {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::ProposalMismatch,
        );
    }
    if receipt.remote_execution_id_sha256 == [0; 32] {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::MissingRemoteExecutionIdentity,
        );
    }
    if receipt.host_identity_sha256 == [0; 32] {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::MissingHostIdentity,
        );
    }
    if receipt.evidence_sha256 == [0; 32] {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::MissingEvidenceDigest,
        );
    }
    if !verifier.verify_authenticated_receipt(receipt) {
        return EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::UnauthenticatedReceipt,
        );
    }
    match receipt.resolution {
        SideEffectResolution::Unknown => EffectReconciliationDecision::NeedExternalReconciliation(
            EffectReconciliationReason::OutcomeUnknown,
        ),
        SideEffectResolution::ConfirmedEffect => {
            EffectReconciliationDecision::RetryBlockedByConfirmedEffect
        }
        SideEffectResolution::ConfirmedNoEffect => {
            EffectReconciliationDecision::ReproposalCandidate(ReproposalCandidate {
                attempt_id_sha256,
                proposal_sha256: context.result_sha256,
                evidence_sha256: receipt.evidence_sha256,
            })
        }
    }
}

fn effect_context_digests(
    scope: &TaskCapabilityScope<'_>,
    proposal: &ToolProposalView<'_>,
) -> Result<ToolResultProvenanceDigests, EffectReconciliationReason> {
    if scope.validate().is_err() {
        return Err(EffectReconciliationReason::InvalidTaskScope);
    }
    if !cogno_core::tool_proposal_within_limits(proposal) {
        return Err(EffectReconciliationReason::OversizedProposal);
    }
    if !scope.permits(proposal) {
        return Err(EffectReconciliationReason::ProposalOutsideScope);
    }
    if scope.capability_class(proposal.capability_id) != Some(CapabilityClass::Effect) {
        return Err(EffectReconciliationReason::NonEffectProposal);
    }
    tool_result_provenance_digests(scope, proposal)
        .ok_or(EffectReconciliationReason::MissingProvenance)
}

fn hash_reference(domain: &[u8], value: &str) -> Option<[u8; 32]> {
    if value.is_empty()
        || value.len() > MAX_REFERENCE_BYTES
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return None;
    }
    let length = u64::try_from(value.len()).ok()?;
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(length.to_le_bytes());
    hash.update(value.as_bytes());
    let digest = hash.finalize();
    let mut output = [0; 32];
    output.copy_from_slice(&digest);
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{
        CapabilityClassification, CapabilityId, ReasonCode, TaskExecutionProvenance, ToolId,
        TypedArgument, WorkspaceSnapshotSha256,
    };

    static TOOLS: &[ToolId] = &[ToolId(7)];
    static EFFECTS: &[cogno_core::CapabilityClassification] = &[CapabilityClassification::new(
        CapabilityId(9),
        CapabilityClass::Effect,
    )];
    static READS: &[CapabilityClassification] = &[CapabilityClassification::new(
        CapabilityId(9),
        CapabilityClass::Read,
    )];

    struct AcceptReceipt;

    impl RemoteOpsEffectReceiptVerifier for AcceptReceipt {
        fn verify_authenticated_receipt(&self, _: &RemoteOpsEffectReceipt) -> bool {
            true
        }
    }

    struct RejectReceipt;

    impl RemoteOpsEffectReceiptVerifier for RejectReceipt {
        fn verify_authenticated_receipt(&self, _: &RemoteOpsEffectReceipt) -> bool {
            false
        }
    }

    fn scope<'a>(task_binding: &'a [u8], workspace: u8, model: u8) -> TaskCapabilityScope<'a> {
        TaskCapabilityScope::new_with_provenance(
            task_binding,
            TOOLS,
            EFFECTS,
            TaskExecutionProvenance::new(WorkspaceSnapshotSha256([workspace; 32]), [model; 32]),
        )
    }

    fn proposal<'a>(arguments: &'a [TypedArgument<'a>]) -> ToolProposalView<'a> {
        ToolProposalView {
            tool_id: ToolId(7),
            capability_id: CapabilityId(9),
            arguments,
            justification_code: ReasonCode(3),
        }
    }

    fn receipt(
        scope: &TaskCapabilityScope<'_>,
        proposal: &ToolProposalView<'_>,
        resolution: SideEffectResolution,
    ) -> RemoteOpsEffectReceipt {
        RemoteOpsEffectReceipt::new(
            "attempt-1",
            scope,
            proposal,
            "remote-execution-7",
            "thor-01",
            [0x55; 32],
            resolution,
        )
        .expect("valid receipt")
    }

    #[test]
    fn confirmed_no_effect_yields_only_a_digest_only_reproposal_candidate() {
        let scope = scope(b"task-1", 0x11, 0x22);
        let arguments = [TypedArgument::Bytes(b"effect-payload")];
        let proposal = proposal(&arguments);
        let receipt = receipt(&scope, &proposal, SideEffectResolution::ConfirmedNoEffect);
        let decision =
            reconcile_ambiguous_effect("attempt-1", &scope, &proposal, &receipt, &AcceptReceipt);
        let EffectReconciliationDecision::ReproposalCandidate(candidate) = decision else {
            panic!("confirmed no-effect receipt should yield a candidate");
        };
        assert_ne!(candidate.attempt_id_sha256(), [0; 32]);
        assert_ne!(candidate.proposal_sha256(), [0; 32]);
        assert_eq!(candidate.evidence_sha256(), [0x55; 32]);
    }

    #[test]
    fn unknown_and_confirmed_effect_outcomes_never_yield_candidates() {
        let scope = scope(b"task-1", 0x11, 0x22);
        let arguments = [TypedArgument::Bytes(b"effect-payload")];
        let proposal = proposal(&arguments);
        let unknown = receipt(&scope, &proposal, SideEffectResolution::Unknown);
        assert_eq!(
            reconcile_ambiguous_effect("attempt-1", &scope, &proposal, &unknown, &AcceptReceipt),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::OutcomeUnknown
            )
        );
        let applied = receipt(&scope, &proposal, SideEffectResolution::ConfirmedEffect);
        assert_eq!(
            reconcile_ambiguous_effect("attempt-1", &scope, &proposal, &applied, &AcceptReceipt),
            EffectReconciliationDecision::RetryBlockedByConfirmedEffect
        );
    }

    #[test]
    fn unauthenticated_receipts_fail_closed() {
        let scope = scope(b"task-1", 0x11, 0x22);
        let arguments = [TypedArgument::Bytes(b"effect-payload")];
        let proposal = proposal(&arguments);
        let receipt = receipt(&scope, &proposal, SideEffectResolution::ConfirmedNoEffect);
        assert_eq!(
            reconcile_ambiguous_effect("attempt-1", &scope, &proposal, &receipt, &RejectReceipt),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::UnauthenticatedReceipt
            )
        );
    }

    #[test]
    fn changed_attempt_task_workspace_model_or_proposal_is_not_replayable() {
        let original_scope = scope(b"task-1", 0x11, 0x22);
        let original_arguments = [TypedArgument::Bytes(b"effect-payload")];
        let original_proposal = proposal(&original_arguments);
        let receipt = receipt(
            &original_scope,
            &original_proposal,
            SideEffectResolution::ConfirmedNoEffect,
        );

        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-2",
                &original_scope,
                &original_proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::AttemptMismatch
            )
        );

        let changed_task = scope(b"task-2", 0x11, 0x22);
        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-1",
                &changed_task,
                &original_proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::TaskBindingMismatch
            )
        );

        let changed_workspace = scope(b"task-1", 0x33, 0x22);
        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-1",
                &changed_workspace,
                &original_proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::WorkspaceMismatch
            )
        );

        let changed_model = scope(b"task-1", 0x11, 0x44);
        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-1",
                &changed_model,
                &original_proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::ModelMismatch
            )
        );

        let changed_arguments = [TypedArgument::Bytes(b"different-effect")];
        let changed_proposal = proposal(&changed_arguments);
        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-1",
                &original_scope,
                &changed_proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::ProposalMismatch
            )
        );
    }

    #[test]
    fn malformed_attempt_and_non_effect_proposals_fail_closed() {
        let scope = scope(b"task-1", 0x11, 0x22);
        let arguments = [TypedArgument::Bytes(b"effect-payload")];
        let proposal = proposal(&arguments);
        let receipt = receipt(&scope, &proposal, SideEffectResolution::ConfirmedNoEffect);
        assert_eq!(
            reconcile_ambiguous_effect("", &scope, &proposal, &receipt, &AcceptReceipt),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::InvalidAttemptId
            )
        );

        let read_scope = TaskCapabilityScope::new_with_provenance(
            b"task-1",
            TOOLS,
            READS,
            TaskExecutionProvenance::new(WorkspaceSnapshotSha256([0x11; 32]), [0x22; 32]),
        );
        assert_eq!(
            reconcile_ambiguous_effect(
                "attempt-1",
                &read_scope,
                &proposal,
                &receipt,
                &AcceptReceipt
            ),
            EffectReconciliationDecision::NeedExternalReconciliation(
                EffectReconciliationReason::NonEffectProposal
            )
        );
    }
}
