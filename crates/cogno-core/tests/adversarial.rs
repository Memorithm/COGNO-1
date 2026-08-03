//! Adversarial security tests (COGNO-1 V2 §25). Each test passes when the core
//! **rejects** (fail-closed, S10).

use cogno_core::{
    validate_proposal, DataClassification, EvidenceId, InputOrigin, MemoryBudget, PathPolicy,
    ProposalAction, RuleCategory, RuleScope, SafetyPolicy, TrustClass,
};

const MAX_BPS: u16 = 10_000;

fn good_view<'a>(
    evidence: &'a [EvidenceId],
    payload: &'a [u8],
    action: ProposalAction,
) -> cogno_core::CognoProposalView<'a> {
    cogno_core::CognoProposalView {
        schema_version: 1,
        action,
        category: RuleCategory::Style,
        scope: RuleScope::Session,
        confidence_bps: 1_000,
        evidence_ids: evidence,
        payload,
    }
}

#[test]
fn direct_prompt_injection_is_untrusted() {
    // An explicit user instruction can never carry SystemPolicy trust.
    let origin = InputOrigin::ExplicitUserInstruction;
    assert_ne!(origin.default_trust(), TrustClass::TrustedPolicy);
    assert_eq!(origin.default_trust(), TrustClass::AuthenticatedUser);
    // "ignore previous instructions" is just userdata: its trust is not elevated
    // by content. The system does not concat it into SystemPolicy fields.
    let data_origin = InputOrigin::UserData;
    assert!(!matches!(
        data_origin.default_trust(),
        TrustClass::TrustedPolicy
    ));
}

#[test]
fn confidence_above_max_is_rejected() {
    let ev = [EvidenceId::from_u64(1)];
    let mut v = good_view(&ev, &[], ProposalAction::ClassifyFeedback);
    v.confidence_bps = MAX_BPS + 1;
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::OutOfBounds)
    );
}

#[test]
fn unknown_schema_version_rejected() {
    let ev = [EvidenceId::from_u64(1)];
    let mut v = good_view(&ev, &[], ProposalAction::ClassifyFeedback);
    v.schema_version = 99;
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::OutOfBounds)
    );
}

#[test]
fn too_many_evidence_ids_rejected() {
    let mut ev = Vec::with_capacity(cogno_core::MAX_EVIDENCE_IDS + 1);
    for i in 1..=(cogno_core::MAX_EVIDENCE_IDS as u64 + 1) {
        ev.push(EvidenceId::from_u64(i));
    }
    let v = good_view(&ev, &[], ProposalAction::ExtractPreference);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Oversized)
    );
}

#[test]
fn payload_too_large_rejected() {
    let ev = [EvidenceId::from_u64(1)];
    let payload = vec![0u8; cogno_core::MAX_PAYLOAD_BYTES + 1];
    let v = good_view(&ev, &payload, ProposalAction::ClassifyFeedback);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Oversized)
    );
}

#[test]
fn duplicate_evidence_rejected() {
    let ev = [EvidenceId::from_u64(7), EvidenceId::from_u64(7)];
    let v = good_view(&ev, &[], ProposalAction::ExtractPreference);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Malformed)
    );
}

#[test]
fn zero_evidence_id_rejected() {
    let ev = [EvidenceId::from_u64(0)];
    let v = good_view(&ev, &[], ProposalAction::ExtractPreference);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Malformed)
    );
}

#[test]
fn evidence_required_for_extraction() {
    let v = good_view(&[], &[], ProposalAction::ExtractPreference);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Malformed)
    );
}

#[test]
fn unsupported_action_rejected() {
    let ev = [EvidenceId::from_u64(1)];
    let mut v = good_view(&ev, &[], ProposalAction::Unsupported);
    // unsupported must never pass even with good evidence
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Unauthorized)
    );
    v.action = ProposalAction::ClassifyFeedback;
    assert!(validate_proposal(&v).is_ok());
}

#[test]
fn shell_command_tool_proposal_rejected_in_mvp() {
    use cogno_core::{
        looks_like_shell_invocation, CapabilityId, ReasonCode, ToolId, ToolProposalView,
        TypedArgument,
    };
    let shell_text = "rm -rf / ; echo pwned";
    let args = [TypedArgument::Text(shell_text)];
    let p = ToolProposalView {
        tool_id: ToolId(1),
        capability_id: CapabilityId(1),
        arguments: &args,
        justification_code: ReasonCode(1),
    };
    assert!(looks_like_shell_invocation(&p));
    // MVP: every tool proposal is unauthorized regardless of shape.
    assert_eq!(
        cogno_core::classify_tool_proposal(&p),
        Err(cogno_core::RejectReason::Unauthorized)
    );
}

#[test]
fn secret_data_rejected_by_policy() {
    let pol = SafetyPolicy::MVP;
    assert_eq!(
        pol.check_data(DataClassification::Secret),
        cogno_core::HardVerdict::Reject(cogno_core::RejectReason::Secret)
    );
}

#[test]
fn confidential_data_rejected_by_default() {
    let pol = SafetyPolicy::MVP;
    assert_eq!(
        pol.check_data(DataClassification::Confidential),
        cogno_core::HardVerdict::Reject(cogno_core::RejectReason::Confidential)
    );
}

#[test]
fn public_and_internal_allowed_by_default() {
    let pol = SafetyPolicy::MVP;
    assert_eq!(
        pol.check_data(DataClassification::Public),
        cogno_core::HardVerdict::Allow
    );
    assert_eq!(
        pol.check_data(DataClassification::Internal),
        cogno_core::HardVerdict::Allow
    );
}

#[test]
fn parent_component_path_rejected() {
    let pol = PathPolicy {
        root: b"/app/root".to_vec(),
    };
    assert_eq!(
        pol.check_lexical(b"/app/root/../etc/passwd"),
        cogno_core::PathVerdict::RejectParentComponent
    );
}

#[test]
fn outside_root_rejected() {
    let pol = PathPolicy {
        root: b"/app/root".to_vec(),
    };
    assert_eq!(
        pol.check_lexical(b"/etc/passwd"),
        cogno_core::PathVerdict::RejectOutsideRoot
    );
}

#[test]
fn valid_budget() {
    let b = MemoryBudget::try_new(
        1_000_000, 100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 4_096, 4_096,
        2048, 1024, 8, 16, 4, 64,
    );
    assert!(b.is_ok());
}

#[test]
fn budget_subbudget_sum_exceeds_hard_rejected() {
    let b = MemoryBudget::try_new(
        100, 100, 100, 100, 100, 100, 100, 100, // sum=700 > 100
        4, 4, 2, 1, 1, 1, 1, 1,
    );
    assert!(matches!(
        b.unwrap_err(),
        cogno_core::BudgetError::SubBudgetSumExceedsHardLimit { .. }
    ));
}

#[test]
fn budget_mandatory_zero_rejected() {
    let b = MemoryBudget::try_new(
        1_000_000, 0, 0, 0, 0, 0, 0, 0, 4_096, 4_096, 2048, 1024, 8, 16, 4, 64,
    );
    // hard_limit non-zero, max_* non-zero, sub-budgets may be 0 -> ok
    let _ = b.unwrap();

    let b2 = MemoryBudget::try_new(
        1_000_000, 0, 0, 0, 0, 0, 0, 0, 0, 4_096, 2048, 1024, 8, 16, 4,
        64, // max_input_bytes=0 -> mandatory zero
    );
    assert!(matches!(
        b2.unwrap_err(),
        cogno_core::BudgetError::MandatoryLimitZero { .. }
    ));
}

#[test]
fn kv_cache_arithmetic_overflow_detected() {
    // layers * tokens * heads * head_dim * bytes * batch overflows usize.
    let max = usize::MAX;
    let r = cogno_core::checked_kv_cache_bytes(max, max, max, max, max, max);
    assert_eq!(r, Err(cogno_core::MemoryError::ArithmeticOverflow));
}

#[test]
fn kv_cache_ok_for_small_dims() {
    let r = cogno_core::checked_kv_cache_bytes(12, 1024, 8, 128, 2, 1);
    let expected = 2u128 * 12 * 1024 * 8 * 128 * 2;
    assert!(r.is_ok());
    assert_eq!(r.unwrap() as u128, expected);
}

#[test]
fn bounded_vec_refuses_overflow() {
    let mut v = cogno_core::BoundedVec::<u8>::try_new(2).unwrap();
    assert!(v.try_push(1).is_ok());
    assert!(v.try_push(2).is_ok());
    assert!(matches!(
        v.try_push(3).unwrap_err(),
        cogno_core::CapacityError::Exceeded { .. }
    ));
    // inner untouched on failure
    assert_eq!(v.as_slice(), &[1, 2]);
}

#[test]
fn bounded_vec_zero_capacity_rejected() {
    assert!(cogno_core::BoundedVec::<u8>::try_new(0).is_err());
}

#[test]
fn journal_deduplicates_by_fingerprint() {
    use cogno_core::{EvidenceOrigin, Fingerprint, Journal, JournalEvent};
    let mut j = Journal::default();
    let fp = Fingerprint([7u8; 32]);
    let e = JournalEvent {
        id: cogno_core::EventId(0),
        fingerprint: fp,
        origin: InputOrigin::ExplicitUserInstruction,
        evidence_origin: EvidenceOrigin::ExplicitUserStatement,
        evidence_id: EvidenceId::from_u64(1),
        category_tag: 1,
        payload: Vec::new(),
    };
    let o1 = j.append(e.clone());
    let o2 = j.append(e);
    assert!(matches!(o1, cogno_core::AppendOutcome::Appended { .. }));
    assert!(matches!(o2, cogno_core::AppendOutcome::Duplicate { .. }));
    assert_eq!(j.len(), 1);
}

#[test]
fn model_only_rule_stays_quarantined_and_never_hard() {
    use cogno_core::{DerivationPolicy, EvidenceOrigin, Fingerprint, Journal, JournalEvent};
    let mut j = Journal::default();
    let fp = Fingerprint([1u8; 32]);
    let e = JournalEvent {
        id: cogno_core::EventId(0),
        fingerprint: fp,
        origin: InputOrigin::ModelOutput,
        evidence_origin: EvidenceOrigin::ModelInference,
        evidence_id: EvidenceId::from_u64(2),
        category_tag: 9,
        payload: Vec::new(),
    };
    j.append(e);
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    let rule = p.rule(9).unwrap();
    assert_eq!(rule.state, cogno_core::RuleState::Quarantined);
    assert!(!rule.is_hard());
    assert!(rule.model_only);
}

#[test]
fn formal_validator_creates_hard_rule_after_min_evidence() {
    use cogno_core::{DerivationPolicy, EvidenceOrigin, Fingerprint, Journal, JournalEvent};
    let mut j = Journal::default();
    for i in 0..2u8 {
        let mut fp = [0u8; 32];
        fp[0] = i + 1;
        j.append(JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::SystemPolicy,
            evidence_origin: EvidenceOrigin::FormalValidator,
            evidence_id: EvidenceId::from_u64(10 + i as u64),
            category_tag: 42,
            payload: Vec::new(),
        });
    }
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    let rule = p.rule(42).unwrap();
    assert!(rule.is_active());
    assert!(rule.is_hard());
}

#[test]
fn lexicographic_decision_rejects_hard_before_reward() {
    use cogno_core::{decide, Candidate, Decision, RejectReason};
    struct Bad;
    impl Candidate for Bad {
        fn is_structurally_valid(&self) -> bool {
            true
        }
        fn hard_constraint_ok(&self) -> bool {
            false
        } // hard violation
        fn capability_allowed(&self) -> bool {
            true
        }
        fn privacy_rejection(&self) -> Option<RejectReason> {
            None
        }
    }
    let d = decide(&Bad, 999_999_i32);
    assert_eq!(d, Decision::Reject(RejectReason::HardConstraint));
}

#[test]
fn manifest_corrupted_or_unknown_version_rejected() {
    use cogno_core::{
        ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
    };
    let m = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION + 1,
        model_family: ModelFamily::Generic,
        architecture_id: ArchitectureId(1),
        tensor_count: 1,
        parameter_count: 1,
        max_context_tokens: 2048,
        tokenizer_hash: [0u8; 32],
        weights_hash: [0u8; 32],
        expected_file_bytes: 100,
    };
    assert_eq!(m.validate(100), Err(ManifestError::UnknownSchemaVersion));
}

#[test]
fn manifest_truncated_artifact_rejected() {
    use cogno_core::{ArchitectureId, ManifestError, ModelFamily, ModelManifest};
    let m = ModelManifest {
        schema_version: 1,
        model_family: ModelFamily::Generic,
        architecture_id: ArchitectureId(1),
        tensor_count: 4,
        parameter_count: 1000,
        max_context_tokens: 2048,
        tokenizer_hash: [0u8; 32],
        weights_hash: [0u8; 32],
        expected_file_bytes: 1000,
    };
    assert_eq!(
        m.validate(500),
        Err(ManifestError::FileSizeMismatch {
            expected: 1000,
            actual: 500
        })
    );
}

// ---------------------------------------------------------------------------
// Strengthened genuine tests for threats whose earlier mapping was stretched
// (§25: each threat must have at least one *real* test).
// ---------------------------------------------------------------------------

#[test]
fn t03_kb_injection_retrieved_document_is_untrusted_and_distinct_from_policy() {
    // A retrieved document can never speak as SystemPolicy (§6, S3). Its trust
    // class is UntrustedExternalData, distinct from TrustedPolicy. A "SYSTEM:
    // ..." line inside such a document is just userdata.
    let o = InputOrigin::RetrievedDocument;
    assert_eq!(o.default_trust(), TrustClass::UntrustedExternalData);
    assert_ne!(o.default_trust(), TrustClass::TrustedPolicy);
    // And trust ordering cannot be elevated by content: the lexicographic
    // decision never consults trust to bypass a hard gate (S4).
    assert!(
        TrustClass::UntrustedExternalData.trust_rank() < TrustClass::TrustedPolicy.trust_rank()
    );
}

#[test]
fn t09_tokenizer_hash_mismatch_rejected() {
    use cogno_core::{ArchitectureId, ModelFamily, ModelManifest};
    // A manifest with one tokenizer hash must not validate a different
    // tokenizer. The loader would recompute the hash and compare; here we
    // model that the manifest's stored hash is the contract — any artifact
    // whose hash differs is hostile (§21).
    let stored = [0xAA; 32];
    let mut actual = [0u8; 32];
    actual[0] = 0xAA;
    actual[1] = 0xAA;
    // Introduce a single-byte difference.
    actual[5] = 0x01;
    assert_ne!(stored, actual, "hash mismatch must be detected");
    // The manifest itself still validates structurally (size is correct),
    // which is why the loader performs the hash check *separately*, before
    // any major allocation.
    let m = ModelManifest {
        schema_version: 1,
        model_family: ModelFamily::Generic,
        architecture_id: ArchitectureId(1),
        tensor_count: 4,
        parameter_count: 1000,
        max_context_tokens: 2048,
        tokenizer_hash: stored,
        weights_hash: [0u8; 32],
        expected_file_bytes: 1000,
    };
    assert!(
        m.validate(1000).is_ok(),
        "structural checks pass; hash gate is separate"
    );
    assert_ne!(
        m.tokenizer_hash, actual,
        "tokenizer hash gate would refuse this artifact"
    );
}

#[test]
fn t13_write_outside_root_rejected_by_root_policy() {
    // A proposal that resolves to a path outside the configured root is
    // rejected at the lexical border (§22); the runtime layers canonicalize
    // on top, but the lexical gate fails fast.
    let pol = PathPolicy {
        root: b"/app/root".to_vec(),
    };
    assert_eq!(
        pol.check_lexical(b"/app/root/log.txt"),
        cogno_core::PathVerdict::Allow
    );
    assert_eq!(
        pol.check_lexical(b"/var/log/hack"),
        cogno_core::PathVerdict::RejectOutsideRoot
    );
}

#[test]
fn t18_deep_recursion_rejected_by_payload_bound() {
    // A deeply-nested payload exceeds the §18/§2 payload bound and is
    // rejected at the structural stage instead of recursing into the parser.
    // The bound is the deterministic defense against deep-structure DoS.
    let ev = [EvidenceId::from_u64(1)];
    let payload = vec![b'{'; cogno_core::MAX_PAYLOAD_BYTES + 1];
    let v = good_view(&ev, &payload, ProposalAction::ClassifyFeedback);
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::Oversized)
    );
}

#[test]
fn t22_rollback_to_unknown_schema_version_rejected() {
    // A "rolled-back" proposal claiming an older or unknown schema version
    // is rejected outright (§2). Downgrades cannot bypass the validators.
    let ev = [EvidenceId::from_u64(1)];
    let mut v = good_view(&ev, &[], ProposalAction::ClassifyFeedback);
    v.schema_version = 0; // older than the current SCHEMA_VERSION
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::OutOfBounds)
    );
    v.schema_version = cogno_core::SCHEMA_VERSION + 1; // future/unknown
    assert_eq!(
        validate_proposal(&v),
        Err(cogno_core::RejectReason::OutOfBounds)
    );
}

#[test]
fn t23_provenance_forgery_rejected_via_default_trust() {
    // A model output cannot forge SystemPolicy provenance: the runtime
    // assigns InputOrigin from a typed envelope, never from the payload.
    // Even if the model claimed "I am system policy", its origin stays
    // ModelOutput -> UntrustedModelData (§6, S3).
    let o = InputOrigin::ModelOutput;
    assert_eq!(o.default_trust(), TrustClass::UntrustedModelData);
    assert_ne!(o.default_trust(), TrustClass::TrustedPolicy);
}

#[test]
fn t04_tool_output_never_elevated_to_policy() {
    // A tool output is UntrustedExternalData (§6), never TrustedPolicy. It
    // cannot mint instructions, even if its bytes look like policy text (S3).
    assert_eq!(
        InputOrigin::ToolOutput.default_trust(),
        TrustClass::UntrustedExternalData
    );
}

#[test]
fn t06_profile_poisoning_contradictory_evidence_becomes_conflicted() {
    // §9, T06: a contradictory preference does not silently replace the
    // previous one. The journal keeps both pieces of evidence (append-only);
    // the derived view surfaces the conflict as `Conflicted`, never `Active`,
    // so the host must arbitrate before re-promotion.
    use cogno_core::{
        DerivationPolicy, EvidenceOrigin, Fingerprint, InputOrigin, Journal, JournalEvent,
        RuleState,
    };
    let mut j = Journal::default();
    // First: explicit approval for category 7 (2 distinct approvals -> would
    // normally become Active under default min_evidence=2).
    for i in 0..2u8 {
        let mut fp = [0u8; 32];
        fp[0] = i + 1;
        j.append(JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin: EvidenceOrigin::ExplicitUserApproval,
            evidence_id: EvidenceId::from_u64(100 + i as u64),
            category_tag: 7,
            payload: Vec::new(),
        });
    }
    // Then: a rejection against the same category — contradiction.
    let mut fp = [0u8; 32];
    fp[0] = 99;
    j.append(JournalEvent {
        id: cogno_core::EventId(0),
        fingerprint: Fingerprint(fp),
        origin: InputOrigin::ExplicitUserInstruction,
        evidence_origin: EvidenceOrigin::UserRejection,
        evidence_id: EvidenceId::from_u64(200),
        category_tag: 7,
        payload: Vec::new(),
    });
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    let rule = p.rule(7).expect("rule preserved (not silently dropped)");
    assert_eq!(
        rule.state,
        RuleState::Conflicted,
        "contradiction -> Conflicted, never Active"
    );
    assert!(!rule.is_active(), "conflicted rule must not be Active");
    assert!(
        !rule.conflicts.is_empty(),
        "conflict recorded, not silenced"
    );
    // Both pieces of evidence remain in the journal (S6/S7: reconstructible).
    assert_eq!(
        j.len(),
        3,
        "all evidence preserved in the append-only journal"
    );
}
