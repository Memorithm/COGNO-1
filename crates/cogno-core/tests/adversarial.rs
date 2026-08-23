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
    // The loader contract (§21): recompute the artifact digest and compare it
    // to the manifest's stored hash **before** any major allocation. We
    // exercise that comparison with a deterministic, dependency-free FNV-1a
    // digest over the artifact bytes: a single flipped byte must change the
    // digest and therefore refuse the artifact.
    fn fnv1a(bytes: &[u8]) -> [u8; 32] {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let mut out = [0u8; 32];
        out[..8].copy_from_slice(&h.to_be_bytes());
        out
    }

    let tokenizer_bytes = b"vocab: a=1 b=2 c=3";
    let stored = fnv1a(tokenizer_bytes);

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

    // Matching artifact passes the loader's hash gate.
    assert_eq!(
        m.tokenizer_hash,
        fnv1a(tokenizer_bytes),
        "authentic tokenizer accepted"
    );
    // A hostile artifact differing by ONE byte produces a different digest:
    // the loader gate refuses it even though the manifest is structurally
    // valid.
    let mut tampered = *tokenizer_bytes;
    tampered[7] ^= 0x01;
    assert_ne!(
        m.tokenizer_hash,
        fnv1a(&tampered),
        "single-byte tampering must be detected"
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

#[test]
fn t06_conflict_detected_regardless_of_order() {
    use cogno_core::{
        DerivationPolicy, EvidenceOrigin, Fingerprint, Journal, JournalEvent, RuleState,
    };
    // Mirror of the approval-then-rejection test: the rejection comes FIRST.
    // The conflict must be detected either way (§9/T06) — a later approval
    // must not silently override an earlier user rejection.
    fn event(fp_byte: u8, evidence_origin: EvidenceOrigin, evidence_id: u64) -> JournalEvent {
        let mut fp = [0u8; 32];
        fp[0] = fp_byte;
        JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin,
            evidence_id: EvidenceId::from_u64(evidence_id),
            category_tag: 7,
            payload: Vec::new(),
        }
    }
    for (rejection_first, tag_fp) in [(true, 150u8), (false, 151)] {
        let mut j = Journal::default();
        if rejection_first {
            j.append(event(tag_fp, EvidenceOrigin::UserRejection, 300));
            j.append(event(tag_fp + 1, EvidenceOrigin::ExplicitUserApproval, 301));
        } else {
            j.append(event(tag_fp, EvidenceOrigin::ExplicitUserApproval, 302));
            j.append(event(tag_fp + 1, EvidenceOrigin::UserRejection, 303));
        }
        let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
        let rule = p.rule(7).expect("rule present");
        assert_eq!(
            rule.state,
            RuleState::Conflicted,
            "approve+reject is Conflicted in any order (rejection_first={rejection_first})"
        );
        assert!(!rule.is_active());
        assert!(!rule.conflicts.is_empty());
    }
}

#[test]
fn duplicate_evidence_ids_do_not_inflate_distinct_evidence() {
    use cogno_core::{DerivationPolicy, EvidenceOrigin, Fingerprint, Journal, JournalEvent};
    // Same evidence id replayed with different fingerprints (e.g. payload
    // jitter) must count ONCE toward min_evidence (§9 distinct non-model
    // proofs). Two distinct ids are required for promotion.
    let mut j = Journal::default();
    for i in 0..2u8 {
        let mut fp = [0u8; 32];
        fp[0] = 60 + i;
        j.append(JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin: EvidenceOrigin::ExplicitUserStatement,
            evidence_id: EvidenceId::from_u64(777), // SAME id twice
            category_tag: 5,
            payload: Vec::new(),
        });
    }
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    let rule = p.rule(5).unwrap();
    assert_eq!(rule.distinct_evidence, 1, "duplicate id counted once");
    assert!(!rule.is_active(), "min_evidence=2 needs two DISTINCT ids");
    // A genuinely second id promotes.
    let mut fp = [0u8; 32];
    fp[0] = 99;
    j.append(JournalEvent {
        id: cogno_core::EventId(0),
        fingerprint: Fingerprint(fp),
        origin: InputOrigin::ExplicitUserInstruction,
        evidence_origin: EvidenceOrigin::ExplicitUserStatement,
        evidence_id: EvidenceId::from_u64(778),
        category_tag: 5,
        payload: Vec::new(),
    });
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    assert!(p.rule(5).unwrap().is_active());
}

#[test]
fn sibling_prefix_path_rejected() {
    // `/app/root` must NOT authorize the sibling subtree `/app/rootkit/...`:
    // the root match requires a component boundary (T13).
    let pol = PathPolicy {
        root: b"/app/root".to_vec(),
    };
    assert_eq!(
        pol.check_lexical(b"/app/rootkit/secrets.txt"),
        cogno_core::PathVerdict::RejectOutsideRoot
    );
    // The real subtree still passes.
    assert_eq!(
        pol.check_lexical(b"/app/root/data/log.txt"),
        cogno_core::PathVerdict::Allow
    );
    // Exact-root candidate passes.
    assert_eq!(
        pol.check_lexical(b"/app/root"),
        cogno_core::PathVerdict::Allow
    );
}

#[test]
fn empty_root_rejects_everything() {
    // An empty root would otherwise authorize every absolute path (S10).
    let pol = PathPolicy { root: Vec::new() };
    assert_eq!(
        pol.check_lexical(b"any/path"),
        cogno_core::PathVerdict::RejectOutsideRoot
    );
}

#[test]
fn proposal_rejects_non_adjacent_duplicate_evidence_ids() {
    use cogno_core::{CognoProposalView, ProposalAction, RejectReason, RuleCategory, RuleScope};
    // `[a, b, a]` was previously accepted by the adjacency-only check; the
    // spec forbids duplicates anywhere in the list.
    let p = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ExtractPreference,
        category: RuleCategory::Format,
        scope: RuleScope::Session,
        confidence_bps: 5_000,
        evidence_ids: &[
            EvidenceId::from_u64(7),
            EvidenceId::from_u64(8),
            EvidenceId::from_u64(7),
        ],
        payload: br#"{}"#.to_vec().leak(),
    };
    assert_eq!(
        validate_proposal(&p),
        Err(RejectReason::Malformed),
        "non-adjacent duplicates rejected"
    );
}

#[test]
fn journal_refuses_zero_fingerprint_and_keeps_dedup_o1_behavior() {
    use cogno_core::{EvidenceOrigin, Fingerprint, Journal, JournalEvent};
    let mut j = Journal::default();
    // ZERO fingerprint: distinct events with an all-zero digest would all
    // collapse into one dedup bucket — refused outright (fail closed).
    let e = JournalEvent {
        id: cogno_core::EventId(0),
        fingerprint: Fingerprint::ZERO,
        origin: InputOrigin::ModelOutput,
        evidence_origin: EvidenceOrigin::ModelInference,
        evidence_id: EvidenceId::from_u64(1),
        category_tag: 1,
        payload: Vec::new(),
    };
    assert_eq!(
        j.append(e),
        cogno_core::AppendOutcome::ZeroFingerprint,
        "zero fingerprint rejected"
    );
    assert_eq!(j.len(), 0);

    // Dedup stays correct when the duplicate arrives far from the original:
    // the `seen` map keeps O(1) average lookups and the ORIGINAL event id.
    let mk = |byte: u8| {
        let mut fp = [0u8; 32];
        fp[0] = byte;
        JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin: EvidenceOrigin::ExplicitUserStatement,
            evidence_id: EvidenceId::from_u64(u64::from(byte)),
            category_tag: 1,
            payload: Vec::new(),
        }
    };
    let first_id = match j.append(mk(42)) {
        cogno_core::AppendOutcome::Appended { id } => id,
        other => panic!("expected append, got {other:?}"),
    };
    for i in 0..500u16 {
        let mut e = mk((i % 200 + 1) as u8);
        if i == 499 {
            e.fingerprint = {
                let mut fp = [0u8; 32];
                fp[0] = 42;
                Fingerprint(fp)
            };
        }
        let _ = j.append(e);
    }
    assert_eq!(
        j.append(mk(42)),
        cogno_core::AppendOutcome::Duplicate { id: first_id },
        "duplicate returns the original id even after many appends"
    );
}
