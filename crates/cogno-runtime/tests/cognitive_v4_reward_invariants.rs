use cogno_core::{
    CognoProposalView, DataClassification, EvidenceId, EvidenceOrigin, InputOrigin, KvCachePolicy,
    MemoryBudget, ProposalAction, QueueFullPolicy, RejectReason, Reward, RewardParams,
    RuleCategory, RuleScope, SafetyPolicy, TrustClass,
};
use cogno_model::{
    review_sequence_cognitive_model_for_meta, SequenceCognitiveExample,
    SequenceCognitiveMetaReviewConfig, SequenceCognitiveMetaReviewPolicy,
    SequenceCognitiveReviewCorpus, SequenceCognitiveReviewExample, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    CognitiveDecisionCandidate, CognitiveObservationInput, CognitivePostHardGateError,
    CognitivePreferenceRelation, CognitiveRewardError, CognitiveSoftAdjustmentPolicy,
    CognitiveSoftTargets, GenerationBoundControlledRestartCognitiveModel, HostMetaAttestation,
    HostModelPromotionAttestation, HostModelRestartActivationAttestation, PipelineParams, Runtime,
    RuntimeConfig,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-v4-reward-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn runtime() -> Runtime {
    let budget = MemoryBudget::try_new(
        1024, 100, 100, 100, 100, 100, 100, 100, 256, 256, 64, 32, 8, 16, 4, 16,
    )
    .expect("bounded budget");
    Runtime::try_new(RuntimeConfig {
        budget,
        reserved_bytes: 0,
        kv_capacity_tokens: 64,
        kv_policy: KvCachePolicy::RejectOnOverflow,
        queue_capacity: 8,
        queue_policy: QueueFullPolicy::RejectNewest,
    })
    .expect("runtime")
}

fn example(index: u8, class: usize, contradicts: bool) -> SequenceCognitiveReviewExample {
    let alpha = class == 0;
    let positive = if alpha {
        b"alpha memory"
    } else {
        b"omega memory"
    };
    let negative = if alpha {
        b"omega memory"
    } else {
        b"alpha memory"
    };
    SequenceCognitiveReviewExample::new(
        SequenceCognitiveExample {
            classification_payload: format!("class-{class}-sample-{index}").into_bytes(),
            classification_target: class,
            preferred: format!("preferred-{class}-{index}").into_bytes(),
            dispreferred: format!("rejected-{class}-{index}").into_bytes(),
            symbolic_payload: format!("rule-{class}-{index}").into_bytes(),
            rule_satisfied: vec![alpha, !alpha, true, false],
            contradiction_left: format!("claim-{index}").into_bytes(),
            contradiction_right: format!("counter-{index}").into_bytes(),
            contradicts,
            retrieval_query: if alpha {
                b"alpha".to_vec()
            } else {
                b"omega".to_vec()
            },
            retrieval_candidates: vec![positive.to_vec(), negative.to_vec()],
            retrieval_positive_idx: 0,
        },
        InputOrigin::ExplicitUserInstruction,
        EvidenceOrigin::ExplicitUserApproval,
    )
}

fn prepare_v4_runtime(root: &Path) -> (Runtime, [u8; 32]) {
    let mut corpus = SequenceCognitiveReviewCorpus::with_seed(2701);
    for (index, class, contradiction) in [
        (0, 0, false),
        (1, 1, true),
        (2, 0, true),
        (3, 1, false),
        (4, 0, false),
        (5, 1, true),
        (6, 0, true),
        (7, 1, false),
    ] {
        assert!(
            corpus
                .add(
                    example(index, class, contradiction),
                    DataClassification::Internal,
                )
                .expect("classified V4 review data")
        );
    }
    let train = cogno_model::CorpusSplit {
        kind: SplitKind::Train,
        indices: vec![0, 1, 2, 3],
    };
    let validation = cogno_model::CorpusSplit {
        kind: SplitKind::Validation,
        indices: vec![4, 5],
    };
    let test = cogno_model::CorpusSplit {
        kind: SplitKind::Test,
        indices: vec![6, 7],
    };
    let mut config = SequenceCognitiveMetaReviewConfig::default();
    config.model.epochs = 2;
    config.model.cognitive.encoder.max_tokens = 64;
    config.model.cognitive.encoder.embedding_dim = 8;
    config.model.cognitive.encoder.hidden_dim = 8;
    config.model.max_retrieval_candidates = 2;
    let review = review_sequence_cognitive_model_for_meta(
        &corpus,
        &train,
        &validation,
        &test,
        config,
        SequenceCognitiveMetaReviewPolicy {
            minimum_validation_task_accuracy_bps: 0,
            minimum_test_task_accuracy_bps: 0,
            maximum_classification_regression_bps: 10_000,
        },
    )
    .expect("review")
    .into_eligible()
    .expect("eligible");

    let mut runtime = runtime();
    let receipt = runtime
        .activate_meta_from_review(
            &review,
            HostMetaAttestation::attest_validated_scalar_engine_and_frozen_reference_policy(),
        )
        .expect("activate Meta");
    let commit = commit_reviewed_model_generation(
        root,
        1,
        &review,
        HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence(),
    )
    .expect("persist");
    assert_eq!(receipt.candidate_artifact_sha256, commit.artifact_sha256);

    let seal = GenerationBoundControlledRestartCognitiveModel::prepare(
        load_persisted_model_generation_selection(root).expect("replay"),
        HostModelRestartActivationAttestation::approve_verified_v4_for_restart_initialization(),
    )
    .expect("restart seal");
    runtime
        .install_controlled_restart_cognitive_model(seal)
        .expect("install");
    assert_eq!(
        runtime.meta_candidate_digest(),
        Some(commit.artifact_sha256)
    );
    assert_eq!(
        runtime.cognitive_model_artifact_sha256(),
        Some(commit.artifact_sha256)
    );
    let report = runtime.report();
    assert!(report.meta_active);
    assert_eq!(report.meta_candidate_digest, Some(commit.artifact_sha256));
    assert!(report.cognitive_model_loaded);
    assert_eq!(report.cognitive_model_generation, Some(1));
    assert_eq!(
        report.cognitive_model_artifact_sha256,
        Some(commit.artifact_sha256)
    );
    assert!(report.cognitive_model_meta_bound);
    (runtime, commit.artifact_sha256)
}

fn cognitive_input<'a>(candidates: &'a [&'a [u8]]) -> CognitiveObservationInput<'a> {
    CognitiveObservationInput {
        classification_payload: b"class-0-live",
        preference_left: b"preferred-live",
        preference_right: b"rejected-live",
        symbolic_payload: b"rule-live",
        contradiction_left: b"claim-live",
        contradiction_right: b"counter-live",
        retrieval_query: b"alpha",
        retrieval_candidates: candidates,
    }
}

fn targets() -> CognitiveSoftTargets<'static> {
    CognitiveSoftTargets {
        classification_class: Some(0),
        preference: Some(CognitivePreferenceRelation::Left),
        symbolic_satisfied: Some(&[true, false, true, false]),
        contradiction: Some(false),
        retrieval_index: Some(0),
    }
}

fn pipeline_params<'a>(
    proposal: CognoProposalView<'a>,
    data_class: DataClassification,
) -> PipelineParams<'a> {
    PipelineParams {
        policy: SafetyPolicy::MVP,
        reward_params: RewardParams::DEFAULT,
        data_class,
        trust: TrustClass::AuthenticatedUser,
        reward_signal: Reward::ZERO,
        proposal: Some(proposal),
    }
}

#[test]
fn report_fails_closed_without_cognitive_v4_state() {
    let runtime = runtime();
    let report = runtime.report();
    assert!(!report.meta_active);
    assert_eq!(report.meta_candidate_digest, None);
    assert!(!report.cognitive_model_loaded);
    assert_eq!(report.cognitive_model_generation, None);
    assert_eq!(report.cognitive_model_artifact_sha256, None);
    assert!(!report.cognitive_model_meta_bound);
}

#[test]
fn hard_rejection_never_mints_cognitive_observation_or_reward() {
    let root = root("hard-reject");
    let (mut runtime, _digest) = prepare_v4_runtime(&root);
    let evidence = [EvidenceId::from_u64(1)];
    let proposal = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: RuleCategory::Behavior,
        scope: RuleScope::Session,
        confidence_bps: 10_000,
        evidence_ids: &evidence,
        payload: b"clean payload",
    };
    let candidates: [&[u8]; 2] = [b"alpha memory", b"omega memory"];

    let result = runtime.run_pipeline_with_cognitive_soft_reward(
        pipeline_params(proposal, DataClassification::Secret),
        cognitive_input(&candidates),
        CognitiveSoftAdjustmentPolicy::default(),
        targets(),
    );
    assert!(matches!(
        result,
        Err(CognitiveRewardError::PostHard(
            CognitivePostHardGateError::PipelineRejected {
                stage: "hard",
                reason: RejectReason::Secret,
            }
        ))
    ));
    assert_eq!(runtime.rejections, 1);
    assert_eq!(runtime.admissions, 0);
    assert_eq!(runtime.audit.len(), 1);
    assert_eq!(runtime.audit.records[0].decision, "reject");
    assert!(runtime
        .audit
        .records
        .iter()
        .all(|record| record.cognitive.is_none() && record.cognitive_reward.is_none()));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn eligible_candidate_gets_checked_bounded_generation_bound_reward() {
    let root = root("eligible");
    let (mut runtime, digest) = prepare_v4_runtime(&root);
    let evidence = [EvidenceId::from_u64(1)];
    let proposal = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: RuleCategory::Behavior,
        scope: RuleScope::Session,
        confidence_bps: 5_000,
        evidence_ids: &evidence,
        payload: b"clean payload",
    };
    let candidates: [&[u8]; 2] = [b"alpha memory", b"omega memory"];

    let applied = runtime
        .run_pipeline_with_cognitive_soft_reward(
            pipeline_params(proposal, DataClassification::Internal),
            cognitive_input(&candidates),
            CognitiveSoftAdjustmentPolicy::default(),
            targets(),
        )
        .expect("bounded cognitive reward");

    assert!(applied.delta().unsigned_abs() <= 25);
    assert_eq!(
        applied.final_score().points,
        applied
            .base_score()
            .points
            .checked_add(applied.delta())
            .expect("bounded addition")
    );
    assert!(applied.hard_constraints_preserved());
    assert_eq!(applied.model_generation(), 1);
    assert_eq!(applied.model_artifact_sha256(), digest);
    assert_eq!(applied.meta_candidate_digest(), digest);
    assert_eq!(runtime.admissions, 1);
    assert_eq!(runtime.rejections, 0);

    let reward_audit = runtime
        .audit
        .records
        .iter()
        .find_map(|record| record.cognitive_reward.as_ref())
        .expect("cognitive reward audit");
    assert_eq!(reward_audit.base_score, applied.base_score().points);
    assert_eq!(reward_audit.final_score, applied.final_score().points);
    assert_eq!(reward_audit.delta, applied.delta());
    assert_eq!(reward_audit.model_artifact_sha256, digest);
    assert_eq!(reward_audit.meta_candidate_digest, digest);
    assert!(reward_audit.hard_constraints_preserved);
    assert_eq!(
        runtime
            .audit
            .records
            .iter()
            .filter(|record| record.cognitive.is_some())
            .count(),
        1
    );
    assert_eq!(
        runtime
            .audit
            .records
            .iter()
            .filter(|record| record.cognitive_reward.is_some())
            .count(),
        1
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn real_sealed_rewards_are_compared_and_audited_deterministically() {
    let root = root("decision");
    let (mut runtime, digest) = prepare_v4_runtime(&root);
    let evidence = [EvidenceId::from_u64(1)];
    let candidates: [&[u8]; 2] = [b"alpha memory", b"omega memory"];
    let strong = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: RuleCategory::Behavior,
        scope: RuleScope::Session,
        confidence_bps: 7_000,
        evidence_ids: &evidence,
        payload: b"strong clean payload",
    };
    let weak = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: RuleCategory::Behavior,
        scope: RuleScope::Session,
        confidence_bps: 2_000,
        evidence_ids: &evidence,
        payload: b"weak clean payload",
    };

    let strong_reward = runtime
        .run_pipeline_with_cognitive_soft_reward(
            pipeline_params(strong, DataClassification::Internal),
            cognitive_input(&candidates),
            CognitiveSoftAdjustmentPolicy::default(),
            targets(),
        )
        .expect("strong reward");
    let weak_reward = runtime
        .run_pipeline_with_cognitive_soft_reward(
            pipeline_params(weak, DataClassification::Internal),
            cognitive_input(&candidates),
            CognitiveSoftAdjustmentPolicy::default(),
            targets(),
        )
        .expect("weak reward");
    assert!(strong_reward.final_score() > weak_reward.final_score());

    let decision = runtime
        .decide_with_cognitive_rewards(&[
            CognitiveDecisionCandidate {
                candidate_id: 20,
                applied: &strong_reward,
            },
            CognitiveDecisionCandidate {
                candidate_id: 10,
                applied: &weak_reward,
            },
        ])
        .expect("cognitive decision");

    assert_eq!(decision.selected_candidate_id, 20);
    assert_eq!(decision.selected_final_score, strong_reward.final_score());
    assert_eq!(decision.model_generation, 1);
    assert_eq!(decision.model_artifact_sha256, digest);
    assert_eq!(decision.meta_candidate_digest, digest);
    assert!(decision.hard_constraints_preserved);
    assert_eq!(
        decision
            .candidates
            .iter()
            .map(|trace| trace.candidate_id)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );

    let decision_audit = runtime
        .audit
        .records
        .iter()
        .filter_map(|record| record.cognitive_decision.as_ref())
        .next_back()
        .expect("cognitive decision audit");
    assert_eq!(decision_audit, &decision);
    assert_eq!(
        runtime
            .audit
            .records
            .iter()
            .filter(|record| record.decision == "cognitive_decision")
            .count(),
        1
    );

    fs::remove_dir_all(root).expect("cleanup");
}
