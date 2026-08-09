use cogno_core::{
    DataClassification, EvidenceOrigin, InputOrigin, KvCachePolicy, MemoryBudget, QueueFullPolicy,
};
use cogno_model::{
    review_sequence_cognitive_model_for_meta, SequenceCognitiveExample,
    SequenceCognitiveMetaReviewConfig, SequenceCognitiveMetaReviewPolicy,
    SequenceCognitiveReviewCorpus, SequenceCognitiveReviewExample, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    ControlledRestartModelAuthority, GenerationBoundControlledRestartCognitiveModel,
    HostModelPromotionAttestation, HostModelRestartActivationAttestation, Runtime, RuntimeConfig,
    RuntimeModelInstallError,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("cogno-v4-restart-{}-{nonce}", std::process::id()))
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

fn observation(index: u8, class: usize, contradicts: bool) -> SequenceCognitiveReviewExample {
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

fn reviewed_v4() -> cogno_model::EligibleSequenceCognitiveMetaModelReview {
    let mut corpus = SequenceCognitiveReviewCorpus::with_seed(2401);
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
        assert!(corpus
            .add(
                observation(index, class, contradiction),
                DataClassification::Internal,
            )
            .expect("classified V4 review data"));
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
    review_sequence_cognitive_model_for_meta(
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
    .expect("V4 review")
    .into_eligible()
    .expect("sealed V4 proof")
}

#[test]
fn persisted_v4_is_generation_bound_and_installed_only_once() {
    let root = temp_root();
    let review = reviewed_v4();
    let commit = commit_reviewed_model_generation(
        &root,
        1,
        &review,
        HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence(),
    )
    .expect("persist reviewed V4");
    let first = GenerationBoundControlledRestartCognitiveModel::prepare(
        load_persisted_model_generation_selection(&root).expect("first replay"),
        HostModelRestartActivationAttestation::approve_verified_v4_for_restart_initialization(),
    )
    .expect("first restart seal");
    let second = GenerationBoundControlledRestartCognitiveModel::prepare(
        load_persisted_model_generation_selection(&root).expect("second replay"),
        HostModelRestartActivationAttestation::approve_verified_v4_for_restart_initialization(),
    )
    .expect("second restart seal");

    assert_eq!(first.generation(), 1);
    assert_eq!(first.artifact_sha256(), commit.artifact_sha256);
    assert_eq!(
        first.authority(),
        ControlledRestartModelAuthority::RestartInitializationOnly
    );
    assert!(!first.model().capabilities.is_empty());

    let mut runtime = runtime();
    runtime
        .install_controlled_restart_cognitive_model(first)
        .expect("one-shot V4 install");
    assert_eq!(runtime.cognitive_model_generation(), Some(1));
    assert_eq!(
        runtime.cognitive_model_artifact_sha256(),
        Some(commit.artifact_sha256)
    );
    assert!(runtime.cognitive_model().is_some());
    assert!(!runtime.tools.tools_enabled);
    assert_eq!(
        runtime.install_controlled_restart_cognitive_model(second),
        Err(RuntimeModelInstallError::AlreadyInstalled)
    );
    fs::remove_dir_all(root).expect("cleanup");
}
