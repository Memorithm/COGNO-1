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
    CognitiveObservationError, CognitiveObservationInput,
    GenerationBoundControlledRestartCognitiveModel, HostModelPromotionAttestation,
    HostModelRestartActivationAttestation, Runtime, RuntimeConfig,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-v4-observation-{}-{nonce}",
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

fn install_v4(runtime: &mut Runtime, root: &PathBuf) -> [u8; 32] {
    let mut corpus = SequenceCognitiveReviewCorpus::with_seed(2503);
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
                example(index, class, contradiction),
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
    let commit = commit_reviewed_model_generation(
        root,
        1,
        &review,
        HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence(),
    )
    .expect("persist");
    let seal = GenerationBoundControlledRestartCognitiveModel::prepare(
        load_persisted_model_generation_selection(root).expect("replay"),
        HostModelRestartActivationAttestation::approve_verified_v4_for_restart_initialization(),
    )
    .expect("restart seal");
    runtime
        .install_controlled_restart_cognitive_model(seal)
        .expect("install");
    commit.artifact_sha256
}

#[test]
fn observation_requires_installed_v4_and_is_audited_as_non_authoritative() {
    let root = root();
    let mut runtime = runtime();
    let candidates: [&[u8]; 2] = [b"alpha memory", b"omega memory"];
    let input = CognitiveObservationInput {
        classification_payload: b"class-0-live",
        preference_left: b"preferred-live",
        preference_right: b"rejected-live",
        symbolic_payload: b"rule-live",
        contradiction_left: b"claim-live",
        contradiction_right: b"counter-live",
        retrieval_query: b"alpha",
        retrieval_candidates: &candidates,
    };
    assert_eq!(
        runtime.observe_cognitive(input),
        Err(CognitiveObservationError::NoModelInstalled)
    );
    assert!(runtime.audit.is_empty());

    let digest = install_v4(&mut runtime, &root);
    let observation = runtime.observe_cognitive(input).expect("observation");
    assert_eq!(observation.model_generation, 1);
    assert_eq!(observation.model_artifact_sha256, digest);
    assert!(!observation.authoritative);
    assert!(observation.classification_confidence_bps <= 10_000);
    assert!(observation
        .classification_probabilities_bps
        .iter()
        .all(|value| *value <= 10_000));
    assert!(observation
        .symbolic_satisfaction_bps
        .iter()
        .all(|value| *value <= 10_000));
    assert!(observation.contradiction_bps <= 10_000);
    assert_eq!(runtime.audit.len(), 1);
    assert_eq!(
        runtime.audit.records[0].cognitive.as_ref(),
        Some(&observation)
    );
    assert_eq!(runtime.audit.records[0].decision, "cognitive_observation");
    assert!(runtime.audit.records[0].reason.is_none());
    assert!(runtime.audit.records[0].taste.is_none());
    assert!(!runtime.tools.tools_enabled);

    fs::remove_dir_all(root).expect("cleanup");
}
