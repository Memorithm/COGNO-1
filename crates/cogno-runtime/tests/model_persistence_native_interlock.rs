use cogno_core::{EvidenceOrigin, InputOrigin};
use cogno_model::{
    review_neural_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample,
    MetaNeuralReviewPolicy, NeuralConfig, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, HostModelPromotionAttestation, ModelPersistenceError,
};
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-model-public-native-lock-{}-{nonce}",
        std::process::id()
    ))
}

fn eligible_review() -> cogno_model::EligibleMetaModelReview {
    let mut corpus = Corpus::with_seed(7);
    for (label, payload) in [
        (Label(0), b"alpha train one".as_slice()),
        (Label(1), b"omega train one"),
        (Label(0), b"alpha train two"),
        (Label(1), b"omega train two"),
        (Label(0), b"alpha validation"),
        (Label(1), b"omega validation"),
        (Label(0), b"alpha test"),
        (Label(1), b"omega test"),
    ] {
        assert!(corpus.add(LabeledExample::new(
            label,
            payload.to_vec(),
            InputOrigin::ExplicitUserInstruction,
            EvidenceOrigin::ExplicitUserApproval,
        )));
    }
    let train = CorpusSplit {
        kind: SplitKind::Train,
        indices: vec![0, 1, 2, 3],
    };
    let validation = CorpusSplit {
        kind: SplitKind::Validation,
        indices: vec![4, 5],
    };
    let test = CorpusSplit {
        kind: SplitKind::Test,
        indices: vec![6, 7],
    };
    review_neural_model_for_meta(
        &corpus,
        &train,
        &validation,
        &test,
        NeuralConfig {
            input_dim: 64,
            epochs: 64,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        },
        MetaNeuralReviewPolicy {
            minimum_validation_accuracy_bps: 5_000,
            minimum_test_accuracy_bps: 5_000,
            maximum_regression_bps: 5_000,
            artifact_max_context_tokens: 2_048,
        },
    )
    .expect("review")
    .into_eligible()
    .expect("eligible")
}

fn host() -> HostModelPromotionAttestation {
    HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence()
}

#[test]
fn public_commit_fails_closed_while_native_interlock_is_held() {
    let root = root();
    std::fs::create_dir_all(&root).expect("root");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".MODEL-COMMIT.lock"))
        .expect("lock file");
    lock.try_lock().expect("hold native lock");

    let review = eligible_review();
    let error = commit_reviewed_model_generation(&root, 1, &review, host())
        .expect_err("contended commit must fail closed");
    assert!(matches!(
        error,
        ModelPersistenceError::Io(ref io_error) if io_error.kind() == ErrorKind::WouldBlock
    ));
    assert!(!root.join("MODEL_CURRENT").exists());
    assert!(!root.join("model-generation-1").exists());
    assert!(!root.join(".model-stage-1").exists());

    drop(lock);
    let commit =
        commit_reviewed_model_generation(&root, 1, &review, host()).expect("released lock commits");
    assert_eq!(commit.generation, 1);
    assert!(root.join("MODEL_CURRENT").is_file());
    assert!(root.join("model-generation-1").is_dir());

    std::fs::remove_dir_all(root).expect("cleanup");
}
