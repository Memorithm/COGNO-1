use cogno_core::{EvidenceOrigin, InputOrigin};
use cogno_model::{
    review_neural_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample,
    MetaNeuralReviewPolicy, NeuralConfig, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    HostModelPromotionAttestation, ModelPersistenceError,
};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-model-crash-resume-{label}-{}-{nonce}",
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

fn copy_generation(source_root: &Path, target_root: &Path, generation: u64) {
    let source = source_root.join(format!("model-generation-{generation}"));
    let target = target_root.join(format!("model-generation-{generation}"));
    fs::create_dir_all(&target).expect("target generation");
    for name in ["model.bin", "model.manifest", "generation.manifest"] {
        fs::copy(source.join(name), target.join(name)).expect("copy generation artifact");
    }
}

#[test]
fn exact_published_genesis_is_resumed_and_stale_current_temp_is_replaced() {
    let source_root = root("source-genesis");
    let target_root = root("target-genesis");
    let review = eligible_review();

    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source commit");
    fs::create_dir_all(&target_root).expect("target root");
    copy_generation(&source_root, &target_root, 1);
    fs::write(target_root.join(".MODEL_CURRENT.tmp"), b"stale\n").expect("stale temp");

    let recovered = commit_reviewed_model_generation(&target_root, 1, &review, host())
        .expect("exact published generation must resume");
    assert_eq!(recovered.generation, 1);
    assert_eq!(
        fs::read_to_string(target_root.join("MODEL_CURRENT")).expect("current"),
        "1\n"
    );
    assert!(!target_root.join(".MODEL_CURRENT.tmp").exists());
    let selection =
        load_persisted_model_generation_selection(&target_root).expect("replayed selection");
    assert_eq!(selection.selected_generation, 1);

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}

#[test]
fn exact_published_successor_resumes_only_from_selected_predecessor() {
    let source_root = root("source-successor");
    let target_root = root("target-successor");
    let review = eligible_review();

    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source first");
    commit_reviewed_model_generation(&source_root, 2, &review, host()).expect("source second");

    fs::create_dir_all(&target_root).expect("target root");
    copy_generation(&source_root, &target_root, 1);
    fs::write(target_root.join("MODEL_CURRENT"), b"1\n").expect("selected predecessor");
    copy_generation(&source_root, &target_root, 2);

    let recovered = commit_reviewed_model_generation(&target_root, 2, &review, host())
        .expect("exact successor must resume");
    assert_eq!(recovered.generation, 2);
    assert_eq!(
        fs::read_to_string(target_root.join("MODEL_CURRENT")).expect("current"),
        "2\n"
    );
    let selection =
        load_persisted_model_generation_selection(&target_root).expect("replayed selection");
    assert_eq!(selection.selected_generation, 2);
    assert_eq!(selection.chain.len(), 2);

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}

#[test]
fn tampered_published_generation_is_never_selected_during_recovery() {
    let source_root = root("source-tamper");
    let target_root = root("target-tamper");
    let review = eligible_review();

    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source commit");
    fs::create_dir_all(&target_root).expect("target root");
    copy_generation(&source_root, &target_root, 1);
    let artifact_path = target_root.join("model-generation-1/model.bin");
    let mut bytes = fs::read(&artifact_path).expect("artifact");
    bytes[0] ^= 1;
    fs::write(&artifact_path, bytes).expect("tamper");

    let error = commit_reviewed_model_generation(&target_root, 1, &review, host())
        .expect_err("tampered generation must fail closed");
    assert!(matches!(
        error,
        ModelPersistenceError::Io(ref io_error) if io_error.kind() == ErrorKind::InvalidData
    ));
    assert!(!target_root.join("MODEL_CURRENT").exists());

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}
