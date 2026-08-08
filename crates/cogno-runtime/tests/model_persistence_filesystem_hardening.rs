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
        "cogno-model-fs-hardening-{label}-{}-{nonce}",
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

fn assert_invalid_data(error: ModelPersistenceError) {
    assert!(matches!(
        error,
        ModelPersistenceError::Io(ref io_error) if io_error.kind() == ErrorKind::InvalidData
    ));
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
fn current_pointer_must_be_a_regular_file() {
    let root = root("current-directory");
    fs::create_dir_all(root.join("MODEL_CURRENT")).expect("directory posing as current");

    let error = load_persisted_model_generation_selection(&root)
        .expect_err("directory MODEL_CURRENT must fail closed");
    assert_invalid_data(error);

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn finalized_generation_must_be_a_real_directory() {
    let root = root("generation-file");
    fs::create_dir_all(&root).expect("root");
    fs::write(root.join("MODEL_CURRENT"), b"1\n").expect("current");
    fs::write(root.join("model-generation-1"), b"not-a-directory").expect("fake generation");

    let error = load_persisted_model_generation_selection(&root)
        .expect_err("regular file generation must fail closed");
    assert_invalid_data(error);

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn malformed_reserved_generation_name_is_rejected() {
    let root = root("reserved-name");
    fs::create_dir_all(root.join("model-generation-01")).expect("reserved directory");

    let error = load_persisted_model_generation_selection(&root)
        .expect_err("non-canonical reserved generation name must fail closed");
    assert_invalid_data(error);

    fs::remove_dir_all(root).expect("cleanup");
}

#[cfg(unix)]
#[test]
fn symlink_commit_lock_is_rejected_before_opening() {
    use std::os::unix::fs::symlink;

    let persistence_root = root("lock-symlink");
    let outside = root("lock-target");
    fs::create_dir_all(&persistence_root).expect("root");
    fs::write(&outside, b"external lock target").expect("outside file");
    symlink(&outside, persistence_root.join(".MODEL-COMMIT.lock")).expect("lock symlink");

    let review = eligible_review();
    let error = commit_reviewed_model_generation(&persistence_root, 1, &review, host())
        .expect_err("symlink lock must fail closed");
    assert_invalid_data(error);
    assert_eq!(
        fs::read(&outside).expect("outside remains readable"),
        b"external lock target"
    );
    assert!(!persistence_root.join("MODEL_CURRENT").exists());
    assert!(!persistence_root.join("model-generation-1").exists());

    fs::remove_dir_all(persistence_root).expect("cleanup");
    fs::remove_file(outside).expect("outside cleanup");
}

#[cfg(unix)]
#[test]
fn symlink_generation_directory_is_rejected_during_replay() {
    use std::os::unix::fs::symlink;

    let source_root = root("generation-symlink-source");
    let target_root = root("generation-symlink-target");
    let review = eligible_review();
    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source commit");

    fs::create_dir_all(&target_root).expect("target root");
    fs::write(target_root.join("MODEL_CURRENT"), b"1\n").expect("target current");
    symlink(
        source_root.join("model-generation-1"),
        target_root.join("model-generation-1"),
    )
    .expect("generation symlink");

    let error = load_persisted_model_generation_selection(&target_root)
        .expect_err("symlink generation must fail closed");
    assert_invalid_data(error);

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}

#[cfg(unix)]
#[test]
fn symlink_artifact_is_rejected_before_hostile_loader_reads_it() {
    use std::os::unix::fs::symlink;

    let source_root = root("artifact-symlink-source");
    let target_root = root("artifact-symlink-target");
    let review = eligible_review();
    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source commit");

    fs::create_dir_all(&target_root).expect("target root");
    fs::write(target_root.join("MODEL_CURRENT"), b"1\n").expect("target current");
    copy_generation(&source_root, &target_root, 1);
    let target_artifact = target_root.join("model-generation-1/model.bin");
    fs::remove_file(&target_artifact).expect("remove copied artifact");
    symlink(
        source_root.join("model-generation-1/model.bin"),
        &target_artifact,
    )
    .expect("artifact symlink");

    let error = load_persisted_model_generation_selection(&target_root)
        .expect_err("symlink artifact must fail closed");
    assert_invalid_data(error);

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}
