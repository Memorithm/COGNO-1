use cogno_core::{EvidenceOrigin, InputOrigin};
use cogno_model::{
    review_sequence_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample, LoadedNeuralModel,
    SequenceMetaReviewConfig, SequenceMetaReviewPolicy, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    HostModelPromotionAttestation,
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
        "cogno-sequence-persistence-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn reviewed_sequence() -> cogno_model::EligibleSequenceMetaModelReview {
    let mut corpus = Corpus::with_seed(401);
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
    let mut config = SequenceMetaReviewConfig::default();
    config.classifier.encoder.max_tokens = 64;
    config.classifier.encoder.embedding_dim = 8;
    config.classifier.encoder.hidden_dim = 12;
    config.classifier.encoder.seed = 409;
    config.classifier.num_classes = 2;
    config.classifier.head_seed = 419;
    config.epochs = 16;
    config.learning_rate = 0.02;
    config.baseline_feature_dim = 64;

    review_sequence_model_for_meta(
        &corpus,
        &CorpusSplit {
            kind: SplitKind::Train,
            indices: vec![0, 1, 2, 3],
        },
        &CorpusSplit {
            kind: SplitKind::Validation,
            indices: vec![4, 5],
        },
        &CorpusSplit {
            kind: SplitKind::Test,
            indices: vec![6, 7],
        },
        config,
        SequenceMetaReviewPolicy {
            minimum_validation_accuracy_bps: 0,
            minimum_test_accuracy_bps: 0,
            maximum_regression_bps: 10_000,
        },
    )
    .expect("sequence review")
    .into_eligible()
    .expect("sealed sequence review")
}

fn host() -> HostModelPromotionAttestation {
    HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence()
}

fn copy_generation(source_root: &Path, target_root: &Path, generation: u64) {
    let source = source_root.join(format!("model-generation-{generation}"));
    let target = target_root.join(format!("model-generation-{generation}"));
    fs::create_dir_all(&target).expect("target generation");
    for name in ["model.bin", "model.manifest", "generation.manifest"] {
        fs::copy(source.join(name), target.join(name)).expect("copy generation file");
    }
}

#[test]
fn sequence_v3_review_persists_and_replays_as_sequence_v3() {
    let root = root("normal");
    let review = reviewed_sequence();
    let expected_digest = review.artifact().manifest.weights_hash;
    commit_reviewed_model_generation(&root, 1, &review, host()).expect("sequence commit");

    let selection = load_persisted_model_generation_selection(&root).expect("sequence replay");
    assert_eq!(selection.selected_generation, 1);
    assert_eq!(
        selection.selected_artifact.manifest.weights_hash,
        expected_digest
    );
    match selection.selected_model {
        LoadedNeuralModel::SequenceV3(model) => {
            assert_eq!(
                model.parameter_count(),
                usize::try_from(review.artifact().manifest.parameter_count)
                    .expect("bounded parameter count")
            );
        }
        _ => panic!("expected SequenceV3 persisted selection"),
    }

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn exact_published_sequence_v3_generation_resumes_after_crash_window() {
    let source_root = root("source-crash");
    let target_root = root("target-crash");
    let review = reviewed_sequence();

    commit_reviewed_model_generation(&source_root, 1, &review, host()).expect("source commit");
    fs::create_dir_all(&target_root).expect("target root");
    copy_generation(&source_root, &target_root, 1);
    fs::write(target_root.join(".MODEL_CURRENT.tmp"), b"stale\n").expect("stale temp");

    let recovered = commit_reviewed_model_generation(&target_root, 1, &review, host())
        .expect("exact sequence generation must resume");
    assert_eq!(recovered.generation, 1);
    assert_eq!(
        fs::read_to_string(target_root.join("MODEL_CURRENT")).expect("current"),
        "1\n"
    );
    assert!(!target_root.join(".MODEL_CURRENT.tmp").exists());
    let selection = load_persisted_model_generation_selection(&target_root).expect("replay");
    assert!(matches!(
        selection.selected_model,
        LoadedNeuralModel::SequenceV3(_)
    ));

    fs::remove_dir_all(source_root).expect("source cleanup");
    fs::remove_dir_all(target_root).expect("target cleanup");
}
