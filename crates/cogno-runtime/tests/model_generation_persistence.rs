//! Public end-to-end coverage for controlled neural model persistence.

use cogno_core::{EvidenceOrigin, InputOrigin};
use cogno_model::{
    review_neural_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample, LoadedNeuralModel,
    MetaNeuralReviewPolicy, NeuralConfig, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    HostModelPromotionAttestation, ModelPersistenceError,
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
        "cogno-public-model-persistence-{}-{nonce}",
        std::process::id()
    ))
}

fn eligible_review() -> cogno_model::EligibleMetaModelReview {
    let mut corpus = Corpus::with_seed(71);
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
    review_neural_model_for_meta(
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
fn public_commit_and_restart_replay_are_deterministic() {
    let root = root();
    let review = eligible_review();
    let first = commit_reviewed_model_generation(&root, 1, &review, host()).expect("generation 1");
    let first_replay = load_persisted_model_generation_selection(&root).expect("first replay");
    assert_eq!(first_replay.selected_generation, 1);
    assert_eq!(first_replay.chain.len(), 1);
    assert_eq!(
        first_replay.selected_artifact.manifest.weights_hash,
        first.artifact_sha256
    );

    commit_reviewed_model_generation(&root, 2, &review, host()).expect("generation 2");
    let left = load_persisted_model_generation_selection(&root).expect("left replay");
    let right = load_persisted_model_generation_selection(&root).expect("right replay");
    assert_eq!(left.selected_generation, 2);
    assert_eq!(left.chain, right.chain);
    assert_eq!(left.selected_artifact, right.selected_artifact);
    match (&left.selected_model, &right.selected_model) {
        (LoadedNeuralModel::LinearV1(left), LoadedNeuralModel::LinearV1(right)) => {
            assert_eq!(left.weights(), right.weights());
        }
        _ => panic!("expected persisted v1 model"),
    }
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn restart_replay_rejects_missing_intermediate_generation() {
    let root = root();
    let review = eligible_review();
    commit_reviewed_model_generation(&root, 1, &review, host()).expect("generation 1");
    commit_reviewed_model_generation(&root, 2, &review, host()).expect("generation 2");
    fs::remove_dir_all(root.join("model-generation-1")).expect("remove predecessor");
    assert!(matches!(
        load_persisted_model_generation_selection(&root),
        Err(ModelPersistenceError::MissingGeneration(1))
    ));
    fs::remove_dir_all(root).expect("cleanup");
}
