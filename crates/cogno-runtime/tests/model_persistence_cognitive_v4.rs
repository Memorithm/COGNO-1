use cogno_core::{DataClassification, EvidenceOrigin, InputOrigin};
use cogno_model::{
    review_sequence_cognitive_model_for_meta, LoadedNeuralModel, SequenceCognitiveExample,
    SequenceCognitiveMetaReviewConfig, SequenceCognitiveMetaReviewPolicy,
    SequenceCognitiveReviewCorpus, SequenceCognitiveReviewExample, SplitKind,
};
use cogno_runtime::{
    commit_reviewed_model_generation, load_persisted_model_generation_selection,
    HostModelPromotionAttestation,
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
        "cogno-model-persistence-v4-{}-{nonce}",
        std::process::id()
    ))
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

#[test]
fn reviewed_cognitive_v4_commits_and_replays_exact_architecture() {
    let mut corpus = SequenceCognitiveReviewCorpus::with_seed(2203);
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
                    observation(index, class, contradiction),
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
    config.model.cognitive.encoder.embedding_dim = 8;
    config.model.cognitive.encoder.hidden_dim = 8;
    config.model.cognitive.encoder.max_tokens = 64;
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
    .expect("eligible V4 proof");

    let root = root();
    let commit = commit_reviewed_model_generation(
        &root,
        1,
        &review,
        HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence(),
    )
    .expect("persist V4");
    assert_eq!(
        commit.artifact_sha256,
        review.artifact().manifest.weights_hash
    );

    let selection = load_persisted_model_generation_selection(&root).expect("replay V4");
    assert_eq!(selection.selected_generation, 1);
    assert_eq!(selection.chain.len(), 1);
    assert_eq!(
        selection.selected_artifact.manifest.weights_hash,
        review.artifact().manifest.weights_hash
    );
    match selection.selected_model {
        LoadedNeuralModel::SequenceCognitiveV4(state) => {
            assert_eq!(state.max_retrieval_candidates(), 2);
            assert_eq!(
                state.parameter_count(),
                usize::try_from(review.artifact().manifest.parameter_count)
                    .expect("parameter count")
            );
        }
        _ => panic!("expected persisted sequence cognitive V4 model"),
    }
    fs::remove_dir_all(root).expect("cleanup");
}
