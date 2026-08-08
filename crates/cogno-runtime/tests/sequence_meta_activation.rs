use cogno_core::{EvidenceOrigin, InputOrigin, KvCachePolicy, MemoryBudget, QueueFullPolicy};
use cogno_model::{
    review_sequence_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample,
    SequenceMetaReviewConfig, SequenceMetaReviewPolicy, SplitKind,
};
use cogno_runtime::{
    ControlledMetaActivationError, HostMetaAttestation, MetaActivationAuthority, Runtime,
    RuntimeConfig,
};

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

fn reviewed_sequence_candidate() -> cogno_model::EligibleSequenceMetaModelReview {
    let mut corpus = Corpus::with_seed(307);
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
    let mut config = SequenceMetaReviewConfig::default();
    config.classifier.encoder.max_tokens = 64;
    config.classifier.encoder.embedding_dim = 8;
    config.classifier.encoder.hidden_dim = 12;
    config.classifier.encoder.seed = 311;
    config.classifier.num_classes = 2;
    config.classifier.head_seed = 313;
    config.epochs = 16;
    config.learning_rate = 0.02;
    config.baseline_feature_dim = 64;

    review_sequence_model_for_meta(
        &corpus,
        &train,
        &validation,
        &test,
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

fn host() -> HostMetaAttestation {
    HostMetaAttestation::attest_validated_scalar_engine_and_frozen_reference_policy()
}

#[test]
fn sealed_sequence_v3_review_activates_only_objective_authority() {
    let review = reviewed_sequence_candidate();
    let expected_digest = review.artifact().manifest.weights_hash;
    let mut runtime = runtime();
    let receipt = runtime
        .activate_meta_from_review(&review, host())
        .expect("controlled Meta activation");

    assert!(runtime.meta_is_active());
    assert_eq!(runtime.meta_candidate_digest(), Some(expected_digest));
    assert_eq!(receipt.candidate_artifact_sha256, expected_digest);
    assert_eq!(receipt.authority, MetaActivationAuthority::ObjectiveOnly);
    assert!(!runtime.tools.tools_enabled);

    assert_eq!(
        runtime.activate_meta_from_review(&review, host()),
        Err(ControlledMetaActivationError::AlreadyActive)
    );
}
