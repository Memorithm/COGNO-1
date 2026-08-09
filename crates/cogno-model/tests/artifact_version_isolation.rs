use cogno_core::{DataClassification, EvidenceOrigin, InputOrigin};
use cogno_model::{
    encode_mlp_neural_artifact, encode_neural_artifact, load_mlp_neural_artifact,
    load_neural_artifact, Corpus, CorpusSplit, Label, LabeledExample, MlpNeuralArtifactError,
    MlpNeuralConfig, MlpNeuralTrainer, NeuralArtifactError, NeuralConfig, NeuralTrainer, SplitKind,
};

fn corpus() -> (Corpus, CorpusSplit) {
    let mut corpus = Corpus::with_seed(53);
    for (label, payload) in [
        (Label(0), b"alpha-one".as_slice()),
        (Label(0), b"alpha-two"),
        (Label(1), b"omega-one"),
        (Label(1), b"omega-two"),
    ] {
        assert!(corpus
            .add_classified(
                LabeledExample::new(
                    label,
                    payload.to_vec(),
                    InputOrigin::ExplicitUserInstruction,
                    EvidenceOrigin::ExplicitUserApproval,
                ),
                DataClassification::Internal,
            )
            .expect("classified training data"));
    }
    let split = CorpusSplit {
        kind: SplitKind::Train,
        indices: (0..corpus.len()).collect(),
    };
    (corpus, split)
}

#[test]
fn linear_v1_artifact_is_rejected_by_mlp_v2_decoder() {
    let (corpus, split) = corpus();
    let trainer = NeuralTrainer::try_new(NeuralConfig {
        input_dim: 64,
        epochs: 8,
        learning_rate: 0.02,
        max_payload_bytes: 128,
    })
    .expect("linear trainer");
    let model = trainer.train(&corpus, &split).expect("linear train").0;
    let artifact = encode_neural_artifact(&model, 2_048).expect("linear artifact");

    assert!(matches!(
        load_mlp_neural_artifact(&artifact.manifest, &artifact.bytes),
        Err(MlpNeuralArtifactError::UnsupportedArchitecture)
    ));
}

#[test]
fn mlp_v2_artifact_is_rejected_by_linear_v1_decoder() {
    let (corpus, split) = corpus();
    let trainer = MlpNeuralTrainer::try_new(MlpNeuralConfig {
        input_dim: 64,
        hidden_dim: 16,
        epochs: 8,
        learning_rate: 0.02,
        max_payload_bytes: 128,
        initialization_seed: 59,
    })
    .expect("MLP trainer");
    let model = trainer.train(&corpus, &split).expect("MLP train").0;
    let artifact = encode_mlp_neural_artifact(&model, 2_048).expect("MLP artifact");

    assert!(matches!(
        load_neural_artifact(&artifact.manifest, &artifact.bytes),
        Err(NeuralArtifactError::UnsupportedArchitecture)
    ));
}
