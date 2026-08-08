//! Versioned neural artifact dispatch without reinterpretation.
//!
//! The external manifest's architecture id selects one exact hostile decoder.
//! V1 linear, v2 MLP and v3 sequence artifacts are never reinterpreted as one
//! another. This keeps restart/replay compatibility explicit.

use crate::artifact::{load_neural_artifact, NeuralArtifactError, NEURAL_ARCHITECTURE_ID};
use crate::mlp::MlpNeuralModel;
use crate::mlp_artifact::{
    load_mlp_neural_artifact, MlpNeuralArtifactError, MLP_NEURAL_ARCHITECTURE_ID,
};
use crate::neural::NeuralModel;
use crate::sequence_artifact::{
    load_sequence_neural_artifact, SequenceNeuralArtifactError, SEQUENCE_NEURAL_ARCHITECTURE_ID,
};
use cogno_core::ModelManifest;
use cogno_scirust::SequenceClassifier;

/// Successfully decoded model generation, tagged by its canonical architecture.
#[derive(Clone, Debug)]
pub enum LoadedNeuralModel {
    LinearV1(NeuralModel),
    MlpV2(MlpNeuralModel),
    SequenceV3(SequenceClassifier),
}

impl LoadedNeuralModel {
    #[must_use]
    pub const fn architecture_id(&self) -> cogno_core::ArchitectureId {
        match self {
            Self::LinearV1(_) => NEURAL_ARCHITECTURE_ID,
            Self::MlpV2(_) => MLP_NEURAL_ARCHITECTURE_ID,
            Self::SequenceV3(_) => SEQUENCE_NEURAL_ARCHITECTURE_ID,
        }
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::LinearV1(model) => model.parameter_count(),
            Self::MlpV2(model) => model.parameter_count(),
            Self::SequenceV3(model) => model.parameter_count(),
        }
    }
}

/// Versioned decoding error preserving the exact selected hostile loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionedNeuralArtifactError {
    UnsupportedArchitecture,
    LinearV1(NeuralArtifactError),
    MlpV2(MlpNeuralArtifactError),
    SequenceV3(SequenceNeuralArtifactError),
}

/// Decode only the architecture explicitly named by the external manifest.
pub fn load_versioned_neural_artifact(
    manifest: &ModelManifest,
    bytes: &[u8],
) -> Result<LoadedNeuralModel, VersionedNeuralArtifactError> {
    if manifest.architecture_id == NEURAL_ARCHITECTURE_ID {
        return load_neural_artifact(manifest, bytes)
            .map(LoadedNeuralModel::LinearV1)
            .map_err(VersionedNeuralArtifactError::LinearV1);
    }
    if manifest.architecture_id == MLP_NEURAL_ARCHITECTURE_ID {
        return load_mlp_neural_artifact(manifest, bytes)
            .map(LoadedNeuralModel::MlpV2)
            .map_err(VersionedNeuralArtifactError::MlpV2);
    }
    if manifest.architecture_id == SEQUENCE_NEURAL_ARCHITECTURE_ID {
        return load_sequence_neural_artifact(manifest, bytes)
            .map(LoadedNeuralModel::SequenceV3)
            .map_err(VersionedNeuralArtifactError::SequenceV3);
    }
    Err(VersionedNeuralArtifactError::UnsupportedArchitecture)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encode_mlp_neural_artifact, encode_neural_artifact, encode_sequence_neural_artifact,
        Corpus, CorpusSplit, Label, LabeledExample, MlpNeuralConfig, MlpNeuralTrainer, NeuralConfig,
        NeuralTrainer, SplitKind, BYTE_TOKENIZER_VOCAB_SIZE,
    };
    use cogno_core::{ArchitectureId, EvidenceOrigin, InputOrigin};
    use cogno_scirust::{SequenceClassifierConfig, SequenceEncoderConfig};

    fn corpus() -> (Corpus, CorpusSplit) {
        let mut corpus = Corpus::with_seed(37);
        for (label, payload) in [
            (Label(0), b"alpha-one".as_slice()),
            (Label(0), b"alpha-two"),
            (Label(1), b"omega-one"),
            (Label(1), b"omega-two"),
        ] {
            assert!(corpus.add(LabeledExample::new(
                label,
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        let split = CorpusSplit {
            kind: SplitKind::Train,
            indices: (0..corpus.examples.len()).collect(),
        };
        (corpus, split)
    }

    #[test]
    fn dispatch_preserves_v1_linear_compatibility() {
        let (corpus, split) = corpus();
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 32,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("trainer");
        let model = trainer.train(&corpus, &split).expect("train").0;
        let artifact = encode_neural_artifact(&model, 2_048).expect("artifact");
        let loaded = load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes)
            .expect("versioned v1 load");
        assert!(matches!(loaded, LoadedNeuralModel::LinearV1(_)));
        assert_eq!(loaded.architecture_id(), NEURAL_ARCHITECTURE_ID);
        assert_eq!(loaded.parameter_count(), model.parameter_count());
    }

    #[test]
    fn dispatch_selects_v2_mlp_without_reinterpreting_v1() {
        let (corpus, split) = corpus();
        let trainer = MlpNeuralTrainer::try_new(MlpNeuralConfig {
            input_dim: 64,
            hidden_dim: 16,
            epochs: 48,
            learning_rate: 0.02,
            max_payload_bytes: 128,
            initialization_seed: 41,
        })
        .expect("trainer");
        let model = trainer.train(&corpus, &split).expect("train").0;
        let artifact = encode_mlp_neural_artifact(&model, 2_048).expect("artifact");
        let loaded = load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes)
            .expect("versioned v2 load");
        assert!(matches!(loaded, LoadedNeuralModel::MlpV2(_)));
        assert_eq!(loaded.architecture_id(), MLP_NEURAL_ARCHITECTURE_ID);
        assert_eq!(loaded.parameter_count(), model.parameter_count());
    }

    #[test]
    fn dispatch_selects_v3_sequence_model() {
        let model = SequenceClassifier::try_new(SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                max_tokens: 32,
                embedding_dim: 8,
                hidden_dim: 12,
                seed: 131,
            },
            num_classes: 3,
            head_seed: 137,
        })
        .expect("sequence classifier");
        let artifact = encode_sequence_neural_artifact(&model).expect("artifact");
        let loaded = load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes)
            .expect("versioned v3 load");
        assert!(matches!(loaded, LoadedNeuralModel::SequenceV3(_)));
        assert_eq!(loaded.architecture_id(), SEQUENCE_NEURAL_ARCHITECTURE_ID);
        assert_eq!(loaded.parameter_count(), model.parameter_count());
    }

    #[test]
    fn unknown_architecture_is_rejected_before_decoder_selection() {
        let (corpus, split) = corpus();
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 1,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("trainer");
        let model = trainer.train(&corpus, &split).expect("train").0;
        let mut artifact = encode_neural_artifact(&model, 2_048).expect("artifact");
        artifact.manifest.architecture_id = ArchitectureId(u32::MAX);
        assert!(matches!(
            load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(VersionedNeuralArtifactError::UnsupportedArchitecture)
        ));
    }
}
