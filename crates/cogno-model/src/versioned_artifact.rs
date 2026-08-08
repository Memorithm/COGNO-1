//! Versioned neural artifact dispatch without reinterpretation.
//!
//! The external manifest's architecture id selects one exact hostile decoder.
//! A v1 linear artifact is never parsed as an MLP and a v2 MLP artifact is
//! never parsed as v1. This keeps restart/replay compatibility explicit for
//! later runtime integration.

use crate::artifact::{load_neural_artifact, NeuralArtifactError, NEURAL_ARCHITECTURE_ID};
use crate::mlp::MlpNeuralModel;
use crate::mlp_artifact::{
    load_mlp_neural_artifact, MlpNeuralArtifactError, MLP_NEURAL_ARCHITECTURE_ID,
};
use crate::neural::NeuralModel;
use cogno_core::ModelManifest;

/// Successfully decoded model generation, tagged by its canonical architecture.
#[derive(Clone, Debug)]
pub enum LoadedNeuralModel {
    LinearV1(NeuralModel),
    MlpV2(MlpNeuralModel),
}

impl LoadedNeuralModel {
    #[must_use]
    pub const fn architecture_id(&self) -> cogno_core::ArchitectureId {
        match self {
            Self::LinearV1(_) => NEURAL_ARCHITECTURE_ID,
            Self::MlpV2(_) => MLP_NEURAL_ARCHITECTURE_ID,
        }
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::LinearV1(model) => model.parameter_count(),
            Self::MlpV2(model) => model.parameter_count(),
        }
    }
}

/// Versioned decoding error preserving the exact selected hostile loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionedNeuralArtifactError {
    UnsupportedArchitecture,
    LinearV1(NeuralArtifactError),
    MlpV2(MlpNeuralArtifactError),
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
    Err(VersionedNeuralArtifactError::UnsupportedArchitecture)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encode_mlp_neural_artifact, encode_neural_artifact, Corpus, CorpusSplit, Label,
        LabeledExample, MlpNeuralConfig, MlpNeuralTrainer, NeuralConfig, NeuralTrainer, SplitKind,
    };
    use cogno_core::{ArchitectureId, EvidenceOrigin, InputOrigin};

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
        assert_eq!(
            load_versioned_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(VersionedNeuralArtifactError::UnsupportedArchitecture)
        );
    }
}
