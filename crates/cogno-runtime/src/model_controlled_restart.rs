//! Controlled-restart seal for a persisted shared-cognitive V4 model.
//!
//! Persistence/replay proves the immutable generation chain and hostile artifact
//! bytes. This module adds a distinct, explicit host activation attestation and
//! binds the read-only model to the exact selected generation before runtime
//! installation. It never hot-swaps an already running model.

use crate::model_persistence::PersistedModelGenerationSelection;
use cogno_model::{
    LoadedNeuralModel, SciRustSequenceCognitiveReadOnlyModel, SequenceCognitiveModelError,
    SEQUENCE_COGNITIVE_ARCHITECTURE_ID,
};

/// Explicit host authority to admit one already-persisted V4 selection during
/// runtime initialization. It grants neither tool nor hard-rule authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostModelRestartActivationAttestation {
    _private: (),
}

impl HostModelRestartActivationAttestation {
    #[must_use]
    pub const fn approve_verified_v4_for_restart_initialization() -> Self {
        Self { _private: () }
    }
}

/// Authority carried by a generation-bound V4 restart seal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlledRestartModelAuthority {
    RestartInitializationOnly,
}

/// One verified V4 read-only model sealed to an immutable persisted generation.
#[derive(Debug)]
pub struct GenerationBoundControlledRestartCognitiveModel {
    model: SciRustSequenceCognitiveReadOnlyModel,
    generation: u64,
    generation_manifest_sha256: [u8; 32],
    artifact_sha256: [u8; 32],
    validation_accuracy_bps: u16,
    test_accuracy_bps: u16,
    authority: ControlledRestartModelAuthority,
}

impl GenerationBoundControlledRestartCognitiveModel {
    /// Consume a fully replayed persisted selection and bind V4 activation to
    /// its exact selected generation and artifact digest.
    pub fn prepare(
        selection: PersistedModelGenerationSelection,
        _host: HostModelRestartActivationAttestation,
    ) -> Result<Self, ControlledRestartModelError> {
        let selected_manifest = selection
            .chain
            .selected()
            .copied()
            .ok_or(ControlledRestartModelError::NoSelectedGeneration)?;
        if selected_manifest.generation != selection.selected_generation {
            return Err(ControlledRestartModelError::SelectedGenerationMismatch {
                expected: selection.selected_generation,
                actual: selected_manifest.generation,
            });
        }
        if selected_manifest.artifact_sha256 != selection.selected_artifact.manifest.weights_hash {
            return Err(ControlledRestartModelError::ArtifactDigestMismatch);
        }
        if selection.selected_artifact.manifest.architecture_id
            != SEQUENCE_COGNITIVE_ARCHITECTURE_ID
        {
            return Err(ControlledRestartModelError::UnsupportedArchitecture);
        }
        let state = match selection.selected_model {
            LoadedNeuralModel::SequenceCognitiveV4(state) => state,
            _ => return Err(ControlledRestartModelError::UnsupportedArchitecture),
        };
        let model = state
            .into_read_only_model()
            .map_err(ControlledRestartModelError::Model)?;
        Ok(Self {
            model,
            generation: selected_manifest.generation,
            generation_manifest_sha256: selected_manifest.digest(),
            artifact_sha256: selected_manifest.artifact_sha256,
            validation_accuracy_bps: selected_manifest.validation_accuracy_bps,
            test_accuracy_bps: selected_manifest.test_accuracy_bps,
            authority: ControlledRestartModelAuthority::RestartInitializationOnly,
        })
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn generation_manifest_sha256(&self) -> [u8; 32] {
        self.generation_manifest_sha256
    }

    #[must_use]
    pub const fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }

    #[must_use]
    pub const fn validation_accuracy_bps(&self) -> u16 {
        self.validation_accuracy_bps
    }

    #[must_use]
    pub const fn test_accuracy_bps(&self) -> u16 {
        self.test_accuracy_bps
    }

    #[must_use]
    pub const fn authority(&self) -> ControlledRestartModelAuthority {
        self.authority
    }

    #[must_use]
    pub const fn model(&self) -> &SciRustSequenceCognitiveReadOnlyModel {
        &self.model
    }

    #[must_use]
    pub fn into_model(self) -> SciRustSequenceCognitiveReadOnlyModel {
        self.model
    }
}

/// Fail-closed preparation errors for controlled V4 restart activation.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlledRestartModelError {
    NoSelectedGeneration,
    SelectedGenerationMismatch { expected: u64, actual: u64 },
    ArtifactDigestMismatch,
    UnsupportedArchitecture,
    Model(SequenceCognitiveModelError),
}
