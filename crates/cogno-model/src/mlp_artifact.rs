//! Canonical hostile artifact format for the bounded nonlinear MLP.
//!
//! Version 1 remains owned by [`crate::artifact`] and is never reinterpreted.
//! This module defines a separate architecture id, magic, fixed header and
//! four-tensor layout for the nonlinear model. All dimensions and exact byte
//! counts are validated before any weight allocation.

use crate::artifact::{neural_tokenizer_hash, EncodedNeuralArtifact, MAX_NEURAL_CONTEXT_TOKENS};
use crate::mlp::{MlpNeuralModel, MAX_MLP_HIDDEN_FEATURES, MIN_MLP_HIDDEN_FEATURES};
use crate::neural::{
    MAX_NEURAL_FEATURES, MAX_NEURAL_LABELS, MAX_NEURAL_PAYLOAD_BYTES, MIN_NEURAL_FEATURES,
};
use cogno_core::{
    ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
};
use cogno_scirust::{MlpConfig, MAX_MLP_PARAMETERS};
use sha2::{Digest, Sha256};
use std::mem::size_of;

/// Fixed magic for nonlinear neural artifacts.
pub const MLP_NEURAL_ARTIFACT_MAGIC: [u8; 8] = *b"CGMLP002";
/// Binary schema version for nonlinear artifacts.
pub const MLP_NEURAL_ARTIFACT_VERSION: u16 = 2;
/// COGNO architecture id for the bounded one-hidden-layer MLP.
pub const MLP_NEURAL_ARCHITECTURE_ID: ArchitectureId = ArchitectureId(0x4d4c_5032);
/// The artifact stores W1, b1, W2 and b2 as four logical tensors.
pub const MLP_NEURAL_TENSOR_COUNT: u32 = 4;
/// Fixed header size before the first tensor scalar.
pub const MLP_NEURAL_ARTIFACT_HEADER_BYTES: usize = 44;
/// Maximum accepted nonlinear artifact size.
pub const MAX_MLP_NEURAL_ARTIFACT_BYTES: usize =
    MLP_NEURAL_ARTIFACT_HEADER_BYTES + MAX_MLP_PARAMETERS * size_of::<f32>();

/// Fail-closed errors for MLP artifact encoding and loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpNeuralArtifactError {
    Manifest(ManifestError),
    UnsupportedManifestSchema,
    UnsupportedModelFamily,
    UnsupportedArchitecture,
    InvalidTensorCount,
    InvalidContext,
    InvalidParameterCount,
    InvalidTokenizerHash,
    TruncatedHeader,
    InvalidMagic,
    UnsupportedArtifactVersion,
    HeaderMismatch,
    ArtifactSizeOverflow,
    ArtifactTooLarge { actual: u64, maximum: u64 },
    WeightHashMismatch,
    NonFiniteWeight { index: usize },
    AllocationFailed,
}

impl From<ManifestError> for MlpNeuralArtifactError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Encode a frozen nonlinear model into the canonical four-tensor artifact.
pub fn encode_mlp_neural_artifact(
    model: &MlpNeuralModel,
    max_context_tokens: u32,
) -> Result<EncodedNeuralArtifact, MlpNeuralArtifactError> {
    if max_context_tokens == 0 || max_context_tokens > MAX_NEURAL_CONTEXT_TOKENS {
        return Err(MlpNeuralArtifactError::InvalidContext);
    }
    let layout = validate_model_shape(
        model.input_dim(),
        model.hidden_dim(),
        model.num_labels(),
        model.max_payload_bytes(),
        model.parameter_count(),
    )?;
    validate_weights(model)?;

    let weight_bytes = layout
        .parameters
        .checked_mul(size_of::<f32>())
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let artifact_bytes = MLP_NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    if artifact_bytes > MAX_MLP_NEURAL_ARTIFACT_BYTES {
        return Err(MlpNeuralArtifactError::ArtifactSizeOverflow);
    }

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(artifact_bytes)
        .map_err(|_| MlpNeuralArtifactError::AllocationFailed)?;
    bytes.extend_from_slice(&MLP_NEURAL_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&MLP_NEURAL_ARTIFACT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&MLP_NEURAL_ARCHITECTURE_ID.0.to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(model.input_dim())
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u32::try_from(model.hidden_dim())
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&model.num_labels().to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(model.max_payload_bytes())
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&model.initialization_seed().to_le_bytes());
    bytes.extend_from_slice(
        &u64::try_from(layout.parameters)
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    debug_assert_eq!(bytes.len(), MLP_NEURAL_ARTIFACT_HEADER_BYTES);

    append_f32s(&mut bytes, model.input_hidden());
    append_f32s(&mut bytes, model.hidden_bias());
    append_f32s(&mut bytes, model.hidden_output());
    append_f32s(&mut bytes, model.output_bias());
    debug_assert_eq!(bytes.len(), artifact_bytes);

    let manifest = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model_family: ModelFamily::Generic,
        architecture_id: MLP_NEURAL_ARCHITECTURE_ID,
        tensor_count: MLP_NEURAL_TENSOR_COUNT,
        parameter_count: u64::try_from(layout.parameters)
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?,
        max_context_tokens,
        tokenizer_hash: neural_tokenizer_hash(),
        weights_hash: sha256(&bytes),
        expected_file_bytes: u64::try_from(bytes.len())
            .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?,
    };
    Ok(EncodedNeuralArtifact { bytes, manifest })
}

/// Decode an MLP artifact only after manifest, shape, hash and byte validation.
pub fn load_mlp_neural_artifact(
    manifest: &ModelManifest,
    bytes: &[u8],
) -> Result<MlpNeuralModel, MlpNeuralArtifactError> {
    let file_bytes =
        u64::try_from(bytes.len()).map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    manifest.validate(file_bytes)?;
    validate_manifest(manifest)?;
    if sha256(bytes) != manifest.weights_hash {
        return Err(MlpNeuralArtifactError::WeightHashMismatch);
    }
    if bytes.len() < MLP_NEURAL_ARTIFACT_HEADER_BYTES {
        return Err(MlpNeuralArtifactError::TruncatedHeader);
    }
    if bytes[..8] != MLP_NEURAL_ARTIFACT_MAGIC {
        return Err(MlpNeuralArtifactError::InvalidMagic);
    }

    let artifact_version = read_u16(bytes, 8)?;
    if artifact_version != MLP_NEURAL_ARTIFACT_VERSION {
        return Err(MlpNeuralArtifactError::UnsupportedArtifactVersion);
    }
    let architecture_id = ArchitectureId(read_u32(bytes, 10)?);
    let input_dim = usize::try_from(read_u32(bytes, 14)?)
        .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let hidden_dim = usize::try_from(read_u32(bytes, 18)?)
        .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let num_labels = read_u16(bytes, 22)?;
    let max_payload_bytes = usize::try_from(read_u32(bytes, 24)?)
        .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let initialization_seed = read_u64(bytes, 28)?;
    let header_parameters = usize::try_from(read_u64(bytes, 36)?)
        .map_err(|_| MlpNeuralArtifactError::InvalidParameterCount)?;

    if architecture_id != manifest.architecture_id || architecture_id != MLP_NEURAL_ARCHITECTURE_ID
    {
        return Err(MlpNeuralArtifactError::HeaderMismatch);
    }
    let layout = validate_model_shape(
        input_dim,
        hidden_dim,
        num_labels,
        max_payload_bytes,
        header_parameters,
    )?;
    let manifest_parameters = usize::try_from(manifest.parameter_count)
        .map_err(|_| MlpNeuralArtifactError::InvalidParameterCount)?;
    if header_parameters != manifest_parameters {
        return Err(MlpNeuralArtifactError::InvalidParameterCount);
    }

    let weight_bytes = layout
        .parameters
        .checked_mul(size_of::<f32>())
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let expected_bytes = MLP_NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    if expected_bytes != bytes.len() {
        return Err(MlpNeuralArtifactError::HeaderMismatch);
    }

    let mut offset = MLP_NEURAL_ARTIFACT_HEADER_BYTES;
    let mut scalar_index = 0usize;
    let input_hidden = decode_f32s(bytes, &mut offset, layout.input_hidden, &mut scalar_index)?;
    let hidden_bias = decode_f32s(bytes, &mut offset, layout.hidden_bias, &mut scalar_index)?;
    let hidden_output = decode_f32s(bytes, &mut offset, layout.hidden_output, &mut scalar_index)?;
    let output_bias = decode_f32s(bytes, &mut offset, layout.output_bias, &mut scalar_index)?;
    if offset != bytes.len() || scalar_index != layout.parameters {
        return Err(MlpNeuralArtifactError::HeaderMismatch);
    }

    MlpNeuralModel::from_verified_parts(
        input_dim,
        hidden_dim,
        num_labels,
        max_payload_bytes,
        initialization_seed,
        input_hidden,
        hidden_bias,
        hidden_output,
        output_bias,
    )
    .map_err(|_| MlpNeuralArtifactError::HeaderMismatch)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MlpLayout {
    input_hidden: usize,
    hidden_bias: usize,
    hidden_output: usize,
    output_bias: usize,
    parameters: usize,
}

fn validate_manifest(manifest: &ModelManifest) -> Result<(), MlpNeuralArtifactError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(MlpNeuralArtifactError::UnsupportedManifestSchema);
    }
    if manifest.model_family != ModelFamily::Generic {
        return Err(MlpNeuralArtifactError::UnsupportedModelFamily);
    }
    if manifest.architecture_id != MLP_NEURAL_ARCHITECTURE_ID {
        return Err(MlpNeuralArtifactError::UnsupportedArchitecture);
    }
    if manifest.tensor_count != MLP_NEURAL_TENSOR_COUNT {
        return Err(MlpNeuralArtifactError::InvalidTensorCount);
    }
    if manifest.max_context_tokens == 0 || manifest.max_context_tokens > MAX_NEURAL_CONTEXT_TOKENS {
        return Err(MlpNeuralArtifactError::InvalidContext);
    }
    let max_parameters = u64::try_from(MAX_MLP_PARAMETERS)
        .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.parameter_count == 0 || manifest.parameter_count > max_parameters {
        return Err(MlpNeuralArtifactError::InvalidParameterCount);
    }
    let maximum_bytes = u64::try_from(MAX_MLP_NEURAL_ARTIFACT_BYTES)
        .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.expected_file_bytes > maximum_bytes {
        return Err(MlpNeuralArtifactError::ArtifactTooLarge {
            actual: manifest.expected_file_bytes,
            maximum: maximum_bytes,
        });
    }
    if manifest.tokenizer_hash != neural_tokenizer_hash() {
        return Err(MlpNeuralArtifactError::InvalidTokenizerHash);
    }
    Ok(())
}

fn validate_model_shape(
    input_dim: usize,
    hidden_dim: usize,
    num_labels: u16,
    max_payload_bytes: usize,
    parameters: usize,
) -> Result<MlpLayout, MlpNeuralArtifactError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&input_dim)
        || !(MIN_MLP_HIDDEN_FEATURES..=MAX_MLP_HIDDEN_FEATURES).contains(&hidden_dim)
        || num_labels == 0
        || num_labels > MAX_NEURAL_LABELS
        || max_payload_bytes == 0
        || max_payload_bytes > MAX_NEURAL_PAYLOAD_BYTES
        || parameters == 0
        || parameters > MAX_MLP_PARAMETERS
    {
        return Err(MlpNeuralArtifactError::InvalidParameterCount);
    }
    let expected = MlpConfig {
        input_dim,
        hidden_dim,
        output_dim: usize::from(num_labels),
        seed: 0,
    }
    .parameter_count()
    .map_err(|_| MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    if expected != parameters {
        return Err(MlpNeuralArtifactError::InvalidParameterCount);
    }
    let input_hidden = input_dim
        .checked_mul(hidden_dim)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let hidden_output = hidden_dim
        .checked_mul(usize::from(num_labels))
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    Ok(MlpLayout {
        input_hidden,
        hidden_bias: hidden_dim,
        hidden_output,
        output_bias: usize::from(num_labels),
        parameters,
    })
}

fn validate_weights(model: &MlpNeuralModel) -> Result<(), MlpNeuralArtifactError> {
    let mut index = 0usize;
    for weights in [
        model.input_hidden(),
        model.hidden_bias(),
        model.hidden_output(),
        model.output_bias(),
    ] {
        for &weight in weights {
            if !weight.is_finite() {
                return Err(MlpNeuralArtifactError::NonFiniteWeight { index });
            }
            index = index
                .checked_add(1)
                .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
        }
    }
    if index != model.parameter_count() {
        return Err(MlpNeuralArtifactError::InvalidParameterCount);
    }
    Ok(())
}

fn append_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
    for &value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn decode_f32s(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
    scalar_index: &mut usize,
) -> Result<Vec<f32>, MlpNeuralArtifactError> {
    let section_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let end = offset
        .checked_add(section_bytes)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let encoded = bytes
        .get(*offset..end)
        .ok_or(MlpNeuralArtifactError::TruncatedHeader)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| MlpNeuralArtifactError::AllocationFailed)?;
    for chunk in encoded.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(MlpNeuralArtifactError::NonFiniteWeight {
                index: *scalar_index,
            });
        }
        values.push(value);
        *scalar_index = scalar_index
            .checked_add(1)
            .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    }
    *offset = end;
    Ok(values)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, MlpNeuralArtifactError> {
    let end = offset
        .checked_add(2)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(MlpNeuralArtifactError::TruncatedHeader)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, MlpNeuralArtifactError> {
    let end = offset
        .checked_add(4)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(MlpNeuralArtifactError::TruncatedHeader)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, MlpNeuralArtifactError> {
    let end = offset
        .checked_add(8)
        .ok_or(MlpNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(MlpNeuralArtifactError::TruncatedHeader)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Corpus, CorpusSplit, Label, LabeledExample, MlpNeuralConfig, MlpNeuralTrainer, SplitKind,
    };
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn trained_model() -> MlpNeuralModel {
        let mut corpus = Corpus::with_seed(27);
        for (label, payload) in [
            (Label(0), b"alpha-alpha".as_slice()),
            (Label(0), b"alpha-style"),
            (Label(1), b"omega-omega"),
            (Label(1), b"omega-policy"),
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
        let trainer = MlpNeuralTrainer::try_new(MlpNeuralConfig {
            input_dim: 64,
            hidden_dim: 16,
            epochs: 48,
            learning_rate: 0.02,
            max_payload_bytes: 128,
            initialization_seed: 29,
        })
        .expect("trainer");
        trainer.train(&corpus, &split).expect("train").0
    }

    #[test]
    fn canonical_mlp_artifact_roundtrips_exact_tensors() {
        let model = trained_model();
        let left = encode_mlp_neural_artifact(&model, 2_048).expect("encode");
        let right = encode_mlp_neural_artifact(&model, 2_048).expect("encode repeat");
        assert_eq!(left, right);
        assert_eq!(left.manifest.architecture_id, MLP_NEURAL_ARCHITECTURE_ID);
        assert_eq!(left.manifest.tensor_count, MLP_NEURAL_TENSOR_COUNT);
        let loaded = load_mlp_neural_artifact(&left.manifest, &left.bytes).expect("load");
        assert_eq!(loaded, model);
    }

    #[test]
    fn tampered_mlp_bytes_fail_hash_before_decode() {
        let model = trained_model();
        let mut artifact = encode_mlp_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[MLP_NEURAL_ARTIFACT_HEADER_BYTES] ^= 0x80;
        assert!(matches!(
            load_mlp_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(MlpNeuralArtifactError::WeightHashMismatch)
        ));
    }

    #[test]
    fn valid_hash_with_non_finite_mlp_weight_fails_closed() {
        let model = trained_model();
        let mut artifact = encode_mlp_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[MLP_NEURAL_ARTIFACT_HEADER_BYTES..MLP_NEURAL_ARTIFACT_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_mlp_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(MlpNeuralArtifactError::NonFiniteWeight { index: 0 })
        ));
    }

    #[test]
    fn hostile_hidden_dimension_fails_before_weight_allocation() {
        let model = trained_model();
        let mut artifact = encode_mlp_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[18..22].copy_from_slice(
            &(u32::try_from(MAX_MLP_HIDDEN_FEATURES + 1).expect("u32")).to_le_bytes(),
        );
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_mlp_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(MlpNeuralArtifactError::InvalidParameterCount)
        ));
    }
}
