//! Canonical hostile artifact format for the trainable sequence classifier.
//!
//! V1 linear and v2 MLP artifacts retain their exact decoders. This module
//! introduces a separate v3 architecture bound to the deterministic byte
//! tokenizer and stores five trainable tensors: token embeddings, positional
//! embeddings, encoder mixing weights, classifier head weights and head bias.
//! All manifest fields, dimensions and exact byte counts are checked before
//! tensor allocation.

use crate::artifact::EncodedNeuralArtifact;
use crate::tokenizer::{
    byte_tokenizer_hash, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS,
};
use cogno_core::{
    ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
};
use cogno_scirust::{
    SequenceClassifier, SequenceClassifierConfig, SequenceEncoder, SequenceEncoderConfig,
    MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_CLASSIFIER_PARAMETERS, MAX_SEQUENCE_EMBEDDING_DIM,
    MAX_SEQUENCE_HIDDEN_DIM,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;

/// Fixed magic for sequence-classifier artifacts.
pub const SEQUENCE_NEURAL_ARTIFACT_MAGIC: [u8; 8] = *b"CGSEQ003";
/// Binary schema version for sequence-classifier artifacts.
pub const SEQUENCE_NEURAL_ARTIFACT_VERSION: u16 = 3;
/// COGNO architecture id for the bounded sequence classifier (`SEQ3`).
pub const SEQUENCE_NEURAL_ARCHITECTURE_ID: ArchitectureId = ArchitectureId(0x5345_5133);
/// Token embeddings, position embeddings, mixing weights, head weights, bias.
pub const SEQUENCE_NEURAL_TENSOR_COUNT: u32 = 5;
/// Fixed bytes before the first scalar tensor payload.
pub const SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES: usize = 56;
/// Maximum accepted v3 artifact size.
pub const MAX_SEQUENCE_NEURAL_ARTIFACT_BYTES: usize = SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES
    + MAX_SEQUENCE_CLASSIFIER_PARAMETERS * size_of::<f32>();

/// Fail-closed errors for sequence artifact encoding and loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceNeuralArtifactError {
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

impl From<ManifestError> for SequenceNeuralArtifactError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Encode a frozen sequence classifier into the canonical five-tensor v3
/// artifact. The manifest context is the exact configured sequence length.
pub fn encode_sequence_neural_artifact(
    model: &SequenceClassifier,
) -> Result<EncodedNeuralArtifact, SequenceNeuralArtifactError> {
    let config = model.config();
    let layout = validate_config(config, model.parameter_count())?;
    validate_weights(model)?;

    let weight_bytes = layout
        .parameters
        .checked_mul(size_of::<f32>())
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let artifact_bytes = SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    if artifact_bytes > MAX_SEQUENCE_NEURAL_ARTIFACT_BYTES {
        return Err(SequenceNeuralArtifactError::ArtifactTooLarge {
            actual: u64::try_from(artifact_bytes)
                .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
            maximum: u64::try_from(MAX_SEQUENCE_NEURAL_ARTIFACT_BYTES)
                .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
        });
    }

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(artifact_bytes)
        .map_err(|_| SequenceNeuralArtifactError::AllocationFailed)?;
    bytes.extend_from_slice(&SEQUENCE_NEURAL_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&SEQUENCE_NEURAL_ARTIFACT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&SEQUENCE_NEURAL_ARCHITECTURE_ID.0.to_le_bytes());
    put_u32(&mut bytes, config.encoder.vocab_size)?;
    put_u32(&mut bytes, config.encoder.max_tokens)?;
    put_u32(&mut bytes, config.encoder.embedding_dim)?;
    put_u32(&mut bytes, config.encoder.hidden_dim)?;
    bytes.extend_from_slice(
        &u16::try_from(config.num_classes)
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&config.encoder.seed.to_le_bytes());
    bytes.extend_from_slice(&config.head_seed.to_le_bytes());
    bytes.extend_from_slice(
        &u64::try_from(layout.parameters)
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    debug_assert_eq!(bytes.len(), SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES);

    append_f32s(&mut bytes, model.encoder().token_embeddings());
    append_f32s(&mut bytes, model.encoder().position_embeddings());
    append_f32s(&mut bytes, model.encoder().mixing_weights());
    append_f32s(&mut bytes, model.head_weights());
    append_f32s(&mut bytes, model.head_bias());
    debug_assert_eq!(bytes.len(), artifact_bytes);

    let manifest = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model_family: ModelFamily::Generic,
        architecture_id: SEQUENCE_NEURAL_ARCHITECTURE_ID,
        tensor_count: SEQUENCE_NEURAL_TENSOR_COUNT,
        parameter_count: u64::try_from(layout.parameters)
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
        max_context_tokens: u32::try_from(config.encoder.max_tokens)
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
        tokenizer_hash: byte_tokenizer_hash(),
        weights_hash: sha256(&bytes),
        expected_file_bytes: u64::try_from(bytes.len())
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
    };
    Ok(EncodedNeuralArtifact { bytes, manifest })
}

/// Decode v3 only after manifest, hash, shape and exact-length validation.
pub fn load_sequence_neural_artifact(
    manifest: &ModelManifest,
    bytes: &[u8],
) -> Result<SequenceClassifier, SequenceNeuralArtifactError> {
    let file_bytes = u64::try_from(bytes.len())
        .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    manifest.validate(file_bytes)?;
    validate_manifest(manifest)?;
    if sha256(bytes) != manifest.weights_hash {
        return Err(SequenceNeuralArtifactError::WeightHashMismatch);
    }
    if bytes.len() < SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES {
        return Err(SequenceNeuralArtifactError::TruncatedHeader);
    }
    if bytes[..8] != SEQUENCE_NEURAL_ARTIFACT_MAGIC {
        return Err(SequenceNeuralArtifactError::InvalidMagic);
    }

    let version = read_u16(bytes, 8)?;
    if version != SEQUENCE_NEURAL_ARTIFACT_VERSION {
        return Err(SequenceNeuralArtifactError::UnsupportedArtifactVersion);
    }
    let architecture_id = ArchitectureId(read_u32(bytes, 10)?);
    let vocab_size = read_usize_u32(bytes, 14)?;
    let max_tokens = read_usize_u32(bytes, 18)?;
    let embedding_dim = read_usize_u32(bytes, 22)?;
    let hidden_dim = read_usize_u32(bytes, 26)?;
    let num_classes = usize::from(read_u16(bytes, 30)?);
    let encoder_seed = read_u64(bytes, 32)?;
    let head_seed = read_u64(bytes, 40)?;
    let header_parameters = usize::try_from(read_u64(bytes, 48)?)
        .map_err(|_| SequenceNeuralArtifactError::InvalidParameterCount)?;

    if architecture_id != manifest.architecture_id
        || architecture_id != SEQUENCE_NEURAL_ARCHITECTURE_ID
    {
        return Err(SequenceNeuralArtifactError::HeaderMismatch);
    }
    let config = SequenceClassifierConfig {
        encoder: SequenceEncoderConfig {
            vocab_size,
            max_tokens,
            embedding_dim,
            hidden_dim,
            seed: encoder_seed,
        },
        num_classes,
        head_seed,
    };
    let layout = validate_config(config, header_parameters)?;
    let manifest_parameters = usize::try_from(manifest.parameter_count)
        .map_err(|_| SequenceNeuralArtifactError::InvalidParameterCount)?;
    if manifest_parameters != header_parameters
        || usize::try_from(manifest.max_context_tokens)
            .map_err(|_| SequenceNeuralArtifactError::InvalidContext)?
            != max_tokens
    {
        return Err(SequenceNeuralArtifactError::HeaderMismatch);
    }

    let expected_bytes = SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(
            layout
                .parameters
                .checked_mul(size_of::<f32>())
                .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?,
        )
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    if expected_bytes != bytes.len() {
        return Err(SequenceNeuralArtifactError::HeaderMismatch);
    }

    let mut offset = SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES;
    let mut scalar_index = 0usize;
    let token_embeddings = decode_f32s(
        bytes,
        &mut offset,
        layout.token_embeddings,
        &mut scalar_index,
    )?;
    let position_embeddings = decode_f32s(
        bytes,
        &mut offset,
        layout.position_embeddings,
        &mut scalar_index,
    )?;
    let mixing_weights = decode_f32s(
        bytes,
        &mut offset,
        layout.mixing_weights,
        &mut scalar_index,
    )?;
    let head_weights = decode_f32s(
        bytes,
        &mut offset,
        layout.head_weights,
        &mut scalar_index,
    )?;
    let head_bias = decode_f32s(bytes, &mut offset, layout.head_bias, &mut scalar_index)?;
    if offset != bytes.len() || scalar_index != layout.parameters {
        return Err(SequenceNeuralArtifactError::HeaderMismatch);
    }

    let encoder = SequenceEncoder::from_parts(
        config.encoder,
        token_embeddings,
        position_embeddings,
        mixing_weights,
    )
    .map_err(|_| SequenceNeuralArtifactError::HeaderMismatch)?;
    SequenceClassifier::from_parts(config, encoder, head_weights, head_bias)
        .map_err(|_| SequenceNeuralArtifactError::HeaderMismatch)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SequenceLayout {
    token_embeddings: usize,
    position_embeddings: usize,
    mixing_weights: usize,
    head_weights: usize,
    head_bias: usize,
    parameters: usize,
}

fn validate_manifest(manifest: &ModelManifest) -> Result<(), SequenceNeuralArtifactError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(SequenceNeuralArtifactError::UnsupportedManifestSchema);
    }
    if manifest.model_family != ModelFamily::Generic {
        return Err(SequenceNeuralArtifactError::UnsupportedModelFamily);
    }
    if manifest.architecture_id != SEQUENCE_NEURAL_ARCHITECTURE_ID {
        return Err(SequenceNeuralArtifactError::UnsupportedArchitecture);
    }
    if manifest.tensor_count != SEQUENCE_NEURAL_TENSOR_COUNT {
        return Err(SequenceNeuralArtifactError::InvalidTensorCount);
    }
    if manifest.max_context_tokens < 2
        || manifest.max_context_tokens
            > u32::try_from(MAX_BYTE_TOKENIZER_TOKENS)
                .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?
    {
        return Err(SequenceNeuralArtifactError::InvalidContext);
    }
    if manifest.tokenizer_hash != byte_tokenizer_hash() {
        return Err(SequenceNeuralArtifactError::InvalidTokenizerHash);
    }
    let max_parameters = u64::try_from(MAX_SEQUENCE_CLASSIFIER_PARAMETERS)
        .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.parameter_count == 0 || manifest.parameter_count > max_parameters {
        return Err(SequenceNeuralArtifactError::InvalidParameterCount);
    }
    let maximum_bytes = u64::try_from(MAX_SEQUENCE_NEURAL_ARTIFACT_BYTES)
        .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.expected_file_bytes > maximum_bytes {
        return Err(SequenceNeuralArtifactError::ArtifactTooLarge {
            actual: manifest.expected_file_bytes,
            maximum: maximum_bytes,
        });
    }
    Ok(())
}

fn validate_config(
    config: SequenceClassifierConfig,
    parameters: usize,
) -> Result<SequenceLayout, SequenceNeuralArtifactError> {
    if config.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.encoder.max_tokens)
        || config.encoder.embedding_dim == 0
        || config.encoder.embedding_dim > MAX_SEQUENCE_EMBEDDING_DIM
        || config.encoder.hidden_dim == 0
        || config.encoder.hidden_dim > MAX_SEQUENCE_HIDDEN_DIM
        || config.num_classes < 2
        || config.num_classes > MAX_SEQUENCE_CLASSES
        || parameters == 0
        || parameters > MAX_SEQUENCE_CLASSIFIER_PARAMETERS
    {
        return Err(SequenceNeuralArtifactError::InvalidParameterCount);
    }
    let expected = config
        .parameter_count()
        .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    if expected != parameters {
        return Err(SequenceNeuralArtifactError::InvalidParameterCount);
    }
    let token_embeddings = config
        .encoder
        .vocab_size
        .checked_mul(config.encoder.embedding_dim)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let position_embeddings = config
        .encoder
        .max_tokens
        .checked_mul(config.encoder.embedding_dim)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let mixing_weights = config
        .encoder
        .embedding_dim
        .checked_mul(config.encoder.hidden_dim)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let head_weights = config
        .encoder
        .hidden_dim
        .checked_mul(config.num_classes)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    Ok(SequenceLayout {
        token_embeddings,
        position_embeddings,
        mixing_weights,
        head_weights,
        head_bias: config.num_classes,
        parameters,
    })
}

fn validate_weights(model: &SequenceClassifier) -> Result<(), SequenceNeuralArtifactError> {
    let mut index = 0usize;
    for values in [
        model.encoder().token_embeddings(),
        model.encoder().position_embeddings(),
        model.encoder().mixing_weights(),
        model.head_weights(),
        model.head_bias(),
    ] {
        for &value in values {
            if !value.is_finite() {
                return Err(SequenceNeuralArtifactError::NonFiniteWeight { index });
            }
            index = index
                .checked_add(1)
                .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
        }
    }
    if index != model.parameter_count() {
        return Err(SequenceNeuralArtifactError::InvalidParameterCount);
    }
    Ok(())
}

fn put_u32(bytes: &mut Vec<u8>, value: usize) -> Result<(), SequenceNeuralArtifactError> {
    bytes.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
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
) -> Result<Vec<f32>, SequenceNeuralArtifactError> {
    let section_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let end = offset
        .checked_add(section_bytes)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let encoded = bytes
        .get(*offset..end)
        .ok_or(SequenceNeuralArtifactError::TruncatedHeader)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| SequenceNeuralArtifactError::AllocationFailed)?;
    for chunk in encoded.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(SequenceNeuralArtifactError::NonFiniteWeight {
                index: *scalar_index,
            });
        }
        values.push(value);
        *scalar_index = scalar_index
            .checked_add(1)
            .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    }
    *offset = end;
    Ok(values)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, SequenceNeuralArtifactError> {
    let end = offset
        .checked_add(2)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(SequenceNeuralArtifactError::TruncatedHeader)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, SequenceNeuralArtifactError> {
    let end = offset
        .checked_add(4)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(SequenceNeuralArtifactError::TruncatedHeader)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_usize_u32(bytes: &[u8], offset: usize) -> Result<usize, SequenceNeuralArtifactError> {
    usize::try_from(read_u32(bytes, offset)?)
        .map_err(|_| SequenceNeuralArtifactError::ArtifactSizeOverflow)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, SequenceNeuralArtifactError> {
    let end = offset
        .checked_add(8)
        .ok_or(SequenceNeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(SequenceNeuralArtifactError::TruncatedHeader)?;
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

    fn model() -> SequenceClassifier {
        SequenceClassifier::try_new(SequenceClassifierConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                max_tokens: 32,
                embedding_dim: 8,
                hidden_dim: 12,
                seed: 113,
            },
            num_classes: 3,
            head_seed: 127,
        })
        .expect("bounded sequence classifier")
    }

    #[test]
    fn sequence_v3_roundtrips_exactly() {
        let model = model();
        let left = encode_sequence_neural_artifact(&model).expect("encode");
        let right = encode_sequence_neural_artifact(&model).expect("repeat encode");
        assert_eq!(left, right);
        assert_eq!(left.manifest.architecture_id, SEQUENCE_NEURAL_ARCHITECTURE_ID);
        assert_eq!(left.manifest.tensor_count, SEQUENCE_NEURAL_TENSOR_COUNT);
        assert_eq!(left.manifest.tokenizer_hash, byte_tokenizer_hash());
        assert_eq!(left.manifest.max_context_tokens, 32);
        let loaded = load_sequence_neural_artifact(&left.manifest, &left.bytes).expect("load");
        assert_eq!(loaded, model);
    }

    #[test]
    fn tampered_sequence_bytes_fail_hash_before_decode() {
        let model = model();
        let mut artifact = encode_sequence_neural_artifact(&model).expect("encode");
        artifact.bytes[SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES] ^= 0x80;
        assert!(matches!(
            load_sequence_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceNeuralArtifactError::WeightHashMismatch)
        ));
    }

    #[test]
    fn valid_hash_with_non_finite_sequence_weight_fails_closed() {
        let model = model();
        let mut artifact = encode_sequence_neural_artifact(&model).expect("encode");
        artifact.bytes
            [SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES..SEQUENCE_NEURAL_ARTIFACT_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_sequence_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceNeuralArtifactError::NonFiniteWeight { index: 0 })
        ));
    }

    #[test]
    fn wrong_tokenizer_hash_is_rejected() {
        let model = model();
        let mut artifact = encode_sequence_neural_artifact(&model).expect("encode");
        artifact.manifest.tokenizer_hash = [0x55; 32];
        assert!(matches!(
            load_sequence_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceNeuralArtifactError::InvalidTokenizerHash)
        ));
    }

    #[test]
    fn hostile_vocabulary_fails_before_tensor_allocation() {
        let model = model();
        let mut artifact = encode_sequence_neural_artifact(&model).expect("encode");
        artifact.bytes[14..18].copy_from_slice(&260u32.to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_sequence_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceNeuralArtifactError::InvalidParameterCount)
        ));
    }
}
