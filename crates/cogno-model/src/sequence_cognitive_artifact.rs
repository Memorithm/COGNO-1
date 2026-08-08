//! Canonical hostile artifact format for the shared sequence cognitive model.
//!
//! V4 persists one shared sequence encoder plus classification, preference,
//! symbolic-satisfaction and contradiction heads. Retrieval has no independent
//! tensor: it scores the same shared representation directly. The persisted
//! retrieval-candidate cap remains part of the artifact header because it is an
//! inference boundary, not a training-only detail.
//!
//! Decoding yields a verified [`SequenceCognitiveArtifactState`]. Activation as
//! a runtime model remains a separate authority-controlled step.

use crate::artifact::EncodedNeuralArtifact;
use crate::tokenizer::{byte_tokenizer_hash, BYTE_TOKENIZER_VOCAB_SIZE, MAX_BYTE_TOKENIZER_TOKENS};
use cogno_core::{
    ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
};
use cogno_scirust::{
    SequenceCognitiveConfig, SequenceCognitiveHeads, SequenceEncoder, SequenceEncoderConfig,
    COGNITIVE_CONTRADICTION_CLASSES, MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_COGNITIVE_PARAMETERS,
    MAX_SEQUENCE_EMBEDDING_DIM, MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_RETRIEVAL_CANDIDATES,
    MAX_SEQUENCE_SYMBOLIC_RULES,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;

/// Fixed magic for the shared cognitive V4 artifact.
pub const SEQUENCE_COGNITIVE_ARTIFACT_MAGIC: [u8; 8] = *b"CGCOG004";
/// Binary schema version for the shared cognitive artifact.
pub const SEQUENCE_COGNITIVE_ARTIFACT_VERSION: u16 = 4;
/// COGNO architecture id for the shared sequence cognitive model (`COG4`).
pub const SEQUENCE_COGNITIVE_ARCHITECTURE_ID: ArchitectureId = ArchitectureId(0x434F_4734);
/// Encoder(3) + classification(2) + preference(2) + symbolic(2) + contradiction(2).
pub const SEQUENCE_COGNITIVE_TENSOR_COUNT: u32 = 11;
/// Fixed bytes before the first scalar tensor payload.
pub const SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES: usize = 88;
/// Maximum accepted V4 artifact size.
pub const MAX_SEQUENCE_COGNITIVE_ARTIFACT_BYTES: usize = SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES
    + MAX_SEQUENCE_COGNITIVE_PARAMETERS * size_of::<f32>();

/// Verified persisted state. Runtime activation is intentionally separate.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceCognitiveArtifactState {
    heads: SequenceCognitiveHeads,
    max_retrieval_candidates: usize,
}

impl SequenceCognitiveArtifactState {
    #[must_use]
    pub const fn heads(&self) -> &SequenceCognitiveHeads {
        &self.heads
    }

    #[must_use]
    pub const fn max_retrieval_candidates(&self) -> usize {
        self.max_retrieval_candidates
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.heads.parameter_count()
    }

    #[must_use]
    pub fn into_parts(self) -> (SequenceCognitiveHeads, usize) {
        (self.heads, self.max_retrieval_candidates)
    }
}

/// Fail-closed errors for V4 encoding and loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceCognitiveArtifactError {
    Manifest(ManifestError),
    UnsupportedManifestSchema,
    UnsupportedModelFamily,
    UnsupportedArchitecture,
    InvalidTensorCount,
    InvalidContext,
    InvalidParameterCount,
    InvalidRetrievalCandidateCap,
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

impl From<ManifestError> for SequenceCognitiveArtifactError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Encode one verified shared cognitive parameterization into canonical V4.
pub fn encode_sequence_cognitive_artifact(
    heads: &SequenceCognitiveHeads,
    max_retrieval_candidates: usize,
) -> Result<EncodedNeuralArtifact, SequenceCognitiveArtifactError> {
    let config = heads.config();
    let layout = validate_config(config, heads.parameter_count(), max_retrieval_candidates)?;
    validate_weights(heads)?;

    let weight_bytes = layout
        .parameters
        .checked_mul(size_of::<f32>())
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    let artifact_bytes = SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    if artifact_bytes > MAX_SEQUENCE_COGNITIVE_ARTIFACT_BYTES {
        return Err(SequenceCognitiveArtifactError::ArtifactTooLarge {
            actual: to_u64(artifact_bytes)?,
            maximum: to_u64(MAX_SEQUENCE_COGNITIVE_ARTIFACT_BYTES)?,
        });
    }

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(artifact_bytes)
        .map_err(|_| SequenceCognitiveArtifactError::AllocationFailed)?;
    bytes.extend_from_slice(&SEQUENCE_COGNITIVE_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&SEQUENCE_COGNITIVE_ARTIFACT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&SEQUENCE_COGNITIVE_ARCHITECTURE_ID.0.to_le_bytes());
    put_u32(&mut bytes, config.encoder.vocab_size)?;
    put_u32(&mut bytes, config.encoder.max_tokens)?;
    put_u32(&mut bytes, config.encoder.embedding_dim)?;
    put_u32(&mut bytes, config.encoder.hidden_dim)?;
    put_u16(&mut bytes, config.num_classes)?;
    put_u16(&mut bytes, config.num_rules)?;
    put_u16(&mut bytes, max_retrieval_candidates)?;
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&config.encoder.seed.to_le_bytes());
    bytes.extend_from_slice(&config.classification_seed.to_le_bytes());
    bytes.extend_from_slice(&config.preference_seed.to_le_bytes());
    bytes.extend_from_slice(&config.symbolic_seed.to_le_bytes());
    bytes.extend_from_slice(&config.contradiction_seed.to_le_bytes());
    bytes.extend_from_slice(&to_u64(layout.parameters)?.to_le_bytes());
    debug_assert_eq!(bytes.len(), SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES);

    append_f32s(&mut bytes, heads.encoder().token_embeddings());
    append_f32s(&mut bytes, heads.encoder().position_embeddings());
    append_f32s(&mut bytes, heads.encoder().mixing_weights());
    append_f32s(&mut bytes, heads.classification_weights());
    append_f32s(&mut bytes, heads.classification_bias());
    append_f32s(&mut bytes, heads.preference_weights());
    append_f32s(&mut bytes, heads.preference_bias());
    append_f32s(&mut bytes, heads.symbolic_weights());
    append_f32s(&mut bytes, heads.symbolic_bias());
    append_f32s(&mut bytes, heads.contradiction_weights());
    append_f32s(&mut bytes, heads.contradiction_bias());
    debug_assert_eq!(bytes.len(), artifact_bytes);

    let manifest = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model_family: ModelFamily::Generic,
        architecture_id: SEQUENCE_COGNITIVE_ARCHITECTURE_ID,
        tensor_count: SEQUENCE_COGNITIVE_TENSOR_COUNT,
        parameter_count: to_u64(layout.parameters)?,
        max_context_tokens: u32::try_from(config.encoder.max_tokens)
            .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)?,
        tokenizer_hash: byte_tokenizer_hash(),
        weights_hash: sha256(&bytes),
        expected_file_bytes: to_u64(bytes.len())?,
    };
    Ok(EncodedNeuralArtifact { bytes, manifest })
}

/// Decode V4 only after manifest, checksum, header, shape and exact-size checks.
pub fn load_sequence_cognitive_artifact(
    manifest: &ModelManifest,
    bytes: &[u8],
) -> Result<SequenceCognitiveArtifactState, SequenceCognitiveArtifactError> {
    let file_bytes = to_u64(bytes.len())?;
    manifest.validate(file_bytes)?;
    validate_manifest(manifest)?;
    if sha256(bytes) != manifest.weights_hash {
        return Err(SequenceCognitiveArtifactError::WeightHashMismatch);
    }
    if bytes.len() < SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES {
        return Err(SequenceCognitiveArtifactError::TruncatedHeader);
    }
    if bytes[..8] != SEQUENCE_COGNITIVE_ARTIFACT_MAGIC {
        return Err(SequenceCognitiveArtifactError::InvalidMagic);
    }
    if read_u16(bytes, 8)? != SEQUENCE_COGNITIVE_ARTIFACT_VERSION {
        return Err(SequenceCognitiveArtifactError::UnsupportedArtifactVersion);
    }

    let architecture_id = ArchitectureId(read_u32(bytes, 10)?);
    if architecture_id != SEQUENCE_COGNITIVE_ARCHITECTURE_ID
        || architecture_id != manifest.architecture_id
    {
        return Err(SequenceCognitiveArtifactError::HeaderMismatch);
    }
    let vocab_size = read_usize_u32(bytes, 14)?;
    let max_tokens = read_usize_u32(bytes, 18)?;
    let embedding_dim = read_usize_u32(bytes, 22)?;
    let hidden_dim = read_usize_u32(bytes, 26)?;
    let num_classes = usize::from(read_u16(bytes, 30)?);
    let num_rules = usize::from(read_u16(bytes, 32)?);
    let max_retrieval_candidates = usize::from(read_u16(bytes, 34)?);
    if read_u32(bytes, 36)? != 0 {
        return Err(SequenceCognitiveArtifactError::HeaderMismatch);
    }
    let config = SequenceCognitiveConfig {
        encoder: SequenceEncoderConfig {
            vocab_size,
            max_tokens,
            embedding_dim,
            hidden_dim,
            seed: read_u64(bytes, 40)?,
        },
        num_classes,
        num_rules,
        classification_seed: read_u64(bytes, 48)?,
        preference_seed: read_u64(bytes, 56)?,
        symbolic_seed: read_u64(bytes, 64)?,
        contradiction_seed: read_u64(bytes, 72)?,
    };
    let header_parameters = usize::try_from(read_u64(bytes, 80)?)
        .map_err(|_| SequenceCognitiveArtifactError::InvalidParameterCount)?;
    let layout = validate_config(config, header_parameters, max_retrieval_candidates)?;

    let manifest_parameters = usize::try_from(manifest.parameter_count)
        .map_err(|_| SequenceCognitiveArtifactError::InvalidParameterCount)?;
    if manifest_parameters != header_parameters
        || usize::try_from(manifest.max_context_tokens)
            .map_err(|_| SequenceCognitiveArtifactError::InvalidContext)?
            != max_tokens
    {
        return Err(SequenceCognitiveArtifactError::HeaderMismatch);
    }
    let expected_bytes = SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES
        .checked_add(
            layout
                .parameters
                .checked_mul(size_of::<f32>())
                .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?,
        )
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    if expected_bytes != bytes.len() {
        return Err(SequenceCognitiveArtifactError::HeaderMismatch);
    }

    let mut offset = SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES;
    let mut scalar_index = 0usize;
    let token_embeddings = decode_f32s(bytes, &mut offset, layout.token_embeddings, &mut scalar_index)?;
    let position_embeddings =
        decode_f32s(bytes, &mut offset, layout.position_embeddings, &mut scalar_index)?;
    let mixing_weights = decode_f32s(bytes, &mut offset, layout.mixing_weights, &mut scalar_index)?;
    let classification_weights =
        decode_f32s(bytes, &mut offset, layout.classification_weights, &mut scalar_index)?;
    let classification_bias =
        decode_f32s(bytes, &mut offset, layout.classification_bias, &mut scalar_index)?;
    let preference_weights =
        decode_f32s(bytes, &mut offset, layout.preference_weights, &mut scalar_index)?;
    let preference_bias =
        decode_f32s(bytes, &mut offset, layout.preference_bias, &mut scalar_index)?;
    let symbolic_weights =
        decode_f32s(bytes, &mut offset, layout.symbolic_weights, &mut scalar_index)?;
    let symbolic_bias = decode_f32s(bytes, &mut offset, layout.symbolic_bias, &mut scalar_index)?;
    let contradiction_weights =
        decode_f32s(bytes, &mut offset, layout.contradiction_weights, &mut scalar_index)?;
    let contradiction_bias =
        decode_f32s(bytes, &mut offset, layout.contradiction_bias, &mut scalar_index)?;
    if offset != bytes.len() || scalar_index != layout.parameters {
        return Err(SequenceCognitiveArtifactError::HeaderMismatch);
    }

    let encoder = SequenceEncoder::from_parts(
        config.encoder,
        token_embeddings,
        position_embeddings,
        mixing_weights,
    )
    .map_err(|_| SequenceCognitiveArtifactError::HeaderMismatch)?;
    let heads = SequenceCognitiveHeads::from_parts(
        config,
        encoder,
        classification_weights,
        classification_bias,
        preference_weights,
        preference_bias,
        symbolic_weights,
        symbolic_bias,
        contradiction_weights,
        contradiction_bias,
    )
    .map_err(|_| SequenceCognitiveArtifactError::HeaderMismatch)?;
    Ok(SequenceCognitiveArtifactState {
        heads,
        max_retrieval_candidates,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CognitiveLayout {
    token_embeddings: usize,
    position_embeddings: usize,
    mixing_weights: usize,
    classification_weights: usize,
    classification_bias: usize,
    preference_weights: usize,
    preference_bias: usize,
    symbolic_weights: usize,
    symbolic_bias: usize,
    contradiction_weights: usize,
    contradiction_bias: usize,
    parameters: usize,
}

fn validate_manifest(manifest: &ModelManifest) -> Result<(), SequenceCognitiveArtifactError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(SequenceCognitiveArtifactError::UnsupportedManifestSchema);
    }
    if manifest.model_family != ModelFamily::Generic {
        return Err(SequenceCognitiveArtifactError::UnsupportedModelFamily);
    }
    if manifest.architecture_id != SEQUENCE_COGNITIVE_ARCHITECTURE_ID {
        return Err(SequenceCognitiveArtifactError::UnsupportedArchitecture);
    }
    if manifest.tensor_count != SEQUENCE_COGNITIVE_TENSOR_COUNT {
        return Err(SequenceCognitiveArtifactError::InvalidTensorCount);
    }
    if manifest.max_context_tokens < 3
        || manifest.max_context_tokens
            > u32::try_from(MAX_BYTE_TOKENIZER_TOKENS)
                .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)?
    {
        return Err(SequenceCognitiveArtifactError::InvalidContext);
    }
    if manifest.tokenizer_hash != byte_tokenizer_hash() {
        return Err(SequenceCognitiveArtifactError::InvalidTokenizerHash);
    }
    let max_parameters = to_u64(MAX_SEQUENCE_COGNITIVE_PARAMETERS)?;
    if manifest.parameter_count == 0 || manifest.parameter_count > max_parameters {
        return Err(SequenceCognitiveArtifactError::InvalidParameterCount);
    }
    let maximum_bytes = to_u64(MAX_SEQUENCE_COGNITIVE_ARTIFACT_BYTES)?;
    if manifest.expected_file_bytes > maximum_bytes {
        return Err(SequenceCognitiveArtifactError::ArtifactTooLarge {
            actual: manifest.expected_file_bytes,
            maximum: maximum_bytes,
        });
    }
    Ok(())
}

fn validate_config(
    config: SequenceCognitiveConfig,
    parameters: usize,
    max_retrieval_candidates: usize,
) -> Result<CognitiveLayout, SequenceCognitiveArtifactError> {
    if config.encoder.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE
        || !(3..=MAX_BYTE_TOKENIZER_TOKENS).contains(&config.encoder.max_tokens)
        || config.encoder.embedding_dim == 0
        || config.encoder.embedding_dim > MAX_SEQUENCE_EMBEDDING_DIM
        || config.encoder.hidden_dim == 0
        || config.encoder.hidden_dim > MAX_SEQUENCE_HIDDEN_DIM
        || config.num_classes < 2
        || config.num_classes > MAX_SEQUENCE_CLASSES
        || config.num_rules == 0
        || config.num_rules > MAX_SEQUENCE_SYMBOLIC_RULES
        || parameters == 0
        || parameters > MAX_SEQUENCE_COGNITIVE_PARAMETERS
    {
        return Err(SequenceCognitiveArtifactError::InvalidParameterCount);
    }
    if !(2..=MAX_SEQUENCE_RETRIEVAL_CANDIDATES).contains(&max_retrieval_candidates) {
        return Err(SequenceCognitiveArtifactError::InvalidRetrievalCandidateCap);
    }
    let expected = config
        .parameter_count()
        .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    if expected != parameters {
        return Err(SequenceCognitiveArtifactError::InvalidParameterCount);
    }

    let token_embeddings = checked_mul(config.encoder.vocab_size, config.encoder.embedding_dim)?;
    let position_embeddings = checked_mul(config.encoder.max_tokens, config.encoder.embedding_dim)?;
    let mixing_weights = checked_mul(config.encoder.embedding_dim, config.encoder.hidden_dim)?;
    let classification_weights = checked_mul(config.encoder.hidden_dim, config.num_classes)?;
    let symbolic_weights = checked_mul(config.encoder.hidden_dim, config.num_rules)?;
    let contradiction_weights = checked_mul(
        config.encoder.hidden_dim,
        COGNITIVE_CONTRADICTION_CLASSES,
    )?;
    Ok(CognitiveLayout {
        token_embeddings,
        position_embeddings,
        mixing_weights,
        classification_weights,
        classification_bias: config.num_classes,
        preference_weights: config.encoder.hidden_dim,
        preference_bias: 1,
        symbolic_weights,
        symbolic_bias: config.num_rules,
        contradiction_weights,
        contradiction_bias: COGNITIVE_CONTRADICTION_CLASSES,
        parameters,
    })
}

fn validate_weights(heads: &SequenceCognitiveHeads) -> Result<(), SequenceCognitiveArtifactError> {
    let mut index = 0usize;
    for values in [
        heads.encoder().token_embeddings(),
        heads.encoder().position_embeddings(),
        heads.encoder().mixing_weights(),
        heads.classification_weights(),
        heads.classification_bias(),
        heads.preference_weights(),
        heads.preference_bias(),
        heads.symbolic_weights(),
        heads.symbolic_bias(),
        heads.contradiction_weights(),
        heads.contradiction_bias(),
    ] {
        for &value in values {
            if !value.is_finite() {
                return Err(SequenceCognitiveArtifactError::NonFiniteWeight { index });
            }
            index = index
                .checked_add(1)
                .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
        }
    }
    if index != heads.parameter_count() {
        return Err(SequenceCognitiveArtifactError::InvalidParameterCount);
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize) -> Result<usize, SequenceCognitiveArtifactError> {
    left.checked_mul(right)
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)
}

fn to_u64(value: usize) -> Result<u64, SequenceCognitiveArtifactError> {
    u64::try_from(value).map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)
}

fn put_u16(bytes: &mut Vec<u8>, value: usize) -> Result<(), SequenceCognitiveArtifactError> {
    bytes.extend_from_slice(
        &u16::try_from(value)
            .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn put_u32(bytes: &mut Vec<u8>, value: usize) -> Result<(), SequenceCognitiveArtifactError> {
    bytes.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)?
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
) -> Result<Vec<f32>, SequenceCognitiveArtifactError> {
    let section_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    let end = offset
        .checked_add(section_bytes)
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    let encoded = bytes
        .get(*offset..end)
        .ok_or(SequenceCognitiveArtifactError::TruncatedHeader)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| SequenceCognitiveArtifactError::AllocationFailed)?;
    for chunk in encoded.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(SequenceCognitiveArtifactError::NonFiniteWeight {
                index: *scalar_index,
            });
        }
        values.push(value);
        *scalar_index = scalar_index
            .checked_add(1)
            .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    }
    *offset = end;
    Ok(values)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, SequenceCognitiveArtifactError> {
    let slice = read_exact(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, SequenceCognitiveArtifactError> {
    let slice = read_exact(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, SequenceCognitiveArtifactError> {
    let slice = read_exact(bytes, offset, 8)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_usize_u32(bytes: &[u8], offset: usize) -> Result<usize, SequenceCognitiveArtifactError> {
    usize::try_from(read_u32(bytes, offset)?)
        .map_err(|_| SequenceCognitiveArtifactError::ArtifactSizeOverflow)
}

fn read_exact(
    bytes: &[u8],
    offset: usize,
    len: usize,
) -> Result<&[u8], SequenceCognitiveArtifactError> {
    let end = offset
        .checked_add(len)
        .ok_or(SequenceCognitiveArtifactError::ArtifactSizeOverflow)?;
    bytes
        .get(offset..end)
        .ok_or(SequenceCognitiveArtifactError::TruncatedHeader)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heads() -> SequenceCognitiveHeads {
        SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                max_tokens: 32,
                embedding_dim: 8,
                hidden_dim: 12,
                seed: 1709,
            },
            num_classes: 3,
            num_rules: 4,
            classification_seed: 1721,
            preference_seed: 1723,
            symbolic_seed: 1733,
            contradiction_seed: 1741,
        })
        .expect("bounded cognitive heads")
    }

    #[test]
    fn cognitive_v4_roundtrips_exactly() {
        let heads = heads();
        let left = encode_sequence_cognitive_artifact(&heads, 8).expect("encode");
        let right = encode_sequence_cognitive_artifact(&heads, 8).expect("repeat encode");
        assert_eq!(left, right);
        assert_eq!(
            left.manifest.architecture_id,
            SEQUENCE_COGNITIVE_ARCHITECTURE_ID
        );
        assert_eq!(left.manifest.tensor_count, SEQUENCE_COGNITIVE_TENSOR_COUNT);
        assert_eq!(left.manifest.tokenizer_hash, byte_tokenizer_hash());
        let loaded = load_sequence_cognitive_artifact(&left.manifest, &left.bytes).expect("load");
        assert_eq!(loaded.heads(), &heads);
        assert_eq!(loaded.max_retrieval_candidates(), 8);
        assert_eq!(loaded.parameter_count(), heads.parameter_count());
    }

    #[test]
    fn tampered_cognitive_bytes_fail_hash_before_decode() {
        let heads = heads();
        let mut artifact = encode_sequence_cognitive_artifact(&heads, 8).expect("encode");
        artifact.bytes[SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES] ^= 0x80;
        assert!(matches!(
            load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceCognitiveArtifactError::WeightHashMismatch)
        ));
    }

    #[test]
    fn valid_hash_with_non_finite_cognitive_weight_fails_closed() {
        let heads = heads();
        let mut artifact = encode_sequence_cognitive_artifact(&heads, 8).expect("encode");
        artifact.bytes[SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES
            ..SEQUENCE_COGNITIVE_ARTIFACT_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceCognitiveArtifactError::NonFiniteWeight { index: 0 })
        ));
    }

    #[test]
    fn hostile_retrieval_cap_fails_closed_after_valid_hash() {
        let heads = heads();
        let mut artifact = encode_sequence_cognitive_artifact(&heads, 8).expect("encode");
        artifact.bytes[34..36].copy_from_slice(&1u16.to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert_eq!(
            load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes),
            Err(SequenceCognitiveArtifactError::InvalidRetrievalCandidateCap)
        );
    }
}
