//! Canonical, fail-closed neural model artifact format.
//!
//! The artifact is deliberately simple: one fixed 32-byte header followed by
//! one dense little-endian `f32` weight matrix. The loader validates the
//! external [`ModelManifest`] and every dimension/hash relationship before it
//! allocates the weight vector. No names, paths, scripts, plugins, or code are
//! embedded in the format.

use crate::neural::{
    NeuralModel, MAX_NEURAL_FEATURES, MAX_NEURAL_LABELS, MAX_NEURAL_PARAMETERS,
    MAX_NEURAL_PAYLOAD_BYTES, MIN_NEURAL_FEATURES,
};
use cogno_core::{
    ArchitectureId, ManifestError, ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;

/// Fixed binary prefix. The full header is exactly 32 bytes.
pub const NEURAL_ARTIFACT_MAGIC: [u8; 8] = *b"COGNNA01";
/// Binary artifact schema version.
pub const NEURAL_ARTIFACT_VERSION: u16 = 1;
/// Number of bytes before the first weight.
pub const NEURAL_ARTIFACT_HEADER_BYTES: usize = 32;
/// Maximum canonical artifact size accepted before hashing or parsing.
pub const MAX_NEURAL_ARTIFACT_BYTES: usize =
    NEURAL_ARTIFACT_HEADER_BYTES + MAX_NEURAL_PARAMETERS * size_of::<f32>();
/// COGNO-1 architecture id for the bounded hashed-feature linear classifier.
pub const NEURAL_ARCHITECTURE_ID: ArchitectureId = ArchitectureId(0x434f_474e);
/// This format contains exactly one logical dense weight matrix.
pub const NEURAL_TENSOR_COUNT: u32 = 1;
/// Upper bound accepted from `ModelManifest::max_context_tokens`.
pub const MAX_NEURAL_CONTEXT_TOKENS: u32 = 1_048_576;
/// Canonical description of the hard-coded deterministic byte feature encoder.
pub const NEURAL_TOKENIZER_DESCRIPTOR: &[u8] = b"cogno-byte-hash-features-v1";

/// Canonical encoded artifact and its matching manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedNeuralArtifact {
    pub bytes: Vec<u8>,
    pub manifest: ModelManifest,
}

/// Fail-closed errors for artifact encoding and hostile artifact loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuralArtifactError {
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

impl From<ManifestError> for NeuralArtifactError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// SHA-256 fingerprint of the deterministic feature encoder contract.
#[must_use]
pub fn neural_tokenizer_hash() -> [u8; 32] {
    sha256(NEURAL_TOKENIZER_DESCRIPTOR)
}

/// Encode a trained model into the canonical non-executable artifact format.
pub fn encode_neural_artifact(
    model: &NeuralModel,
    max_context_tokens: u32,
) -> Result<EncodedNeuralArtifact, NeuralArtifactError> {
    if max_context_tokens == 0 || max_context_tokens > MAX_NEURAL_CONTEXT_TOKENS {
        return Err(NeuralArtifactError::InvalidContext);
    }
    validate_model_shape(
        model.input_dim(),
        model.num_labels(),
        model.max_payload_bytes(),
        model.parameter_count(),
    )?;
    for (index, weight) in model.weights().iter().copied().enumerate() {
        if !weight.is_finite() {
            return Err(NeuralArtifactError::NonFiniteWeight { index });
        }
    }

    let weight_bytes = model
        .parameter_count()
        .checked_mul(size_of::<f32>())
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    let artifact_bytes = NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    if artifact_bytes > MAX_NEURAL_ARTIFACT_BYTES {
        return Err(NeuralArtifactError::ArtifactSizeOverflow);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(artifact_bytes)
        .map_err(|_| NeuralArtifactError::AllocationFailed)?;

    bytes.extend_from_slice(&NEURAL_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&NEURAL_ARTIFACT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&NEURAL_ARCHITECTURE_ID.0.to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(model.input_dim())
            .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&model.num_labels().to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(model.max_payload_bytes())
            .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u64::try_from(model.parameter_count())
            .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?
            .to_le_bytes(),
    );
    debug_assert_eq!(bytes.len(), NEURAL_ARTIFACT_HEADER_BYTES);
    for weight in model.weights() {
        bytes.extend_from_slice(&weight.to_bits().to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), artifact_bytes);

    let manifest = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model_family: ModelFamily::Generic,
        architecture_id: NEURAL_ARCHITECTURE_ID,
        tensor_count: NEURAL_TENSOR_COUNT,
        parameter_count: u64::try_from(model.parameter_count())
            .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?,
        max_context_tokens,
        tokenizer_hash: neural_tokenizer_hash(),
        // `weights_hash` fingerprints the complete canonical weights artifact,
        // including its shape/header metadata, not only the f32 payload.
        weights_hash: sha256(&bytes),
        expected_file_bytes: u64::try_from(bytes.len())
            .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?,
    };
    Ok(EncodedNeuralArtifact { bytes, manifest })
}

/// Load only a canonical artifact that exactly matches its external manifest.
///
/// All untrusted counts are checked against hard maxima and against the actual
/// byte length before the `Vec<f32>` allocation occurs.
pub fn load_neural_artifact(
    manifest: &ModelManifest,
    bytes: &[u8],
) -> Result<NeuralModel, NeuralArtifactError> {
    let file_bytes =
        u64::try_from(bytes.len()).map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?;
    manifest.validate(file_bytes)?;
    validate_manifest(manifest)?;
    if sha256(bytes) != manifest.weights_hash {
        return Err(NeuralArtifactError::WeightHashMismatch);
    }
    if bytes.len() < NEURAL_ARTIFACT_HEADER_BYTES {
        return Err(NeuralArtifactError::TruncatedHeader);
    }
    if bytes[..8] != NEURAL_ARTIFACT_MAGIC {
        return Err(NeuralArtifactError::InvalidMagic);
    }

    let artifact_version = read_u16(bytes, 8)?;
    if artifact_version != NEURAL_ARTIFACT_VERSION {
        return Err(NeuralArtifactError::UnsupportedArtifactVersion);
    }
    let architecture_id = ArchitectureId(read_u32(bytes, 10)?);
    let input_dim = usize::try_from(read_u32(bytes, 14)?)
        .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?;
    let num_labels = read_u16(bytes, 18)?;
    let max_payload_bytes = usize::try_from(read_u32(bytes, 20)?)
        .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?;
    let header_parameters = usize::try_from(read_u64(bytes, 24)?)
        .map_err(|_| NeuralArtifactError::InvalidParameterCount)?;

    if architecture_id != manifest.architecture_id || architecture_id != NEURAL_ARCHITECTURE_ID {
        return Err(NeuralArtifactError::HeaderMismatch);
    }
    validate_model_shape(input_dim, num_labels, max_payload_bytes, header_parameters)?;
    let manifest_parameters = usize::try_from(manifest.parameter_count)
        .map_err(|_| NeuralArtifactError::InvalidParameterCount)?;
    if header_parameters != manifest_parameters {
        return Err(NeuralArtifactError::InvalidParameterCount);
    }

    let weight_bytes = header_parameters
        .checked_mul(size_of::<f32>())
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    let expected_bytes = NEURAL_ARTIFACT_HEADER_BYTES
        .checked_add(weight_bytes)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    if expected_bytes != bytes.len() {
        return Err(NeuralArtifactError::HeaderMismatch);
    }
    let encoded_weights = &bytes[NEURAL_ARTIFACT_HEADER_BYTES..];

    let mut weights = Vec::new();
    weights
        .try_reserve_exact(header_parameters)
        .map_err(|_| NeuralArtifactError::AllocationFailed)?;
    for (index, chunk) in encoded_weights.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let weight = f32::from_bits(bits);
        if !weight.is_finite() {
            return Err(NeuralArtifactError::NonFiniteWeight { index });
        }
        weights.push(weight);
    }

    NeuralModel::from_verified_parts(input_dim, num_labels, max_payload_bytes, weights)
        .map_err(|_| NeuralArtifactError::HeaderMismatch)
}

fn validate_manifest(manifest: &ModelManifest) -> Result<(), NeuralArtifactError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(NeuralArtifactError::UnsupportedManifestSchema);
    }
    if manifest.model_family != ModelFamily::Generic {
        return Err(NeuralArtifactError::UnsupportedModelFamily);
    }
    if manifest.architecture_id != NEURAL_ARCHITECTURE_ID {
        return Err(NeuralArtifactError::UnsupportedArchitecture);
    }
    if manifest.tensor_count != NEURAL_TENSOR_COUNT {
        return Err(NeuralArtifactError::InvalidTensorCount);
    }
    if manifest.max_context_tokens == 0 || manifest.max_context_tokens > MAX_NEURAL_CONTEXT_TOKENS {
        return Err(NeuralArtifactError::InvalidContext);
    }
    let max_parameters = u64::try_from(MAX_NEURAL_PARAMETERS)
        .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.parameter_count == 0 || manifest.parameter_count > max_parameters {
        return Err(NeuralArtifactError::InvalidParameterCount);
    }
    let maximum_bytes = u64::try_from(MAX_NEURAL_ARTIFACT_BYTES)
        .map_err(|_| NeuralArtifactError::ArtifactSizeOverflow)?;
    if manifest.expected_file_bytes > maximum_bytes {
        return Err(NeuralArtifactError::ArtifactTooLarge {
            actual: manifest.expected_file_bytes,
            maximum: maximum_bytes,
        });
    }
    if manifest.tokenizer_hash != neural_tokenizer_hash() {
        return Err(NeuralArtifactError::InvalidTokenizerHash);
    }
    Ok(())
}

fn validate_model_shape(
    input_dim: usize,
    num_labels: u16,
    max_payload_bytes: usize,
    parameters: usize,
) -> Result<(), NeuralArtifactError> {
    if !(MIN_NEURAL_FEATURES..=MAX_NEURAL_FEATURES).contains(&input_dim)
        || num_labels == 0
        || num_labels > MAX_NEURAL_LABELS
        || max_payload_bytes == 0
        || max_payload_bytes > MAX_NEURAL_PAYLOAD_BYTES
        || parameters == 0
        || parameters > MAX_NEURAL_PARAMETERS
    {
        return Err(NeuralArtifactError::InvalidParameterCount);
    }
    let expected = usize::from(num_labels)
        .checked_mul(input_dim)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    if expected != parameters {
        return Err(NeuralArtifactError::InvalidParameterCount);
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, NeuralArtifactError> {
    let end = offset
        .checked_add(2)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(NeuralArtifactError::TruncatedHeader)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, NeuralArtifactError> {
    let end = offset
        .checked_add(4)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(NeuralArtifactError::TruncatedHeader)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, NeuralArtifactError> {
    let end = offset
        .checked_add(8)
        .ok_or(NeuralArtifactError::ArtifactSizeOverflow)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(NeuralArtifactError::TruncatedHeader)?;
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
        Corpus, CorpusSplit, Label, LabeledExample, NeuralConfig, NeuralTrainer, SplitKind,
    };
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn trained_model() -> NeuralModel {
        let mut corpus = Corpus::with_seed(9);
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
        let trainer = NeuralTrainer::try_new(NeuralConfig {
            input_dim: 64,
            epochs: 32,
            learning_rate: 0.02,
            max_payload_bytes: 128,
        })
        .expect("trainer");
        trainer.train(&corpus, &split).expect("train").0
    }

    #[test]
    fn canonical_artifact_roundtrips_exact_weights() {
        let model = trained_model();
        let left = encode_neural_artifact(&model, 2_048).expect("encode");
        let right = encode_neural_artifact(&model, 2_048).expect("encode repeat");
        assert_eq!(left, right);
        let loaded = load_neural_artifact(&left.manifest, &left.bytes).expect("load");
        assert_eq!(loaded.input_dim(), model.input_dim());
        assert_eq!(loaded.num_labels(), model.num_labels());
        assert_eq!(loaded.max_payload_bytes(), model.max_payload_bytes());
        assert_eq!(loaded.weights(), model.weights());
    }

    #[test]
    fn tampered_weight_bytes_fail_hash_before_decode() {
        let model = trained_model();
        let mut artifact = encode_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[NEURAL_ARTIFACT_HEADER_BYTES] ^= 0x80;
        assert!(matches!(
            load_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(NeuralArtifactError::WeightHashMismatch)
        ));
    }

    #[test]
    fn valid_hash_with_nan_weight_is_still_rejected() {
        let model = trained_model();
        let mut artifact = encode_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[NEURAL_ARTIFACT_HEADER_BYTES..NEURAL_ARTIFACT_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        artifact.manifest.weights_hash = sha256(&artifact.bytes);
        assert!(matches!(
            load_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(NeuralArtifactError::NonFiniteWeight { index: 0 })
        ));
    }

    #[test]
    fn hostile_parameter_bomb_is_rejected_before_hash_or_weight_allocation() {
        let model = trained_model();
        let mut artifact = encode_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        artifact.manifest.parameter_count = u64::MAX;
        assert!(matches!(
            load_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(NeuralArtifactError::InvalidParameterCount)
        ));
    }

    #[test]
    fn header_is_cryptographically_bound_to_manifest_hash() {
        let model = trained_model();
        let mut artifact = encode_neural_artifact(&model, 2_048).expect("encode");
        artifact.bytes[14] ^= 1;
        assert!(matches!(
            load_neural_artifact(&artifact.manifest, &artifact.bytes),
            Err(NeuralArtifactError::WeightHashMismatch)
        ));
    }

    #[test]
    fn tokenizer_architecture_and_file_size_are_bound() {
        let model = trained_model();
        let artifact = encode_neural_artifact(&model, 2_048).expect("encode");

        let mut tokenizer = artifact.manifest;
        tokenizer.tokenizer_hash[0] ^= 1;
        assert!(matches!(
            load_neural_artifact(&tokenizer, &artifact.bytes),
            Err(NeuralArtifactError::InvalidTokenizerHash)
        ));

        let mut architecture = artifact.manifest;
        architecture.architecture_id = ArchitectureId(0);
        assert!(matches!(
            load_neural_artifact(&architecture, &artifact.bytes),
            Err(NeuralArtifactError::UnsupportedArchitecture)
        ));

        let truncated = &artifact.bytes[..artifact.bytes.len() - 1];
        assert!(matches!(
            load_neural_artifact(&artifact.manifest, truncated),
            Err(NeuralArtifactError::Manifest(
                ManifestError::FileSizeMismatch { .. }
            ))
        ));
    }
}
