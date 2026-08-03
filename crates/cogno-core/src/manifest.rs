//! Hostile model artifact manifest (COGNO-1 V2 §21).
//!
//! The loader treats weights and tokenizer as hostile. Before any major
//! allocation it verifies: file size, fingerprint, schema version, tensor
//! count, dimensions, numeric types, dimension multiplications (checked
//! arithmetic), rejects duplicated names, out-of-range data, unknown schema
//! fields when the schema requires it, and unsupported architectures. The
//! model format never allows code execution during loading. The core never
//! downloads weights from the network.

/// Model family. Loader rejects everything not listed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    Generic,
    TransformerDecoder,
    TransformerEncoderDecoder,
    MixtureOfExperts,
}

/// Opaque architecture id. The loader rejects unknown ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArchitectureId(pub u32);

/// Manifest schema version. Bumped only on incompatible changes.
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Errors raised while validating a manifest against an actual artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestError {
    UnknownSchemaVersion,
    FileSizeMismatch { expected: u64, actual: u64 },
    InvalidTensorCount,
    InvalidContext,
    HashMismatch,
    UnsupportedArchitecture,
}

/// Artifact manifest (§21). Hashes are SHA-256 (32 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelManifest {
    pub schema_version: u16,
    pub model_family: ModelFamily,
    pub architecture_id: ArchitectureId,
    pub tensor_count: u32,
    pub parameter_count: u64,
    pub max_context_tokens: u32,
    pub tokenizer_hash: [u8; 32],
    pub weights_hash: [u8; 32],
    pub expected_file_bytes: u64,
}

impl ModelManifest {
    /// Validate the manifest's cheap invariants against an actual file size.
    /// Hash and per-tensor checks happen in the loader where the bytes are
    /// streamed; this is the pre-allocation gate.
    pub fn validate(&self, file_bytes: u64) -> Result<(), ManifestError> {
        if self.schema_version == 0 || self.schema_version > MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::UnknownSchemaVersion);
        }
        if self.expected_file_bytes != file_bytes {
            return Err(ManifestError::FileSizeMismatch {
                expected: self.expected_file_bytes,
                actual: file_bytes,
            });
        }
        if self.tensor_count == 0 {
            return Err(ManifestError::InvalidTensorCount);
        }
        if self.max_context_tokens == 0 {
            return Err(ManifestError::InvalidContext);
        }
        Ok(())
    }
}
