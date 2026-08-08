//! Immutable generation chain for controlled neural-model promotion.
//!
//! This namespace is deliberately separate from scientific-taste generations.
//! Each generation binds the exact neural artifact, its external
//! `ModelManifest`, and the held-out metrics that justified persistence.

use sha2::{Digest, Sha256};

/// Digest used by the first model generation.
pub const MODEL_GENESIS_DIGEST: [u8; 32] = [0; 32];
/// Fixed canonical byte width of one generation manifest.
pub const MODEL_GENERATION_MANIFEST_BYTES: usize = 8 + 32 * 3 + 2 + 2;

/// Immutable manifest for one persisted neural-model candidate generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelGenerationManifest {
    pub generation: u64,
    pub previous_manifest_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    pub model_manifest_sha256: [u8; 32],
    pub validation_accuracy_bps: u16,
    pub test_accuracy_bps: u16,
}

impl ModelGenerationManifest {
    /// Fixed-width canonical encoding used both on disk and for chain hashing.
    #[must_use]
    pub fn canonical_bytes(&self) -> [u8; MODEL_GENERATION_MANIFEST_BYTES] {
        let mut bytes = [0u8; MODEL_GENERATION_MANIFEST_BYTES];
        let mut offset = 0usize;
        put(&mut bytes, &mut offset, &self.generation.to_le_bytes());
        put(&mut bytes, &mut offset, &self.previous_manifest_sha256);
        put(&mut bytes, &mut offset, &self.artifact_sha256);
        put(&mut bytes, &mut offset, &self.model_manifest_sha256);
        put(
            &mut bytes,
            &mut offset,
            &self.validation_accuracy_bps.to_le_bytes(),
        );
        put(
            &mut bytes,
            &mut offset,
            &self.test_accuracy_bps.to_le_bytes(),
        );
        debug_assert_eq!(offset, MODEL_GENERATION_MANIFEST_BYTES);
        bytes
    }

    /// SHA-256 of the canonical manifest bytes.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }
}

/// Fail-closed generation-chain errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelGenerationError {
    ZeroGeneration,
    NonMonotonicGeneration,
    InvalidGenesisLink,
    PreviousManifestMismatch,
    ZeroArtifactDigest,
    ZeroModelManifestDigest,
    AccuracyOutOfRange,
}

/// In-memory reconstruction of the immutable model generation chain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelGenerationChain {
    manifests: Vec<ModelGenerationManifest>,
}

impl ModelGenerationChain {
    /// Append only the exact next generation with the exact predecessor hash.
    pub fn append(
        &mut self,
        manifest: ModelGenerationManifest,
    ) -> Result<(), ModelGenerationError> {
        validate_manifest(&manifest)?;
        match self.manifests.last() {
            None => {
                if manifest.generation != 1 {
                    return Err(ModelGenerationError::NonMonotonicGeneration);
                }
                if manifest.previous_manifest_sha256 != MODEL_GENESIS_DIGEST {
                    return Err(ModelGenerationError::InvalidGenesisLink);
                }
            }
            Some(previous) => {
                let expected_generation = previous
                    .generation
                    .checked_add(1)
                    .ok_or(ModelGenerationError::NonMonotonicGeneration)?;
                if manifest.generation != expected_generation {
                    return Err(ModelGenerationError::NonMonotonicGeneration);
                }
                if manifest.previous_manifest_sha256 != previous.digest() {
                    return Err(ModelGenerationError::PreviousManifestMismatch);
                }
            }
        }
        self.manifests.push(manifest);
        Ok(())
    }

    #[must_use]
    pub fn selected(&self) -> Option<&ModelGenerationManifest> {
        self.manifests.last()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    #[must_use]
    pub fn manifests(&self) -> &[ModelGenerationManifest] {
        &self.manifests
    }
}

fn validate_manifest(manifest: &ModelGenerationManifest) -> Result<(), ModelGenerationError> {
    if manifest.generation == 0 {
        return Err(ModelGenerationError::ZeroGeneration);
    }
    if manifest.artifact_sha256 == [0; 32] {
        return Err(ModelGenerationError::ZeroArtifactDigest);
    }
    if manifest.model_manifest_sha256 == [0; 32] {
        return Err(ModelGenerationError::ZeroModelManifestDigest);
    }
    if manifest.validation_accuracy_bps > 10_000 || manifest.test_accuracy_bps > 10_000 {
        return Err(ModelGenerationError::AccuracyOutOfRange);
    }
    Ok(())
}

fn put<const N: usize>(target: &mut [u8; N], offset: &mut usize, value: &[u8]) {
    let end = *offset + value.len();
    target[*offset..end].copy_from_slice(value);
    *offset = end;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(generation: u64, previous_manifest_sha256: [u8; 32]) -> ModelGenerationManifest {
        ModelGenerationManifest {
            generation,
            previous_manifest_sha256,
            artifact_sha256: [generation as u8; 32],
            model_manifest_sha256: [generation.saturating_add(1) as u8; 32],
            validation_accuracy_bps: 8_000,
            test_accuracy_bps: 8_000,
        }
    }

    #[test]
    fn chain_is_strictly_monotonic_and_hash_linked() {
        let first = manifest(1, MODEL_GENESIS_DIGEST);
        let second = manifest(2, first.digest());
        let mut chain = ModelGenerationChain::default();
        chain.append(first).expect("first");
        chain.append(second).expect("second");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.selected(), Some(&second));
    }

    #[test]
    fn fork_and_skipped_generation_fail_closed() {
        let first = manifest(1, MODEL_GENESIS_DIGEST);
        let mut chain = ModelGenerationChain::default();
        chain.append(first).expect("first");
        assert_eq!(
            chain.append(manifest(3, first.digest())),
            Err(ModelGenerationError::NonMonotonicGeneration)
        );
        assert_eq!(
            chain.append(manifest(2, [9; 32])),
            Err(ModelGenerationError::PreviousManifestMismatch)
        );
    }
}
