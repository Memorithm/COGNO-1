//! Monotonic, hash-chained scientific-taste generations.
//!
//! A live runtime still installs exactly one profile. Legitimate evolution is
//! represented by a new persisted generation, verified before the next runtime
//! starts. Rollback is explicit and can only target a known generation.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Genesis predecessor digest.
pub const GENESIS_DIGEST: [u8; 32] = [0; 32];

/// One immutable generation manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteGenerationManifest {
    pub generation: u64,
    pub previous_manifest_sha256: [u8; 32],
    pub profile_sha256: [u8; 32],
    pub replay_sha256: [u8; 32],
    pub candidate_report_sha256: [u8; 32],
    pub validation_store_sha256: [u8; 32],
}

impl TasteGenerationManifest {
    /// Canonical deterministic bytes used for hashing and persistence.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 32 * 5);
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes.extend_from_slice(&self.previous_manifest_sha256);
        bytes.extend_from_slice(&self.profile_sha256);
        bytes.extend_from_slice(&self.replay_sha256);
        bytes.extend_from_slice(&self.candidate_report_sha256);
        bytes.extend_from_slice(&self.validation_store_sha256);
        bytes
    }

    /// SHA-256 of canonical manifest bytes.
    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        digest(&self.canonical_bytes())
    }
}

/// Failure while extending or selecting a generation chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteGenerationError {
    ZeroGeneration,
    GenerationAlreadyExists,
    NonMonotonicGeneration,
    PreviousDigestMismatch,
    UnknownRollbackGeneration,
}

/// In-memory index of immutable, verified generation manifests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TasteGenerationChain {
    manifests: BTreeMap<u64, TasteGenerationManifest>,
    selected_generation: Option<u64>,
}

impl TasteGenerationChain {
    /// Append exactly the next generation and verify its predecessor link.
    pub fn append(
        &mut self,
        manifest: TasteGenerationManifest,
    ) -> Result<(), TasteGenerationError> {
        if manifest.generation == 0 {
            return Err(TasteGenerationError::ZeroGeneration);
        }
        if self.manifests.contains_key(&manifest.generation) {
            return Err(TasteGenerationError::GenerationAlreadyExists);
        }

        match self.manifests.last_key_value() {
            None => {
                if manifest.generation != 1 {
                    return Err(TasteGenerationError::NonMonotonicGeneration);
                }
                if manifest.previous_manifest_sha256 != GENESIS_DIGEST {
                    return Err(TasteGenerationError::PreviousDigestMismatch);
                }
            }
            Some((generation, previous)) => {
                if manifest.generation != generation.saturating_add(1) {
                    return Err(TasteGenerationError::NonMonotonicGeneration);
                }
                if manifest.previous_manifest_sha256 != previous.sha256() {
                    return Err(TasteGenerationError::PreviousDigestMismatch);
                }
            }
        }

        self.selected_generation = Some(manifest.generation);
        self.manifests.insert(manifest.generation, manifest);
        Ok(())
    }

    /// Explicitly select a known historical generation for the next startup.
    pub fn select_rollback(&mut self, generation: u64) -> Result<(), TasteGenerationError> {
        if !self.manifests.contains_key(&generation) {
            return Err(TasteGenerationError::UnknownRollbackGeneration);
        }
        self.selected_generation = Some(generation);
        Ok(())
    }

    #[must_use]
    pub fn selected(&self) -> Option<&TasteGenerationManifest> {
        self.selected_generation
            .and_then(|generation| self.manifests.get(&generation))
    }

    #[must_use]
    pub fn latest(&self) -> Option<&TasteGenerationManifest> {
        self.manifests.last_key_value().map(|(_, manifest)| manifest)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(generation: u64, previous: [u8; 32], marker: u8) -> TasteGenerationManifest {
        TasteGenerationManifest {
            generation,
            previous_manifest_sha256: previous,
            profile_sha256: [marker; 32],
            replay_sha256: [marker.wrapping_add(1); 32],
            candidate_report_sha256: [marker.wrapping_add(2); 32],
            validation_store_sha256: [marker.wrapping_add(3); 32],
        }
    }

    #[test]
    fn chain_is_monotonic_and_hash_linked() {
        let mut chain = TasteGenerationChain::default();
        let first = manifest(1, GENESIS_DIGEST, 1);
        let second = manifest(2, first.sha256(), 2);
        chain.append(first).expect("first");
        chain.append(second).expect("second");
        assert_eq!(chain.latest().map(|item| item.generation), Some(2));
    }

    #[test]
    fn stale_or_forked_generation_is_rejected() {
        let mut chain = TasteGenerationChain::default();
        chain
            .append(manifest(1, GENESIS_DIGEST, 1))
            .expect("first");
        assert_eq!(
            chain.append(manifest(3, [9; 32], 3)),
            Err(TasteGenerationError::NonMonotonicGeneration)
        );
    }

    #[test]
    fn rollback_requires_a_known_generation() {
        let mut chain = TasteGenerationChain::default();
        chain
            .append(manifest(1, GENESIS_DIGEST, 1))
            .expect("first");
        assert_eq!(
            chain.select_rollback(9),
            Err(TasteGenerationError::UnknownRollbackGeneration)
        );
        chain.select_rollback(1).expect("known rollback");
        assert_eq!(chain.selected().map(|item| item.generation), Some(1));
    }
}
