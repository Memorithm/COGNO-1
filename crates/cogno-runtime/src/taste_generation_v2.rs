//! Explicit v2 scientific-taste generation manifests with SciRust provenance.
//!
//! The historical [`crate::taste_generation::TasteGenerationManifest`] remains
//! byte-for-byte and hash-for-hash unchanged. V2 uses a distinct canonical
//! domain and adds the digest of `scirust.validation.receipts`, allowing a mixed
//! v1 -> v2 chain without reinterpreting any persisted v1 generation.

use crate::taste_generation::{TasteGenerationError, TasteGenerationManifest, GENESIS_DIGEST};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Explicit schema version for provenance-sealed taste generations.
pub const TASTE_GENERATION_V2_SCHEMA_VERSION: u16 = 2;
/// Fixed v2 persistence/hash domain. Its presence makes a v2 manifest
/// unambiguous from the historical 168-byte v1 representation.
pub const TASTE_GENERATION_V2_MAGIC: [u8; 16] = *b"COGNO-TASTE-GEN2";
/// Canonical byte length of one v2 manifest.
pub const TASTE_GENERATION_V2_MANIFEST_BYTES: usize = 16 + 2 + 8 + 32 * 6;

/// Immutable v2 generation manifest that seals both ordinary validation state
/// and authenticated SciRust execution-provenance receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteGenerationManifestV2 {
    pub schema_version: u16,
    pub generation: u64,
    pub previous_manifest_sha256: [u8; 32],
    pub profile_sha256: [u8; 32],
    pub replay_sha256: [u8; 32],
    pub candidate_report_sha256: [u8; 32],
    pub validation_store_sha256: [u8; 32],
    pub scirust_validation_receipts_sha256: [u8; 32],
}

impl TasteGenerationManifestV2 {
    /// Construct one v2 manifest with an explicit, non-forgeable schema value.
    #[must_use]
    pub const fn new(
        generation: u64,
        previous_manifest_sha256: [u8; 32],
        profile_sha256: [u8; 32],
        replay_sha256: [u8; 32],
        candidate_report_sha256: [u8; 32],
        validation_store_sha256: [u8; 32],
        scirust_validation_receipts_sha256: [u8; 32],
    ) -> Self {
        Self {
            schema_version: TASTE_GENERATION_V2_SCHEMA_VERSION,
            generation,
            previous_manifest_sha256,
            profile_sha256,
            replay_sha256,
            candidate_report_sha256,
            validation_store_sha256,
            scirust_validation_receipts_sha256,
        }
    }

    /// Canonical deterministic v2 bytes used for hashing and persistence.
    ///
    /// This format is deliberately unrelated to the v1 byte layout:
    /// `magic || schema || generation || six SHA-256 fields`.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TASTE_GENERATION_V2_MANIFEST_BYTES);
        bytes.extend_from_slice(&TASTE_GENERATION_V2_MAGIC);
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes.extend_from_slice(&self.previous_manifest_sha256);
        bytes.extend_from_slice(&self.profile_sha256);
        bytes.extend_from_slice(&self.replay_sha256);
        bytes.extend_from_slice(&self.candidate_report_sha256);
        bytes.extend_from_slice(&self.validation_store_sha256);
        bytes.extend_from_slice(&self.scirust_validation_receipts_sha256);
        debug_assert_eq!(bytes.len(), TASTE_GENERATION_V2_MANIFEST_BYTES);
        bytes
    }

    /// SHA-256 of the explicitly domain-separated v2 canonical bytes.
    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    /// Validate invariants that distinguish a real v2 manifest from malformed
    /// or caller-mutated data before chain admission/persistence.
    pub fn validate(&self) -> Result<(), TasteGenerationV2Error> {
        if self.schema_version != TASTE_GENERATION_V2_SCHEMA_VERSION {
            return Err(TasteGenerationV2Error::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if self.generation == 0 {
            return Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::ZeroGeneration,
            ));
        }
        if self.scirust_validation_receipts_sha256 == [0; 32] {
            return Err(TasteGenerationV2Error::ZeroSciRustReceiptsDigest);
        }
        Ok(())
    }
}

/// Either historical v1 semantics or explicit v2 provenance-sealed semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionedTasteGenerationManifest {
    V1(TasteGenerationManifest),
    V2(TasteGenerationManifestV2),
}

impl VersionedTasteGenerationManifest {
    #[must_use]
    pub const fn generation(&self) -> u64 {
        match self {
            Self::V1(manifest) => manifest.generation,
            Self::V2(manifest) => manifest.generation,
        }
    }

    #[must_use]
    pub const fn previous_manifest_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(manifest) => manifest.previous_manifest_sha256,
            Self::V2(manifest) => manifest.previous_manifest_sha256,
        }
    }

    /// Hash using the manifest's own historical/versioned semantics.
    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(manifest) => manifest.sha256(),
            Self::V2(manifest) => manifest.sha256(),
        }
    }

    #[must_use]
    pub const fn has_scirust_provenance(&self) -> bool {
        matches!(self, Self::V2(_))
    }
}

/// Failure while validating or extending a mixed v1/v2 chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteGenerationV2Error {
    UnsupportedSchema(u16),
    ZeroSciRustReceiptsDigest,
    Generation(TasteGenerationError),
}

/// Version-aware chain that preserves the exact hash semantics of each
/// generation while permitting a one-way migration from v1 to v2.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VersionedTasteGenerationChain {
    manifests: BTreeMap<u64, VersionedTasteGenerationManifest>,
    selected_generation: Option<u64>,
    v2_started: bool,
}

impl VersionedTasteGenerationChain {
    /// Append exactly the next generation. Once any v2 generation exists, a v1
    /// generation can never be appended again: provenance sealing is monotonic.
    pub fn append(
        &mut self,
        manifest: VersionedTasteGenerationManifest,
    ) -> Result<(), TasteGenerationV2Error> {
        if let VersionedTasteGenerationManifest::V2(v2) = &manifest {
            v2.validate()?;
        }
        let generation = manifest.generation();
        if generation == 0 {
            return Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::ZeroGeneration,
            ));
        }
        if self.manifests.contains_key(&generation) {
            return Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::GenerationAlreadyExists,
            ));
        }
        if self.v2_started && matches!(manifest, VersionedTasteGenerationManifest::V1(_)) {
            return Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::NonMonotonicGeneration,
            ));
        }

        match self.manifests.last_key_value() {
            None => {
                if generation != 1 {
                    return Err(TasteGenerationV2Error::Generation(
                        TasteGenerationError::NonMonotonicGeneration,
                    ));
                }
                if manifest.previous_manifest_sha256() != GENESIS_DIGEST {
                    return Err(TasteGenerationV2Error::Generation(
                        TasteGenerationError::PreviousDigestMismatch,
                    ));
                }
            }
            Some((previous_generation, previous)) => {
                if generation != previous_generation.saturating_add(1) {
                    return Err(TasteGenerationV2Error::Generation(
                        TasteGenerationError::NonMonotonicGeneration,
                    ));
                }
                if manifest.previous_manifest_sha256() != previous.sha256() {
                    return Err(TasteGenerationV2Error::Generation(
                        TasteGenerationError::PreviousDigestMismatch,
                    ));
                }
            }
        }

        if matches!(manifest, VersionedTasteGenerationManifest::V2(_)) {
            self.v2_started = true;
        }
        self.selected_generation = Some(generation);
        self.manifests.insert(generation, manifest);
        Ok(())
    }

    pub fn select_rollback(&mut self, generation: u64) -> Result<(), TasteGenerationV2Error> {
        if !self.manifests.contains_key(&generation) {
            return Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::UnknownRollbackGeneration,
            ));
        }
        self.selected_generation = Some(generation);
        Ok(())
    }

    #[must_use]
    pub fn selected(&self) -> Option<&VersionedTasteGenerationManifest> {
        self.selected_generation
            .and_then(|generation| self.manifests.get(&generation))
    }

    #[must_use]
    pub fn latest(&self) -> Option<&VersionedTasteGenerationManifest> {
        self.manifests
            .last_key_value()
            .map(|(_, manifest)| manifest)
    }

    #[must_use]
    pub const fn v2_started(&self) -> bool {
        self.v2_started
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

#[cfg(test)]
mod tests {
    use super::*;

    fn v1(generation: u64, previous: [u8; 32], marker: u8) -> TasteGenerationManifest {
        TasteGenerationManifest {
            generation,
            previous_manifest_sha256: previous,
            profile_sha256: [marker; 32],
            replay_sha256: [marker.wrapping_add(1); 32],
            candidate_report_sha256: [marker.wrapping_add(2); 32],
            validation_store_sha256: [marker.wrapping_add(3); 32],
        }
    }

    fn v2(generation: u64, previous: [u8; 32], marker: u8) -> TasteGenerationManifestV2 {
        TasteGenerationManifestV2::new(
            generation,
            previous,
            [marker; 32],
            [marker.wrapping_add(1); 32],
            [marker.wrapping_add(2); 32],
            [marker.wrapping_add(3); 32],
            [marker.wrapping_add(4); 32],
        )
    }

    #[test]
    fn historical_v1_hash_semantics_are_unchanged() {
        let manifest = v1(1, GENESIS_DIGEST, 1);
        let versioned = VersionedTasteGenerationManifest::V1(manifest.clone());
        assert_eq!(versioned.sha256(), manifest.sha256());
        assert_eq!(manifest.canonical_bytes().len(), 8 + 32 * 5);
    }

    #[test]
    fn v2_bytes_are_explicitly_domain_separated_and_fixed_width() {
        let manifest = v2(1, GENESIS_DIGEST, 1);
        let bytes = manifest.canonical_bytes();
        assert_eq!(bytes.len(), TASTE_GENERATION_V2_MANIFEST_BYTES);
        assert_eq!(&bytes[..16], &TASTE_GENERATION_V2_MAGIC);
        assert_eq!(
            u16::from_le_bytes(bytes[16..18].try_into().expect("schema")),
            TASTE_GENERATION_V2_SCHEMA_VERSION
        );
    }

    #[test]
    fn mixed_chain_can_upgrade_v1_to_v2_without_rehashing_v1() {
        let first = v1(1, GENESIS_DIGEST, 1);
        let second = v2(2, first.sha256(), 2);
        let mut chain = VersionedTasteGenerationChain::default();
        chain
            .append(VersionedTasteGenerationManifest::V1(first))
            .expect("v1");
        chain
            .append(VersionedTasteGenerationManifest::V2(second))
            .expect("v2");
        assert!(chain.v2_started());
        assert_eq!(chain.latest().map(|item| item.generation()), Some(2));
        assert!(chain
            .latest()
            .is_some_and(|item| item.has_scirust_provenance()));
    }

    #[test]
    fn chain_cannot_downgrade_back_to_v1_after_v2() {
        let first = v2(1, GENESIS_DIGEST, 1);
        let second = v1(2, first.sha256(), 2);
        let mut chain = VersionedTasteGenerationChain::default();
        chain
            .append(VersionedTasteGenerationManifest::V2(first))
            .expect("v2");
        assert_eq!(
            chain.append(VersionedTasteGenerationManifest::V1(second)),
            Err(TasteGenerationV2Error::Generation(
                TasteGenerationError::NonMonotonicGeneration
            ))
        );
    }

    #[test]
    fn zero_receipt_digest_is_never_valid_v2_provenance() {
        let mut manifest = v2(1, GENESIS_DIGEST, 1);
        manifest.scirust_validation_receipts_sha256 = [0; 32];
        assert_eq!(
            manifest.validate(),
            Err(TasteGenerationV2Error::ZeroSciRustReceiptsDigest)
        );
    }

    #[test]
    fn caller_mutated_schema_is_rejected() {
        let mut manifest = v2(1, GENESIS_DIGEST, 1);
        manifest.schema_version = 99;
        assert_eq!(
            manifest.validate(),
            Err(TasteGenerationV2Error::UnsupportedSchema(99))
        );
    }
}
