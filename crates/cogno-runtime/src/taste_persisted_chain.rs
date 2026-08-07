//! Fail-closed replay of the persisted scientific-taste generation chain.
//!
//! [`crate::taste_cycle::commit_taste_cycle`] advances `CURRENT` only after a
//! complete generation is durably committed. This module provides the inverse
//! restart path: it reads `CURRENT`, reconstructs every manifest from genesis
//! through the selected generation, verifies the hash chain, and re-hashes all
//! four persisted artifacts before returning a selected chain.

use crate::taste_generation::{
    TasteGenerationChain, TasteGenerationError, TasteGenerationManifest,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_BYTES: usize = 8 + 32 * 5;
const MAX_PERSISTED_GENERATIONS: u64 = 4_096;
const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Verified persisted generation state selected by `CURRENT`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedTasteGenerationSelection {
    pub chain: TasteGenerationChain,
    pub selected_generation: u64,
    pub selected_generation_path: PathBuf,
}

/// Failure while replaying persisted scientific-taste generations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistedTasteGenerationError {
    CannotReadCurrent,
    InvalidCurrent,
    TooManyGenerations,
    MissingGeneration(u64),
    InvalidManifestSize(u64),
    InvalidArtifactSize {
        generation: u64,
        artifact: &'static str,
    },
    CannotReadArtifact {
        generation: u64,
        artifact: &'static str,
    },
    ArtifactDigestMismatch {
        generation: u64,
        artifact: &'static str,
    },
    InvalidGenerationChain(TasteGenerationError),
}

/// Reconstruct and verify the complete persisted chain selected by `CURRENT`.
pub fn load_persisted_taste_generation_selection(
    root: impl AsRef<Path>,
) -> Result<PersistedTasteGenerationSelection, PersistedTasteGenerationError> {
    let root = root.as_ref();
    let current = fs::read_to_string(root.join("CURRENT"))
        .map_err(|_| PersistedTasteGenerationError::CannotReadCurrent)?;
    let selected_generation = parse_current(&current)?;
    if selected_generation > MAX_PERSISTED_GENERATIONS {
        return Err(PersistedTasteGenerationError::TooManyGenerations);
    }

    let mut chain = TasteGenerationChain::default();
    for generation in 1..=selected_generation {
        let generation_path = root.join(format!("generation-{generation}"));
        if !generation_path.is_dir() {
            return Err(PersistedTasteGenerationError::MissingGeneration(generation));
        }
        let manifest = read_manifest(&generation_path, generation)?;
        verify_generation_artifacts(&generation_path, &manifest)?;
        chain
            .append(manifest)
            .map_err(PersistedTasteGenerationError::InvalidGenerationChain)?;
    }

    Ok(PersistedTasteGenerationSelection {
        chain,
        selected_generation,
        selected_generation_path: root.join(format!("generation-{selected_generation}")),
    })
}

fn parse_current(current: &str) -> Result<u64, PersistedTasteGenerationError> {
    let bytes = current.as_bytes();
    if bytes.is_empty() || !bytes.ends_with(b"\n") || bytes[..bytes.len() - 1].contains(&b'\n') {
        return Err(PersistedTasteGenerationError::InvalidCurrent);
    }
    let digits = &current[..current.len() - 1];
    if digits.is_empty()
        || digits.len() > 1 && digits.starts_with('0')
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(PersistedTasteGenerationError::InvalidCurrent);
    }
    let generation = digits
        .parse::<u64>()
        .map_err(|_| PersistedTasteGenerationError::InvalidCurrent)?;
    if generation == 0 {
        return Err(PersistedTasteGenerationError::InvalidCurrent);
    }
    Ok(generation)
}

fn read_manifest(
    generation_path: &Path,
    expected_generation: u64,
) -> Result<TasteGenerationManifest, PersistedTasteGenerationError> {
    let bytes = fs::read(generation_path.join("generation.manifest"))
        .map_err(|_| PersistedTasteGenerationError::MissingGeneration(expected_generation))?;
    if bytes.len() != MANIFEST_BYTES {
        return Err(PersistedTasteGenerationError::InvalidManifestSize(
            expected_generation,
        ));
    }

    let generation = u64::from_le_bytes(bytes[0..8].try_into().expect("fixed slice"));
    if generation != expected_generation {
        return Err(PersistedTasteGenerationError::InvalidGenerationChain(
            TasteGenerationError::NonMonotonicGeneration,
        ));
    }
    let mut offset = 8;
    let previous_manifest_sha256 = take_digest(&bytes, &mut offset);
    let profile_sha256 = take_digest(&bytes, &mut offset);
    let replay_sha256 = take_digest(&bytes, &mut offset);
    let candidate_report_sha256 = take_digest(&bytes, &mut offset);
    let validation_store_sha256 = take_digest(&bytes, &mut offset);

    Ok(TasteGenerationManifest {
        generation,
        previous_manifest_sha256,
        profile_sha256,
        replay_sha256,
        candidate_report_sha256,
        validation_store_sha256,
    })
}

fn take_digest(bytes: &[u8], offset: &mut usize) -> [u8; 32] {
    let start = *offset;
    let end = start + 32;
    *offset = end;
    bytes[start..end].try_into().expect("fixed manifest digest")
}

fn verify_generation_artifacts(
    generation_path: &Path,
    manifest: &TasteGenerationManifest,
) -> Result<(), PersistedTasteGenerationError> {
    verify_artifact(
        generation_path,
        manifest.generation,
        "taste.profile",
        "profile",
        manifest.profile_sha256,
    )?;
    verify_artifact(
        generation_path,
        manifest.generation,
        "replay.json",
        "replay report",
        manifest.replay_sha256,
    )?;
    verify_artifact(
        generation_path,
        manifest.generation,
        "candidates.json",
        "candidate report",
        manifest.candidate_report_sha256,
    )?;
    verify_artifact(
        generation_path,
        manifest.generation,
        "taste.validations",
        "validation store",
        manifest.validation_store_sha256,
    )?;
    Ok(())
}

fn verify_artifact(
    generation_path: &Path,
    generation: u64,
    file_name: &str,
    artifact: &'static str,
    expected: [u8; 32],
) -> Result<(), PersistedTasteGenerationError> {
    let path = generation_path.join(file_name);
    let metadata =
        fs::metadata(&path).map_err(|_| PersistedTasteGenerationError::CannotReadArtifact {
            generation,
            artifact,
        })?;
    let length = usize::try_from(metadata.len()).map_err(|_| {
        PersistedTasteGenerationError::InvalidArtifactSize {
            generation,
            artifact,
        }
    })?;
    if length == 0 || length > MAX_ARTIFACT_BYTES {
        return Err(PersistedTasteGenerationError::InvalidArtifactSize {
            generation,
            artifact,
        });
    }
    let bytes = fs::read(path).map_err(|_| PersistedTasteGenerationError::CannotReadArtifact {
        generation,
        artifact,
    })?;
    let observed: [u8; 32] = Sha256::digest(&bytes).into();
    if observed != expected {
        return Err(PersistedTasteGenerationError::ArtifactDigestMismatch {
            generation,
            artifact,
        });
    }
    Ok(())
}
