//! Transactional persistence for the SCIAGENT → evaluation → validation →
//! replay → profile hand-off.
//!
//! Upstream components produce bounded artifacts. This module verifies their
//! digests and proves that each new generation extends the currently selected
//! persisted hash chain before writing a complete staging directory, fsyncing
//! it, atomically renaming it into place, then atomically advancing `CURRENT`.
//! Partial, skipped, or forked generations are never selected.

use crate::taste_generation::{TasteGenerationManifest, GENESIS_DIGEST};
use crate::taste_persisted_chain::{
    load_persisted_taste_generation_selection, PersistedTasteGenerationError,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAX_CYCLE_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Complete set of artifacts produced by one scientific-taste cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCycleArtifacts {
    pub candidate_report: Vec<u8>,
    pub validation_store: Vec<u8>,
    pub replay_report: Vec<u8>,
    pub profile: Vec<u8>,
}

/// Successfully committed generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCycleCommit {
    pub generation: u64,
    pub generation_path: PathBuf,
    pub manifest_sha256: [u8; 32],
}

/// Transaction failure. All failures occur before `CURRENT` is advanced.
#[derive(Debug)]
pub enum TasteCycleError {
    InvalidArtifactSize(&'static str),
    DigestMismatch(&'static str),
    ZeroGeneration,
    ExistingHistoryForGenesis,
    NonMonotonicPersistedGeneration,
    PreviousManifestDigestMismatch,
    InvalidPersistedPredecessor(PersistedTasteGenerationError),
    GenerationAlreadyExists,
    Io(io::Error),
}

impl From<io::Error> for TasteCycleError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Commit one verified cycle atomically under `root`.
pub fn commit_taste_cycle(
    root: impl AsRef<Path>,
    manifest: &TasteGenerationManifest,
    artifacts: &TasteCycleArtifacts,
) -> Result<TasteCycleCommit, TasteCycleError> {
    validate_artifact(
        "candidate report",
        &artifacts.candidate_report,
        manifest.candidate_report_sha256,
    )?;
    validate_artifact(
        "validation store",
        &artifacts.validation_store,
        manifest.validation_store_sha256,
    )?;
    validate_artifact(
        "replay report",
        &artifacts.replay_report,
        manifest.replay_sha256,
    )?;
    validate_artifact("profile", &artifacts.profile, manifest.profile_sha256)?;

    let root = root.as_ref();
    fs::create_dir_all(root)?;

    let final_path = root.join(format!("generation-{}", manifest.generation));
    if final_path.exists() {
        return Err(TasteCycleError::GenerationAlreadyExists);
    }
    verify_persisted_predecessor(root, manifest)?;

    let staging_path = root.join(format!(".generation-{}.tmp", manifest.generation));
    if staging_path.exists() {
        fs::remove_dir_all(&staging_path)?;
    }
    fs::create_dir(&staging_path)?;

    let result = (|| {
        write_synced(
            &staging_path.join("candidates.json"),
            &artifacts.candidate_report,
        )?;
        write_synced(
            &staging_path.join("taste.validations"),
            &artifacts.validation_store,
        )?;
        write_synced(&staging_path.join("replay.json"), &artifacts.replay_report)?;
        write_synced(&staging_path.join("taste.profile"), &artifacts.profile)?;
        write_synced(
            &staging_path.join("generation.manifest"),
            &manifest.canonical_bytes(),
        )?;
        sync_directory(&staging_path)?;
        fs::rename(&staging_path, &final_path)?;
        sync_directory(root)?;

        let current_tmp = root.join(".CURRENT.tmp");
        let current = root.join("CURRENT");
        write_synced(
            &current_tmp,
            format!("{}\n", manifest.generation).as_bytes(),
        )?;
        fs::rename(&current_tmp, &current)?;
        sync_directory(root)?;
        Ok::<(), TasteCycleError>(())
    })();

    if result.is_err() && staging_path.exists() {
        let _ = fs::remove_dir_all(&staging_path);
    }
    result?;

    Ok(TasteCycleCommit {
        generation: manifest.generation,
        generation_path: final_path,
        manifest_sha256: manifest.sha256(),
    })
}

fn verify_persisted_predecessor(
    root: &Path,
    manifest: &TasteGenerationManifest,
) -> Result<(), TasteCycleError> {
    if manifest.generation == 0 {
        return Err(TasteCycleError::ZeroGeneration);
    }

    if manifest.generation == 1 {
        if root.join("CURRENT").exists() {
            return Err(TasteCycleError::ExistingHistoryForGenesis);
        }
        if manifest.previous_manifest_sha256 != GENESIS_DIGEST {
            return Err(TasteCycleError::PreviousManifestDigestMismatch);
        }
        return Ok(());
    }

    let selection = load_persisted_taste_generation_selection(root)
        .map_err(TasteCycleError::InvalidPersistedPredecessor)?;
    if selection.selected_generation.checked_add(1) != Some(manifest.generation) {
        return Err(TasteCycleError::NonMonotonicPersistedGeneration);
    }
    let previous = selection
        .chain
        .selected()
        .ok_or(TasteCycleError::NonMonotonicPersistedGeneration)?;
    if previous.sha256() != manifest.previous_manifest_sha256 {
        return Err(TasteCycleError::PreviousManifestDigestMismatch);
    }
    Ok(())
}

fn validate_artifact(
    label: &'static str,
    bytes: &[u8],
    expected_digest: [u8; 32],
) -> Result<(), TasteCycleError> {
    if bytes.is_empty() || bytes.len() > MAX_CYCLE_ARTIFACT_BYTES {
        return Err(TasteCycleError::InvalidArtifactSize(label));
    }
    let observed: [u8; 32] = Sha256::digest(bytes).into();
    if observed != expected_digest {
        return Err(TasteCycleError::DigestMismatch(label));
    }
    Ok(())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), TasteCycleError> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), TasteCycleError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn digest(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-taste-cycle-{}-{nonce}", std::process::id()))
    }

    fn artifacts(marker: &str) -> TasteCycleArtifacts {
        TasteCycleArtifacts {
            candidate_report: format!("candidates-{marker}").into_bytes(),
            validation_store: format!("validations-{marker}").into_bytes(),
            replay_report: format!("replay-{marker}").into_bytes(),
            profile: format!("profile-{marker}").into_bytes(),
        }
    }

    fn manifest(
        generation: u64,
        previous_manifest_sha256: [u8; 32],
        artifacts: &TasteCycleArtifacts,
    ) -> TasteGenerationManifest {
        TasteGenerationManifest {
            generation,
            previous_manifest_sha256,
            profile_sha256: digest(&artifacts.profile),
            replay_sha256: digest(&artifacts.replay_report),
            candidate_report_sha256: digest(&artifacts.candidate_report),
            validation_store_sha256: digest(&artifacts.validation_store),
        }
    }

    #[test]
    fn complete_cycle_is_selected_atomically() {
        let root = root();
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        let committed = commit_taste_cycle(&root, &manifest, &artifacts).expect("commit");
        assert!(committed.generation_path.join("taste.profile").is_file());
        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("current"),
            "1\n"
        );
        assert!(!root.join(".generation-1.tmp").exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn digest_mismatch_fails_before_any_generation_is_visible() {
        let root = root();
        let artifacts = artifacts("one");
        let mut manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        manifest.profile_sha256 = [9; 32];
        assert!(matches!(
            commit_taste_cycle(&root, &manifest, &artifacts),
            Err(TasteCycleError::DigestMismatch("profile"))
        ));
        assert!(!root.join("CURRENT").exists());
    }

    #[test]
    fn skipped_generation_cannot_advance_current() {
        let root = root();
        let first_artifacts = artifacts("one");
        let first = manifest(1, GENESIS_DIGEST, &first_artifacts);
        commit_taste_cycle(&root, &first, &first_artifacts).expect("first");

        let third_artifacts = artifacts("three");
        let third = manifest(3, first.sha256(), &third_artifacts);
        assert!(matches!(
            commit_taste_cycle(&root, &third, &third_artifacts),
            Err(TasteCycleError::NonMonotonicPersistedGeneration)
        ));
        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("current"),
            "1\n"
        );
        assert!(!root.join("generation-3").exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn forked_previous_digest_cannot_advance_current() {
        let root = root();
        let first_artifacts = artifacts("one");
        let first = manifest(1, GENESIS_DIGEST, &first_artifacts);
        commit_taste_cycle(&root, &first, &first_artifacts).expect("first");

        let second_artifacts = artifacts("two");
        let second = manifest(2, [7; 32], &second_artifacts);
        assert!(matches!(
            commit_taste_cycle(&root, &second, &second_artifacts),
            Err(TasteCycleError::PreviousManifestDigestMismatch)
        ));
        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("current"),
            "1\n"
        );
        assert!(!root.join("generation-2").exists());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
