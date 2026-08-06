//! Transactional persistence for the SCIAGENT → evaluation → validation →
//! replay → profile hand-off.
//!
//! Upstream components produce bounded artifacts. This module verifies their
//! digests against a monotonic generation manifest, writes a complete staging
//! directory, fsyncs it, atomically renames it into place, then atomically
//! advances `CURRENT`. Partial generations are never selected.

use crate::taste_generation::TasteGenerationManifest;
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
    use crate::taste_generation::GENESIS_DIGEST;
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

    #[test]
    fn complete_cycle_is_selected_atomically() {
        let root = root();
        let artifacts = TasteCycleArtifacts {
            candidate_report: b"candidates".to_vec(),
            validation_store: b"validations".to_vec(),
            replay_report: b"replay".to_vec(),
            profile: b"profile".to_vec(),
        };
        let manifest = TasteGenerationManifest {
            generation: 1,
            previous_manifest_sha256: GENESIS_DIGEST,
            profile_sha256: digest(&artifacts.profile),
            replay_sha256: digest(&artifacts.replay_report),
            candidate_report_sha256: digest(&artifacts.candidate_report),
            validation_store_sha256: digest(&artifacts.validation_store),
        };
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
        let artifacts = TasteCycleArtifacts {
            candidate_report: b"candidates".to_vec(),
            validation_store: b"validations".to_vec(),
            replay_report: b"replay".to_vec(),
            profile: b"profile".to_vec(),
        };
        let manifest = TasteGenerationManifest {
            generation: 1,
            previous_manifest_sha256: GENESIS_DIGEST,
            profile_sha256: [9; 32],
            replay_sha256: digest(&artifacts.replay_report),
            candidate_report_sha256: digest(&artifacts.candidate_report),
            validation_store_sha256: digest(&artifacts.validation_store),
        };
        assert!(matches!(
            commit_taste_cycle(&root, &manifest, &artifacts),
            Err(TasteCycleError::DigestMismatch("profile"))
        ));
        assert!(!root.join("CURRENT").exists());
    }
}
