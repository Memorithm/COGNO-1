//! Atomic disk commit for v2 scientific-taste generations.
//!
//! V1 commit semantics remain untouched. V2 persists one additional immutable
//! artifact, `scirust-validation-receipts.bin`, and seals its exact bytes in the
//! explicit [`TasteGenerationManifestV2`] digest.

use crate::taste_generation::GENESIS_DIGEST;
use crate::taste_generation_v2::{TasteGenerationManifestV2, TasteGenerationV2Error};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const CANDIDATE_REPORT_FILE: &str = "candidate-report.bin";
const VALIDATION_STORE_FILE: &str = "validation-store.bin";
const SCIRUST_RECEIPTS_FILE: &str = "scirust-validation-receipts.bin";
const REPLAY_REPORT_FILE: &str = "replay-report.bin";
const PROFILE_FILE: &str = "profile.bin";
const MANIFEST_FILE: &str = "generation.manifest";
const CURRENT_FILE: &str = "CURRENT";

/// Immutable artifacts committed together for one provenance-sealed generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCycleArtifactsV2 {
    pub candidate_report: Vec<u8>,
    pub validation_store: Vec<u8>,
    pub scirust_validation_receipts: Vec<u8>,
    pub replay_report: Vec<u8>,
    pub profile: Vec<u8>,
}

/// Successful durable v2 generation commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCycleCommitV2 {
    pub generation: u64,
    pub manifest_sha256: [u8; 32],
    pub directory: PathBuf,
}

/// Fail-closed v2 generation persistence error.
#[derive(Debug)]
pub enum TasteCycleV2Error {
    Io(io::Error),
    Manifest(TasteGenerationV2Error),
    EmptyArtifact(&'static str),
    DigestMismatch(&'static str),
    InvalidCurrentPointer,
    CurrentGenerationMismatch { expected: u64, actual: u64 },
    UnexpectedCurrentForGenesis,
    MissingPreviousGeneration,
    PreviousManifestDigestMismatch,
    GenerationAlreadyExists,
    StagingAlreadyExists,
}

impl From<io::Error> for TasteCycleV2Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Atomically commit one v2 generation after validating every artifact digest
/// and the exact hash of the currently selected predecessor manifest.
pub fn commit_taste_cycle_v2(
    root: impl AsRef<Path>,
    manifest: &TasteGenerationManifestV2,
    artifacts: &TasteCycleArtifactsV2,
) -> Result<TasteCycleCommitV2, TasteCycleV2Error> {
    manifest.validate().map_err(TasteCycleV2Error::Manifest)?;
    validate_artifacts(manifest, artifacts)?;

    let root = root.as_ref();
    fs::create_dir_all(root)?;
    validate_predecessor(root, manifest)?;

    let final_dir = root.join(format!("generation-{}", manifest.generation));
    if final_dir.exists() {
        return Err(TasteCycleV2Error::GenerationAlreadyExists);
    }
    let stage_dir = root.join(format!(
        "generation-{}.tmp-{}",
        manifest.generation,
        std::process::id()
    ));
    if stage_dir.exists() {
        return Err(TasteCycleV2Error::StagingAlreadyExists);
    }
    fs::create_dir(&stage_dir)?;

    let commit_result = (|| {
        write_synced(&stage_dir.join(CANDIDATE_REPORT_FILE), &artifacts.candidate_report)?;
        write_synced(&stage_dir.join(VALIDATION_STORE_FILE), &artifacts.validation_store)?;
        write_synced(
            &stage_dir.join(SCIRUST_RECEIPTS_FILE),
            &artifacts.scirust_validation_receipts,
        )?;
        write_synced(&stage_dir.join(REPLAY_REPORT_FILE), &artifacts.replay_report)?;
        write_synced(&stage_dir.join(PROFILE_FILE), &artifacts.profile)?;
        write_synced(&stage_dir.join(MANIFEST_FILE), &manifest.canonical_bytes())?;
        File::open(&stage_dir)?.sync_all()?;
        fs::rename(&stage_dir, &final_dir)?;
        File::open(root)?.sync_all()?;
        write_current(root, manifest.generation)?;
        Ok::<(), TasteCycleV2Error>(())
    })();

    if let Err(error) = commit_result {
        if stage_dir.exists() {
            let _ = fs::remove_dir_all(&stage_dir);
        }
        return Err(error);
    }

    Ok(TasteCycleCommitV2 {
        generation: manifest.generation,
        manifest_sha256: manifest.sha256(),
        directory: final_dir,
    })
}

fn validate_artifacts(
    manifest: &TasteGenerationManifestV2,
    artifacts: &TasteCycleArtifactsV2,
) -> Result<(), TasteCycleV2Error> {
    for (name, bytes) in [
        ("candidate_report", artifacts.candidate_report.as_slice()),
        ("validation_store", artifacts.validation_store.as_slice()),
        (
            "scirust_validation_receipts",
            artifacts.scirust_validation_receipts.as_slice(),
        ),
        ("replay_report", artifacts.replay_report.as_slice()),
        ("profile", artifacts.profile.as_slice()),
    ] {
        if bytes.is_empty() {
            return Err(TasteCycleV2Error::EmptyArtifact(name));
        }
    }

    verify_digest(
        "candidate_report",
        &artifacts.candidate_report,
        manifest.candidate_report_sha256,
    )?;
    verify_digest(
        "validation_store",
        &artifacts.validation_store,
        manifest.validation_store_sha256,
    )?;
    verify_digest(
        "scirust_validation_receipts",
        &artifacts.scirust_validation_receipts,
        manifest.scirust_validation_receipts_sha256,
    )?;
    verify_digest(
        "replay_report",
        &artifacts.replay_report,
        manifest.replay_sha256,
    )?;
    verify_digest("profile", &artifacts.profile, manifest.profile_sha256)?;
    Ok(())
}

fn verify_digest(
    name: &'static str,
    bytes: &[u8],
    expected: [u8; 32],
) -> Result<(), TasteCycleV2Error> {
    let actual: [u8; 32] = Sha256::digest(bytes).into();
    if actual == expected {
        Ok(())
    } else {
        Err(TasteCycleV2Error::DigestMismatch(name))
    }
}

fn validate_predecessor(
    root: &Path,
    manifest: &TasteGenerationManifestV2,
) -> Result<(), TasteCycleV2Error> {
    let current = read_current(root)?;
    if manifest.generation == 1 {
        if current.is_some() {
            return Err(TasteCycleV2Error::UnexpectedCurrentForGenesis);
        }
        if manifest.previous_manifest_sha256 != GENESIS_DIGEST {
            return Err(TasteCycleV2Error::PreviousManifestDigestMismatch);
        }
        return Ok(());
    }

    let expected_generation = manifest.generation - 1;
    let current_generation = current.ok_or(TasteCycleV2Error::MissingPreviousGeneration)?;
    if current_generation != expected_generation {
        return Err(TasteCycleV2Error::CurrentGenerationMismatch {
            expected: expected_generation,
            actual: current_generation,
        });
    }
    let previous_manifest = root
        .join(format!("generation-{current_generation}"))
        .join(MANIFEST_FILE);
    let previous_bytes = fs::read(previous_manifest)
        .map_err(|_| TasteCycleV2Error::MissingPreviousGeneration)?;
    let previous_sha256: [u8; 32] = Sha256::digest(previous_bytes).into();
    if previous_sha256 != manifest.previous_manifest_sha256 {
        return Err(TasteCycleV2Error::PreviousManifestDigestMismatch);
    }
    Ok(())
}

fn read_current(root: &Path) -> Result<Option<u64>, TasteCycleV2Error> {
    let path = root.join(CURRENT_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)?;
    let generation = text
        .trim()
        .parse::<u64>()
        .map_err(|_| TasteCycleV2Error::InvalidCurrentPointer)?;
    if generation == 0 {
        return Err(TasteCycleV2Error::InvalidCurrentPointer);
    }
    Ok(Some(generation))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), TasteCycleV2Error> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_current(root: &Path, generation: u64) -> Result<(), TasteCycleV2Error> {
    let temporary = root.join(format!("CURRENT.tmp-{}", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    writeln!(file, "{generation}")?;
    file.sync_all()?;
    fs::rename(&temporary, root.join(CURRENT_FILE))?;
    File::open(root)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{commit_taste_cycle, TasteCycleArtifacts, TasteGenerationManifest};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-taste-cycle-v2-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn digest(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn artifacts(marker: u8) -> TasteCycleArtifactsV2 {
        TasteCycleArtifactsV2 {
            candidate_report: vec![marker, b'c'],
            validation_store: vec![marker, b'v'],
            scirust_validation_receipts: vec![marker, b's'],
            replay_report: vec![marker, b'r'],
            profile: vec![marker, b'p'],
        }
    }

    fn manifest(
        generation: u64,
        previous: [u8; 32],
        artifacts: &TasteCycleArtifactsV2,
    ) -> TasteGenerationManifestV2 {
        TasteGenerationManifestV2::new(
            generation,
            previous,
            digest(&artifacts.profile),
            digest(&artifacts.replay_report),
            digest(&artifacts.candidate_report),
            digest(&artifacts.validation_store),
            digest(&artifacts.scirust_validation_receipts),
        )
    }

    #[test]
    fn commits_all_v2_artifacts_and_manifest() {
        let root = root("commit");
        let artifacts = artifacts(1);
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        let committed = commit_taste_cycle_v2(&root, &manifest, &artifacts).expect("commit");

        assert_eq!(committed.generation, 1);
        assert_eq!(committed.manifest_sha256, manifest.sha256());
        assert_eq!(
            fs::read(committed.directory.join(SCIRUST_RECEIPTS_FILE)).expect("receipts"),
            artifacts.scirust_validation_receipts
        );
        assert_eq!(
            fs::read(committed.directory.join(MANIFEST_FILE)).expect("manifest"),
            manifest.canonical_bytes()
        );
        assert_eq!(fs::read_to_string(root.join(CURRENT_FILE)).expect("current"), "1\n");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn v2_can_follow_existing_v1_generation_by_exact_v1_hash() {
        let root = root("upgrade");
        let v1_artifacts = TasteCycleArtifacts {
            candidate_report: b"candidate-v1".to_vec(),
            validation_store: b"validations-v1".to_vec(),
            replay_report: b"replay-v1".to_vec(),
            profile: b"profile-v1".to_vec(),
        };
        let v1 = TasteGenerationManifest {
            generation: 1,
            previous_manifest_sha256: GENESIS_DIGEST,
            profile_sha256: digest(&v1_artifacts.profile),
            replay_sha256: digest(&v1_artifacts.replay_report),
            candidate_report_sha256: digest(&v1_artifacts.candidate_report),
            validation_store_sha256: digest(&v1_artifacts.validation_store),
        };
        commit_taste_cycle(&root, &v1, &v1_artifacts).expect("v1 commit");

        let v2_artifacts = artifacts(2);
        let v2 = manifest(2, v1.sha256(), &v2_artifacts);
        commit_taste_cycle_v2(&root, &v2, &v2_artifacts).expect("v2 commit");
        assert_eq!(fs::read_to_string(root.join(CURRENT_FILE)).expect("current"), "2\n");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn wrong_receipt_digest_fails_without_generation_directory() {
        let root = root("digest");
        let artifacts = artifacts(1);
        let mut manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        manifest.scirust_validation_receipts_sha256 = [9; 32];
        assert!(matches!(
            commit_taste_cycle_v2(&root, &manifest, &artifacts),
            Err(TasteCycleV2Error::DigestMismatch(
                "scirust_validation_receipts"
            ))
        ));
        assert!(!root.join("generation-1").exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn stale_previous_digest_fails_before_staging() {
        let root = root("previous");
        let first_artifacts = artifacts(1);
        let first = manifest(1, GENESIS_DIGEST, &first_artifacts);
        commit_taste_cycle_v2(&root, &first, &first_artifacts).expect("first");

        let second_artifacts = artifacts(2);
        let second = manifest(2, [7; 32], &second_artifacts);
        assert!(matches!(
            commit_taste_cycle_v2(&root, &second, &second_artifacts),
            Err(TasteCycleV2Error::PreviousManifestDigestMismatch)
        ));
        assert!(!root.join("generation-2").exists());
        assert_eq!(fs::read_to_string(root.join(CURRENT_FILE)).expect("current"), "1\n");
        fs::remove_dir_all(root).expect("cleanup");
    }
}
