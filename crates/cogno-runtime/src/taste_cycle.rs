//! Transactional persistence for the SCIAGENT → evaluation → validation →
//! replay → profile hand-off.
//!
//! Upstream components produce bounded artifacts. This module verifies their
//! digests and proves that each new generation extends the currently selected
//! persisted hash chain before writing a complete staging directory, fsyncing
//! it, atomically renaming it into place, then atomically advancing `CURRENT`.
//! Partial, skipped, or forked generations are never selected. A crash after
//! the generation rename but before `CURRENT` advances can be resumed only
//! when every already-persisted byte exactly matches the requested commit.
//! Concurrent commit or recovery attempts are serialized by an atomic,
//! fail-closed inter-process commit interlock. On Linux only, a residual lock
//! is reclaimed automatically when `/proc` proves its original owner is gone.

use crate::taste_generation::{TasteGenerationManifest, GENESIS_DIGEST};
use crate::taste_persisted_chain::{
    load_persisted_taste_generation_selection, PersistedTasteGenerationError,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};

const MAX_CYCLE_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const COMMIT_INTERLOCK_FILE: &str = ".TASTE-COMMIT.lock";
const COMMIT_INTERLOCK_V2: &str = "cogno-taste-commit-v2";

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
    RecoveryManifestMismatch,
    RecoveryArtifactMismatch(&'static str),
    CommitInterlockHeld,
    InvalidCommitInterlockIdentity,
    Io(io::Error),
}

impl From<io::Error> for TasteCycleError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
struct TasteCommitInterlock {
    path: PathBuf,
}

impl TasteCommitInterlock {
    fn acquire(root: &Path) -> Result<Self, TasteCycleError> {
        let path = root.join(COMMIT_INTERLOCK_FILE);
        match Self::create(root, &path) {
            Ok(interlock) => Ok(interlock),
            Err(TasteCycleError::CommitInterlockHeld) => {
                if recover_stale_commit_interlock(&path)? {
                    Self::create(root, &path)
                } else {
                    Err(TasteCycleError::CommitInterlockHeld)
                }
            }
            Err(error) => Err(error),
        }
    }

    fn create(root: &Path, path: &Path) -> Result<Self, TasteCycleError> {
        let mut file = match OpenOptions::new().create_new(true).write(true).open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                return Err(TasteCycleError::CommitInterlockHeld);
            }
            Err(error) => return Err(TasteCycleError::Io(error)),
        };
        let interlock = Self {
            path: path.to_path_buf(),
        };
        file.write_all(&commit_interlock_identity()?)?;
        file.sync_all()?;
        sync_directory(root)?;
        Ok(interlock)
    }
}

impl Drop for TasteCommitInterlock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LinuxCommitInterlockIdentity {
    boot_id: String,
    pid: u32,
    start_ticks: u64,
}

#[cfg(target_os = "linux")]
fn commit_interlock_identity() -> Result<Vec<u8>, TasteCycleError> {
    let identity = LinuxCommitInterlockIdentity {
        boot_id: linux_boot_id()?,
        pid: std::process::id(),
        start_ticks: linux_process_start_ticks(std::process::id())?
            .ok_or(TasteCycleError::InvalidCommitInterlockIdentity)?,
    };
    Ok(format!(
        "{COMMIT_INTERLOCK_V2}\nboot_id={}\npid={}\nstart_ticks={}\n",
        identity.boot_id, identity.pid, identity.start_ticks
    )
    .into_bytes())
}

#[cfg(not(target_os = "linux"))]
fn commit_interlock_identity() -> Result<Vec<u8>, TasteCycleError> {
    Ok(format!("{COMMIT_INTERLOCK_V2}\nplatform=unverified\n").into_bytes())
}

#[cfg(target_os = "linux")]
fn recover_stale_commit_interlock(path: &Path) -> Result<bool, TasteCycleError> {
    let bytes = fs::read(path)?;
    let Some(identity) = parse_linux_commit_interlock_identity(&bytes) else {
        return Ok(false);
    };
    if identity.boot_id != linux_boot_id()? {
        remove_stale_interlock(path)?;
        return Ok(true);
    }
    match linux_process_start_ticks(identity.pid)? {
        None => {
            remove_stale_interlock(path)?;
            Ok(true)
        }
        Some(start_ticks) if start_ticks != identity.start_ticks => {
            remove_stale_interlock(path)?;
            Ok(true)
        }
        Some(_) => Ok(false),
    }
}

#[cfg(not(target_os = "linux"))]
fn recover_stale_commit_interlock(_path: &Path) -> Result<bool, TasteCycleError> {
    Ok(false)
}

#[cfg(target_os = "linux")]
fn remove_stale_interlock(path: &Path) -> Result<(), TasteCycleError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(TasteCycleError::Io(error)),
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_commit_interlock_identity(bytes: &[u8]) -> Option<LinuxCommitInterlockIdentity> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.lines();
    if lines.next()? != COMMIT_INTERLOCK_V2 {
        return None;
    }
    let boot_id = lines.next()?.strip_prefix("boot_id=")?.to_owned();
    let pid = lines.next()?.strip_prefix("pid=")?.parse().ok()?;
    let start_ticks = lines.next()?.strip_prefix("start_ticks=")?.parse().ok()?;
    if boot_id.is_empty() || lines.next().is_some() {
        return None;
    }
    Some(LinuxCommitInterlockIdentity {
        boot_id,
        pid,
        start_ticks,
    })
}

#[cfg(target_os = "linux")]
fn linux_boot_id() -> Result<String, TasteCycleError> {
    let value = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let value = value.trim();
    if value.is_empty() || value.contains('\n') || value.contains('\r') {
        return Err(TasteCycleError::InvalidCommitInterlockIdentity);
    }
    Ok(value.to_owned())
}

#[cfg(target_os = "linux")]
fn linux_process_start_ticks(pid: u32) -> Result<Option<u64>, TasteCycleError> {
    let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(TasteCycleError::Io(error)),
    };
    let close_paren = stat
        .rfind(')')
        .ok_or(TasteCycleError::InvalidCommitInterlockIdentity)?;
    let fields = stat
        .get(close_paren + 1..)
        .ok_or(TasteCycleError::InvalidCommitInterlockIdentity)?;
    let start_ticks = fields
        .split_whitespace()
        .nth(19)
        .ok_or(TasteCycleError::InvalidCommitInterlockIdentity)?
        .parse()
        .map_err(|_| TasteCycleError::InvalidCommitInterlockIdentity)?;
    Ok(Some(start_ticks))
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
    let _interlock = TasteCommitInterlock::acquire(root)?;

    let final_path = root.join(format!("generation-{}", manifest.generation));
    if final_path.exists() {
        if current_points_to(root, manifest.generation)? {
            return Err(TasteCycleError::GenerationAlreadyExists);
        }
        verify_persisted_predecessor(root, manifest)?;
        verify_recoverable_generation(&final_path, manifest, artifacts)?;
        advance_current(root, manifest.generation)?;
        return Ok(TasteCycleCommit {
            generation: manifest.generation,
            generation_path: final_path,
            manifest_sha256: manifest.sha256(),
        });
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
        advance_current(root, manifest.generation)
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

fn current_points_to(root: &Path, generation: u64) -> Result<bool, TasteCycleError> {
    match fs::read(root.join("CURRENT")) {
        Ok(bytes) => Ok(bytes == format!("{generation}\n").as_bytes()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(TasteCycleError::Io(error)),
    }
}

fn verify_recoverable_generation(
    generation_path: &Path,
    manifest: &TasteGenerationManifest,
    artifacts: &TasteCycleArtifacts,
) -> Result<(), TasteCycleError> {
    let persisted_manifest = fs::read(generation_path.join("generation.manifest"))?;
    if persisted_manifest != manifest.canonical_bytes() {
        return Err(TasteCycleError::RecoveryManifestMismatch);
    }
    verify_recovery_artifact(
        &generation_path.join("candidates.json"),
        "candidate report",
        &artifacts.candidate_report,
    )?;
    verify_recovery_artifact(
        &generation_path.join("taste.validations"),
        "validation store",
        &artifacts.validation_store,
    )?;
    verify_recovery_artifact(
        &generation_path.join("replay.json"),
        "replay report",
        &artifacts.replay_report,
    )?;
    verify_recovery_artifact(
        &generation_path.join("taste.profile"),
        "profile",
        &artifacts.profile,
    )?;
    Ok(())
}

fn verify_recovery_artifact(
    path: &Path,
    label: &'static str,
    expected: &[u8],
) -> Result<(), TasteCycleError> {
    let persisted = fs::read(path)?;
    if persisted != expected {
        return Err(TasteCycleError::RecoveryArtifactMismatch(label));
    }
    Ok(())
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

fn advance_current(root: &Path, generation: u64) -> Result<(), TasteCycleError> {
    let current_tmp = root.join(".CURRENT.tmp");
    if current_tmp.exists() {
        fs::remove_file(&current_tmp)?;
    }
    write_synced(&current_tmp, format!("{generation}\n").as_bytes())?;
    fs::rename(&current_tmp, root.join("CURRENT"))?;
    sync_directory(root)?;
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

    fn write_complete_generation(
        root: &Path,
        manifest: &TasteGenerationManifest,
        artifacts: &TasteCycleArtifacts,
    ) {
        let generation_path = root.join(format!("generation-{}", manifest.generation));
        fs::create_dir_all(&generation_path).expect("generation dir");
        fs::write(
            generation_path.join("candidates.json"),
            &artifacts.candidate_report,
        )
        .expect("candidates");
        fs::write(
            generation_path.join("taste.validations"),
            &artifacts.validation_store,
        )
        .expect("validations");
        fs::write(
            generation_path.join("replay.json"),
            &artifacts.replay_report,
        )
        .expect("replay");
        fs::write(generation_path.join("taste.profile"), &artifacts.profile).expect("profile");
        fs::write(
            generation_path.join("generation.manifest"),
            manifest.canonical_bytes(),
        )
        .expect("manifest");
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
        assert!(!root.join(COMMIT_INTERLOCK_FILE).exists());
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
    fn active_interlock_blocks_commit_before_state_mutation() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        let interlock = TasteCommitInterlock::acquire(&root).expect("interlock");
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);

        assert!(matches!(
            commit_taste_cycle(&root, &manifest, &artifacts),
            Err(TasteCycleError::CommitInterlockHeld)
        ));
        assert!(!root.join("CURRENT").exists());
        assert!(!root.join("generation-1").exists());

        drop(interlock);
        commit_taste_cycle(&root, &manifest, &artifacts).expect("commit after release");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn interlock_is_released_after_fail_closed_predecessor_error() {
        let root = root();
        let artifacts = artifacts("one");
        let invalid = manifest(1, [7; 32], &artifacts);
        assert!(matches!(
            commit_taste_cycle(&root, &invalid, &artifacts),
            Err(TasteCycleError::PreviousManifestDigestMismatch)
        ));
        assert!(!root.join(COMMIT_INTERLOCK_FILE).exists());

        let valid = manifest(1, GENESIS_DIGEST, &artifacts);
        commit_taste_cycle(&root, &valid, &artifacts).expect("retry");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stale_linux_interlock_is_reclaimed_when_owner_is_proven_absent() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        let stale = format!(
            "{COMMIT_INTERLOCK_V2}\nboot_id={}\npid={}\nstart_ticks=1\n",
            linux_boot_id().expect("boot id"),
            u32::MAX
        );
        fs::write(root.join(COMMIT_INTERLOCK_FILE), stale).expect("stale lock");
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);

        commit_taste_cycle(&root, &manifest, &artifacts).expect("recover stale lock");
        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("current"),
            "1\n"
        );
        assert!(!root.join(COMMIT_INTERLOCK_FILE).exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn malformed_linux_interlock_remains_fail_closed() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        fs::write(root.join(COMMIT_INTERLOCK_FILE), b"untrusted\n").expect("lock");
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);

        assert!(matches!(
            commit_taste_cycle(&root, &manifest, &artifacts),
            Err(TasteCycleError::CommitInterlockHeld)
        ));
        assert!(!root.join("CURRENT").exists());
        assert!(!root.join("generation-1").exists());
        assert!(root.join(COMMIT_INTERLOCK_FILE).exists());
        fs::remove_dir_all(root).expect("cleanup");
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

    #[test]
    fn complete_unselected_generation_resumes_current_after_crash() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        write_complete_generation(&root, &manifest, &artifacts);
        fs::write(root.join(".CURRENT.tmp"), b"1\n").expect("stale current temp");

        let committed = commit_taste_cycle(&root, &manifest, &artifacts).expect("resume");
        assert_eq!(committed.generation, 1);
        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("current"),
            "1\n"
        );
        assert!(!root.join(".CURRENT.tmp").exists());
        assert!(!root.join(COMMIT_INTERLOCK_FILE).exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn crash_recovery_refuses_existing_bytes_that_do_not_match_transaction() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        let artifacts = artifacts("one");
        let manifest = manifest(1, GENESIS_DIGEST, &artifacts);
        write_complete_generation(&root, &manifest, &artifacts);
        fs::write(
            root.join("generation-1/taste.profile"),
            b"different-profile",
        )
        .expect("tamper");

        assert!(matches!(
            commit_taste_cycle(&root, &manifest, &artifacts),
            Err(TasteCycleError::RecoveryArtifactMismatch("profile"))
        ));
        assert!(!root.join("CURRENT").exists());
        assert!(!root.join(COMMIT_INTERLOCK_FILE).exists());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
