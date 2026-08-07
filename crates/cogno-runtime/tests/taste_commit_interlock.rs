use cogno_runtime::{
    commit_taste_cycle, TasteCycleArtifacts, TasteCycleError, TasteGenerationManifest,
    GENESIS_DIGEST,
};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-taste-interlock-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn held_commit_interlock_preserves_persisted_state() {
    let root = root();
    fs::create_dir_all(&root).expect("root");
    let lock_path = root.join(".TASTE-COMMIT.lock");
    let _lock = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock_path)
        .expect("lock");
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

    assert!(matches!(
        commit_taste_cycle(&root, &manifest, &artifacts),
        Err(TasteCycleError::CommitInterlockHeld)
    ));
    assert!(!root.join("CURRENT").exists());
    assert!(!root.join("generation-1").exists());
    assert!(!root.join(".generation-1.tmp").exists());

    drop(_lock);
    fs::remove_file(lock_path).expect("remove lock");
    commit_taste_cycle(&root, &manifest, &artifacts).expect("commit");
    assert_eq!(
        fs::read_to_string(root.join("CURRENT")).expect("current"),
        "1\n"
    );
    fs::remove_dir_all(root).expect("cleanup");
}
