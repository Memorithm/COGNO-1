#![cfg(target_os = "linux")]

use cogno_runtime::{
    commit_taste_cycle, TasteCycleArtifacts, TasteCycleError, TasteGenerationManifest,
    GENESIS_DIGEST,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CHILD_ROOT_ENV: &str = "COGNO_SYSTEM_INTERLOCK_CHILD_ROOT";
const LOCK_FILE: &str = ".TASTE-COMMIT.lock";
const READY_FILE: &str = ".child-ready";

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-system-interlock-{}-{nonce}",
        std::process::id()
    ))
}

fn linux_boot_id() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .expect("boot id")
        .trim()
        .to_owned()
}

fn linux_start_ticks(pid: u32) -> u64 {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).expect("process stat");
    let close_paren = stat.rfind(')').expect("stat comm terminator");
    stat[close_paren + 1..]
        .split_whitespace()
        .nth(19)
        .expect("start ticks field")
        .parse()
        .expect("start ticks integer")
}

fn write_live_lock(root: &Path) {
    let pid = std::process::id();
    let identity = format!(
        "cogno-taste-commit-v2\nboot_id={}\npid={pid}\nstart_ticks={}\n",
        linux_boot_id(),
        linux_start_ticks(pid)
    );
    fs::write(root.join(LOCK_FILE), identity).expect("write child lock");
    fs::write(root.join(READY_FILE), b"ready\n").expect("write readiness marker");
}

fn wait_until_child_ready(root: &Path, child: &mut Child) {
    for _ in 0..200 {
        if root.join(READY_FILE).is_file() {
            return;
        }
        if let Some(status) = child.try_wait().expect("child status") {
            panic!("lock-holder child exited before readiness: {status}");
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("lock-holder child never became ready");
}

fn artifacts() -> TasteCycleArtifacts {
    TasteCycleArtifacts {
        candidate_report: b"system-candidates".to_vec(),
        validation_store: b"system-validations".to_vec(),
        replay_report: b"system-replay".to_vec(),
        profile: b"system-profile".to_vec(),
    }
}

fn manifest(artifacts: &TasteCycleArtifacts) -> TasteGenerationManifest {
    TasteGenerationManifest {
        generation: 1,
        previous_manifest_sha256: GENESIS_DIGEST,
        profile_sha256: digest(&artifacts.profile),
        replay_sha256: digest(&artifacts.replay_report),
        candidate_report_sha256: digest(&artifacts.candidate_report),
        validation_store_sha256: digest(&artifacts.validation_store),
    }
}

#[test]
fn child_holds_interlock() {
    let Ok(root) = std::env::var(CHILD_ROOT_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    fs::create_dir_all(&root).expect("child root");
    write_live_lock(&root);
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

#[test]
fn live_process_blocks_then_dead_owner_is_recovered() {
    let root = root();
    fs::create_dir_all(&root).expect("root");
    let current_exe = std::env::current_exe().expect("current test executable");
    let mut child = Command::new(current_exe)
        .arg("--exact")
        .arg("child_holds_interlock")
        .arg("--nocapture")
        .env(CHILD_ROOT_ENV, &root)
        .spawn()
        .expect("spawn lock-holder child");
    wait_until_child_ready(&root, &mut child);

    let artifacts = artifacts();
    let manifest = manifest(&artifacts);
    assert!(matches!(
        commit_taste_cycle(&root, &manifest, &artifacts),
        Err(TasteCycleError::CommitInterlockHeld)
    ));
    assert!(!root.join("CURRENT").exists());
    assert!(!root.join("generation-1").exists());

    child.kill().expect("kill lock-holder child");
    let _ = child.wait().expect("reap lock-holder child");
    fs::remove_file(root.join(READY_FILE)).expect("remove readiness marker");

    commit_taste_cycle(&root, &manifest, &artifacts).expect("recover dead owner and commit");
    assert_eq!(
        fs::read_to_string(root.join("CURRENT")).expect("current"),
        "1\n"
    );
    assert!(!root.join(LOCK_FILE).exists());
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn vanished_pid_is_distinct_from_proc_read_error() {
    let missing = fs::read_to_string(format!("/proc/{}/stat", u32::MAX));
    assert!(matches!(missing, Err(error) if error.kind() == ErrorKind::NotFound));
}
