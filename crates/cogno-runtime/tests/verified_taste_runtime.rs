//! End-to-end attachment of a verified scientific taste profile to Runtime.

use cogno_core::{KvCachePolicy, MemoryBudget, QueueFullPolicy};
use cogno_runtime::{Runtime, RuntimeConfig, VerifiedTasteProfile};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-runtime-taste-{}-{nonce}",
        std::process::id()
    ))
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("string write");
    }
    output
}

fn runtime_config() -> RuntimeConfig {
    RuntimeConfig {
        budget: MemoryBudget::try_new(
            1024, 100, 100, 100, 100, 100, 100, 100, 256, 256, 64, 32, 8, 16, 4, 16,
        )
        .expect("budget"),
        reserved_bytes: 0,
        kv_capacity_tokens: 64,
        kv_policy: KvCachePolicy::RejectOnOverflow,
        queue_capacity: 8,
        queue_policy: QueueFullPolicy::RejectNewest,
    }
}

fn write_verified_fixture(root: &Path) -> (PathBuf, PathBuf) {
    fs::create_dir_all(root).expect("root");
    let replay_path = root.join("taste.replay.json");
    let candidates_path = root.join("taste.candidates.json");
    let validations_path = root.join("taste.validations");
    let profile_path = root.join("taste.profile");

    let candidates = b"quarantined-candidates";
    let validations = b"deterministic-validations";
    fs::write(&candidates_path, candidates).expect("candidates");
    fs::write(&validations_path, validations).expect("validations");

    let candidate_sha256 = digest_hex(candidates);
    let validation_sha256 = digest_hex(validations);
    let active = serde_json::json!({
        "preference_id": 42,
        "state": "active",
        "confidence_bps": 8200,
        "non_model_confirmations": 2,
        "validations": 2,
        "can_activate": true
    });
    let quarantined = serde_json::json!({
        "preference_id": 99,
        "state": "quarantined",
        "confidence_bps": 5000,
        "non_model_confirmations": 0,
        "validations": 0,
        "can_activate": false
    });
    let replay = serde_json::json!({
        "schema_version": 1,
        "authority": "deterministic_replay_only",
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 2,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [active.clone(), quarantined.clone()]
    });
    let replay_bytes = serde_json::to_vec(&replay).expect("replay json");
    fs::write(&replay_path, &replay_bytes).expect("replay");

    let profile = serde_json::json!({
        "schema_version": 1,
        "authority": "derived_cache_only",
        "replay_sha256": digest_hex(&replay_bytes),
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 2,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [active, quarantined]
    });
    fs::write(
        profile_path,
        serde_json::to_vec(&profile).expect("profile json"),
    )
    .expect("profile");

    (replay_path, candidates_path)
}

#[test]
fn runtime_exposes_only_verified_active_preferences() {
    let root = temp_root();
    let (replay_path, candidates_path) = write_verified_fixture(&root);
    let profile = VerifiedTasteProfile::load(&replay_path, &candidates_path, &root)
        .expect("verified profile");

    let mut runtime = Runtime::try_new(runtime_config()).expect("runtime");
    assert!(runtime.verified_taste_profile().is_none());
    assert!(runtime.active_taste_preferences().is_empty());

    runtime.install_verified_taste_profile(profile);

    let active = runtime.active_taste_preferences();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].preference_id, 42);
    assert_eq!(active[0].confidence_bps, 8200);
    assert_eq!(active[0].non_model_confirmations, 2);
    assert_eq!(active[0].validations, 2);
    assert_eq!(runtime.active_taste_preference(42), Some(&active[0]));
    assert_eq!(runtime.active_taste_preference(99), None);

    fs::remove_dir_all(root).expect("cleanup");
}
