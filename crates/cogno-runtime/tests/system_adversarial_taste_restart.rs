//! System-level adversarial coverage for the scientific-taste restart authority chain.

use cogno_core::{KvCachePolicy, MemoryBudget, QueueFullPolicy};
use cogno_runtime::{
    build_taste_restart_manifest, ControlledRestartTasteError, ControlledRestartTasteProfile,
    Runtime, RuntimeConfig, RuntimeTasteProfileError, TasteCampaignReviewAuthority,
    TasteCampaignReviewDisposition, TasteCampaignReviewReport, TastePreferenceReview,
    TasteRestartManifest, TasteRestartPreference, VerifiedTasteProfile,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const RESTART_MANIFEST_DOMAIN: &[u8; 16] = b"COGNO-RST-MAN-V1";

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-system-adversarial-restart-{}-{nonce}",
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

fn write_profile_fixture(root: &Path) -> (PathBuf, PathBuf) {
    fs::create_dir_all(root).expect("root");
    let replay_path = root.join("taste.replay.json");
    let candidates_path = root.join("taste.candidates.json");
    let validations_path = root.join("taste.validations");
    let profile_path = root.join("taste.profile");

    let candidates = b"system-adversarial-candidates";
    let validations = b"system-adversarial-validations";
    fs::write(&candidates_path, candidates).expect("candidates");
    fs::write(&validations_path, validations).expect("validations");

    let candidate_sha256 = digest_hex(candidates);
    let validation_sha256 = digest_hex(validations);
    let active_a = serde_json::json!({
        "preference_id": 11,
        "state": "active",
        "confidence_bps": 8200,
        "non_model_confirmations": 2,
        "validations": 3,
        "can_activate": true
    });
    let active_b = serde_json::json!({
        "preference_id": 22,
        "state": "active",
        "confidence_bps": 7900,
        "non_model_confirmations": 3,
        "validations": 4,
        "can_activate": true
    });
    let quarantined_model = serde_json::json!({
        "preference_id": 99,
        "state": "quarantined",
        "confidence_bps": 9900,
        "non_model_confirmations": 0,
        "validations": 1,
        "can_activate": false
    });
    let replay = serde_json::json!({
        "schema_version": 1,
        "authority": "deterministic_replay_only",
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 8,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [active_a.clone(), active_b.clone(), quarantined_model.clone()]
    });
    let replay_bytes = serde_json::to_vec(&replay).expect("replay json");
    fs::write(&replay_path, &replay_bytes).expect("replay");

    let profile = serde_json::json!({
        "schema_version": 1,
        "authority": "derived_cache_only",
        "replay_sha256": digest_hex(&replay_bytes),
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 8,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [active_a, active_b, quarantined_model]
    });
    fs::write(
        profile_path,
        serde_json::to_vec(&profile).expect("profile json"),
    )
    .expect("profile");

    (replay_path, candidates_path)
}

fn eligible_review(confidence_a: u16, confidence_b: u16) -> TasteCampaignReviewReport {
    TasteCampaignReviewReport {
        first_generation: 40,
        last_generation: 45,
        generations: 6,
        benchmark_cases: 16,
        baseline_total_regret: 160,
        taste_total_regret: 40,
        improved_cases: 12,
        regressed_cases: 0,
        quarantined_model_observations: 7,
        preferences: vec![
            TastePreferenceReview {
                preference_id: 22,
                latest_confidence_bps: confidence_b,
                blockers: Vec::new(),
                eligible: true,
            },
            TastePreferenceReview {
                preference_id: 11,
                latest_confidence_bps: confidence_a,
                blockers: Vec::new(),
                eligible: true,
            },
        ],
        disposition: TasteCampaignReviewDisposition::EligibleForControlledRestartReview,
        authority: TasteCampaignReviewAuthority::ReviewOnly,
    }
}

fn load_profile(root: &Path, replay: &Path, candidates: &Path) -> VerifiedTasteProfile {
    VerifiedTasteProfile::load(replay, candidates, root).expect("verified profile")
}

fn hash_preferences(preferences: &[TasteRestartPreference]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RESTART_MANIFEST_DOMAIN);
    hasher.update(b"preferences");
    hasher.update((preferences.len() as u64).to_le_bytes());
    for preference in preferences {
        hasher.update(preference.preference_id.to_le_bytes());
        hasher.update(preference.confidence_bps.to_le_bytes());
    }
    hasher.finalize().into()
}

fn rehash_manifest(manifest: &mut TasteRestartManifest) {
    manifest.preferences_sha256 = hash_preferences(&manifest.preferences);
    let mut hasher = Sha256::new();
    hasher.update(RESTART_MANIFEST_DOMAIN);
    hasher.update(1u16.to_le_bytes());
    hasher.update(manifest.first_generation.to_le_bytes());
    hasher.update(manifest.last_generation.to_le_bytes());
    hasher.update((manifest.generations as u64).to_le_bytes());
    hasher.update(manifest.campaign_sha256);
    hasher.update(manifest.benchmark_sha256);
    hasher.update(manifest.longitudinal_sha256);
    hasher.update(manifest.preferences_sha256);
    hasher.update((manifest.preferences.len() as u64).to_le_bytes());
    for preference in &manifest.preferences {
        hasher.update(preference.preference_id.to_le_bytes());
        hasher.update(preference.confidence_bps.to_le_bytes());
    }
    manifest.manifest_sha256 = hasher.finalize().into();
}

#[test]
fn full_restart_chain_exposes_only_reviewed_active_preferences() {
    let root = temp_root();
    let (replay, candidates) = write_profile_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let manifest =
        build_taste_restart_manifest(&eligible_review(8_200, 7_900), [1; 32], [2; 32], [3; 32])
            .expect("manifest");
    let sealed = ControlledRestartTasteProfile::prepare(&manifest, profile).expect("seal");
    let mut runtime = Runtime::try_new(runtime_config()).expect("runtime");

    runtime
        .install_controlled_restart_taste_profile(sealed)
        .expect("install");

    assert_eq!(runtime.active_taste_preferences().len(), 2);
    assert!(runtime.active_taste_preference(11).is_some());
    assert!(runtime.active_taste_preference(22).is_some());
    assert!(runtime.active_taste_preference(99).is_none());

    let replacement_profile = load_profile(&root, &replay, &candidates);
    let replacement = ControlledRestartTasteProfile::prepare(&manifest, replacement_profile)
        .expect("replacement seal");
    assert_eq!(
        runtime.install_controlled_restart_taste_profile(replacement),
        Err(RuntimeTasteProfileError::AlreadyInstalled)
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn reviewed_confidence_mismatch_is_rejected_before_runtime_install() {
    let root = temp_root();
    let (replay, candidates) = write_profile_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let manifest =
        build_taste_restart_manifest(&eligible_review(8_201, 7_900), [1; 32], [2; 32], [3; 32])
            .expect("manifest");

    assert_eq!(
        ControlledRestartTasteProfile::prepare(&manifest, profile).unwrap_err(),
        ControlledRestartTasteError::ConfidenceMismatch(11)
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn noncanonical_but_rehashed_manifest_is_rejected_at_restart_boundary() {
    let root = temp_root();
    let (replay, candidates) = write_profile_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let mut manifest =
        build_taste_restart_manifest(&eligible_review(8_200, 7_900), [1; 32], [2; 32], [3; 32])
            .expect("manifest");

    manifest.preferences.reverse();
    rehash_manifest(&mut manifest);

    assert_eq!(
        ControlledRestartTasteProfile::prepare(&manifest, profile).unwrap_err(),
        ControlledRestartTasteError::NonCanonicalPreferenceOrder
    );

    fs::remove_dir_all(root).expect("cleanup");
}
