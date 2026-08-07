//! Cross-artifact provenance coverage for generation-bound controlled restart.

use cogno_runtime::{
    build_taste_restart_manifest, GenerationBoundControlledRestartTasteError,
    GenerationBoundControlledRestartTasteProfile, TasteCampaignReviewAuthority,
    TasteCampaignReviewDisposition, TasteCampaignReviewReport, TasteGenerationManifest,
    TastePreferenceReview, VerifiedTasteProfile, GENESIS_DIGEST,
};
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
        "cogno-generation-bound-restart-{}-{nonce}",
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

fn write_verified_fixture(root: &Path) -> (PathBuf, PathBuf) {
    fs::create_dir_all(root).expect("root");
    let replay_path = root.join("replay.json");
    let candidate_path = root.join("candidates.json");
    let validation_path = root.join("taste.validations");
    let profile_path = root.join("taste.profile");

    let candidates = b"generation-bound-candidates";
    let validations = b"generation-bound-validations";
    fs::write(&candidate_path, candidates).expect("candidates");
    fs::write(&validation_path, validations).expect("validations");

    let candidate_sha256 = digest_hex(candidates);
    let validation_sha256 = digest_hex(validations);
    let preference = serde_json::json!({
        "preference_id": 41,
        "state": "active",
        "confidence_bps": 8300,
        "non_model_confirmations": 3,
        "validations": 4,
        "can_activate": true
    });
    let replay = serde_json::json!({
        "schema_version": 1,
        "authority": "deterministic_replay_only",
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 4,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [preference.clone()]
    });
    let replay_bytes = serde_json::to_vec(&replay).expect("replay json");
    fs::write(&replay_path, &replay_bytes).expect("replay");

    let profile = serde_json::json!({
        "schema_version": 1,
        "authority": "derived_cache_only",
        "replay_sha256": digest_hex(&replay_bytes),
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 4,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [preference]
    });
    fs::write(
        profile_path,
        serde_json::to_vec(&profile).expect("profile json"),
    )
    .expect("profile");

    (replay_path, candidate_path)
}

fn load_profile(root: &Path, replay: &Path, candidates: &Path) -> VerifiedTasteProfile {
    VerifiedTasteProfile::load(replay, candidates, root).expect("verified profile")
}

fn restart_manifest() -> cogno_runtime::TasteRestartManifest {
    let review = TasteCampaignReviewReport {
        first_generation: 1,
        last_generation: 1,
        generations: 1,
        benchmark_cases: 4,
        baseline_total_regret: 40,
        taste_total_regret: 10,
        improved_cases: 3,
        regressed_cases: 0,
        quarantined_model_observations: 1,
        preferences: vec![TastePreferenceReview {
            preference_id: 41,
            latest_confidence_bps: 8_300,
            blockers: Vec::new(),
            eligible: true,
        }],
        disposition: TasteCampaignReviewDisposition::EligibleForControlledRestartReview,
        authority: TasteCampaignReviewAuthority::ReviewOnly,
    };
    build_taste_restart_manifest(&review, [1; 32], [2; 32], [3; 32]).expect("restart manifest")
}

fn generation(profile: &VerifiedTasteProfile) -> TasteGenerationManifest {
    TasteGenerationManifest {
        generation: 1,
        previous_manifest_sha256: GENESIS_DIGEST,
        profile_sha256: profile.profile_sha256(),
        replay_sha256: profile.replay_sha256(),
        candidate_report_sha256: profile.candidate_report_sha256(),
        validation_store_sha256: profile.validation_store_sha256(),
    }
}

#[test]
fn exact_selected_generation_binds_restart_provenance() {
    let root = temp_root();
    let (replay, candidates) = write_verified_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let generation = generation(&profile);
    let expected_generation_sha256 = generation.sha256();
    let manifest = restart_manifest();

    let bound = GenerationBoundControlledRestartTasteProfile::prepare(
        &manifest,
        &generation,
        profile,
    )
    .expect("generation-bound restart");

    assert_eq!(bound.generation(), 1);
    assert_eq!(bound.generation_manifest_sha256(), expected_generation_sha256);
    assert_eq!(bound.restart_manifest_sha256(), manifest.manifest_sha256);
    assert_eq!(bound.active_preferences()[0].preference_id, 41);
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn stale_profile_digest_is_rejected_before_restart_sealing() {
    let root = temp_root();
    let (replay, candidates) = write_verified_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let mut selected = generation(&profile);
    selected.profile_sha256 = [9; 32];

    assert_eq!(
        GenerationBoundControlledRestartTasteProfile::prepare(
            &restart_manifest(),
            &selected,
            profile
        )
        .unwrap_err(),
        GenerationBoundControlledRestartTasteError::ProfileDigestMismatch
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn cross_generation_replay_digest_is_rejected() {
    let root = temp_root();
    let (replay, candidates) = write_verified_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let mut selected = generation(&profile);
    selected.replay_sha256 = [7; 32];

    assert_eq!(
        GenerationBoundControlledRestartTasteProfile::prepare(
            &restart_manifest(),
            &selected,
            profile
        )
        .unwrap_err(),
        GenerationBoundControlledRestartTasteError::ReplayDigestMismatch
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn cross_generation_validation_digest_is_rejected() {
    let root = temp_root();
    let (replay, candidates) = write_verified_fixture(&root);
    let profile = load_profile(&root, &replay, &candidates);
    let mut selected = generation(&profile);
    selected.validation_store_sha256 = [5; 32];

    assert_eq!(
        GenerationBoundControlledRestartTasteProfile::prepare(
            &restart_manifest(),
            &selected,
            profile
        )
        .unwrap_err(),
        GenerationBoundControlledRestartTasteError::ValidationStoreDigestMismatch
    );
    fs::remove_dir_all(root).expect("cleanup");
}
