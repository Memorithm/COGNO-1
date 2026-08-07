use cogno_runtime::{
    build_taste_restart_manifest, ControlledRestartTasteAuthority, ControlledRestartTasteError,
    ControlledRestartTasteProfile, TasteCampaignReviewAuthority, TasteCampaignReviewDisposition,
    TasteCampaignReviewReport, TastePreferenceReview, VerifiedTasteProfile,
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
        "cogno-controlled-restart-{}-{nonce}",
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

fn write_verified_profile(root: &Path, confidence_bps: u16) -> (PathBuf, PathBuf) {
    fs::create_dir_all(root).expect("root");
    let replay_path = root.join("replay.json");
    let candidate_path = root.join("candidates.json");
    let validation_path = root.join("taste.validations");
    let profile_path = root.join("taste.profile");

    let candidate_bytes = b"controlled-restart-candidates";
    let validation_bytes = b"controlled-restart-validations";
    fs::write(&candidate_path, candidate_bytes).expect("candidate");
    fs::write(&validation_path, validation_bytes).expect("validation");

    let candidate_digest = digest_hex(candidate_bytes);
    let validation_digest = digest_hex(validation_bytes);
    let preference = serde_json::json!({
        "preference_id": 7,
        "state": "active",
        "confidence_bps": confidence_bps,
        "non_model_confirmations": 2,
        "validations": 2,
        "can_activate": true
    });
    let replay = serde_json::json!({
        "schema_version": 1,
        "authority": "deterministic_replay_only",
        "candidate_report_sha256": candidate_digest,
        "validation_store_sha256": validation_digest,
        "validation_records": 2,
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
        "candidate_report_sha256": candidate_digest,
        "validation_store_sha256": validation_digest,
        "validation_records": 2,
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

fn eligible_review(confidence_bps: u16) -> TasteCampaignReviewReport {
    TasteCampaignReviewReport {
        first_generation: 10,
        last_generation: 12,
        generations: 3,
        benchmark_cases: 4,
        baseline_total_regret: 40,
        taste_total_regret: 10,
        improved_cases: 3,
        regressed_cases: 0,
        quarantined_model_observations: 1,
        preferences: vec![TastePreferenceReview {
            preference_id: 7,
            latest_confidence_bps: confidence_bps,
            blockers: Vec::new(),
            eligible: true,
        }],
        disposition: TasteCampaignReviewDisposition::EligibleForControlledRestartReview,
        authority: TasteCampaignReviewAuthority::ReviewOnly,
    }
}

#[test]
fn exact_review_and_verified_profile_are_sealed_for_restart() {
    let root = temp_root();
    let (replay_path, candidate_path) = write_verified_profile(&root, 8_000);
    let profile = VerifiedTasteProfile::load(&replay_path, &candidate_path, &root).expect("profile");
    let manifest = build_taste_restart_manifest(&eligible_review(8_000), [1; 32], [2; 32], [3; 32])
        .expect("manifest");

    let sealed = ControlledRestartTasteProfile::prepare(&manifest, profile).expect("sealed");

    assert_eq!(
        sealed.authority(),
        ControlledRestartTasteAuthority::RestartInitializationOnly
    );
    assert_eq!(sealed.manifest_sha256(), manifest.manifest_sha256);
    assert_eq!(sealed.first_generation(), 10);
    assert_eq!(sealed.last_generation(), 12);
    assert_eq!(sealed.generations(), 3);
    assert_eq!(sealed.active_preferences().len(), 1);
    assert_eq!(sealed.active_preferences()[0].preference_id, 7);
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn confidence_disagreement_fails_closed() {
    let root = temp_root();
    let (replay_path, candidate_path) = write_verified_profile(&root, 8_000);
    let profile = VerifiedTasteProfile::load(&replay_path, &candidate_path, &root).expect("profile");
    let manifest = build_taste_restart_manifest(&eligible_review(8_100), [1; 32], [2; 32], [3; 32])
        .expect("manifest");

    assert_eq!(
        ControlledRestartTasteProfile::prepare(&manifest, profile).unwrap_err(),
        ControlledRestartTasteError::ConfidenceMismatch(7)
    );
    fs::remove_dir_all(root).expect("cleanup");
}
