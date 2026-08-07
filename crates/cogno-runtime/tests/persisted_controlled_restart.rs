//! End-to-end persisted selection -> verified profile -> generation-bound restart seal.

use cogno_runtime::{
    build_taste_restart_manifest, commit_taste_cycle,
    prepare_persisted_controlled_restart_taste_profile, PersistedControlledRestartTasteError,
    PersistedTasteGenerationError, TasteCampaignReviewAuthority, TasteCampaignReviewDisposition,
    TasteCampaignReviewReport, TasteCycleArtifacts, TasteGenerationManifest, TastePreferenceReview,
    GENESIS_DIGEST,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-persisted-controlled-restart-{}-{nonce}",
        std::process::id()
    ))
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest(bytes) {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("string write");
    }
    output
}

fn artifacts() -> TasteCycleArtifacts {
    let candidate_report = b"persisted-candidates".to_vec();
    let validation_store = b"persisted-validations".to_vec();
    let candidate_sha256 = digest_hex(&candidate_report);
    let validation_sha256 = digest_hex(&validation_store);
    let preference = serde_json::json!({
        "preference_id": 71,
        "state": "active",
        "confidence_bps": 8400,
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
    let replay_report = serde_json::to_vec(&replay).expect("replay");
    let profile = serde_json::json!({
        "schema_version": 1,
        "authority": "derived_cache_only",
        "replay_sha256": digest_hex(&replay_report),
        "candidate_report_sha256": candidate_sha256,
        "validation_store_sha256": validation_sha256,
        "validation_records": 4,
        "minimum_non_model_confirmations": 2,
        "activation_threshold_bps": 7000,
        "preferences": [preference]
    });

    TasteCycleArtifacts {
        candidate_report,
        validation_store,
        replay_report,
        profile: serde_json::to_vec(&profile).expect("profile"),
    }
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
            preference_id: 71,
            latest_confidence_bps: 8_400,
            blockers: Vec::new(),
            eligible: true,
        }],
        disposition: TasteCampaignReviewDisposition::EligibleForControlledRestartReview,
        authority: TasteCampaignReviewAuthority::ReviewOnly,
    };
    build_taste_restart_manifest(&review, [1; 32], [2; 32], [3; 32]).expect("manifest")
}

fn commit_generation(root: &PathBuf) {
    let artifacts = artifacts();
    let manifest = TasteGenerationManifest {
        generation: 1,
        previous_manifest_sha256: GENESIS_DIGEST,
        profile_sha256: digest(&artifacts.profile),
        replay_sha256: digest(&artifacts.replay_report),
        candidate_report_sha256: digest(&artifacts.candidate_report),
        validation_store_sha256: digest(&artifacts.validation_store),
    };
    commit_taste_cycle(root, &manifest, &artifacts).expect("commit");
}

#[test]
fn persisted_selected_generation_produces_restart_seal() {
    let root = root();
    commit_generation(&root);

    let sealed = prepare_persisted_controlled_restart_taste_profile(&root, &restart_manifest())
        .expect("persisted seal");
    assert_eq!(sealed.generation(), 1);
    assert_eq!(sealed.active_preferences().len(), 1);
    assert_eq!(sealed.active_preferences()[0].preference_id, 71);

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn tampered_selected_profile_fails_before_restart_sealing() {
    let root = root();
    commit_generation(&root);
    fs::write(root.join("generation-1/taste.profile"), b"tampered").expect("tamper");

    assert_eq!(
        prepare_persisted_controlled_restart_taste_profile(&root, &restart_manifest()).unwrap_err(),
        PersistedControlledRestartTasteError::PersistedGeneration(
            PersistedTasteGenerationError::ArtifactDigestMismatch {
                generation: 1,
                artifact: "profile",
            }
        )
    );

    fs::remove_dir_all(root).expect("cleanup");
}
