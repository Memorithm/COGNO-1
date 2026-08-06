//! Verify a derived taste profile against all source artifacts before exposure.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const PROFILE_AUTHORITY: &str = "derived_cache_only";
const REPLAY_AUTHORITY: &str = "deterministic_replay_only";
const VERIFIED_AUTHORITY: &str = "verified_profile_view_only";
const VALIDATION_FILE: &str = "taste.validations";
const PROFILE_FILE: &str = "taste.profile";
const MAX_ARTIFACT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ProfilePreference {
    preference_id: u64,
    state: String,
    confidence_bps: u16,
    non_model_confirmations: u32,
    validations: usize,
    can_activate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct PersistedProfile {
    schema_version: u16,
    authority: String,
    replay_sha256: String,
    candidate_report_sha256: String,
    validation_store_sha256: String,
    validation_records: usize,
    minimum_non_model_confirmations: u32,
    activation_threshold_bps: u16,
    preferences: Vec<ProfilePreference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ReplayReport {
    schema_version: u16,
    authority: String,
    candidate_report_sha256: String,
    validation_store_sha256: String,
    validation_records: usize,
    minimum_non_model_confirmations: u32,
    activation_threshold_bps: u16,
    preferences: Vec<ProfilePreference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct VerifiedPreference {
    preference_id: u64,
    confidence_bps: u16,
    non_model_confirmations: u32,
    validations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct VerifiedProfileView {
    schema_version: u16,
    authority: &'static str,
    profile_sha256: String,
    replay_sha256: String,
    candidate_report_sha256: String,
    validation_store_sha256: String,
    active_preferences: Vec<VerifiedPreference>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-profile-verify: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let replay_path = args
        .next()
        .ok_or("usage: cogno-taste-profile-verify REPLAY_JSON CANDIDATES_JSON STORE_ROOT")?;
    let candidates_path = args
        .next()
        .ok_or("usage: cogno-taste-profile-verify REPLAY_JSON CANDIDATES_JSON STORE_ROOT")?;
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-profile-verify REPLAY_JSON CANDIDATES_JSON STORE_ROOT")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let store_root = Path::new(&store_root);
    let profile_path = store_root.join(PROFILE_FILE);
    let validation_path = store_root.join(VALIDATION_FILE);

    let profile_bytes = read_bounded(&profile_path, "taste profile")?;
    let replay_bytes = read_bounded(Path::new(&replay_path), "replay report")?;
    let candidate_bytes = read_bounded(Path::new(&candidates_path), "candidate report")?;
    let validation_bytes = read_bounded(&validation_path, "validation store")?;

    let profile: PersistedProfile = serde_json::from_slice(&profile_bytes)
        .map_err(|error| format!("invalid taste profile: {error}"))?;
    let replay: ReplayReport = serde_json::from_slice(&replay_bytes)
        .map_err(|error| format!("invalid replay report: {error}"))?;

    validate_profile(&profile)?;
    validate_replay(&replay)?;

    verify_digest(&profile.replay_sha256, &replay_bytes, "replay report")?;
    verify_digest(
        &profile.candidate_report_sha256,
        &candidate_bytes,
        "candidate report",
    )?;
    verify_digest(
        &profile.validation_store_sha256,
        &validation_bytes,
        "validation store",
    )?;
    verify_profile_matches_replay(&profile, &replay)?;

    let active_preferences = profile
        .preferences
        .into_iter()
        .filter(|preference| preference.state == "active" && preference.can_activate)
        .map(|preference| VerifiedPreference {
            preference_id: preference.preference_id,
            confidence_bps: preference.confidence_bps,
            non_model_confirmations: preference.non_model_confirmations,
            validations: preference.validations,
        })
        .collect();

    let view = VerifiedProfileView {
        schema_version: SCHEMA_VERSION,
        authority: VERIFIED_AUTHORITY,
        profile_sha256: digest_hex(&profile_bytes),
        replay_sha256: digest_hex(&replay_bytes),
        candidate_report_sha256: digest_hex(&candidate_bytes),
        validation_store_sha256: digest_hex(&validation_bytes),
        active_preferences,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&view)
            .map_err(|error| format!("cannot serialize verified profile: {error}"))?
    );
    Ok(())
}

fn validate_profile(profile: &PersistedProfile) -> Result<(), String> {
    if profile.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported profile schema version {}",
            profile.schema_version
        ));
    }
    if profile.authority != PROFILE_AUTHORITY {
        return Err("profile authority must be derived_cache_only".to_string());
    }
    validate_policy_and_preferences(
        profile.activation_threshold_bps,
        profile.minimum_non_model_confirmations,
        &profile.preferences,
    )
}

fn validate_replay(replay: &ReplayReport) -> Result<(), String> {
    if replay.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported replay schema version {}",
            replay.schema_version
        ));
    }
    if replay.authority != REPLAY_AUTHORITY {
        return Err("replay authority must be deterministic_replay_only".to_string());
    }
    validate_policy_and_preferences(
        replay.activation_threshold_bps,
        replay.minimum_non_model_confirmations,
        &replay.preferences,
    )
}

fn validate_policy_and_preferences(
    activation_threshold_bps: u16,
    minimum_non_model_confirmations: u32,
    preferences: &[ProfilePreference],
) -> Result<(), String> {
    if activation_threshold_bps > 10_000 {
        return Err("activation threshold exceeds 10000".to_string());
    }
    if minimum_non_model_confirmations == 0 {
        return Err("minimum non-model confirmations must be non-zero".to_string());
    }
    for preference in preferences {
        if preference.confidence_bps > 10_000 {
            return Err(format!(
                "preference {} confidence exceeds 10000",
                preference.preference_id
            ));
        }
        if preference.state == "active" {
            if !preference.can_activate {
                return Err(format!(
                    "active preference {} is not activatable",
                    preference.preference_id
                ));
            }
            if preference.non_model_confirmations < minimum_non_model_confirmations {
                return Err(format!(
                    "active preference {} lacks confirmations",
                    preference.preference_id
                ));
            }
            if preference.confidence_bps < activation_threshold_bps {
                return Err(format!(
                    "active preference {} is below confidence threshold",
                    preference.preference_id
                ));
            }
        } else if preference.can_activate {
            return Err(format!(
                "non-active preference {} cannot be activatable",
                preference.preference_id
            ));
        }
    }
    Ok(())
}

fn verify_profile_matches_replay(
    profile: &PersistedProfile,
    replay: &ReplayReport,
) -> Result<(), String> {
    if profile.candidate_report_sha256 != replay.candidate_report_sha256 {
        return Err("candidate digest differs between profile and replay".to_string());
    }
    if profile.validation_store_sha256 != replay.validation_store_sha256 {
        return Err("validation digest differs between profile and replay".to_string());
    }
    if profile.validation_records != replay.validation_records {
        return Err("validation record count differs between profile and replay".to_string());
    }
    if profile.minimum_non_model_confirmations != replay.minimum_non_model_confirmations {
        return Err("confirmation policy differs between profile and replay".to_string());
    }
    if profile.activation_threshold_bps != replay.activation_threshold_bps {
        return Err("activation threshold differs between profile and replay".to_string());
    }
    if profile.preferences != replay.preferences {
        return Err("preference contents differ between profile and replay".to_string());
    }
    Ok(())
}

fn verify_digest(expected: &str, bytes: &[u8], label: &str) -> Result<(), String> {
    let observed = digest_hex(bytes);
    if expected != observed {
        return Err(format!("{label} digest does not match profile"));
    }
    Ok(())
}

fn read_bounded(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    let length = usize::try_from(metadata.len())
        .map_err(|_| format!("{label} length does not fit usize"))?;
    if length == 0 || length > MAX_ARTIFACT_BYTES {
        return Err(format!("{label} size is outside permitted bounds"));
    }
    fs::read(path).map_err(|error| format!("cannot read {label} {}: {error}", path.display()))
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("string write");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preference() -> ProfilePreference {
        ProfilePreference {
            preference_id: 7,
            state: "active".to_string(),
            confidence_bps: 8_000,
            non_model_confirmations: 2,
            validations: 2,
            can_activate: true,
        }
    }

    fn profile() -> PersistedProfile {
        PersistedProfile {
            schema_version: 1,
            authority: PROFILE_AUTHORITY.to_string(),
            replay_sha256: "ab".repeat(32),
            candidate_report_sha256: "cd".repeat(32),
            validation_store_sha256: "ef".repeat(32),
            validation_records: 2,
            minimum_non_model_confirmations: 2,
            activation_threshold_bps: 7_000,
            preferences: vec![preference()],
        }
    }

    fn replay() -> ReplayReport {
        ReplayReport {
            schema_version: 1,
            authority: REPLAY_AUTHORITY.to_string(),
            candidate_report_sha256: "cd".repeat(32),
            validation_store_sha256: "ef".repeat(32),
            validation_records: 2,
            minimum_non_model_confirmations: 2,
            activation_threshold_bps: 7_000,
            preferences: vec![preference()],
        }
    }

    #[test]
    fn valid_active_profile_is_accepted() {
        assert_eq!(validate_profile(&profile()), Ok(()));
    }

    #[test]
    fn active_profile_requires_confirmations() {
        let mut item = profile();
        item.preferences[0].non_model_confirmations = 1;
        assert!(validate_profile(&item).is_err());
    }

    #[test]
    fn active_profile_requires_threshold() {
        let mut item = profile();
        item.preferences[0].confidence_bps = 6_999;
        assert!(validate_profile(&item).is_err());
    }

    #[test]
    fn profile_contents_must_match_replay() {
        let mut item = profile();
        item.preferences[0].confidence_bps = 9_999;
        assert!(verify_profile_matches_replay(&item, &replay()).is_err());
    }

    #[test]
    fn valid_profile_matches_replay() {
        assert_eq!(verify_profile_matches_replay(&profile(), &replay()), Ok(()));
    }

    #[test]
    fn digest_is_stable() {
        assert_eq!(
            digest_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
