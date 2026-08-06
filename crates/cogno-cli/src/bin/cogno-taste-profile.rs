//! Persist a verified deterministic taste replay as an atomic derived profile.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u16 = 1;
const PROFILE_AUTHORITY: &str = "derived_cache_only";
const REPLAY_AUTHORITY: &str = "deterministic_replay_only";
const VALIDATION_FILE: &str = "taste.validations";
const PROFILE_FILE: &str = "taste.profile";
const TEMP_PROFILE_FILE: &str = ".taste.profile.tmp";
const MAX_PROFILE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct ReplayPreference {
    preference_id: u64,
    state: String,
    confidence_bps: u16,
    non_model_confirmations: u32,
    validations: usize,
    can_activate: bool,
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
    preferences: Vec<ReplayPreference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct PersistedProfile {
    schema_version: u16,
    authority: String,
    replay_sha256: String,
    candidate_report_sha256: String,
    validation_store_sha256: String,
    validation_records: usize,
    minimum_non_model_confirmations: u32,
    activation_threshold_bps: u16,
    preferences: Vec<ReplayPreference>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-profile: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let replay_path = args.next().ok_or(
        "usage: cogno-taste-profile REPLAY_JSON CANDIDATES_JSON STORE_ROOT",
    )?;
    let candidates_path = args.next().ok_or(
        "usage: cogno-taste-profile REPLAY_JSON CANDIDATES_JSON STORE_ROOT",
    )?;
    let store_root = args.next().ok_or(
        "usage: cogno-taste-profile REPLAY_JSON CANDIDATES_JSON STORE_ROOT",
    )?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let replay_bytes = read_bounded(Path::new(&replay_path), "replay report")?;
    let replay: ReplayReport = serde_json::from_slice(&replay_bytes)
        .map_err(|error| format!("invalid replay report: {error}"))?;
    validate_replay(&replay)?;

    let candidate_bytes = read_bounded(Path::new(&candidates_path), "candidate report")?;
    let candidate_digest = hex_digest(Sha256::digest(&candidate_bytes).into());
    if replay.candidate_report_sha256 != candidate_digest {
        return Err("candidate report digest does not match replay".to_string());
    }

    let store_root = Path::new(&store_root);
    let validation_path = store_root.join(VALIDATION_FILE);
    let validation_bytes = read_bounded(&validation_path, "validation store")?;
    let validation_digest = hex_digest(Sha256::digest(&validation_bytes).into());
    if replay.validation_store_sha256 != validation_digest {
        return Err("validation store digest does not match replay".to_string());
    }

    let replay_digest = hex_digest(Sha256::digest(&replay_bytes).into());
    let profile = PersistedProfile {
        schema_version: SCHEMA_VERSION,
        authority: PROFILE_AUTHORITY.to_string(),
        replay_sha256: replay_digest,
        candidate_report_sha256: candidate_digest,
        validation_store_sha256: validation_digest,
        validation_records: replay.validation_records,
        minimum_non_model_confirmations: replay.minimum_non_model_confirmations,
        activation_threshold_bps: replay.activation_threshold_bps,
        preferences: replay.preferences,
    };

    fs::create_dir_all(store_root)
        .map_err(|error| format!("cannot create profile directory: {error}"))?;
    let profile_path = store_root.join(PROFILE_FILE);
    write_profile_atomically(store_root, &profile_path, &profile)?;
    verify_profile(&profile_path, &profile)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&profile)
            .map_err(|error| format!("cannot serialize profile report: {error}"))?
    );
    eprintln!("profile_path={}", profile_path.display());
    Ok(())
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
    validate_digest(&replay.candidate_report_sha256, "candidate report")?;
    validate_digest(&replay.validation_store_sha256, "validation store")?;
    for preference in &replay.preferences {
        if preference.confidence_bps > 10_000 {
            return Err(format!(
                "preference {} confidence exceeds 10000",
                preference.preference_id
            ));
        }
        let state_allows_activation = preference.state == "active";
        if preference.can_activate != state_allows_activation {
            return Err(format!(
                "preference {} activation flag disagrees with state",
                preference.preference_id
            ));
        }
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("invalid {label} sha256"));
    }
    Ok(())
}

fn read_bounded(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    let length = usize::try_from(metadata.len())
        .map_err(|_| format!("{label} length does not fit usize"))?;
    if length == 0 || length > MAX_PROFILE_BYTES {
        return Err(format!("{label} size is outside permitted bounds"));
    }
    fs::read(path).map_err(|error| format!("cannot read {label} {}: {error}", path.display()))
}

fn write_profile_atomically(
    store_root: &Path,
    profile_path: &Path,
    profile: &PersistedProfile,
) -> Result<(), String> {
    let mut encoded = serde_json::to_vec_pretty(profile)
        .map_err(|error| format!("cannot encode profile: {error}"))?;
    encoded.push(b'\n');
    if encoded.len() > MAX_PROFILE_BYTES {
        return Err("encoded profile exceeds size limit".to_string());
    }

    let temporary_path = store_root.join(TEMP_PROFILE_FILE);
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary_path)
            .map_err(|error| format!("cannot create temporary profile: {error}"))?;
        file.write_all(&encoded)
            .map_err(|error| format!("cannot write temporary profile: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync temporary profile: {error}"))?;
        fs::rename(&temporary_path, profile_path)
            .map_err(|error| format!("cannot atomically replace profile: {error}"))?;
        sync_directory(store_root)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("cannot sync profile directory: {error}"))
}

fn verify_profile(path: &Path, expected: &PersistedProfile) -> Result<(), String> {
    let bytes = read_bounded(path, "persisted profile")?;
    let observed: PersistedProfile = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot decode persisted profile: {error}"))?;
    if &observed != expected {
        return Err("persisted profile differs from derived profile".to_string());
    }
    Ok(())
}

fn hex_digest(digest: [u8; 32]) -> String {
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

    fn replay() -> ReplayReport {
        ReplayReport {
            schema_version: 1,
            authority: REPLAY_AUTHORITY.to_string(),
            candidate_report_sha256: "ab".repeat(32),
            validation_store_sha256: "cd".repeat(32),
            validation_records: 2,
            minimum_non_model_confirmations: 2,
            activation_threshold_bps: 7_000,
            preferences: vec![ReplayPreference {
                preference_id: 7,
                state: "active".to_string(),
                confidence_bps: 8_000,
                non_model_confirmations: 2,
                validations: 2,
                can_activate: true,
            }],
        }
    }

    #[test]
    fn valid_replay_is_accepted() {
        assert_eq!(validate_replay(&replay()), Ok(()));
    }

    #[test]
    fn activation_must_match_state() {
        let mut item = replay();
        item.preferences[0].state = "candidate".to_string();
        assert!(validate_replay(&item).is_err());
    }

    #[test]
    fn elevated_replay_authority_is_rejected() {
        let mut item = replay();
        item.authority = "active_policy".to_string();
        assert!(validate_replay(&item).is_err());
    }

    #[test]
    fn digest_hex_is_stable() {
        assert_eq!(hex_digest([0xef; 32]), "ef".repeat(32));
    }
}
