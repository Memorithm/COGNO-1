//! Fail-closed loading of a derived scientific taste profile.
//!
//! The profile is a cache only. Every load verifies its complete contents
//! against the deterministic replay report and verifies all source digests
//! before exposing active preferences to the runtime.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u16 = 1;
const PROFILE_AUTHORITY: &str = "derived_cache_only";
const REPLAY_AUTHORITY: &str = "deterministic_replay_only";
const PROFILE_FILE: &str = "taste.profile";
const VALIDATION_FILE: &str = "taste.validations";
const MAX_ARTIFACT_BYTES: usize = 4 * 1024 * 1024;

/// A runtime-safe preference whose activation was verified from persisted evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedTastePreference {
    /// Stable preference identifier.
    pub preference_id: u64,
    /// Deterministic confidence in basis points.
    pub confidence_bps: u16,
    /// Number of confirmations that did not originate from a model.
    pub non_model_confirmations: u32,
    /// Total number of validation records applied.
    pub validations: usize,
}

/// Read-only runtime view of a verified taste profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTasteProfile {
    profile_sha256: [u8; 32],
    replay_sha256: [u8; 32],
    candidate_report_sha256: [u8; 32],
    validation_store_sha256: [u8; 32],
    active_preferences: Vec<VerifiedTastePreference>,
}

impl VerifiedTasteProfile {
    /// Load and verify a profile and all artifacts from which it was derived.
    pub fn load(
        replay_path: impl AsRef<Path>,
        candidate_report_path: impl AsRef<Path>,
        store_root: impl AsRef<Path>,
    ) -> Result<Self, VerifiedTasteProfileError> {
        let store_root = store_root.as_ref();
        let profile_path = store_root.join(PROFILE_FILE);
        let validation_path = store_root.join(VALIDATION_FILE);

        let profile_bytes = read_bounded(&profile_path, ArtifactKind::Profile)?;
        let replay_bytes = read_bounded(replay_path.as_ref(), ArtifactKind::Replay)?;
        let candidate_bytes = read_bounded(
            candidate_report_path.as_ref(),
            ArtifactKind::CandidateReport,
        )?;
        let validation_bytes = read_bounded(&validation_path, ArtifactKind::ValidationStore)?;

        let profile: PersistedProfile = serde_json::from_slice(&profile_bytes)
            .map_err(|_| VerifiedTasteProfileError::InvalidProfileJson)?;
        let replay: ReplayReport = serde_json::from_slice(&replay_bytes)
            .map_err(|_| VerifiedTasteProfileError::InvalidReplayJson)?;

        validate_profile(&profile)?;
        validate_replay(&replay)?;
        verify_profile_matches_replay(&profile, &replay)?;
        verify_digest(
            &profile.replay_sha256,
            &replay_bytes,
            ArtifactKind::Replay,
        )?;
        verify_digest(
            &profile.candidate_report_sha256,
            &candidate_bytes,
            ArtifactKind::CandidateReport,
        )?;
        verify_digest(
            &profile.validation_store_sha256,
            &validation_bytes,
            ArtifactKind::ValidationStore,
        )?;

        let active_preferences = profile
            .preferences
            .into_iter()
            .filter(|preference| preference.state == "active" && preference.can_activate)
            .map(|preference| VerifiedTastePreference {
                preference_id: preference.preference_id,
                confidence_bps: preference.confidence_bps,
                non_model_confirmations: preference.non_model_confirmations,
                validations: preference.validations,
            })
            .collect();

        Ok(Self {
            profile_sha256: digest(&profile_bytes),
            replay_sha256: digest(&replay_bytes),
            candidate_report_sha256: digest(&candidate_bytes),
            validation_store_sha256: digest(&validation_bytes),
            active_preferences,
        })
    }

    /// Active preferences exposed to the runtime. This slice is read-only.
    #[must_use]
    pub fn active_preferences(&self) -> &[VerifiedTastePreference] {
        &self.active_preferences
    }

    /// SHA-256 of the verified profile bytes.
    #[must_use]
    pub const fn profile_sha256(&self) -> [u8; 32] {
        self.profile_sha256
    }

    /// SHA-256 of the deterministic replay bytes.
    #[must_use]
    pub const fn replay_sha256(&self) -> [u8; 32] {
        self.replay_sha256
    }

    /// SHA-256 of the quarantined candidate report.
    #[must_use]
    pub const fn candidate_report_sha256(&self) -> [u8; 32] {
        self.candidate_report_sha256
    }

    /// SHA-256 of the persistent validation journal.
    #[must_use]
    pub const fn validation_store_sha256(&self) -> [u8; 32] {
        self.validation_store_sha256
    }
}

/// Failure while loading or verifying a derived taste profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedTasteProfileError {
    /// An artifact could not be read.
    CannotRead {
        /// Kind of artifact that failed.
        artifact: &'static str,
        /// Path that could not be read.
        path: PathBuf,
    },
    /// An artifact was empty or exceeded the configured bound.
    InvalidArtifactSize {
        /// Kind of artifact with an invalid size.
        artifact: &'static str,
    },
    /// The profile JSON could not be parsed.
    InvalidProfileJson,
    /// The replay JSON could not be parsed.
    InvalidReplayJson,
    /// Unsupported schema or authority was observed.
    InvalidAuthorityOrSchema,
    /// Profile policy fields are inconsistent.
    InvalidActivationPolicy,
    /// A preference violates activation invariants.
    InvalidPreferenceState,
    /// Profile contents differ from deterministic replay contents.
    ProfileReplayMismatch,
    /// A source digest differs from the digest recorded in the profile.
    DigestMismatch {
        /// Kind of artifact whose digest differed.
        artifact: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
enum ArtifactKind {
    Profile,
    Replay,
    CandidateReport,
    ValidationStore,
}

impl ArtifactKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Profile => "taste profile",
            Self::Replay => "replay report",
            Self::CandidateReport => "candidate report",
            Self::ValidationStore => "validation store",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct PreferenceRecord {
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
    preferences: Vec<PreferenceRecord>,
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
    preferences: Vec<PreferenceRecord>,
}

fn validate_profile(profile: &PersistedProfile) -> Result<(), VerifiedTasteProfileError> {
    if profile.schema_version != SCHEMA_VERSION || profile.authority != PROFILE_AUTHORITY {
        return Err(VerifiedTasteProfileError::InvalidAuthorityOrSchema);
    }
    validate_policy(
        profile.minimum_non_model_confirmations,
        profile.activation_threshold_bps,
        &profile.preferences,
    )
}

fn validate_replay(replay: &ReplayReport) -> Result<(), VerifiedTasteProfileError> {
    if replay.schema_version != SCHEMA_VERSION || replay.authority != REPLAY_AUTHORITY {
        return Err(VerifiedTasteProfileError::InvalidAuthorityOrSchema);
    }
    validate_policy(
        replay.minimum_non_model_confirmations,
        replay.activation_threshold_bps,
        &replay.preferences,
    )
}

fn validate_policy(
    minimum_confirmations: u32,
    threshold_bps: u16,
    preferences: &[PreferenceRecord],
) -> Result<(), VerifiedTasteProfileError> {
    if minimum_confirmations == 0 || threshold_bps > 10_000 {
        return Err(VerifiedTasteProfileError::InvalidActivationPolicy);
    }
    for preference in preferences {
        if preference.confidence_bps > 10_000 {
            return Err(VerifiedTasteProfileError::InvalidPreferenceState);
        }
        if preference.state == "active" {
            if !preference.can_activate
                || preference.non_model_confirmations < minimum_confirmations
                || preference.confidence_bps < threshold_bps
            {
                return Err(VerifiedTasteProfileError::InvalidPreferenceState);
            }
        } else if preference.can_activate {
            return Err(VerifiedTasteProfileError::InvalidPreferenceState);
        }
    }
    Ok(())
}

fn verify_profile_matches_replay(
    profile: &PersistedProfile,
    replay: &ReplayReport,
) -> Result<(), VerifiedTasteProfileError> {
    if profile.validation_records != replay.validation_records
        || profile.minimum_non_model_confirmations != replay.minimum_non_model_confirmations
        || profile.activation_threshold_bps != replay.activation_threshold_bps
        || profile.candidate_report_sha256 != replay.candidate_report_sha256
        || profile.validation_store_sha256 != replay.validation_store_sha256
        || profile.preferences != replay.preferences
    {
        return Err(VerifiedTasteProfileError::ProfileReplayMismatch);
    }
    Ok(())
}

fn read_bounded(path: &Path, kind: ArtifactKind) -> Result<Vec<u8>, VerifiedTasteProfileError> {
    let metadata = fs::metadata(path).map_err(|_| VerifiedTasteProfileError::CannotRead {
        artifact: kind.label(),
        path: path.to_path_buf(),
    })?;
    let length = usize::try_from(metadata.len())
        .map_err(|_| VerifiedTasteProfileError::InvalidArtifactSize {
            artifact: kind.label(),
        })?;
    if length == 0 || length > MAX_ARTIFACT_BYTES {
        return Err(VerifiedTasteProfileError::InvalidArtifactSize {
            artifact: kind.label(),
        });
    }
    fs::read(path).map_err(|_| VerifiedTasteProfileError::CannotRead {
        artifact: kind.label(),
        path: path.to_path_buf(),
    })
}

fn verify_digest(
    expected_hex: &str,
    bytes: &[u8],
    kind: ArtifactKind,
) -> Result<(), VerifiedTasteProfileError> {
    if expected_hex != digest_hex(bytes) {
        return Err(VerifiedTasteProfileError::DigestMismatch {
            artifact: kind.label(),
        });
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = digest(bytes);
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-verified-profile-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn profile_tampering_is_rejected() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("root");
        let replay_path = root.join("replay.json");
        let candidate_path = root.join("candidates.json");
        let validation_path = root.join(VALIDATION_FILE);
        let profile_path = root.join(PROFILE_FILE);

        let candidates = b"candidate";
        let validations = b"validation";
        fs::write(&candidate_path, candidates).expect("candidate");
        fs::write(&validation_path, validations).expect("validation");

        let candidate_digest = digest_hex(candidates);
        let validation_digest = digest_hex(validations);
        let preference = serde_json::json!({
            "preference_id": 7,
            "state": "active",
            "confidence_bps": 8000,
            "non_model_confirmations": 2,
            "validations": 2,
            "can_activate": true
        });
        let replay = serde_json::json!({
            "schema_version": 1,
            "authority": REPLAY_AUTHORITY,
            "candidate_report_sha256": candidate_digest,
            "validation_store_sha256": validation_digest,
            "validation_records": 2,
            "minimum_non_model_confirmations": 2,
            "activation_threshold_bps": 7000,
            "preferences": [preference.clone()]
        });
        let replay_bytes = serde_json::to_vec(&replay).expect("replay");
        fs::write(&replay_path, &replay_bytes).expect("replay file");

        let profile = serde_json::json!({
            "schema_version": 1,
            "authority": PROFILE_AUTHORITY,
            "replay_sha256": digest_hex(&replay_bytes),
            "candidate_report_sha256": candidate_digest,
            "validation_store_sha256": validation_digest,
            "validation_records": 2,
            "minimum_non_model_confirmations": 2,
            "activation_threshold_bps": 7000,
            "preferences": [preference]
        });
        fs::write(
            &profile_path,
            serde_json::to_vec(&profile).expect("profile"),
        )
        .expect("profile file");

        let loaded = VerifiedTasteProfile::load(&replay_path, &candidate_path, &root)
            .expect("verified profile");
        assert_eq!(loaded.active_preferences().len(), 1);

        let mut tampered = profile;
        tampered["preferences"][0]["confidence_bps"] = serde_json::json!(9999);
        fs::write(
            &profile_path,
            serde_json::to_vec(&tampered).expect("tampered"),
        )
        .expect("tampered file");

        assert_eq!(
            VerifiedTasteProfile::load(&replay_path, &candidate_path, &root),
            Err(VerifiedTasteProfileError::ProfileReplayMismatch)
        );

        fs::remove_dir_all(root).expect("cleanup");
    }
}
