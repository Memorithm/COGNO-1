//! Deterministic controlled-restart manifest for reviewed scientific taste campaigns.
//!
//! This module converts an eligible [`TasteCampaignReviewReport`] into a sealed,
//! content-addressed review artifact. The manifest is deliberately non-authoritative:
//! it cannot install, replace, or hot-swap a verified runtime taste profile.

use crate::taste_campaign_review::{
    TasteCampaignReviewAuthority, TasteCampaignReviewDisposition, TasteCampaignReviewReport,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Binary domain separator used for deterministic manifest hashing.
const RESTART_MANIFEST_DOMAIN: &[u8; 16] = b"COGNO-RST-MAN-V1";

/// Maximum number of preferences that may appear in one restart manifest.
pub const MAX_RESTART_MANIFEST_PREFERENCES: usize = 4_096;

/// Explicitly non-authoritative classification of the manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteRestartManifestAuthority {
    ReviewArtifactOnly,
}

/// One preference admitted to controlled-restart review.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TasteRestartPreference {
    pub preference_id: u64,
    pub confidence_bps: u16,
}

/// Immutable, content-addressed restart review artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteRestartManifest {
    pub schema_version: u16,
    pub first_generation: u64,
    pub last_generation: u64,
    pub generations: usize,
    pub campaign_sha256: [u8; 32],
    pub benchmark_sha256: [u8; 32],
    pub longitudinal_sha256: [u8; 32],
    pub preferences_sha256: [u8; 32],
    pub preferences: Vec<TasteRestartPreference>,
    pub manifest_sha256: [u8; 32],
    pub authority: TasteRestartManifestAuthority,
}

/// Invalid input to restart-manifest construction or verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteRestartManifestError {
    ReviewNotEligible,
    InvalidReviewAuthority,
    EmptyPreferences,
    TooManyPreferences,
    DuplicatePreference(u64),
    PreferenceNotEligible(u64),
    InvalidConfidence(u64),
    InvalidGenerationRange,
    ZeroCampaignDigest,
    ZeroBenchmarkDigest,
    ZeroLongitudinalDigest,
    PreferencesDigestMismatch,
    ManifestDigestMismatch,
}

/// Build a deterministic restart-review manifest from an eligible campaign review.
pub fn build_taste_restart_manifest(
    review: &TasteCampaignReviewReport,
    campaign_sha256: [u8; 32],
    benchmark_sha256: [u8; 32],
    longitudinal_sha256: [u8; 32],
) -> Result<TasteRestartManifest, TasteRestartManifestError> {
    validate_review(review)?;
    validate_nonzero_digest(
        campaign_sha256,
        TasteRestartManifestError::ZeroCampaignDigest,
    )?;
    validate_nonzero_digest(
        benchmark_sha256,
        TasteRestartManifestError::ZeroBenchmarkDigest,
    )?;
    validate_nonzero_digest(
        longitudinal_sha256,
        TasteRestartManifestError::ZeroLongitudinalDigest,
    )?;

    let mut preferences = review
        .preferences
        .iter()
        .map(|preference| TasteRestartPreference {
            preference_id: preference.preference_id,
            confidence_bps: preference.latest_confidence_bps,
        })
        .collect::<Vec<_>>();
    preferences.sort_unstable();

    let preferences_sha256 = hash_preferences(&preferences);
    let manifest_sha256 = hash_manifest_fields(
        review.first_generation,
        review.last_generation,
        review.generations,
        campaign_sha256,
        benchmark_sha256,
        longitudinal_sha256,
        preferences_sha256,
        &preferences,
    );

    Ok(TasteRestartManifest {
        schema_version: 1,
        first_generation: review.first_generation,
        last_generation: review.last_generation,
        generations: review.generations,
        campaign_sha256,
        benchmark_sha256,
        longitudinal_sha256,
        preferences_sha256,
        preferences,
        manifest_sha256,
        authority: TasteRestartManifestAuthority::ReviewArtifactOnly,
    })
}

/// Verify all deterministic invariants and content digests of a restart manifest.
pub fn verify_taste_restart_manifest(
    manifest: &TasteRestartManifest,
) -> Result<(), TasteRestartManifestError> {
    if manifest.first_generation == 0
        || manifest.last_generation < manifest.first_generation
        || manifest.generations == 0
    {
        return Err(TasteRestartManifestError::InvalidGenerationRange);
    }
    if manifest.preferences.is_empty() {
        return Err(TasteRestartManifestError::EmptyPreferences);
    }
    if manifest.preferences.len() > MAX_RESTART_MANIFEST_PREFERENCES {
        return Err(TasteRestartManifestError::TooManyPreferences);
    }
    validate_nonzero_digest(
        manifest.campaign_sha256,
        TasteRestartManifestError::ZeroCampaignDigest,
    )?;
    validate_nonzero_digest(
        manifest.benchmark_sha256,
        TasteRestartManifestError::ZeroBenchmarkDigest,
    )?;
    validate_nonzero_digest(
        manifest.longitudinal_sha256,
        TasteRestartManifestError::ZeroLongitudinalDigest,
    )?;

    let mut seen = BTreeSet::new();
    for preference in &manifest.preferences {
        if !seen.insert(preference.preference_id) {
            return Err(TasteRestartManifestError::DuplicatePreference(
                preference.preference_id,
            ));
        }
        if preference.confidence_bps > 10_000 {
            return Err(TasteRestartManifestError::InvalidConfidence(
                preference.preference_id,
            ));
        }
    }

    if hash_preferences(&manifest.preferences) != manifest.preferences_sha256 {
        return Err(TasteRestartManifestError::PreferencesDigestMismatch);
    }

    let expected_manifest_sha256 = hash_manifest_fields(
        manifest.first_generation,
        manifest.last_generation,
        manifest.generations,
        manifest.campaign_sha256,
        manifest.benchmark_sha256,
        manifest.longitudinal_sha256,
        manifest.preferences_sha256,
        &manifest.preferences,
    );
    if expected_manifest_sha256 != manifest.manifest_sha256 {
        return Err(TasteRestartManifestError::ManifestDigestMismatch);
    }

    Ok(())
}

fn validate_review(review: &TasteCampaignReviewReport) -> Result<(), TasteRestartManifestError> {
    if review.authority != TasteCampaignReviewAuthority::ReviewOnly {
        return Err(TasteRestartManifestError::InvalidReviewAuthority);
    }
    if review.disposition != TasteCampaignReviewDisposition::EligibleForControlledRestartReview {
        return Err(TasteRestartManifestError::ReviewNotEligible);
    }
    if review.first_generation == 0
        || review.last_generation < review.first_generation
        || review.generations == 0
    {
        return Err(TasteRestartManifestError::InvalidGenerationRange);
    }
    if review.preferences.is_empty() {
        return Err(TasteRestartManifestError::EmptyPreferences);
    }
    if review.preferences.len() > MAX_RESTART_MANIFEST_PREFERENCES {
        return Err(TasteRestartManifestError::TooManyPreferences);
    }

    let mut seen = BTreeSet::new();
    for preference in &review.preferences {
        if !seen.insert(preference.preference_id) {
            return Err(TasteRestartManifestError::DuplicatePreference(
                preference.preference_id,
            ));
        }
        if !preference.eligible || !preference.blockers.is_empty() {
            return Err(TasteRestartManifestError::PreferenceNotEligible(
                preference.preference_id,
            ));
        }
        if preference.latest_confidence_bps > 10_000 {
            return Err(TasteRestartManifestError::InvalidConfidence(
                preference.preference_id,
            ));
        }
    }
    Ok(())
}

fn validate_nonzero_digest(
    digest: [u8; 32],
    error: TasteRestartManifestError,
) -> Result<(), TasteRestartManifestError> {
    if digest == [0; 32] {
        Err(error)
    } else {
        Ok(())
    }
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

#[allow(clippy::too_many_arguments)]
fn hash_manifest_fields(
    first_generation: u64,
    last_generation: u64,
    generations: usize,
    campaign_sha256: [u8; 32],
    benchmark_sha256: [u8; 32],
    longitudinal_sha256: [u8; 32],
    preferences_sha256: [u8; 32],
    preferences: &[TasteRestartPreference],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RESTART_MANIFEST_DOMAIN);
    hasher.update(1u16.to_le_bytes());
    hasher.update(first_generation.to_le_bytes());
    hasher.update(last_generation.to_le_bytes());
    hasher.update((generations as u64).to_le_bytes());
    hasher.update(campaign_sha256);
    hasher.update(benchmark_sha256);
    hasher.update(longitudinal_sha256);
    hasher.update(preferences_sha256);
    hasher.update((preferences.len() as u64).to_le_bytes());
    for preference in preferences {
        hasher.update(preference.preference_id.to_le_bytes());
        hasher.update(preference.confidence_bps.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taste_campaign_review::{TasteCampaignReviewBlocker, TastePreferenceReview};

    fn eligible_review() -> TasteCampaignReviewReport {
        TasteCampaignReviewReport {
            first_generation: 4,
            last_generation: 7,
            generations: 4,
            benchmark_cases: 8,
            baseline_total_regret: 80,
            taste_total_regret: 20,
            improved_cases: 6,
            regressed_cases: 0,
            quarantined_model_observations: 3,
            preferences: vec![
                TastePreferenceReview {
                    preference_id: 9,
                    latest_confidence_bps: 8_200,
                    blockers: Vec::new(),
                    eligible: true,
                },
                TastePreferenceReview {
                    preference_id: 3,
                    latest_confidence_bps: 7_800,
                    blockers: Vec::new(),
                    eligible: true,
                },
            ],
            disposition: TasteCampaignReviewDisposition::EligibleForControlledRestartReview,
            authority: TasteCampaignReviewAuthority::ReviewOnly,
        }
    }

    #[test]
    fn eligible_review_builds_deterministic_manifest() {
        let first = build_taste_restart_manifest(&eligible_review(), [1; 32], [2; 32], [3; 32])
            .expect("manifest");
        let second = build_taste_restart_manifest(&eligible_review(), [1; 32], [2; 32], [3; 32])
            .expect("manifest");
        assert_eq!(first, second);
        assert_eq!(first.preferences[0].preference_id, 3);
        assert_eq!(first.preferences[1].preference_id, 9);
        assert_eq!(
            first.authority,
            TasteRestartManifestAuthority::ReviewArtifactOnly
        );
        assert_eq!(verify_taste_restart_manifest(&first), Ok(()));
    }

    #[test]
    fn held_review_cannot_create_manifest() {
        let mut review = eligible_review();
        review.disposition = TasteCampaignReviewDisposition::Hold;
        assert_eq!(
            build_taste_restart_manifest(&review, [1; 32], [2; 32], [3; 32]),
            Err(TasteRestartManifestError::ReviewNotEligible)
        );
    }

    #[test]
    fn ineligible_preference_fails_closed() {
        let mut review = eligible_review();
        review.preferences[0].eligible = false;
        review.preferences[0].blockers = vec![TasteCampaignReviewBlocker::Weakening];
        assert_eq!(
            build_taste_restart_manifest(&review, [1; 32], [2; 32], [3; 32]),
            Err(TasteRestartManifestError::PreferenceNotEligible(9))
        );
    }

    #[test]
    fn tampered_manifest_digest_is_rejected() {
        let mut manifest =
            build_taste_restart_manifest(&eligible_review(), [1; 32], [2; 32], [3; 32])
                .expect("manifest");
        manifest.preferences[0].confidence_bps = 9_999;
        assert_eq!(
            verify_taste_restart_manifest(&manifest),
            Err(TasteRestartManifestError::PreferencesDigestMismatch)
        );
    }
}
