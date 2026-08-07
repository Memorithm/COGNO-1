//! Controlled-restart gate for scientific taste state.
//!
//! This module is the authority boundary between a non-authoritative restart
//! review manifest and a runtime-installable verified taste profile. It never
//! mutates a live [`crate::Runtime`]. Instead it seals a profile for one-time
//! installation during runtime initialization after proving exact agreement
//! between the reviewed manifest and the independently verified profile.

use crate::taste_generation::TasteGenerationManifest;
use crate::taste_restart_manifest::{
    verify_taste_restart_manifest, TasteRestartManifest, TasteRestartManifestError,
};
use crate::verified_taste_profile::{VerifiedTastePreference, VerifiedTasteProfile};

/// Authority carried by a profile that passed the controlled-restart gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlledRestartTasteAuthority {
    /// The sealed profile may only be consumed during runtime initialization.
    RestartInitializationOnly,
}

/// A verified taste profile sealed for controlled-restart installation.
#[derive(Debug)]
pub struct ControlledRestartTasteProfile {
    profile: VerifiedTasteProfile,
    manifest_sha256: [u8; 32],
    first_generation: u64,
    last_generation: u64,
    generations: usize,
    authority: ControlledRestartTasteAuthority,
}

/// A controlled-restart seal additionally bound to one immutable selected generation.
#[derive(Debug)]
pub struct GenerationBoundControlledRestartTasteProfile {
    sealed: ControlledRestartTasteProfile,
    generation: u64,
    generation_manifest_sha256: [u8; 32],
}

impl GenerationBoundControlledRestartTasteProfile {
    /// Bind an already verified profile to both the restart review and the selected
    /// immutable generation manifest. All persisted profile-source digests must
    /// agree exactly with the generation manifest.
    pub fn prepare(
        manifest: &TasteRestartManifest,
        generation: &TasteGenerationManifest,
        profile: VerifiedTasteProfile,
    ) -> Result<Self, GenerationBoundControlledRestartTasteError> {
        verify_generation_matches_profile(generation, &profile)?;
        let sealed = ControlledRestartTasteProfile::prepare(manifest, profile)
            .map_err(GenerationBoundControlledRestartTasteError::ControlledRestart)?;
        Ok(Self {
            sealed,
            generation: generation.generation,
            generation_manifest_sha256: generation.sha256(),
        })
    }

    /// Persisted generation selected for this controlled restart.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// SHA-256 of the immutable generation manifest bound to this restart.
    #[must_use]
    pub const fn generation_manifest_sha256(&self) -> [u8; 32] {
        self.generation_manifest_sha256
    }

    /// Reviewed restart manifest digest.
    #[must_use]
    pub const fn restart_manifest_sha256(&self) -> [u8; 32] {
        self.sealed.manifest_sha256()
    }

    /// Read-only active preferences carried by the verified profile.
    #[must_use]
    pub fn active_preferences(&self) -> &[VerifiedTastePreference] {
        self.sealed.active_preferences()
    }

    /// Consume the generation-bound wrapper and return the controlled-restart seal.
    #[must_use]
    pub fn into_controlled_restart_profile(self) -> ControlledRestartTasteProfile {
        self.sealed
    }
}

/// Failure while binding a controlled restart to a selected immutable generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationBoundControlledRestartTasteError {
    /// The generation manifest references another profile.
    ProfileDigestMismatch,
    /// The generation manifest references another replay artifact.
    ReplayDigestMismatch,
    /// The generation manifest references another candidate report.
    CandidateReportDigestMismatch,
    /// The generation manifest references another validation journal.
    ValidationStoreDigestMismatch,
    /// The ordinary controlled-restart checks failed.
    ControlledRestart(ControlledRestartTasteError),
}

impl ControlledRestartTasteProfile {
    /// Verify exact manifest/profile agreement and seal the profile for restart.
    pub fn prepare(
        manifest: &TasteRestartManifest,
        profile: VerifiedTasteProfile,
    ) -> Result<Self, ControlledRestartTasteError> {
        verify_taste_restart_manifest(manifest)
            .map_err(ControlledRestartTasteError::InvalidManifest)?;
        verify_canonical_manifest_order(manifest)?;
        verify_profile_matches_manifest(manifest, profile.active_preferences())?;

        Ok(Self {
            profile,
            manifest_sha256: manifest.manifest_sha256,
            first_generation: manifest.first_generation,
            last_generation: manifest.last_generation,
            generations: manifest.generations,
            authority: ControlledRestartTasteAuthority::RestartInitializationOnly,
        })
    }

    /// Digest of the reviewed restart manifest that authorized this seal.
    #[must_use]
    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    /// First campaign generation covered by the reviewed manifest.
    #[must_use]
    pub const fn first_generation(&self) -> u64 {
        self.first_generation
    }

    /// Last campaign generation covered by the reviewed manifest.
    #[must_use]
    pub const fn last_generation(&self) -> u64 {
        self.last_generation
    }

    /// Number of campaign generations covered by the reviewed manifest.
    #[must_use]
    pub const fn generations(&self) -> usize {
        self.generations
    }

    /// Explicit authority of this sealed artifact.
    #[must_use]
    pub const fn authority(&self) -> ControlledRestartTasteAuthority {
        self.authority
    }

    /// Read-only active preferences carried by the verified profile.
    #[must_use]
    pub fn active_preferences(&self) -> &[VerifiedTastePreference] {
        self.profile.active_preferences()
    }

    /// Consume the restart seal and return the already verified profile.
    ///
    /// This is intentionally consuming so the same seal cannot be installed
    /// twice by safe Rust code.
    #[must_use]
    pub fn into_verified_profile(self) -> VerifiedTasteProfile {
        self.profile
    }
}

/// Failure while sealing a verified profile for controlled restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlledRestartTasteError {
    /// The review manifest failed its own deterministic verification.
    InvalidManifest(TasteRestartManifestError),
    /// Manifest preferences are not in the canonical ascending identifier order.
    NonCanonicalPreferenceOrder,
    /// The verified profile and reviewed manifest contain different counts.
    PreferenceCountMismatch,
    /// A reviewed preference is absent from the verified active profile.
    MissingVerifiedPreference(u64),
    /// A verified active preference was not reviewed by the manifest.
    UnexpectedVerifiedPreference(u64),
    /// Confidence differs between reviewed and verified state.
    ConfidenceMismatch(u64),
}

fn verify_generation_matches_profile(
    generation: &TasteGenerationManifest,
    profile: &VerifiedTasteProfile,
) -> Result<(), GenerationBoundControlledRestartTasteError> {
    if generation.profile_sha256 != profile.profile_sha256() {
        return Err(GenerationBoundControlledRestartTasteError::ProfileDigestMismatch);
    }
    if generation.replay_sha256 != profile.replay_sha256() {
        return Err(GenerationBoundControlledRestartTasteError::ReplayDigestMismatch);
    }
    if generation.candidate_report_sha256 != profile.candidate_report_sha256() {
        return Err(GenerationBoundControlledRestartTasteError::CandidateReportDigestMismatch);
    }
    if generation.validation_store_sha256 != profile.validation_store_sha256() {
        return Err(GenerationBoundControlledRestartTasteError::ValidationStoreDigestMismatch);
    }
    Ok(())
}

fn verify_canonical_manifest_order(
    manifest: &TasteRestartManifest,
) -> Result<(), ControlledRestartTasteError> {
    if manifest
        .preferences
        .windows(2)
        .any(|pair| pair[0].preference_id >= pair[1].preference_id)
    {
        return Err(ControlledRestartTasteError::NonCanonicalPreferenceOrder);
    }
    Ok(())
}

fn verify_profile_matches_manifest(
    manifest: &TasteRestartManifest,
    active_preferences: &[VerifiedTastePreference],
) -> Result<(), ControlledRestartTasteError> {
    if manifest.preferences.len() != active_preferences.len() {
        return Err(ControlledRestartTasteError::PreferenceCountMismatch);
    }

    for reviewed in &manifest.preferences {
        let Some(verified) = active_preferences
            .iter()
            .find(|verified| verified.preference_id == reviewed.preference_id)
        else {
            return Err(ControlledRestartTasteError::MissingVerifiedPreference(
                reviewed.preference_id,
            ));
        };
        if verified.confidence_bps != reviewed.confidence_bps {
            return Err(ControlledRestartTasteError::ConfidenceMismatch(
                reviewed.preference_id,
            ));
        }
    }

    for verified in active_preferences {
        if !manifest
            .preferences
            .iter()
            .any(|reviewed| reviewed.preference_id == verified.preference_id)
        {
            return Err(ControlledRestartTasteError::UnexpectedVerifiedPreference(
                verified.preference_id,
            ));
        }
    }

    Ok(())
}
