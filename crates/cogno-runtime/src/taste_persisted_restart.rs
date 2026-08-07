//! End-to-end persisted controlled-restart sealing for scientific taste.
//!
//! This is the nominal restart entry point: replay the persisted generation
//! chain selected by `CURRENT`, load the verified profile only from that
//! selected generation directory, then bind it to the reviewed restart
//! manifest and selected hash-linked generation.

use crate::taste_controlled_restart::{
    GenerationBoundControlledRestartTasteError, GenerationBoundControlledRestartTasteProfile,
};
use crate::taste_persisted_chain::{
    load_persisted_taste_generation_selection, PersistedTasteGenerationError,
};
use crate::taste_restart_manifest::TasteRestartManifest;
use crate::verified_taste_profile::{VerifiedTasteProfile, VerifiedTasteProfileError};
use std::path::Path;

/// Failure while producing a runtime-installable seal from persisted state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistedControlledRestartTasteError {
    /// `CURRENT`, a generation manifest, the hash chain, or a persisted artifact failed verification.
    PersistedGeneration(PersistedTasteGenerationError),
    /// The selected generation's profile/replay source set failed semantic verification.
    VerifiedProfile(VerifiedTasteProfileError),
    /// The reviewed restart manifest or selected-generation binding failed.
    GenerationBound(GenerationBoundControlledRestartTasteError),
}

/// Verify persisted state end-to-end and seal the selected generation for runtime restart.
pub fn prepare_persisted_controlled_restart_taste_profile(
    root: impl AsRef<Path>,
    restart_manifest: &TasteRestartManifest,
) -> Result<GenerationBoundControlledRestartTasteProfile, PersistedControlledRestartTasteError> {
    let selection = load_persisted_taste_generation_selection(root.as_ref())
        .map_err(PersistedControlledRestartTasteError::PersistedGeneration)?;
    let generation_path = &selection.selected_generation_path;
    let profile = VerifiedTasteProfile::load(
        generation_path.join("replay.json"),
        generation_path.join("candidates.json"),
        generation_path,
    )
    .map_err(PersistedControlledRestartTasteError::VerifiedProfile)?;

    GenerationBoundControlledRestartTasteProfile::prepare(
        restart_manifest,
        &selection.chain,
        profile,
    )
    .map_err(PersistedControlledRestartTasteError::GenerationBound)
}
