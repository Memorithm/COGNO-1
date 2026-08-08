//! Filesystem object-type validation for persisted neural model state.
//!
//! The model persistence format is path-based, but a pathname is not itself
//! authority. Before the public commit or replay paths open namespace objects,
//! this module inspects them with `symlink_metadata` and rejects symbolic links
//! and unexpected object kinds fail-closed.
//!
//! This is deliberately a preflight hardening layer, not a claim of complete
//! TOCTOU resistance: safe `std` does not currently provide a portable
//! directory-handle-relative `openat2`-style API. A concurrent hostile actor
//! with write access to the persistence root must therefore remain outside the
//! trust model; cooperative COGNO writers are serialized by the native model
//! commit interlock.

use crate::model_persistence::{self, ModelPersistenceError, PersistedModelGenerationSelection};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

const MODEL_COMMIT_LOCK_FILE: &str = ".MODEL-COMMIT.lock";
const MODEL_CURRENT_FILE: &str = "MODEL_CURRENT";
const MODEL_CURRENT_TMP_FILE: &str = ".MODEL_CURRENT.tmp";
const MODEL_ARTIFACT_FILE: &str = "model.bin";
const MODEL_MANIFEST_FILE: &str = "model.manifest";
const GENERATION_MANIFEST_FILE: &str = "generation.manifest";
const MODEL_GENERATION_PREFIX: &str = "model-generation-";
const MODEL_STAGE_PREFIX: &str = ".model-stage-";
const MAX_PERSISTED_MODEL_GENERATIONS: u64 = 4_096;

/// Replay persisted model generations only after the complete selected
/// namespace has passed object-type and symlink preflight validation.
pub fn load_persisted_model_generation_selection(
    root: impl AsRef<Path>,
) -> Result<PersistedModelGenerationSelection, ModelPersistenceError> {
    let root = root.as_ref();
    validate_model_persistence_tree(root)?;
    model_persistence::load_persisted_model_generation_selection(root)
}

/// Create a missing persistence root, then require the resulting path itself
/// to be a real directory rather than a symlink or another filesystem object.
pub(crate) fn prepare_model_persistence_root(root: &Path) -> Result<(), ModelPersistenceError> {
    match fs::symlink_metadata(root) {
        Ok(metadata) => require_directory_metadata(&metadata, "model persistence root"),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            fs::create_dir_all(root)?;
            let metadata = fs::symlink_metadata(root)?;
            require_directory_metadata(&metadata, "model persistence root")
        }
        Err(error) => Err(ModelPersistenceError::Io(error)),
    }
}

/// Validate every reserved object currently present in the model persistence
/// root. Unknown unrelated entries are ignored; malformed names inside the
/// reserved generation/staging namespaces are rejected.
pub(crate) fn validate_model_persistence_tree(
    root: &Path,
) -> Result<(), ModelPersistenceError> {
    let root_metadata = fs::symlink_metadata(root)?;
    require_directory_metadata(&root_metadata, "model persistence root")?;

    validate_regular_file_if_present(&root.join(MODEL_COMMIT_LOCK_FILE), "model commit lock")?;
    validate_regular_file_if_present(&root.join(MODEL_CURRENT_FILE), "MODEL_CURRENT")?;
    validate_regular_file_if_present(
        &root.join(MODEL_CURRENT_TMP_FILE),
        ".MODEL_CURRENT.tmp",
    )?;

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(suffix) = name.strip_prefix(MODEL_GENERATION_PREFIX) {
            validate_reserved_generation_name(suffix, "model generation")?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            require_directory_metadata(&metadata, "model generation directory")?;
            validate_regular_file(
                &path.join(MODEL_ARTIFACT_FILE),
                "persisted model artifact",
            )?;
            validate_regular_file(
                &path.join(MODEL_MANIFEST_FILE),
                "persisted model manifest",
            )?;
            validate_regular_file(
                &path.join(GENERATION_MANIFEST_FILE),
                "persisted generation manifest",
            )?;
        } else if let Some(suffix) = name.strip_prefix(MODEL_STAGE_PREFIX) {
            validate_reserved_generation_name(suffix, "model staging generation")?;
            let metadata = fs::symlink_metadata(entry.path())?;
            require_directory_metadata(&metadata, "model staging directory")?;
        }
    }
    Ok(())
}

/// Validate the persistent lock rendezvous object before opening it. The file
/// may be absent on first use, but if it exists it must be a real regular file.
pub(crate) fn validate_model_commit_lock_path(root: &Path) -> Result<(), ModelPersistenceError> {
    validate_regular_file_if_present(&root.join(MODEL_COMMIT_LOCK_FILE), "model commit lock")
}

fn validate_reserved_generation_name(
    suffix: &str,
    label: &'static str,
) -> Result<u64, ModelPersistenceError> {
    if suffix.is_empty()
        || suffix.starts_with('0')
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid_filesystem_object(label));
    }
    let generation = suffix
        .parse::<u64>()
        .map_err(|_| invalid_filesystem_object(label))?;
    if generation == 0 || generation > MAX_PERSISTED_MODEL_GENERATIONS {
        return Err(invalid_filesystem_object(label));
    }
    Ok(generation)
}

fn validate_regular_file_if_present(
    path: &Path,
    label: &'static str,
) -> Result<(), ModelPersistenceError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => require_regular_file_metadata(&metadata, label),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ModelPersistenceError::Io(error)),
    }
}

fn validate_regular_file(path: &Path, label: &'static str) -> Result<(), ModelPersistenceError> {
    let metadata = fs::symlink_metadata(path)?;
    require_regular_file_metadata(&metadata, label)
}

fn require_regular_file_metadata(
    metadata: &fs::Metadata,
    label: &'static str,
) -> Result<(), ModelPersistenceError> {
    let file_type = metadata.file_type();
    if file_type.is_symlink() || !file_type.is_file() {
        return Err(invalid_filesystem_object(label));
    }
    Ok(())
}

fn require_directory_metadata(
    metadata: &fs::Metadata,
    label: &'static str,
) -> Result<(), ModelPersistenceError> {
    let file_type = metadata.file_type();
    if file_type.is_symlink() || !file_type.is_dir() {
        return Err(invalid_filesystem_object(label));
    }
    Ok(())
}

fn invalid_filesystem_object(label: &'static str) -> ModelPersistenceError {
    ModelPersistenceError::Io(io::Error::new(
        ErrorKind::InvalidData,
        format!("unexpected filesystem object in model persistence namespace: {label}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_reserved_generation_names_fail_closed() {
        for suffix in ["", "0", "01", "abc", "4097"] {
            assert!(validate_reserved_generation_name(suffix, "generation").is_err());
        }
        assert_eq!(
            validate_reserved_generation_name("4096", "generation").expect("upper bound"),
            4_096
        );
    }
}
