//! Native inter-process serialization and crash recovery for persisted model mutation.
//!
//! Rust 1.97.1 exposes `std::fs::File::try_lock`, so COGNO-1 can rely on an
//! OS-managed exclusive lock instead of a create/delete lock file protocol.
//! The lock file itself is persistent; ownership is attached to the open file
//! descriptor/handle and is released by the OS when the holder exits.
//!
//! A narrow crash window remains after `model-generation-N/` is atomically
//! published but before `MODEL_CURRENT` advances. A retry may resume that
//! transaction only when the already-persisted generation manifest, external
//! model manifest, and model artifact exactly match the reviewed candidate.

use crate::model_generation::{
    ModelGenerationManifest, MODEL_GENERATION_MANIFEST_BYTES, MODEL_GENESIS_DIGEST,
};
use crate::model_persistence::{self, ModelGenerationCommit, ModelPersistenceError};
use cogno_core::{ModelFamily, ModelManifest, MANIFEST_SCHEMA_VERSION};
use cogno_model::{load_neural_artifact, EligibleMetaModelReview};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MODEL_COMMIT_LOCK_FILE: &str = ".MODEL-COMMIT.lock";
const MODEL_CURRENT_FILE: &str = "MODEL_CURRENT";
const MODEL_CURRENT_TMP_FILE: &str = ".MODEL_CURRENT.tmp";
const MODEL_ARTIFACT_FILE: &str = "model.bin";
const MODEL_MANIFEST_FILE: &str = "model.manifest";
const GENERATION_MANIFEST_FILE: &str = "generation.manifest";
const MODEL_MANIFEST_BYTES: usize = 95;

/// Commit the exact next reviewed model generation while holding the
/// root-scoped native inter-process mutation lock.
///
/// Contention never waits: it fails closed with `io::ErrorKind::WouldBlock`.
/// The lock remains held across predecessor replay, staging, publication,
/// crash recovery, and the atomic `MODEL_CURRENT` update because the file
/// handle stays in scope for the complete transaction.
pub fn commit_reviewed_model_generation(
    root: impl AsRef<Path>,
    generation: u64,
    review: &EligibleMetaModelReview,
    host: model_persistence::HostModelPromotionAttestation,
) -> Result<ModelGenerationCommit, ModelPersistenceError> {
    let root = root.as_ref();
    std::fs::create_dir_all(root)?;
    let lock = open_lock(root)?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            return Err(ModelPersistenceError::Io(io::Error::new(
                io::ErrorKind::WouldBlock,
                "model persistence interlock is held",
            )));
        }
        Err(TryLockError::Error(error)) => return Err(ModelPersistenceError::Io(error)),
    }

    if let Some(commit) = try_resume_published_generation(root, generation, review)? {
        return Ok(commit);
    }

    model_persistence::commit_reviewed_model_generation(root, generation, review, host)
}

fn try_resume_published_generation(
    root: &Path,
    generation: u64,
    review: &EligibleMetaModelReview,
) -> Result<Option<ModelGenerationCommit>, ModelPersistenceError> {
    let generation_path = model_generation_path(root, generation);
    if !generation_path.is_dir() || current_points_to(root, generation)? {
        return Ok(None);
    }

    let previous_manifest_sha256 = if generation == 1 {
        if root.join(MODEL_CURRENT_FILE).exists() {
            return Ok(None);
        }
        MODEL_GENESIS_DIGEST
    } else {
        let selection = model_persistence::load_persisted_model_generation_selection(root)?;
        let expected_generation = selection
            .selected_generation
            .checked_add(1)
            .ok_or(ModelPersistenceError::ArithmeticOverflow)?;
        if expected_generation != generation {
            return Ok(None);
        }
        selection
            .chain
            .selected()
            .map(ModelGenerationManifest::digest)
            .ok_or(ModelPersistenceError::MalformedCurrent)?
    };

    let artifact = review.artifact();
    let _verified_model = load_neural_artifact(&artifact.manifest, &artifact.bytes)?;
    let model_manifest_bytes = encode_model_manifest(&artifact.manifest)?;
    let model_manifest_sha256: [u8; 32] = Sha256::digest(model_manifest_bytes).into();
    let generation_manifest = ModelGenerationManifest {
        generation,
        previous_manifest_sha256,
        artifact_sha256: artifact.manifest.weights_hash,
        model_manifest_sha256,
        validation_accuracy_bps: review.validation_accuracy_bps(),
        test_accuracy_bps: review.test_accuracy_bps(),
    };

    verify_exact_file(
        &generation_path.join(GENERATION_MANIFEST_FILE),
        &generation_manifest.canonical_bytes(),
        "persisted generation manifest does not match reviewed recovery candidate",
    )?;
    verify_exact_file(
        &generation_path.join(MODEL_MANIFEST_FILE),
        &model_manifest_bytes,
        "persisted model manifest does not match reviewed recovery candidate",
    )?;
    verify_exact_file(
        &generation_path.join(MODEL_ARTIFACT_FILE),
        &artifact.bytes,
        "persisted model artifact does not match reviewed recovery candidate",
    )?;

    advance_current_after_verified_recovery(root, generation)?;
    Ok(Some(ModelGenerationCommit {
        generation,
        generation_path,
        generation_manifest_sha256: generation_manifest.digest(),
        artifact_sha256: generation_manifest.artifact_sha256,
    }))
}

fn current_points_to(root: &Path, generation: u64) -> Result<bool, ModelPersistenceError> {
    match std::fs::read(root.join(MODEL_CURRENT_FILE)) {
        Ok(bytes) => Ok(bytes == format!("{generation}\n").as_bytes()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(ModelPersistenceError::Io(error)),
    }
}

fn verify_exact_file(
    path: &Path,
    expected: &[u8],
    message: &'static str,
) -> Result<(), ModelPersistenceError> {
    let metadata = std::fs::metadata(path)?;
    let expected_len = u64::try_from(expected.len())
        .map_err(|_| ModelPersistenceError::ArithmeticOverflow)?;
    if metadata.len() != expected_len {
        return Err(recovery_mismatch(message));
    }

    let mut observed = Vec::new();
    observed
        .try_reserve_exact(expected.len())
        .map_err(|_| ModelPersistenceError::ArithmeticOverflow)?;
    let read_limit = expected_len
        .checked_add(1)
        .ok_or(ModelPersistenceError::ArithmeticOverflow)?;
    File::open(path)?
        .take(read_limit)
        .read_to_end(&mut observed)?;
    if observed != expected {
        return Err(recovery_mismatch(message));
    }
    Ok(())
}

fn advance_current_after_verified_recovery(
    root: &Path,
    generation: u64,
) -> Result<(), ModelPersistenceError> {
    let temp = root.join(MODEL_CURRENT_TMP_FILE);
    match std::fs::remove_file(&temp) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(ModelPersistenceError::Io(error)),
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    writeln!(file, "{generation}")?;
    file.sync_all()?;
    std::fs::rename(&temp, root.join(MODEL_CURRENT_FILE))?;
    sync_directory(root)?;
    Ok(())
}

fn encode_model_manifest(
    manifest: &ModelManifest,
) -> Result<[u8; MODEL_MANIFEST_BYTES], ModelPersistenceError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION
        || manifest.model_family != ModelFamily::Generic
    {
        return Err(ModelPersistenceError::UnsupportedModelManifest);
    }

    let mut bytes = [0u8; MODEL_MANIFEST_BYTES];
    let mut offset = 0usize;
    put(&mut bytes, &mut offset, &manifest.schema_version.to_le_bytes());
    put(&mut bytes, &mut offset, &[0]);
    put(
        &mut bytes,
        &mut offset,
        &manifest.architecture_id.0.to_le_bytes(),
    );
    put(&mut bytes, &mut offset, &manifest.tensor_count.to_le_bytes());
    put(
        &mut bytes,
        &mut offset,
        &manifest.parameter_count.to_le_bytes(),
    );
    put(
        &mut bytes,
        &mut offset,
        &manifest.max_context_tokens.to_le_bytes(),
    );
    put(&mut bytes, &mut offset, &manifest.tokenizer_hash);
    put(&mut bytes, &mut offset, &manifest.weights_hash);
    put(
        &mut bytes,
        &mut offset,
        &manifest.expected_file_bytes.to_le_bytes(),
    );
    debug_assert_eq!(offset, MODEL_MANIFEST_BYTES);
    Ok(bytes)
}

fn put<const N: usize>(target: &mut [u8; N], offset: &mut usize, value: &[u8]) {
    let end = *offset + value.len();
    target[*offset..end].copy_from_slice(value);
    *offset = end;
}

fn recovery_mismatch(message: &'static str) -> ModelPersistenceError {
    ModelPersistenceError::Io(io::Error::new(io::ErrorKind::InvalidData, message))
}

fn sync_directory(path: &Path) -> Result<(), ModelPersistenceError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn model_generation_path(root: &Path, generation: u64) -> PathBuf {
    root.join(format!("model-generation-{generation}"))
}

fn open_lock(root: &Path) -> Result<File, ModelPersistenceError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(MODEL_COMMIT_LOCK_FILE))
        .map_err(ModelPersistenceError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_handle_is_refused_and_drop_releases_lock() {
        let root =
            std::env::temp_dir().join(format!("cogno-model-native-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");

        let first = open_lock(&root).expect("first lock file");
        first.try_lock().expect("first native lock");
        let second = open_lock(&root).expect("second lock file");
        assert!(matches!(second.try_lock(), Err(TryLockError::WouldBlock)));
        drop(first);
        second.try_lock().expect("lock released with owner handle");

        drop(second);
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn recovery_mismatch_is_fail_closed_invalid_data() {
        let error = recovery_mismatch("mismatch");
        assert!(matches!(
            error,
            ModelPersistenceError::Io(ref inner) if inner.kind() == io::ErrorKind::InvalidData
        ));
    }

    #[test]
    fn generation_manifest_width_remains_canonical() {
        assert_eq!(
            ModelGenerationManifest {
                generation: 1,
                previous_manifest_sha256: MODEL_GENESIS_DIGEST,
                artifact_sha256: [1; 32],
                model_manifest_sha256: [2; 32],
                validation_accuracy_bps: 8_000,
                test_accuracy_bps: 8_000,
            }
            .canonical_bytes()
            .len(),
            MODEL_GENERATION_MANIFEST_BYTES
        );
    }
}
