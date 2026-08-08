//! Native inter-process serialization for persisted model mutation.
//!
//! Rust 1.97.1 exposes `std::fs::File::try_lock`, so COGNO-1 can rely on an
//! OS-managed exclusive lock instead of a create/delete lock file protocol.
//! The lock file itself is persistent; ownership is attached to the open file
//! descriptor/handle and is released by the OS when the holder exits.

use crate::model_persistence::{self, ModelGenerationCommit, ModelPersistenceError};
use cogno_model::EligibleMetaModelReview;
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::Path;

const MODEL_COMMIT_LOCK_FILE: &str = ".MODEL-COMMIT.lock";

/// Commit the exact next reviewed model generation while holding the
/// root-scoped native inter-process mutation lock.
///
/// Contention never waits: it fails closed with `io::ErrorKind::WouldBlock`.
/// The lock remains held across predecessor replay, staging, publication and
/// the atomic `MODEL_CURRENT` update because the file handle stays in scope for
/// the complete inner transaction.
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
    model_persistence::commit_reviewed_model_generation(root, generation, review, host)
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
        let root = std::env::temp_dir().join(format!(
            "cogno-model-native-lock-{}",
            std::process::id()
        ));
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
}
