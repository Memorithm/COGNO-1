//! Persisted restart-chain replay and tamper coverage.

use cogno_runtime::{
    commit_taste_cycle, load_persisted_taste_generation_selection, PersistedTasteGenerationError,
    TasteCycleArtifacts, TasteGenerationManifest, GENESIS_DIGEST,
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
        "cogno-persisted-chain-{}-{nonce}",
        std::process::id()
    ))
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn artifacts(generation: u64) -> TasteCycleArtifacts {
    TasteCycleArtifacts {
        candidate_report: format!("candidates-{generation}").into_bytes(),
        validation_store: format!("validations-{generation}").into_bytes(),
        replay_report: format!("replay-{generation}").into_bytes(),
        profile: format!("profile-{generation}").into_bytes(),
    }
}

fn manifest(
    generation: u64,
    previous_manifest_sha256: [u8; 32],
    artifacts: &TasteCycleArtifacts,
) -> TasteGenerationManifest {
    TasteGenerationManifest {
        generation,
        previous_manifest_sha256,
        profile_sha256: digest(&artifacts.profile),
        replay_sha256: digest(&artifacts.replay_report),
        candidate_report_sha256: digest(&artifacts.candidate_report),
        validation_store_sha256: digest(&artifacts.validation_store),
    }
}

fn commit_two(root: &PathBuf) -> (TasteGenerationManifest, TasteGenerationManifest) {
    let first_artifacts = artifacts(1);
    let first = manifest(1, GENESIS_DIGEST, &first_artifacts);
    commit_taste_cycle(root, &first, &first_artifacts).expect("generation 1");

    let second_artifacts = artifacts(2);
    let second = manifest(2, first.sha256(), &second_artifacts);
    commit_taste_cycle(root, &second, &second_artifacts).expect("generation 2");
    (first, second)
}

#[test]
fn current_replays_complete_hash_linked_generation_chain() {
    let root = root();
    let (_first, second) = commit_two(&root);

    let selection = load_persisted_taste_generation_selection(&root).expect("selection");
    assert_eq!(selection.selected_generation, 2);
    assert_eq!(selection.chain.len(), 2);
    assert_eq!(selection.chain.selected(), Some(&second));
    assert_eq!(selection.selected_generation_path, root.join("generation-2"));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn artifact_tamper_after_commit_fails_closed() {
    let root = root();
    commit_two(&root);
    fs::write(root.join("generation-2/taste.profile"), b"tampered-profile").expect("tamper");

    assert_eq!(
        load_persisted_taste_generation_selection(&root),
        Err(PersistedTasteGenerationError::ArtifactDigestMismatch {
            generation: 2,
            artifact: "profile",
        })
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn missing_intermediate_generation_fails_closed() {
    let root = root();
    commit_two(&root);
    fs::remove_dir_all(root.join("generation-1")).expect("remove generation");

    assert_eq!(
        load_persisted_taste_generation_selection(&root),
        Err(PersistedTasteGenerationError::MissingGeneration(1))
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn malformed_current_pointer_fails_closed() {
    let root = root();
    commit_two(&root);
    fs::write(root.join("CURRENT"), b"02\n").expect("current tamper");

    assert_eq!(
        load_persisted_taste_generation_selection(&root),
        Err(PersistedTasteGenerationError::InvalidCurrent)
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn tampered_previous_manifest_link_fails_closed() {
    let root = root();
    let (_first, mut second) = commit_two(&root);
    second.previous_manifest_sha256 = [9; 32];
    fs::write(
        root.join("generation-2/generation.manifest"),
        second.canonical_bytes(),
    )
    .expect("manifest tamper");

    assert!(matches!(
        load_persisted_taste_generation_selection(&root),
        Err(PersistedTasteGenerationError::InvalidGenerationChain(_))
    ));

    fs::remove_dir_all(root).expect("cleanup");
}
