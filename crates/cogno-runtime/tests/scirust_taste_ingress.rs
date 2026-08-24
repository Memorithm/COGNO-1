use cogno_runtime::{
    bridge_authenticated_scirust_exchange, HostAuthenticatedSciRustSender,
    PersistentSciRustTasteIngress, PersistentSciRustValidationReceiptStore,
    PersistentTasteValidationStore, SciRustTasteIngressError, StoredTasteValidation,
    StoredValidationOrigin, StoredValidationVerdict,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE_SHA256: &str = "c984c0151e84300875c2aead5764d018f9ef5d09d218ab8f8f1ea9ab7157bec8";

fn root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-scirust-ingress-public-{}-{nonce}",
        std::process::id()
    ))
}

fn scirust_message() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": 1,
        "message_id": "m-1",
        "conversation_id": "c-1",
        "parent_message_id": "request-1",
        "sender": {"kind": "deterministic_kernel", "id": "cuda-decode-local"},
        "recipients": [{"kind": "cogno_communicator", "id": "cogno"}],
        "message_kind": "experiment_result",
        "trust_class": "deterministic_kernel_output",
        "confidence_bps": 10_000,
        "evidence": [],
        "payload": {
            "schema_version": 1,
            "observation_id": 17,
            "preference_id": 42,
            "evidence_id": 117,
            "verdict": "confirmed",
            "confidence_bps": 8_500,
            "canonical_payload": [100, 101, 116, 101, 114, 109, 105, 110, 105, 115, 116, 105, 99]
        },
        "requested_capabilities": [],
        "execution_attestation": {
            "profile": {
                "schema_version": 1,
                "backend": "cuda",
                "device_ordinal": 0,
                "architecture": {"family": "nvidia_gpu", "name": "sm_110"},
                "capability_profile_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                "topology_profile_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                "memory_budget_bytes": 1024,
                "numeric_mode": "bf16-fp32-accum-v1",
                "reproducibility": "numerically_equivalent",
                "kernel_semantic_version": "cuda-decode-v1",
                "sampler_semantic_version": "greedy-v1",
                "model_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
                "tokenizer_sha256": "4444444444444444444444444444444444444444444444444444444444444444"
            },
            "profile_sha256": PROFILE_SHA256
        }
    }))
    .expect("message fixture")
}

#[test]
fn reopen_rejects_cross_origin_duplicate_evidence() {
    let root = root();
    let mut validations = PersistentTasteValidationStore::open(&root).expect("validations");
    validations
        .append(StoredTasteValidation {
            validation_id: 90,
            preference_id: 42,
            evidence_id: 117,
            origin: StoredValidationOrigin::ExplicitUserAction,
            verdict: StoredValidationVerdict::Confirmed,
            confidence_bps: 9_000,
            subject_kind: cogno_runtime::ValidationSubjectKind::Unattributed,
            subject_id_sha256: [0u8; 32],
        })
        .expect("user validation");
    drop(validations);

    let sender =
        HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local").expect("sender");
    let receipt = bridge_authenticated_scirust_exchange(&scirust_message(), &sender, "cogno")
        .expect("bridge");
    let mut receipts = PersistentSciRustValidationReceiptStore::open(&root).expect("receipts");
    receipts.append(&receipt).expect("receipt");
    drop(receipts);

    assert!(matches!(
        PersistentSciRustTasteIngress::open(&root),
        Err(SciRustTasteIngressError::DuplicateTrustedEvidence(117))
    ));
    fs::remove_dir_all(root).expect("cleanup");
}
