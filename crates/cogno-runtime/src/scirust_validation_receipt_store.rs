//! Append-only persistence for authenticated SciRust validation receipts.
//!
//! A verified execution attestation proves which deterministic runtime profile
//! produced an exchange, but a replayed attested message must not become a new
//! independent confirmation. This store therefore persists fixed-width hashes
//! of the authenticated message/provenance envelope together with the resulting
//! validation and rejects conflicting reuse of validation, message, or evidence
//! identities across restarts.

use crate::scientific_exchange::ScientificExchangeDisposition;
use crate::scirust_runtime_bridge::{SciRustExchangeBridgeReceipt, SciRustRuntimeKind};
use crate::taste_validation_store::{
    StoredTasteValidation, StoredValidationOrigin, StoredValidationVerdict,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: [u8; 8] = *b"COGSRX01";
const RECORD_HASH_DOMAIN: &[u8] = b"cogno.scirust-validation-receipt.v1\0";
const MESSAGE_ID_HASH_DOMAIN: &[u8] = b"cogno.scirust-message-id.v1\0";
const CONVERSATION_ID_HASH_DOMAIN: &[u8] = b"cogno.scirust-conversation-id.v1\0";
const SENDER_ID_HASH_DOMAIN: &[u8] = b"cogno.scirust-sender-id.v1\0";
const RECORD_BYTES: usize = 228;

/// Result of persistently recording one authenticated SciRust validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SciRustValidationReceiptAppendOutcome {
    Appended,
    Duplicate,
}

/// Fail-closed persistence/replay error for authenticated SciRust receipts.
#[derive(Debug)]
pub enum SciRustValidationReceiptStoreError {
    Io(io::Error),
    InvalidMagic,
    TruncatedRecord,
    InvalidSenderKind(u8),
    InvalidVerdict(u8),
    ConfidenceOutOfRange(u16),
    DigestMismatch,
    NonValidationDisposition,
    UnexpectedValidationOrigin,
    DuplicateValidationConflict(u64),
    DuplicateMessageConflict,
    DuplicateEvidenceConflict(u64),
}

impl From<io::Error> for SciRustValidationReceiptStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PersistedSciRustValidationReceipt {
    validation: StoredTasteValidation,
    sender_kind: SciRustRuntimeKind,
    message_id_sha256: [u8; 32],
    conversation_id_sha256: [u8; 32],
    sender_id_sha256: [u8; 32],
    execution_profile_sha256: [u8; 32],
    payload_sha256: [u8; 32],
}

#[derive(Debug, Default)]
struct ReplayedReceiptIndexes {
    by_validation_id: BTreeMap<u64, PersistedSciRustValidationReceipt>,
    by_message_id: BTreeMap<[u8; 32], u64>,
    by_evidence_id: BTreeMap<u64, u64>,
}

/// Persistent replay firewall for authenticated SciRust deterministic evidence.
#[derive(Debug)]
pub struct PersistentSciRustValidationReceiptStore {
    path: PathBuf,
    by_validation_id: BTreeMap<u64, PersistedSciRustValidationReceipt>,
    by_message_id: BTreeMap<[u8; 32], u64>,
    by_evidence_id: BTreeMap<u64, u64>,
}

impl PersistentSciRustValidationReceiptStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, SciRustValidationReceiptStoreError> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let path = root.join("scirust.validation.receipts");
        if !path.exists() {
            let mut file = File::create(&path)?;
            file.write_all(&MAGIC)?;
            file.sync_data()?;
        }
        let indexes = replay(&path)?;
        Ok(Self {
            path,
            by_validation_id: indexes.by_validation_id,
            by_message_id: indexes.by_message_id,
            by_evidence_id: indexes.by_evidence_id,
        })
    }

    pub fn append(
        &mut self,
        receipt: &SciRustExchangeBridgeReceipt,
    ) -> Result<SciRustValidationReceiptAppendOutcome, SciRustValidationReceiptStoreError> {
        let record = persisted_record(receipt)?;
        validate_persisted_record(record)?;

        if let Some(existing) = self.by_validation_id.get(&record.validation.validation_id) {
            return if *existing == record {
                Ok(SciRustValidationReceiptAppendOutcome::Duplicate)
            } else {
                Err(
                    SciRustValidationReceiptStoreError::DuplicateValidationConflict(
                        record.validation.validation_id,
                    ),
                )
            };
        }
        if self.by_message_id.contains_key(&record.message_id_sha256) {
            return Err(SciRustValidationReceiptStoreError::DuplicateMessageConflict);
        }
        if self
            .by_evidence_id
            .contains_key(&record.validation.evidence_id)
        {
            return Err(
                SciRustValidationReceiptStoreError::DuplicateEvidenceConflict(
                    record.validation.evidence_id,
                ),
            );
        }

        let encoded = encode(record);
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&encoded)?;
        file.sync_data()?;
        self.by_message_id
            .insert(record.message_id_sha256, record.validation.validation_id);
        self.by_evidence_id.insert(
            record.validation.evidence_id,
            record.validation.validation_id,
        );
        self.by_validation_id
            .insert(record.validation.validation_id, record);
        Ok(SciRustValidationReceiptAppendOutcome::Appended)
    }

    /// Persisted unique deterministic validations, ordered by validation id.
    pub fn validations(&self) -> impl Iterator<Item = StoredTasteValidation> + '_ {
        self.by_validation_id
            .values()
            .map(|record| record.validation)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.by_validation_id.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_validation_id.is_empty()
    }
}

fn persisted_record(
    receipt: &SciRustExchangeBridgeReceipt,
) -> Result<PersistedSciRustValidationReceipt, SciRustValidationReceiptStoreError> {
    let validation = match receipt.disposition() {
        ScientificExchangeDisposition::Validation(validation) => validation,
        ScientificExchangeDisposition::QuarantinedModelObservation
        | ScientificExchangeDisposition::RuntimeObservationOnly
        | ScientificExchangeDisposition::Inconclusive => {
            return Err(SciRustValidationReceiptStoreError::NonValidationDisposition);
        }
    };
    if validation.origin != StoredValidationOrigin::DeterministicEvaluation {
        return Err(SciRustValidationReceiptStoreError::UnexpectedValidationOrigin);
    }
    Ok(PersistedSciRustValidationReceipt {
        validation,
        sender_kind: receipt.sender_kind(),
        message_id_sha256: hash_text(MESSAGE_ID_HASH_DOMAIN, receipt.message_id()),
        conversation_id_sha256: hash_text(CONVERSATION_ID_HASH_DOMAIN, receipt.conversation_id()),
        sender_id_sha256: hash_sender(receipt.sender_kind(), receipt.sender_id()),
        execution_profile_sha256: receipt.execution_profile_sha256(),
        payload_sha256: receipt.payload_sha256(),
    })
}

fn validate_persisted_record(
    record: PersistedSciRustValidationReceipt,
) -> Result<(), SciRustValidationReceiptStoreError> {
    if record.validation.origin != StoredValidationOrigin::DeterministicEvaluation {
        return Err(SciRustValidationReceiptStoreError::UnexpectedValidationOrigin);
    }
    if record.validation.confidence_bps > 10_000 {
        return Err(SciRustValidationReceiptStoreError::ConfidenceOutOfRange(
            record.validation.confidence_bps,
        ));
    }
    Ok(())
}

fn encode(record: PersistedSciRustValidationReceipt) -> Vec<u8> {
    let mut output = Vec::with_capacity(RECORD_BYTES);
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&record.validation.validation_id.to_le_bytes());
    output.extend_from_slice(&record.validation.preference_id.to_le_bytes());
    output.extend_from_slice(&record.validation.evidence_id.to_le_bytes());
    output.push(verdict_code(record.validation.verdict));
    output.extend_from_slice(&record.validation.confidence_bps.to_le_bytes());
    output.push(sender_kind_code(record.sender_kind));
    output.extend_from_slice(&record.message_id_sha256);
    output.extend_from_slice(&record.conversation_id_sha256);
    output.extend_from_slice(&record.sender_id_sha256);
    output.extend_from_slice(&record.execution_profile_sha256);
    output.extend_from_slice(&record.payload_sha256);
    let digest = record_digest(&output);
    output.extend_from_slice(&digest);
    debug_assert_eq!(output.len(), RECORD_BYTES);
    output
}

fn decode(
    bytes: &[u8],
) -> Result<PersistedSciRustValidationReceipt, SciRustValidationReceiptStoreError> {
    if bytes.len() != RECORD_BYTES || bytes[..MAGIC.len()] != MAGIC {
        return Err(SciRustValidationReceiptStoreError::InvalidMagic);
    }
    let expected = record_digest(&bytes[..RECORD_BYTES - 32]);
    if bytes[RECORD_BYTES - 32..] != expected {
        return Err(SciRustValidationReceiptStoreError::DigestMismatch);
    }
    let verdict = match bytes[32] {
        1 => StoredValidationVerdict::Confirmed,
        2 => StoredValidationVerdict::Contradicted,
        value => return Err(SciRustValidationReceiptStoreError::InvalidVerdict(value)),
    };
    let sender_kind = match bytes[35] {
        1 => SciRustRuntimeKind::RuntimeDiscovery,
        2 => SciRustRuntimeKind::DeterministicKernel,
        value => return Err(SciRustValidationReceiptStoreError::InvalidSenderKind(value)),
    };
    let record = PersistedSciRustValidationReceipt {
        validation: StoredTasteValidation {
            validation_id: u64::from_le_bytes(bytes[8..16].try_into().expect("validation id")),
            preference_id: u64::from_le_bytes(bytes[16..24].try_into().expect("preference id")),
            evidence_id: u64::from_le_bytes(bytes[24..32].try_into().expect("evidence id")),
            origin: StoredValidationOrigin::DeterministicEvaluation,
            verdict,
            confidence_bps: u16::from_le_bytes(bytes[33..35].try_into().expect("confidence")),
        },
        sender_kind,
        message_id_sha256: bytes[36..68].try_into().expect("message digest"),
        conversation_id_sha256: bytes[68..100].try_into().expect("conversation digest"),
        sender_id_sha256: bytes[100..132].try_into().expect("sender digest"),
        execution_profile_sha256: bytes[132..164].try_into().expect("profile digest"),
        payload_sha256: bytes[164..196].try_into().expect("payload digest"),
    };
    validate_persisted_record(record)?;
    Ok(record)
}

fn replay(path: &Path) -> Result<ReplayedReceiptIndexes, SciRustValidationReceiptStoreError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < MAGIC.len() || bytes[..MAGIC.len()] != MAGIC {
        return Err(SciRustValidationReceiptStoreError::InvalidMagic);
    }

    let mut indexes = ReplayedReceiptIndexes::default();
    let mut offset = MAGIC.len();
    while offset < bytes.len() {
        if bytes.len() - offset < RECORD_BYTES {
            return Err(SciRustValidationReceiptStoreError::TruncatedRecord);
        }
        let record = decode(&bytes[offset..offset + RECORD_BYTES])?;
        if let Some(existing) = indexes
            .by_validation_id
            .insert(record.validation.validation_id, record)
            && existing != record
        {
            return Err(
                SciRustValidationReceiptStoreError::DuplicateValidationConflict(
                    record.validation.validation_id,
                ),
            );
        }
        if let Some(existing_validation_id) = indexes
            .by_message_id
            .insert(record.message_id_sha256, record.validation.validation_id)
            && existing_validation_id != record.validation.validation_id
        {
            return Err(SciRustValidationReceiptStoreError::DuplicateMessageConflict);
        }
        if let Some(existing_validation_id) = indexes.by_evidence_id.insert(
            record.validation.evidence_id,
            record.validation.validation_id,
        ) && existing_validation_id != record.validation.validation_id
        {
            return Err(
                SciRustValidationReceiptStoreError::DuplicateEvidenceConflict(
                    record.validation.evidence_id,
                ),
            );
        }
        offset += RECORD_BYTES;
    }
    Ok(indexes)
}

fn record_digest(encoded_without_digest: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RECORD_HASH_DOMAIN);
    hasher.update(encoded_without_digest);
    hasher.finalize().into()
}

fn hash_text(domain: &[u8], value: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
    hasher.finalize().into()
}

fn hash_sender(kind: SciRustRuntimeKind, id: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SENDER_ID_HASH_DOMAIN);
    hasher.update([sender_kind_code(kind)]);
    hasher.update((id.len() as u64).to_le_bytes());
    hasher.update(id.as_bytes());
    hasher.finalize().into()
}

const fn sender_kind_code(kind: SciRustRuntimeKind) -> u8 {
    match kind {
        SciRustRuntimeKind::RuntimeDiscovery => 1,
        SciRustRuntimeKind::DeterministicKernel => 2,
    }
}

const fn verdict_code(verdict: StoredValidationVerdict) -> u8 {
    match verdict {
        StoredValidationVerdict::Confirmed => 1,
        StoredValidationVerdict::Contradicted => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bridge_authenticated_scirust_exchange, HostAuthenticatedSciRustSender};
    use serde_json::{json, Value};
    use std::time::{SystemTime, UNIX_EPOCH};

    const PROFILE_SHA256: &str = "c984c0151e84300875c2aead5764d018f9ef5d09d218ab8f8f1ea9ab7157bec8";

    fn root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-scirust-receipts-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn message(
        message_id: &str,
        observation_id: u64,
        evidence_id: u64,
        confidence_bps: u16,
    ) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "message_id": message_id,
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
                "observation_id": observation_id,
                "preference_id": 42,
                "evidence_id": evidence_id,
                "verdict": "confirmed",
                "confidence_bps": confidence_bps,
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

    fn receipt(message: &[u8]) -> SciRustExchangeBridgeReceipt {
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("sender");
        bridge_authenticated_scirust_exchange(message, &sender, "cogno").expect("bridge")
    }

    #[test]
    fn append_reopen_and_exact_replay_is_idempotent() {
        let root = root("replay");
        let receipt = receipt(&message("m-1", 17, 117, 8_500));
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        assert_eq!(
            store.append(&receipt).expect("append"),
            SciRustValidationReceiptAppendOutcome::Appended
        );
        drop(store);

        let mut reopened = PersistentSciRustValidationReceiptStore::open(&root).expect("reopen");
        assert_eq!(reopened.len(), 1);
        assert_eq!(
            reopened.append(&receipt).expect("duplicate"),
            SciRustValidationReceiptAppendOutcome::Duplicate
        );
        assert_eq!(reopened.validations().collect::<Vec<_>>().len(), 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn reused_validation_id_with_changed_provenance_fails_closed() {
        let root = root("validation-conflict");
        let first = receipt(&message("m-1", 17, 117, 8_500));
        let changed = receipt(&message("m-2", 17, 118, 9_000));
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        store.append(&first).expect("first");
        assert!(matches!(
            store.append(&changed),
            Err(SciRustValidationReceiptStoreError::DuplicateValidationConflict(17))
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn reused_message_id_for_new_validation_fails_closed() {
        let root = root("message-conflict");
        let first = receipt(&message("m-1", 17, 117, 8_500));
        let replay = receipt(&message("m-1", 18, 118, 8_500));
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        store.append(&first).expect("first");
        assert!(matches!(
            store.append(&replay),
            Err(SciRustValidationReceiptStoreError::DuplicateMessageConflict)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn reused_evidence_id_for_new_validation_fails_closed() {
        let root = root("evidence-conflict");
        let first = receipt(&message("m-1", 17, 117, 8_500));
        let replay = receipt(&message("m-2", 18, 117, 8_500));
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        store.append(&first).expect("first");
        assert!(matches!(
            store.append(&replay),
            Err(SciRustValidationReceiptStoreError::DuplicateEvidenceConflict(117))
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn corrupted_persisted_receipt_is_rejected_on_reopen() {
        let root = root("corrupt");
        let receipt = receipt(&message("m-1", 17, 117, 8_500));
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        store.append(&receipt).expect("append");
        drop(store);

        let path = root.join("scirust.validation.receipts");
        let mut bytes = fs::read(&path).expect("read");
        bytes[20] ^= 0x01;
        fs::write(&path, bytes).expect("corrupt");
        assert!(matches!(
            PersistentSciRustValidationReceiptStore::open(&root),
            Err(SciRustValidationReceiptStoreError::DigestMismatch)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn non_validation_receipt_is_not_persisted_as_promotion_evidence() {
        let mut value: Value =
            serde_json::from_slice(&message("m-1", 17, 117, 8_500)).expect("fixture");
        value["payload"]["verdict"] = json!("inconclusive");
        let encoded = serde_json::to_vec(&value).expect("encoded");
        let receipt = receipt(&encoded);
        let root = root("inconclusive");
        let mut store = PersistentSciRustValidationReceiptStore::open(&root).expect("store");
        assert!(matches!(
            store.append(&receipt),
            Err(SciRustValidationReceiptStoreError::NonValidationDisposition)
        ));
        assert!(store.is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
