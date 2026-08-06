//! Deterministic snapshots derived from the persistent dialogue event log.
//!
//! A snapshot is a non-authoritative summary. It counts observed senders and
//! message kinds, records payload volume, and commits to the complete event log
//! with SHA-256. It never promotes an observation into policy or taste.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const STORE_MAGIC: [u8; 8] = *b"COGDLG01";
const SNAPSHOT_MAGIC: [u8; 8] = *b"COGSNP01";
const RECORD_HEADER_BYTES: usize = 54;
const MAX_PAYLOAD_BYTES: usize = 16 * 1024;
const SENDER_CLASSES: usize = 5;
const MESSAGE_KINDS: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialogueSnapshot {
    pub schema_version: u16,
    pub events: u64,
    pub payload_bytes: u64,
    pub sender_counts: [u64; SENDER_CLASSES],
    pub message_kind_counts: [u64; MESSAGE_KINDS],
    pub event_log_sha256: [u8; 32],
}

#[derive(Debug)]
pub enum DialogueSnapshotError {
    Io(io::Error),
    MissingEventLog(PathBuf),
    InvalidMagic,
    TruncatedRecord,
    InvalidSender(u8),
    InvalidMessageKind(u8),
    InvalidPayloadLength(u32),
    DigestMismatch,
    CounterOverflow,
}

impl From<io::Error> for DialogueSnapshotError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl DialogueSnapshot {
    pub fn derive(root: impl AsRef<Path>) -> Result<Self, DialogueSnapshotError> {
        let path = root.as_ref().join("dialogue.events");
        if !path.is_file() {
            return Err(DialogueSnapshotError::MissingEventLog(path));
        }
        let bytes = fs::read(&path)?;
        if bytes.len() < STORE_MAGIC.len() || bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
            return Err(DialogueSnapshotError::InvalidMagic);
        }

        let mut snapshot = Self {
            schema_version: 1,
            events: 0,
            payload_bytes: 0,
            sender_counts: [0; SENDER_CLASSES],
            message_kind_counts: [0; MESSAGE_KINDS],
            event_log_sha256: Sha256::digest(&bytes).into(),
        };

        let mut offset = STORE_MAGIC.len();
        while offset < bytes.len() {
            if bytes.len() - offset < RECORD_HEADER_BYTES {
                return Err(DialogueSnapshotError::TruncatedRecord);
            }
            if bytes[offset..offset + STORE_MAGIC.len()] != STORE_MAGIC {
                return Err(DialogueSnapshotError::InvalidMagic);
            }

            let sender = bytes[offset + 8];
            let kind = bytes[offset + 9];
            let payload_len = u32::from_le_bytes(
                bytes[offset + 18..offset + 22]
                    .try_into()
                    .expect("fixed payload length slice"),
            );
            let payload_len_usize = payload_len as usize;
            if payload_len_usize == 0 || payload_len_usize > MAX_PAYLOAD_BYTES {
                return Err(DialogueSnapshotError::InvalidPayloadLength(payload_len));
            }
            let total = RECORD_HEADER_BYTES
                .checked_add(payload_len_usize)
                .ok_or(DialogueSnapshotError::InvalidPayloadLength(payload_len))?;
            if bytes.len() - offset < total {
                return Err(DialogueSnapshotError::TruncatedRecord);
            }

            let sender_index = sender_index(sender)?;
            let kind_index = kind_index(kind)?;
            let payload = &bytes[offset + RECORD_HEADER_BYTES..offset + total];
            let expected = &bytes[offset + 22..offset + 54];
            let observed = record_fingerprint(
                u64::from_le_bytes(
                    bytes[offset + 10..offset + 18]
                        .try_into()
                        .expect("fixed evidence id slice"),
                ),
                sender,
                kind,
                payload,
            );
            if observed.as_slice() != expected {
                return Err(DialogueSnapshotError::DigestMismatch);
            }

            snapshot.events = snapshot
                .events
                .checked_add(1)
                .ok_or(DialogueSnapshotError::CounterOverflow)?;
            snapshot.payload_bytes = snapshot
                .payload_bytes
                .checked_add(payload_len as u64)
                .ok_or(DialogueSnapshotError::CounterOverflow)?;
            snapshot.sender_counts[sender_index] = snapshot.sender_counts[sender_index]
                .checked_add(1)
                .ok_or(DialogueSnapshotError::CounterOverflow)?;
            snapshot.message_kind_counts[kind_index] = snapshot.message_kind_counts[kind_index]
                .checked_add(1)
                .ok_or(DialogueSnapshotError::CounterOverflow)?;
            offset += total;
        }
        Ok(snapshot)
    }

    pub fn write_atomic(&self, root: impl AsRef<Path>) -> Result<PathBuf, DialogueSnapshotError> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let target = root.join("dialogue.snapshot");
        let temporary = root.join("dialogue.snapshot.tmp");
        let encoded = self.encode();
        fs::write(&temporary, encoded)?;
        fs::rename(&temporary, &target)?;
        Ok(target)
    }

    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(8 + 2 + 8 + 8 + 5 * 8 + 10 * 8 + 32);
        output.extend_from_slice(&SNAPSHOT_MAGIC);
        output.extend_from_slice(&self.schema_version.to_le_bytes());
        output.extend_from_slice(&self.events.to_le_bytes());
        output.extend_from_slice(&self.payload_bytes.to_le_bytes());
        for count in self.sender_counts {
            output.extend_from_slice(&count.to_le_bytes());
        }
        for count in self.message_kind_counts {
            output.extend_from_slice(&count.to_le_bytes());
        }
        output.extend_from_slice(&self.event_log_sha256);
        output
    }
}

fn sender_index(value: u8) -> Result<usize, DialogueSnapshotError> {
    match value {
        1..=5 => Ok(usize::from(value - 1)),
        other => Err(DialogueSnapshotError::InvalidSender(other)),
    }
}

fn kind_index(value: u8) -> Result<usize, DialogueSnapshotError> {
    match value {
        1..=10 => Ok(usize::from(value - 1)),
        other => Err(DialogueSnapshotError::InvalidMessageKind(other)),
    }
}

fn record_fingerprint(evidence_id: u64, sender: u8, kind: u8, payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(STORE_MAGIC);
    hasher.update(evidence_id.to_le_bytes());
    hasher.update([sender]);
    hasher.update([kind]);
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PersistentDialogueStore;
    use cogno_core::{ObservedDialogueKind, ObservedSenderClass};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-dialogue-snapshot-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn snapshot_is_deterministic_and_counts_records() {
        let root = temporary_root();
        let mut store = PersistentDialogueStore::open(&root).expect("store");
        store
            .append(
                1,
                ObservedSenderClass::SciAgent,
                ObservedDialogueKind::Hypothesis,
                br#"{"id":"h1"}"#,
            )
            .expect("append hypothesis");
        store
            .append(
                2,
                ObservedSenderClass::RuntimeDiscovery,
                ObservedDialogueKind::ExperimentResult,
                br#"{"auc":0.67}"#,
            )
            .expect("append result");

        let first = DialogueSnapshot::derive(&root).expect("first snapshot");
        let second = DialogueSnapshot::derive(&root).expect("second snapshot");
        assert_eq!(first, second);
        assert_eq!(first.events, 2);
        assert_eq!(first.sender_counts[2], 1);
        assert_eq!(first.sender_counts[3], 1);
        assert_eq!(first.message_kind_counts[1], 1);
        assert_eq!(first.message_kind_counts[5], 1);
        let path = first.write_atomic(&root).expect("write snapshot");
        assert_eq!(fs::read(path).expect("snapshot bytes"), first.encode());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn tampered_event_log_is_rejected() {
        let root = temporary_root();
        let mut store = PersistentDialogueStore::open(&root).expect("store");
        store
            .append(
                1,
                ObservedSenderClass::ExternalModel,
                ObservedDialogueKind::Question,
                br#"{"question":"x"}"#,
            )
            .expect("append");
        let path = root.join("dialogue.events");
        let mut bytes = fs::read(&path).expect("read event log");
        *bytes.last_mut().expect("payload byte") ^= 1;
        fs::write(&path, bytes).expect("tamper");
        assert!(matches!(
            DialogueSnapshot::derive(&root),
            Err(DialogueSnapshotError::DigestMismatch)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
