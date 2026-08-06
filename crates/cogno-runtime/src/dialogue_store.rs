//! Persistent append-only storage for validated dialogue observations.
//!
//! Records are length-prefixed, versioned, checksummed with SHA-256 and
//! synchronised after each append. Replay fails closed on truncation, unknown
//! enum values, invalid lengths or digest mismatch.

use cogno_core::{
    observe_dialogue, DialogueObservation, DialogueObservationError, DialogueObservationOutcome,
    Journal, ObservedDialogueKind, ObservedSenderClass, MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const STORE_MAGIC: [u8; 8] = *b"COGDLG01";
const RECORD_HEADER_BYTES: usize = 8 + 1 + 1 + 8 + 4 + 32;
const MAX_RECORD_BYTES: usize = RECORD_HEADER_BYTES + MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES;

#[derive(Debug)]
pub enum DialogueStoreError {
    Io(io::Error),
    InvalidMagic,
    TruncatedRecord,
    InvalidRecordLength(u32),
    InvalidSender(u8),
    InvalidKind(u8),
    DigestMismatch,
    Observation(DialogueObservationError),
}

impl From<io::Error> for DialogueStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<DialogueObservationError> for DialogueStoreError {
    fn from(error: DialogueObservationError) -> Self {
        Self::Observation(error)
    }
}

#[derive(Debug)]
pub struct PersistentDialogueStore {
    root: PathBuf,
    journal_path: PathBuf,
    journal: Journal,
    fingerprints: BTreeSet<[u8; 32]>,
}

impl PersistentDialogueStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DialogueStoreError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let journal_path = root.join("dialogue.events");

        if !journal_path.exists() {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&journal_path)?;
            file.write_all(&STORE_MAGIC)?;
            file.sync_data()?;
        }

        let mut store = Self {
            root,
            journal_path,
            journal: Journal::default(),
            fingerprints: BTreeSet::new(),
        };
        store.replay()?;
        Ok(store)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.journal.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.journal.is_empty()
    }

    pub fn append(
        &mut self,
        evidence_id: u64,
        sender_class: ObservedSenderClass,
        message_kind: ObservedDialogueKind,
        payload: &[u8],
    ) -> Result<DialogueObservationOutcome, DialogueStoreError> {
        let fingerprint = fingerprint(evidence_id, sender_class, message_kind, payload);
        let observation = DialogueObservation {
            fingerprint,
            evidence_id,
            sender_class,
            message_kind,
            payload: payload.to_vec(),
        };
        observation.validate()?;

        if self.fingerprints.contains(&fingerprint) {
            return Ok(observe_dialogue(&mut self.journal, observation)?);
        }

        let record = encode_record(&observation);
        let mut file = OpenOptions::new().append(true).open(&self.journal_path)?;
        file.write_all(&record)?;
        file.sync_data()?;

        let outcome = observe_dialogue(&mut self.journal, observation)?;
        self.fingerprints.insert(fingerprint);
        Ok(outcome)
    }

    fn replay(&mut self) -> Result<(), DialogueStoreError> {
        self.journal.clear();
        self.fingerprints.clear();

        let mut bytes = Vec::new();
        File::open(&self.journal_path)?.read_to_end(&mut bytes)?;
        if bytes.len() < STORE_MAGIC.len() || bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
            return Err(DialogueStoreError::InvalidMagic);
        }

        let mut offset = STORE_MAGIC.len();
        while offset < bytes.len() {
            if bytes.len() - offset < RECORD_HEADER_BYTES {
                return Err(DialogueStoreError::TruncatedRecord);
            }
            let (observation, consumed) = decode_record(&bytes[offset..])?;
            let fingerprint = observation.fingerprint;
            observe_dialogue(&mut self.journal, observation)?;
            self.fingerprints.insert(fingerprint);
            offset = offset
                .checked_add(consumed)
                .ok_or(DialogueStoreError::InvalidRecordLength(u32::MAX))?;
        }
        Ok(())
    }
}

fn fingerprint(
    evidence_id: u64,
    sender_class: ObservedSenderClass,
    message_kind: ObservedDialogueKind,
    payload: &[u8],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(STORE_MAGIC);
    hasher.update(evidence_id.to_le_bytes());
    hasher.update([sender_to_u8(sender_class)]);
    hasher.update([kind_to_u8(message_kind)]);
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

fn encode_record(observation: &DialogueObservation) -> Vec<u8> {
    let payload_len = observation.payload.len() as u32;
    let mut record = Vec::with_capacity(RECORD_HEADER_BYTES + observation.payload.len());
    record.extend_from_slice(&STORE_MAGIC);
    record.push(sender_to_u8(observation.sender_class));
    record.push(kind_to_u8(observation.message_kind));
    record.extend_from_slice(&observation.evidence_id.to_le_bytes());
    record.extend_from_slice(&payload_len.to_le_bytes());
    record.extend_from_slice(&observation.fingerprint);
    record.extend_from_slice(&observation.payload);
    record
}

fn decode_record(bytes: &[u8]) -> Result<(DialogueObservation, usize), DialogueStoreError> {
    if bytes.len() < RECORD_HEADER_BYTES {
        return Err(DialogueStoreError::TruncatedRecord);
    }
    if bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
        return Err(DialogueStoreError::InvalidMagic);
    }

    let sender_class = sender_from_u8(bytes[8])?;
    let message_kind = kind_from_u8(bytes[9])?;
    let evidence_id = u64::from_le_bytes(bytes[10..18].try_into().expect("fixed slice"));
    let payload_len = u32::from_le_bytes(bytes[18..22].try_into().expect("fixed slice"));
    let payload_len_usize = payload_len as usize;
    if payload_len_usize == 0 || payload_len_usize > MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES {
        return Err(DialogueStoreError::InvalidRecordLength(payload_len));
    }
    let total = RECORD_HEADER_BYTES
        .checked_add(payload_len_usize)
        .filter(|length| *length <= MAX_RECORD_BYTES)
        .ok_or(DialogueStoreError::InvalidRecordLength(payload_len))?;
    if bytes.len() < total {
        return Err(DialogueStoreError::TruncatedRecord);
    }

    let mut expected = [0u8; 32];
    expected.copy_from_slice(&bytes[22..54]);
    let payload = bytes[54..total].to_vec();
    let observed = fingerprint(evidence_id, sender_class, message_kind, &payload);
    if observed != expected {
        return Err(DialogueStoreError::DigestMismatch);
    }

    Ok((
        DialogueObservation {
            fingerprint: expected,
            evidence_id,
            sender_class,
            message_kind,
            payload,
        },
        total,
    ))
}

const fn sender_to_u8(value: ObservedSenderClass) -> u8 {
    match value {
        ObservedSenderClass::Human => 1,
        ObservedSenderClass::ExternalModel => 2,
        ObservedSenderClass::SciAgent => 3,
        ObservedSenderClass::RuntimeDiscovery => 4,
        ObservedSenderClass::DeterministicKernel => 5,
    }
}

fn sender_from_u8(value: u8) -> Result<ObservedSenderClass, DialogueStoreError> {
    match value {
        1 => Ok(ObservedSenderClass::Human),
        2 => Ok(ObservedSenderClass::ExternalModel),
        3 => Ok(ObservedSenderClass::SciAgent),
        4 => Ok(ObservedSenderClass::RuntimeDiscovery),
        5 => Ok(ObservedSenderClass::DeterministicKernel),
        other => Err(DialogueStoreError::InvalidSender(other)),
    }
}

const fn kind_to_u8(value: ObservedDialogueKind) -> u8 {
    match value {
        ObservedDialogueKind::Question => 1,
        ObservedDialogueKind::Hypothesis => 2,
        ObservedDialogueKind::Critique => 3,
        ObservedDialogueKind::Counterexample => 4,
        ObservedDialogueKind::ExperimentRequest => 5,
        ObservedDialogueKind::ExperimentResult => 6,
        ObservedDialogueKind::PreferenceObservation => 7,
        ObservedDialogueKind::Contradiction => 8,
        ObservedDialogueKind::Explanation => 9,
        ObservedDialogueKind::Decision => 10,
    }
}

fn kind_from_u8(value: u8) -> Result<ObservedDialogueKind, DialogueStoreError> {
    match value {
        1 => Ok(ObservedDialogueKind::Question),
        2 => Ok(ObservedDialogueKind::Hypothesis),
        3 => Ok(ObservedDialogueKind::Critique),
        4 => Ok(ObservedDialogueKind::Counterexample),
        5 => Ok(ObservedDialogueKind::ExperimentRequest),
        6 => Ok(ObservedDialogueKind::ExperimentResult),
        7 => Ok(ObservedDialogueKind::PreferenceObservation),
        8 => Ok(ObservedDialogueKind::Contradiction),
        9 => Ok(ObservedDialogueKind::Explanation),
        10 => Ok(ObservedDialogueKind::Decision),
        other => Err(DialogueStoreError::InvalidKind(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn append_reopen_and_deduplicate() {
        let root = temporary_root("dialogue-store");
        {
            let mut store = PersistentDialogueStore::open(&root).expect("open");
            let first = store
                .append(
                    9,
                    ObservedSenderClass::SciAgent,
                    ObservedDialogueKind::Hypothesis,
                    br#"{"message_id":"m1"}"#,
                )
                .expect("append");
            assert!(matches!(first, DialogueObservationOutcome::Appended { .. }));
            assert_eq!(store.len(), 1);
        }
        {
            let mut store = PersistentDialogueStore::open(&root).expect("reopen");
            assert_eq!(store.len(), 1);
            let duplicate = store
                .append(
                    9,
                    ObservedSenderClass::SciAgent,
                    ObservedDialogueKind::Hypothesis,
                    br#"{"message_id":"m1"}"#,
                )
                .expect("duplicate");
            assert!(matches!(
                duplicate,
                DialogueObservationOutcome::Duplicate { .. }
            ));
            assert_eq!(store.len(), 1);
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn tampered_payload_fails_replay() {
        let root = temporary_root("dialogue-corruption");
        let path = root.join("dialogue.events");
        {
            let mut store = PersistentDialogueStore::open(&root).expect("open");
            store
                .append(
                    3,
                    ObservedSenderClass::ExternalModel,
                    ObservedDialogueKind::Question,
                    br#"{"question":"x"}"#,
                )
                .expect("append");
        }
        let mut bytes = fs::read(&path).expect("read");
        let last = bytes.last_mut().expect("payload byte");
        *last ^= 1;
        fs::write(&path, bytes).expect("tamper");
        assert!(matches!(
            PersistentDialogueStore::open(&root),
            Err(DialogueStoreError::DigestMismatch)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
