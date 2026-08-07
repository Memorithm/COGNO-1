//! Persistent append-only storage for non-model taste validations.
//!
//! The store is evidence, not authority. Profile state must always be replayed
//! through `ScientificTasteProfile`; this module only preserves validated,
//! bounded inputs with deterministic SHA-256 fingerprints.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: [u8; 8] = *b"COGVAL01";
const RECORD_BYTES: usize = 68;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredValidationOrigin {
    DeterministicEvaluation,
    ExplicitUserAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredValidationVerdict {
    Confirmed,
    Contradicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredTasteValidation {
    pub validation_id: u64,
    pub preference_id: u64,
    pub evidence_id: u64,
    pub origin: StoredValidationOrigin,
    pub verdict: StoredValidationVerdict,
    pub confidence_bps: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteValidationAppendOutcome {
    Appended,
    Duplicate,
}

#[derive(Debug)]
pub enum TasteValidationStoreError {
    Io(io::Error),
    InvalidMagic,
    TruncatedRecord,
    InvalidOrigin(u8),
    InvalidVerdict(u8),
    ConfidenceOutOfRange(u16),
    DigestMismatch,
    DuplicateValidationConflict(u64),
}

impl From<io::Error> for TasteValidationStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
pub struct PersistentTasteValidationStore {
    path: PathBuf,
    records: BTreeMap<u64, StoredTasteValidation>,
}

impl PersistentTasteValidationStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, TasteValidationStoreError> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let path = root.join("taste.validations");
        if !path.exists() {
            let mut file = File::create(&path)?;
            file.write_all(&MAGIC)?;
            file.sync_data()?;
        }
        let records = replay(&path)?;
        Ok(Self { path, records })
    }

    pub fn append(
        &mut self,
        validation: StoredTasteValidation,
    ) -> Result<TasteValidationAppendOutcome, TasteValidationStoreError> {
        validate(validation)?;
        if let Some(existing) = self.records.get(&validation.validation_id) {
            return if *existing == validation {
                Ok(TasteValidationAppendOutcome::Duplicate)
            } else {
                Err(TasteValidationStoreError::DuplicateValidationConflict(
                    validation.validation_id,
                ))
            };
        }
        let encoded = encode(validation);
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&encoded)?;
        file.sync_data()?;
        self.records.insert(validation.validation_id, validation);
        Ok(TasteValidationAppendOutcome::Appended)
    }

    pub fn records(&self) -> impl Iterator<Item = &StoredTasteValidation> {
        self.records.values()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

fn validate(validation: StoredTasteValidation) -> Result<(), TasteValidationStoreError> {
    if validation.confidence_bps > 10_000 {
        return Err(TasteValidationStoreError::ConfidenceOutOfRange(
            validation.confidence_bps,
        ));
    }
    Ok(())
}

fn encode(validation: StoredTasteValidation) -> Vec<u8> {
    let origin = origin_code(validation.origin);
    let verdict = verdict_code(validation.verdict);
    let digest = fingerprint(validation);
    let mut output = Vec::with_capacity(RECORD_BYTES);
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&validation.validation_id.to_le_bytes());
    output.extend_from_slice(&validation.preference_id.to_le_bytes());
    output.extend_from_slice(&validation.evidence_id.to_le_bytes());
    output.push(origin);
    output.push(verdict);
    output.extend_from_slice(&validation.confidence_bps.to_le_bytes());
    output.extend_from_slice(&digest);
    output
}

fn fingerprint(validation: StoredTasteValidation) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MAGIC);
    hasher.update(validation.validation_id.to_le_bytes());
    hasher.update(validation.preference_id.to_le_bytes());
    hasher.update(validation.evidence_id.to_le_bytes());
    hasher.update([
        origin_code(validation.origin),
        verdict_code(validation.verdict),
    ]);
    hasher.update(validation.confidence_bps.to_le_bytes());
    hasher.finalize().into()
}

const fn origin_code(origin: StoredValidationOrigin) -> u8 {
    match origin {
        StoredValidationOrigin::DeterministicEvaluation => 1,
        StoredValidationOrigin::ExplicitUserAction => 2,
    }
}

const fn verdict_code(verdict: StoredValidationVerdict) -> u8 {
    match verdict {
        StoredValidationVerdict::Confirmed => 1,
        StoredValidationVerdict::Contradicted => 2,
    }
}

fn replay(path: &Path) -> Result<BTreeMap<u64, StoredTasteValidation>, TasteValidationStoreError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < MAGIC.len() || bytes[..MAGIC.len()] != MAGIC {
        return Err(TasteValidationStoreError::InvalidMagic);
    }
    let mut records = BTreeMap::new();
    let mut offset = MAGIC.len();
    while offset < bytes.len() {
        if bytes.len() - offset < RECORD_BYTES {
            return Err(TasteValidationStoreError::TruncatedRecord);
        }
        if bytes[offset..offset + MAGIC.len()] != MAGIC {
            return Err(TasteValidationStoreError::InvalidMagic);
        }
        let validation = StoredTasteValidation {
            validation_id: u64::from_le_bytes(
                bytes[offset + 8..offset + 16]
                    .try_into()
                    .expect("validation id"),
            ),
            preference_id: u64::from_le_bytes(
                bytes[offset + 16..offset + 24]
                    .try_into()
                    .expect("preference id"),
            ),
            evidence_id: u64::from_le_bytes(
                bytes[offset + 24..offset + 32]
                    .try_into()
                    .expect("evidence id"),
            ),
            origin: match bytes[offset + 32] {
                1 => StoredValidationOrigin::DeterministicEvaluation,
                2 => StoredValidationOrigin::ExplicitUserAction,
                value => return Err(TasteValidationStoreError::InvalidOrigin(value)),
            },
            verdict: match bytes[offset + 33] {
                1 => StoredValidationVerdict::Confirmed,
                2 => StoredValidationVerdict::Contradicted,
                value => return Err(TasteValidationStoreError::InvalidVerdict(value)),
            },
            confidence_bps: u16::from_le_bytes(
                bytes[offset + 34..offset + 36]
                    .try_into()
                    .expect("confidence"),
            ),
        };
        validate(validation)?;
        if fingerprint(validation).as_slice() != &bytes[offset + 36..offset + RECORD_BYTES] {
            return Err(TasteValidationStoreError::DigestMismatch);
        }
        if let Some(existing) = records.insert(validation.validation_id, validation)
            && existing != validation
        {
            return Err(TasteValidationStoreError::DuplicateValidationConflict(
                validation.validation_id,
            ));
        }
        offset += RECORD_BYTES;
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-validations-{}-{nonce}", std::process::id()))
    }

    fn validation() -> StoredTasteValidation {
        StoredTasteValidation {
            validation_id: 1,
            preference_id: 7,
            evidence_id: 11,
            origin: StoredValidationOrigin::DeterministicEvaluation,
            verdict: StoredValidationVerdict::Confirmed,
            confidence_bps: 8_000,
        }
    }

    #[test]
    fn append_reopen_and_deduplicate() {
        let root = root();
        let mut store = PersistentTasteValidationStore::open(&root).expect("store");
        assert_eq!(
            store.append(validation()).expect("append"),
            TasteValidationAppendOutcome::Appended
        );
        drop(store);
        let mut reopened = PersistentTasteValidationStore::open(&root).expect("reopen");
        assert_eq!(reopened.len(), 1);
        assert_eq!(
            reopened.append(validation()).expect("duplicate"),
            TasteValidationAppendOutcome::Duplicate
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn conflicting_duplicate_is_rejected() {
        let root = root();
        let mut store = PersistentTasteValidationStore::open(&root).expect("store");
        store.append(validation()).expect("append");
        let mut conflicting = validation();
        conflicting.verdict = StoredValidationVerdict::Contradicted;
        assert!(matches!(
            store.append(conflicting),
            Err(TasteValidationStoreError::DuplicateValidationConflict(1))
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
