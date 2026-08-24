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

const MAGIC: [u8; 8] = *b"COGVAL02";
/// Legacy header (no subject attribution). Files written before attribution
/// are transparently migrated to the current format on open.
const LEGACY_MAGIC: [u8; 8] = *b"COGVAL01";
/// v1 record: no subject kind, no subject hash.
const LEGACY_RECORD_BYTES: usize = 68;
/// v2 record: legacy fields + subject kind (1 byte) + SHA-256 of the subject
/// id (32 bytes). The raw subject id is never persisted — only its digest —
/// so the store stays privacy-preserving while remaining attributable.
const RECORD_BYTES: usize = LEGACY_RECORD_BYTES + 33;

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

/// Class of whoever produced a validation (harness harness attribution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationSubjectKind {
    /// Pre-attribution record or host that could not identify the source.
    Unattributed,
    /// A human operator on the host side.
    HumanHost,
    /// A deterministic evaluator instance.
    DeterministicEvaluator,
    /// Another authenticated agent.
    ExternalAgent,
}

impl ValidationSubjectKind {
    const fn code(self) -> u8 {
        match self {
            Self::Unattributed => 0,
            Self::HumanHost => 1,
            Self::DeterministicEvaluator => 2,
            Self::ExternalAgent => 3,
        }
    }

    const fn from_code(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unattributed),
            1 => Some(Self::HumanHost),
            2 => Some(Self::DeterministicEvaluator),
            3 => Some(Self::ExternalAgent),
            _ => None,
        }
    }
}

/// SHA-256 of `ValidationSubjectKind`-specific subject id. Zeroed when
/// [`ValidationSubjectKind::Unattributed`] (enforced by `validate`).
pub type SubjectIdHash = [u8; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredTasteValidation {
    pub validation_id: u64,
    pub preference_id: u64,
    pub evidence_id: u64,
    pub origin: StoredValidationOrigin,
    pub verdict: StoredValidationVerdict,
    pub confidence_bps: u16,
    /// Harness participant that produced this validation. Attribution only:
    /// every variant remains non-model evidence for promotion purposes.
    pub subject_kind: ValidationSubjectKind,
    /// SHA-256 of the subject id (`[0; 32]` when unattributed).
    pub subject_id_sha256: SubjectIdHash,
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
    InvalidSubjectKind(u8),
    ConfidenceOutOfRange(u16),
    /// Attributed record with a zeroed subject hash, or unattributed record
    /// with a non-zero hash: the pair must be coherent (fail closed).
    IncoherentSubjectAttribution,
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
        } else {
            // Transparent v1 -> v2 migration: legacy records are replayed,
            // marked `Unattributed`, and the file is rewritten atomically in
            // the current format so attribution state is never mixed on disk.
            migrate_legacy_if_needed(&path)?;
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
    let zero_hash = validation.subject_id_sha256 == [0u8; 32];
    match validation.subject_kind {
        ValidationSubjectKind::Unattributed if !zero_hash => {
            return Err(TasteValidationStoreError::IncoherentSubjectAttribution);
        }
        ValidationSubjectKind::Unattributed => {}
        _ if zero_hash => {
            return Err(TasteValidationStoreError::IncoherentSubjectAttribution);
        }
        _ => {}
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
    output.push(validation.subject_kind.code());
    output.extend_from_slice(&validation.subject_id_sha256);
    output.extend_from_slice(&digest);
    output
}

/// Encode a legacy (pre-attribution) record. Test-only: used to fabricate
/// pre-attribution files so the transparent migration stays exercised; live
/// appends always write v2.
#[cfg(test)]
fn encode_legacy_v1(validation: StoredTasteValidation) -> Vec<u8> {
    let mut output = Vec::with_capacity(LEGACY_RECORD_BYTES);
    output.extend_from_slice(&LEGACY_MAGIC);
    output.extend_from_slice(&validation.validation_id.to_le_bytes());
    output.extend_from_slice(&validation.preference_id.to_le_bytes());
    output.extend_from_slice(&validation.evidence_id.to_le_bytes());
    output.push(origin_code(validation.origin));
    output.push(verdict_code(validation.verdict));
    output.extend_from_slice(&validation.confidence_bps.to_le_bytes());
    // Legacy digest: SHA-256 over the v1 fields only (no subject attribution).
    let mut hasher = Sha256::new();
    hasher.update(&output);
    let digest: [u8; 32] = hasher.finalize().into();
    debug_assert_eq!(LEGACY_MAGIC.len() + 28 + digest.len(), LEGACY_RECORD_BYTES);
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
        validation.subject_kind.code(),
    ]);
    hasher.update(validation.confidence_bps.to_le_bytes());
    hasher.update(validation.subject_id_sha256);
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
        let subject_kind = ValidationSubjectKind::from_code(bytes[offset + 36]).ok_or(
            TasteValidationStoreError::InvalidSubjectKind(bytes[offset + 36]),
        )?;
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
            subject_kind,
            subject_id_sha256: bytes[offset + 37..offset + 69]
                .try_into()
                .expect("subject hash"),
        };
        validate(validation)?;
        if fingerprint(validation).as_slice() != &bytes[offset + 69..offset + RECORD_BYTES] {
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

/// Rewrite a legacy `COGVAL01` file into the current format atomically
/// (temp file + rename + fsync). Legacy records become `Unattributed` — the
/// migration never invents provenance.
fn migrate_legacy_if_needed(path: &Path) -> Result<(), TasteValidationStoreError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < LEGACY_MAGIC.len() || bytes[..LEGACY_MAGIC.len()] != LEGACY_MAGIC {
        // Current format (or a corrupt header that replay will reject).
        return Ok(());
    }

    let mut records = BTreeMap::new();
    let mut offset = LEGACY_MAGIC.len();
    while offset < bytes.len() {
        if bytes.len() - offset < LEGACY_RECORD_BYTES {
            return Err(TasteValidationStoreError::TruncatedRecord);
        }
        if bytes[offset..offset + LEGACY_MAGIC.len()] != LEGACY_MAGIC {
            return Err(TasteValidationStoreError::InvalidMagic);
        }
        let legacy = StoredTasteValidation {
            validation_id: u64::from_le_bytes(
                bytes[offset + 8..offset + 16].try_into().expect("id"),
            ),
            preference_id: u64::from_le_bytes(
                bytes[offset + 16..offset + 24].try_into().expect("pref"),
            ),
            evidence_id: u64::from_le_bytes(
                bytes[offset + 24..offset + 32].try_into().expect("ev"),
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
                bytes[offset + 34..offset + 36].try_into().expect("conf"),
            ),
            subject_kind: ValidationSubjectKind::Unattributed,
            subject_id_sha256: [0u8; 32],
        };
        validate(legacy)?;
        // Preserve the v1 integrity guarantee across migration: refuse to
        // launder a corrupted legacy record into the new format.
        let mut hasher = Sha256::new();
        hasher.update(&bytes[offset..offset + 36]);
        let expected: [u8; 32] = hasher.finalize().into();
        if expected.as_slice() != &bytes[offset + 36..offset + LEGACY_RECORD_BYTES] {
            return Err(TasteValidationStoreError::DigestMismatch);
        }
        records.insert(legacy.validation_id, legacy);
        offset += LEGACY_RECORD_BYTES;
    }

    let tmp_path = path.with_extension("validations.tmp");
    {
        let mut out = File::create(&tmp_path)?;
        out.write_all(&MAGIC)?;
        for record in records.values() {
            out.write_all(&encode(*record))?;
        }
        out.sync_all()?;
    }
    fs::rename(&tmp_path, path)?;
    Ok(())
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
            subject_kind: ValidationSubjectKind::DeterministicEvaluator,
            subject_id_sha256: [0xAA; 32],
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

#[cfg(test)]
mod attribution_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(tag: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-validations-{tag}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn attributed(kind: ValidationSubjectKind, hash: [u8; 32]) -> StoredTasteValidation {
        StoredTasteValidation {
            validation_id: 5,
            preference_id: 7,
            evidence_id: 15,
            origin: StoredValidationOrigin::DeterministicEvaluation,
            verdict: StoredValidationVerdict::Confirmed,
            confidence_bps: 8_000,
            subject_kind: kind,
            subject_id_sha256: hash,
        }
    }

    #[test]
    fn incoherent_subject_attribution_is_refused() {
        let root = root("incoherent");
        let mut store = PersistentTasteValidationStore::open(&root).expect("store");
        // Attributed kind but zeroed hash.
        assert!(matches!(
            store.append(attributed(ValidationSubjectKind::HumanHost, [0u8; 32])),
            Err(TasteValidationStoreError::IncoherentSubjectAttribution)
        ));
        // Unattributed kind but non-zero hash.
        assert!(matches!(
            store.append(attributed(ValidationSubjectKind::Unattributed, [7; 32])),
            Err(TasteValidationStoreError::IncoherentSubjectAttribution)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn attributed_records_survive_reopen_and_conflict_by_subject() {
        let root = root("subject-reopen");
        let mut store = PersistentTasteValidationStore::open(&root).expect("store");
        let record = attributed(
            ValidationSubjectKind::HumanHost,
            sha256_for_test(b"operator-1"),
        );
        store.append(record).expect("append");
        drop(store);
        let mut reopened = PersistentTasteValidationStore::open(&root).expect("reopen");
        assert_eq!(reopened.len(), 1);
        let stored = *reopened.records().next().expect("one record");
        assert_eq!(stored.subject_kind, ValidationSubjectKind::HumanHost);
        assert_eq!(stored.subject_id_sha256, sha256_for_test(b"operator-1"));
        // Same validation_id with a DIFFERENT subject is a conflict, not a
        // silent overwrite (S6).
        let mut conflicting = record;
        conflicting.subject_id_sha256 = sha256_for_test(b"operator-2");
        assert!(matches!(
            reopened.append(conflicting),
            Err(TasteValidationStoreError::DuplicateValidationConflict(5))
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn legacy_v1_file_migrates_to_unattributed_v2() {
        let root = root("legacy-migration");
        fs::create_dir_all(&root).expect("dir");
        let path = root.join("taste.validations");
        let legacy_records = [
            StoredTasteValidation {
                validation_id: 1,
                preference_id: 7,
                evidence_id: 11,
                origin: StoredValidationOrigin::DeterministicEvaluation,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 8_000,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            },
            StoredTasteValidation {
                validation_id: 2,
                preference_id: 9,
                evidence_id: 22,
                origin: StoredValidationOrigin::ExplicitUserAction,
                verdict: StoredValidationVerdict::Contradicted,
                confidence_bps: 1_000,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            },
        ];
        {
            let mut file = File::create(&path).expect("create legacy file");
            file.write_all(&LEGACY_MAGIC).expect("magic");
            for record in &legacy_records {
                file.write_all(&encode_legacy_v1(*record)).expect("record");
            }
            file.sync_all().expect("sync");
        }

        // Opening transparently migrates and preserves every record.
        let store = PersistentTasteValidationStore::open(&root).expect("open migrates");
        assert_eq!(store.len(), 2);
        for record in store.records() {
            assert_eq!(
                record.subject_kind,
                ValidationSubjectKind::Unattributed,
                "migration never invents provenance"
            );
        }
        // The on-disk format is now v2 (header replaced), and appends work.
        let header = fs::read(&path).expect("read migrated header")[..8].to_vec();
        assert_eq!(header, MAGIC);
        let mut writable = PersistentTasteValidationStore::open(&root).expect("reopen");
        assert!(matches!(
            writable.append(attributed(
                ValidationSubjectKind::ExternalAgent,
                sha256_for_test(b"agent-9")
            )),
            Ok(TasteValidationAppendOutcome::Appended)
        ));
        assert_eq!(writable.len(), 3);
        fs::remove_dir_all(root).expect("cleanup");
    }

    /// Deterministic stand-in digest for tests (FNV-1a over the bytes).
    fn sha256_for_test(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}
