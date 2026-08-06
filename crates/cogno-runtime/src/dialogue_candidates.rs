//! Quarantined preference candidates derived from the persistent dialogue log.
//!
//! Only explicit `preference_observation` dialogue records are eligible. The
//! resulting candidates are non-authoritative: they carry model/tool provenance
//! and always start quarantined. This module never emits an active taste event.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const STORE_MAGIC: [u8; 8] = *b"COGDLG01";
const RECORD_HEADER_BYTES: usize = 54;
const MAX_PAYLOAD_BYTES: usize = 16 * 1024;
const PREFERENCE_OBSERVATION_KIND: u8 = 7;
const MAX_CANDIDATES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateOrigin {
    HumanObservation,
    ModelInference,
    RuntimeObservation,
    DeterministicKernelObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuarantinedPreferenceCandidate {
    pub preference_id: u64,
    pub evidence_id: u64,
    pub origin: CandidateOrigin,
    pub payload_sha256: [u8; 32],
    pub payload_bytes: u32,
    pub quarantined: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialogueCandidateReport {
    pub schema_version: u16,
    pub records_scanned: u64,
    pub preference_observations: u64,
    pub candidates: Vec<QuarantinedPreferenceCandidate>,
    pub authority: &'static str,
}

#[derive(Debug)]
pub enum DialogueCandidateError {
    Io(io::Error),
    MissingEventLog(PathBuf),
    InvalidMagic,
    TruncatedRecord,
    InvalidSender(u8),
    InvalidMessageKind(u8),
    InvalidPayloadLength(u32),
    DigestMismatch,
    CandidateLimitExceeded,
    CounterOverflow,
}

impl From<io::Error> for DialogueCandidateError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl DialogueCandidateReport {
    pub fn derive(root: impl AsRef<Path>) -> Result<Self, DialogueCandidateError> {
        let path = root.as_ref().join("dialogue.events");
        if !path.is_file() {
            return Err(DialogueCandidateError::MissingEventLog(path));
        }
        let bytes = fs::read(path)?;
        if bytes.len() < STORE_MAGIC.len() || bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
            return Err(DialogueCandidateError::InvalidMagic);
        }

        let mut report = Self {
            schema_version: 1,
            records_scanned: 0,
            preference_observations: 0,
            candidates: Vec::new(),
            authority: "quarantine_only",
        };
        let mut offset = STORE_MAGIC.len();
        while offset < bytes.len() {
            if bytes.len() - offset < RECORD_HEADER_BYTES {
                return Err(DialogueCandidateError::TruncatedRecord);
            }
            if bytes[offset..offset + STORE_MAGIC.len()] != STORE_MAGIC {
                return Err(DialogueCandidateError::InvalidMagic);
            }

            let sender = bytes[offset + 8];
            let kind = bytes[offset + 9];
            let origin = candidate_origin(sender)?;
            validate_message_kind(kind)?;
            let evidence_id = u64::from_le_bytes(
                bytes[offset + 10..offset + 18]
                    .try_into()
                    .expect("fixed evidence id slice"),
            );
            let payload_len = u32::from_le_bytes(
                bytes[offset + 18..offset + 22]
                    .try_into()
                    .expect("fixed payload length slice"),
            );
            let payload_len_usize = payload_len as usize;
            if payload_len_usize == 0 || payload_len_usize > MAX_PAYLOAD_BYTES {
                return Err(DialogueCandidateError::InvalidPayloadLength(payload_len));
            }
            let total = RECORD_HEADER_BYTES
                .checked_add(payload_len_usize)
                .ok_or(DialogueCandidateError::InvalidPayloadLength(payload_len))?;
            if bytes.len() - offset < total {
                return Err(DialogueCandidateError::TruncatedRecord);
            }

            let payload = &bytes[offset + RECORD_HEADER_BYTES..offset + total];
            let expected = &bytes[offset + 22..offset + 54];
            let observed = record_fingerprint(evidence_id, sender, kind, payload);
            if observed.as_slice() != expected {
                return Err(DialogueCandidateError::DigestMismatch);
            }

            report.records_scanned = report
                .records_scanned
                .checked_add(1)
                .ok_or(DialogueCandidateError::CounterOverflow)?;
            if kind == PREFERENCE_OBSERVATION_KIND {
                if report.candidates.len() >= MAX_CANDIDATES {
                    return Err(DialogueCandidateError::CandidateLimitExceeded);
                }
                report.preference_observations = report
                    .preference_observations
                    .checked_add(1)
                    .ok_or(DialogueCandidateError::CounterOverflow)?;
                let payload_sha256: [u8; 32] = Sha256::digest(payload).into();
                let preference_id = u64::from_le_bytes(
                    payload_sha256[..8]
                        .try_into()
                        .expect("SHA-256 prefix is eight bytes"),
                );
                report.candidates.push(QuarantinedPreferenceCandidate {
                    preference_id,
                    evidence_id,
                    origin,
                    payload_sha256,
                    payload_bytes: payload_len,
                    quarantined: true,
                });
            }
            offset += total;
        }
        Ok(report)
    }
}

fn candidate_origin(sender: u8) -> Result<CandidateOrigin, DialogueCandidateError> {
    match sender {
        1 => Ok(CandidateOrigin::HumanObservation),
        2 | 3 => Ok(CandidateOrigin::ModelInference),
        4 => Ok(CandidateOrigin::RuntimeObservation),
        5 => Ok(CandidateOrigin::DeterministicKernelObservation),
        other => Err(DialogueCandidateError::InvalidSender(other)),
    }
}

fn validate_message_kind(kind: u8) -> Result<(), DialogueCandidateError> {
    match kind {
        1..=10 => Ok(()),
        other => Err(DialogueCandidateError::InvalidMessageKind(other)),
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
        std::env::temp_dir().join(format!(
            "cogno-dialogue-candidates-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn model_preference_is_extracted_but_remains_quarantined() {
        let root = temporary_root();
        let mut store = PersistentDialogueStore::open(&root).expect("store");
        store
            .append(
                7,
                ObservedSenderClass::SciAgent,
                ObservedDialogueKind::PreferenceObservation,
                br#"{"preference":"prefer deterministic constant-time features"}"#,
            )
            .expect("append");
        let report = DialogueCandidateReport::derive(&root).expect("candidate report");
        assert_eq!(report.records_scanned, 1);
        assert_eq!(report.preference_observations, 1);
        assert_eq!(report.authority, "quarantine_only");
        assert_eq!(report.candidates.len(), 1);
        assert!(report.candidates[0].quarantined);
        assert_eq!(report.candidates[0].origin, CandidateOrigin::ModelInference);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn hypotheses_do_not_become_preference_candidates() {
        let root = temporary_root();
        let mut store = PersistentDialogueStore::open(&root).expect("store");
        store
            .append(
                8,
                ObservedSenderClass::SciAgent,
                ObservedDialogueKind::Hypothesis,
                br#"{"hypothesis":"entropy predicts unsafe cache reuse"}"#,
            )
            .expect("append");
        let report = DialogueCandidateReport::derive(&root).expect("candidate report");
        assert_eq!(report.records_scanned, 1);
        assert!(report.candidates.is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
