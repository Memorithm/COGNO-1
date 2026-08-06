//! Bounded exchange boundary for SCIAGENT / SciRust scientific-taste evidence.
//!
//! External model output may propose observations but can never become a
//! promotion validation. Only deterministic evaluator output or an explicit
//! user action can cross into the persistent validation format.

use crate::taste_validation_store::{
    StoredTasteValidation, StoredValidationOrigin, StoredValidationVerdict,
};
use sha2::{Digest, Sha256};

/// Maximum canonical payload retained for one exchange record.
pub const MAX_SCIENTIFIC_EXCHANGE_PAYLOAD_BYTES: usize = 64 * 1024;

/// Provenance class attached to an incoming scientific observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificExchangeOrigin {
    SciAgentModel,
    SciRustDeterministicEvaluator,
    ExplicitUserAction,
    RuntimeObservation,
}

/// Verdict proposed by an incoming scientific observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificExchangeVerdict {
    Confirmed,
    Contradicted,
    Inconclusive,
}

/// Bounded cross-component observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScientificExchangeRecord {
    pub observation_id: u64,
    pub preference_id: u64,
    pub evidence_id: u64,
    pub origin: ScientificExchangeOrigin,
    pub verdict: ScientificExchangeVerdict,
    pub confidence_bps: u16,
    pub canonical_payload: Vec<u8>,
}

/// Result of classifying one exchange record at the trust boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificExchangeDisposition {
    QuarantinedModelObservation,
    RuntimeObservationOnly,
    Inconclusive,
    Validation(StoredTasteValidation),
}

/// Invalid exchange input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificExchangeError {
    ZeroObservationId,
    ZeroPreferenceId,
    ZeroEvidenceId,
    ConfidenceOutOfRange(u16),
    EmptyPayload,
    PayloadTooLarge,
}

impl ScientificExchangeRecord {
    /// Deterministic SHA-256 over canonical payload bytes.
    #[must_use]
    pub fn payload_sha256(&self) -> [u8; 32] {
        Sha256::digest(&self.canonical_payload).into()
    }
}

/// Classify one SCIAGENT/SciRust record without granting model authority.
pub fn classify_scientific_exchange(
    record: &ScientificExchangeRecord,
) -> Result<ScientificExchangeDisposition, ScientificExchangeError> {
    validate_record(record)?;

    if matches!(record.verdict, ScientificExchangeVerdict::Inconclusive) {
        return Ok(ScientificExchangeDisposition::Inconclusive);
    }

    match record.origin {
        ScientificExchangeOrigin::SciAgentModel => {
            Ok(ScientificExchangeDisposition::QuarantinedModelObservation)
        }
        ScientificExchangeOrigin::RuntimeObservation => {
            Ok(ScientificExchangeDisposition::RuntimeObservationOnly)
        }
        ScientificExchangeOrigin::SciRustDeterministicEvaluator => Ok(
            ScientificExchangeDisposition::Validation(to_validation(
                record,
                StoredValidationOrigin::DeterministicEvaluation,
            )),
        ),
        ScientificExchangeOrigin::ExplicitUserAction => Ok(
            ScientificExchangeDisposition::Validation(to_validation(
                record,
                StoredValidationOrigin::ExplicitUserAction,
            )),
        ),
    }
}

fn validate_record(record: &ScientificExchangeRecord) -> Result<(), ScientificExchangeError> {
    if record.observation_id == 0 {
        return Err(ScientificExchangeError::ZeroObservationId);
    }
    if record.preference_id == 0 {
        return Err(ScientificExchangeError::ZeroPreferenceId);
    }
    if record.evidence_id == 0 {
        return Err(ScientificExchangeError::ZeroEvidenceId);
    }
    if record.confidence_bps > 10_000 {
        return Err(ScientificExchangeError::ConfidenceOutOfRange(
            record.confidence_bps,
        ));
    }
    if record.canonical_payload.is_empty() {
        return Err(ScientificExchangeError::EmptyPayload);
    }
    if record.canonical_payload.len() > MAX_SCIENTIFIC_EXCHANGE_PAYLOAD_BYTES {
        return Err(ScientificExchangeError::PayloadTooLarge);
    }
    Ok(())
}

fn to_validation(
    record: &ScientificExchangeRecord,
    origin: StoredValidationOrigin,
) -> StoredTasteValidation {
    StoredTasteValidation {
        validation_id: record.observation_id,
        preference_id: record.preference_id,
        evidence_id: record.evidence_id,
        origin,
        verdict: match record.verdict {
            ScientificExchangeVerdict::Confirmed => StoredValidationVerdict::Confirmed,
            ScientificExchangeVerdict::Contradicted => StoredValidationVerdict::Contradicted,
            ScientificExchangeVerdict::Inconclusive => unreachable!("handled before conversion"),
        },
        confidence_bps: record.confidence_bps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(origin: ScientificExchangeOrigin) -> ScientificExchangeRecord {
        ScientificExchangeRecord {
            observation_id: 1,
            preference_id: 7,
            evidence_id: 11,
            origin,
            verdict: ScientificExchangeVerdict::Confirmed,
            confidence_bps: 8_000,
            canonical_payload: b"deterministic scientific evidence".to_vec(),
        }
    }

    #[test]
    fn sciagent_model_output_remains_quarantined() {
        assert_eq!(
            classify_scientific_exchange(&record(ScientificExchangeOrigin::SciAgentModel)),
            Ok(ScientificExchangeDisposition::QuarantinedModelObservation)
        );
    }

    #[test]
    fn scirust_deterministic_evaluation_can_become_validation() {
        let disposition = classify_scientific_exchange(&record(
            ScientificExchangeOrigin::SciRustDeterministicEvaluator,
        ))
        .expect("classification");
        assert!(matches!(
            disposition,
            ScientificExchangeDisposition::Validation(StoredTasteValidation {
                origin: StoredValidationOrigin::DeterministicEvaluation,
                ..
            })
        ));
    }

    #[test]
    fn oversized_payload_fails_closed() {
        let mut oversized = record(ScientificExchangeOrigin::SciAgentModel);
        oversized.canonical_payload = vec![0; MAX_SCIENTIFIC_EXCHANGE_PAYLOAD_BYTES + 1];
        assert_eq!(
            classify_scientific_exchange(&oversized),
            Err(ScientificExchangeError::PayloadTooLarge)
        );
    }
}
