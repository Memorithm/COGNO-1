//! Production ingress for authenticated SciRust scientific evidence.
//!
//! The trust path is deliberately ordered as:
//!
//! ```text
//! authenticated transport
//!   -> SciRust execution-attestation verification
//!   -> append-only attestation receipt persistence
//!   -> ordinary validation-store reconciliation
//!   -> deterministic taste orchestration
//! ```
//!
//! A legacy deterministic validation that lacks an authenticated SciRust receipt
//! is retained for backward-compatible storage/replay tooling but is not exposed
//! by this ingress as trusted SciRust promotion evidence.

use crate::scientific_exchange::ScientificExchangeDisposition;
use crate::scirust_runtime_bridge::{
    bridge_authenticated_scirust_exchange, HostAuthenticatedSciRustSender,
    SciRustRuntimeBridgeError, SciRustRuntimeKind,
};
use crate::scirust_validation_receipt_store::{
    hash_sender, PersistentSciRustValidationReceiptStore, SciRustValidationReceiptAppendOutcome,
    SciRustValidationReceiptStoreError,
};
use crate::taste_orchestrator::{
    orchestrate_scientific_taste_cycle, OrchestratedTasteCandidate, ScientificTasteCycleReport,
    ScientificTasteOrchestratorError,
};
use crate::taste_validation_store::{
    PersistentTasteValidationStore, StoredTasteValidation, StoredValidationOrigin,
    TasteValidationAppendOutcome, TasteValidationStoreError,
};
use std::collections::BTreeSet;
use std::path::Path;

/// Startup reconciliation result between attestation receipts and the ordinary
/// taste-validation store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SciRustTasteIngressRecovery {
    pub restored_validations: usize,
    pub existing_validations: usize,
}

/// Result of ingesting one authenticated SciRust exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SciRustTasteIngressOutcome {
    Inconclusive,
    Validation {
        validation_id: u64,
        receipt: SciRustValidationReceiptAppendOutcome,
        validation: TasteValidationAppendOutcome,
    },
}

/// Failure in the persistent authenticated SciRust taste path.
#[derive(Debug)]
pub enum SciRustTasteIngressError {
    Bridge(SciRustRuntimeBridgeError),
    ReceiptStore(SciRustValidationReceiptStoreError),
    ValidationStore(TasteValidationStoreError),
    Orchestrator(ScientificTasteOrchestratorError),
    DuplicateTrustedEvidence(u64),
    UnexpectedDisposition,
    Poisoned,
}

/// Durable SciRust → COGNO ingress that cannot orchestrate after a partial
/// persistence failure until the stores are reopened and reconciled.
#[derive(Debug)]
pub struct PersistentSciRustTasteIngress {
    receipt_store: PersistentSciRustValidationReceiptStore,
    validation_store: PersistentTasteValidationStore,
    startup_recovery: SciRustTasteIngressRecovery,
    healthy: bool,
}

impl PersistentSciRustTasteIngress {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, SciRustTasteIngressError> {
        let root = root.as_ref();
        let receipt_store = PersistentSciRustValidationReceiptStore::open(root)
            .map_err(SciRustTasteIngressError::ReceiptStore)?;
        let mut validation_store = PersistentTasteValidationStore::open(root)
            .map_err(SciRustTasteIngressError::ValidationStore)?;
        let startup_recovery = reconcile(&receipt_store, &mut validation_store)?;
        let ingress = Self {
            receipt_store,
            validation_store,
            startup_recovery,
            healthy: true,
        };
        ingress.validate_trusted_evidence_uniqueness()?;
        Ok(ingress)
    }

    /// Verify, persist, and reconcile one authenticated SciRust runtime message.
    pub fn ingest(
        &mut self,
        encoded_message: &[u8],
        authenticated_sender: &HostAuthenticatedSciRustSender,
        expected_cogno_recipient_id: &str,
    ) -> Result<SciRustTasteIngressOutcome, SciRustTasteIngressError> {
        self.ensure_healthy()?;
        let receipt = bridge_authenticated_scirust_exchange(
            encoded_message,
            authenticated_sender,
            expected_cogno_recipient_id,
        )
        .map_err(SciRustTasteIngressError::Bridge)?;

        let validation = match receipt.disposition() {
            ScientificExchangeDisposition::Validation(validation) => validation,
            ScientificExchangeDisposition::Inconclusive => {
                return Ok(SciRustTasteIngressOutcome::Inconclusive);
            }
            ScientificExchangeDisposition::QuarantinedModelObservation
            | ScientificExchangeDisposition::RuntimeObservationOnly => {
                return Err(SciRustTasteIngressError::UnexpectedDisposition);
            }
        };
        // Stamp harness attribution from the transport-authenticated sender:
        // a deterministic evaluation delivered by an authenticated kernel is
        // attributed to that exact kernel instance (S6). User-action records
        // relayed here stay unattributed — the human identity is unknown to
        // this path and must never be fabricated.
        let mut validation = validation;
        if matches!(
            validation.origin,
            StoredValidationOrigin::DeterministicEvaluation
        ) && matches!(
            validation.subject_kind,
            crate::taste_validation_store::ValidationSubjectKind::Unattributed
        ) {
            use crate::taste_validation_store::ValidationSubjectKind;
            validation.subject_kind = match authenticated_sender.kind() {
                SciRustRuntimeKind::DeterministicKernel => {
                    ValidationSubjectKind::DeterministicEvaluator
                }
                SciRustRuntimeKind::RuntimeDiscovery => ValidationSubjectKind::ExternalAgent,
            };
            validation.subject_id_sha256 =
                hash_sender(authenticated_sender.kind(), authenticated_sender.id());
        }

        self.preflight_validation(validation)?;
        let receipt_outcome = match self.receipt_store.append(&receipt) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.healthy = false;
                return Err(SciRustTasteIngressError::ReceiptStore(error));
            }
        };
        let validation_outcome = match self.validation_store.append(validation) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.healthy = false;
                return Err(SciRustTasteIngressError::ValidationStore(error));
            }
        };

        Ok(SciRustTasteIngressOutcome::Validation {
            validation_id: validation.validation_id,
            receipt: receipt_outcome,
            validation: validation_outcome,
        })
    }

    /// Trusted promotion evidence for the production path.
    ///
    /// Explicit user actions remain authoritative. Deterministic evaluations are
    /// included only when backed by an authenticated persisted SciRust receipt;
    /// legacy deterministic records without such provenance are excluded.
    pub fn trusted_validations(
        &self,
    ) -> Result<Vec<StoredTasteValidation>, SciRustTasteIngressError> {
        self.ensure_healthy()?;
        let validations = self.collect_trusted_validations();
        ensure_unique_evidence(&validations)?;
        Ok(validations)
    }

    /// Orchestrate only persisted, replay-safe trusted evidence.
    pub fn orchestrate(
        &self,
        candidates: &[OrchestratedTasteCandidate],
    ) -> Result<ScientificTasteCycleReport, SciRustTasteIngressError> {
        let validations = self.trusted_validations()?;
        orchestrate_scientific_taste_cycle(candidates, &validations)
            .map_err(SciRustTasteIngressError::Orchestrator)
    }

    #[must_use]
    pub const fn startup_recovery(&self) -> SciRustTasteIngressRecovery {
        self.startup_recovery
    }

    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        self.healthy
    }

    #[must_use]
    pub fn attested_validation_count(&self) -> usize {
        self.receipt_store.len()
    }

    fn preflight_validation(
        &self,
        validation: StoredTasteValidation,
    ) -> Result<(), SciRustTasteIngressError> {
        for existing in self.validation_store.records() {
            if existing.validation_id == validation.validation_id && *existing != validation {
                return Err(SciRustTasteIngressError::ValidationStore(
                    TasteValidationStoreError::DuplicateValidationConflict(
                        validation.validation_id,
                    ),
                ));
            }
            if existing.origin == StoredValidationOrigin::ExplicitUserAction
                && existing.evidence_id == validation.evidence_id
                && existing.validation_id != validation.validation_id
            {
                return Err(SciRustTasteIngressError::DuplicateTrustedEvidence(
                    validation.evidence_id,
                ));
            }
        }
        Ok(())
    }

    fn collect_trusted_validations(&self) -> Vec<StoredTasteValidation> {
        let mut validations: Vec<_> = self
            .validation_store
            .records()
            .copied()
            .filter(|validation| validation.origin == StoredValidationOrigin::ExplicitUserAction)
            .collect();
        validations.extend(self.receipt_store.validations());
        validations.sort_unstable_by_key(|validation| validation.validation_id);
        validations
    }

    fn validate_trusted_evidence_uniqueness(&self) -> Result<(), SciRustTasteIngressError> {
        ensure_unique_evidence(&self.collect_trusted_validations())
    }

    fn ensure_healthy(&self) -> Result<(), SciRustTasteIngressError> {
        if self.healthy {
            Ok(())
        } else {
            Err(SciRustTasteIngressError::Poisoned)
        }
    }
}

fn reconcile(
    receipt_store: &PersistentSciRustValidationReceiptStore,
    validation_store: &mut PersistentTasteValidationStore,
) -> Result<SciRustTasteIngressRecovery, SciRustTasteIngressError> {
    let mut recovery = SciRustTasteIngressRecovery::default();
    for validation in receipt_store.validations() {
        match validation_store
            .append(validation)
            .map_err(SciRustTasteIngressError::ValidationStore)?
        {
            TasteValidationAppendOutcome::Appended => recovery.restored_validations += 1,
            TasteValidationAppendOutcome::Duplicate => recovery.existing_validations += 1,
        }
    }
    Ok(recovery)
}

fn ensure_unique_evidence(
    validations: &[StoredTasteValidation],
) -> Result<(), SciRustTasteIngressError> {
    let mut evidence_ids = BTreeSet::new();
    for validation in validations {
        if !evidence_ids.insert(validation.evidence_id) {
            return Err(SciRustTasteIngressError::DuplicateTrustedEvidence(
                validation.evidence_id,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taste_validation_store::ValidationSubjectKind;
    use crate::{OrchestratedTasteState, StoredValidationVerdict};
    use serde_json::{json, Value};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const PROFILE_SHA256: &str = "c984c0151e84300875c2aead5764d018f9ef5d09d218ab8f8f1ea9ab7157bec8";

    fn root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-scirust-ingress-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn message(
        message_id: &str,
        observation_id: u64,
        evidence_id: u64,
        confidence_bps: u16,
        canonical_payload: &[u8],
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
                "canonical_payload": canonical_payload
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

    fn sender() -> HostAuthenticatedSciRustSender {
        HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local").expect("sender")
    }

    #[test]
    fn exact_replay_does_not_create_another_confirmation() {
        let root = root("replay");
        let mut ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        let encoded = message("m-1", 17, 117, 8_500, b"deterministic-a");
        assert_eq!(
            ingress.ingest(&encoded, &sender(), "cogno").expect("first"),
            SciRustTasteIngressOutcome::Validation {
                validation_id: 17,
                receipt: SciRustValidationReceiptAppendOutcome::Appended,
                validation: TasteValidationAppendOutcome::Appended,
            }
        );
        assert_eq!(
            ingress
                .ingest(&encoded, &sender(), "cogno")
                .expect("replay"),
            SciRustTasteIngressOutcome::Validation {
                validation_id: 17,
                receipt: SciRustValidationReceiptAppendOutcome::Duplicate,
                validation: TasteValidationAppendOutcome::Duplicate,
            }
        );
        let report = ingress
            .orchestrate(&[OrchestratedTasteCandidate { preference_id: 42 }])
            .expect("orchestrate");
        assert_eq!(report.outcomes[0].confirmations, 1);
        assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Candidate);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn two_distinct_persisted_evidence_payloads_can_activate() {
        let root = root("activate");
        let mut ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        ingress
            .ingest(
                &message("m-1", 17, 117, 8_000, b"deterministic-a"),
                &sender(),
                "cogno",
            )
            .expect("first");
        ingress
            .ingest(
                &message("m-2", 18, 118, 9_000, b"deterministic-b"),
                &sender(),
                "cogno",
            )
            .expect("second");
        let report = ingress
            .orchestrate(&[OrchestratedTasteCandidate { preference_id: 42 }])
            .expect("orchestrate");
        assert_eq!(report.outcomes[0].confirmations, 2);
        assert_eq!(report.outcomes[0].confidence_bps, 8_500);
        assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Active);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn startup_reconciles_receipt_written_before_validation_store() {
        let root = root("recovery");
        let encoded = message("m-1", 17, 117, 8_500, b"deterministic-a");
        let receipt =
            bridge_authenticated_scirust_exchange(&encoded, &sender(), "cogno").expect("bridge");
        let mut receipts = PersistentSciRustValidationReceiptStore::open(&root).expect("receipts");
        receipts.append(&receipt).expect("append receipt");
        drop(receipts);

        let ingress = PersistentSciRustTasteIngress::open(&root).expect("recover");
        assert_eq!(
            ingress.startup_recovery(),
            SciRustTasteIngressRecovery {
                restored_validations: 1,
                existing_validations: 0,
            }
        );
        assert_eq!(ingress.trusted_validations().expect("trusted").len(), 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn unattested_legacy_deterministic_validation_is_not_trusted() {
        let root = root("legacy");
        let mut validations = PersistentTasteValidationStore::open(&root).expect("validations");
        validations
            .append(StoredTasteValidation {
                validation_id: 90,
                preference_id: 42,
                evidence_id: 190,
                origin: StoredValidationOrigin::DeterministicEvaluation,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 10_000,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            })
            .expect("legacy deterministic");
        validations
            .append(StoredTasteValidation {
                validation_id: 91,
                preference_id: 42,
                evidence_id: 191,
                origin: StoredValidationOrigin::ExplicitUserAction,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 9_000,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            })
            .expect("user validation");
        drop(validations);

        let ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        let trusted = ingress.trusted_validations().expect("trusted");
        assert_eq!(trusted.len(), 1);
        assert_eq!(trusted[0].validation_id, 91);
        assert_eq!(
            trusted[0].origin,
            StoredValidationOrigin::ExplicitUserAction
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn explicit_user_and_scirust_cannot_double_count_same_evidence_id() {
        let root = root("cross-origin-evidence");
        let mut validations = PersistentTasteValidationStore::open(&root).expect("validations");
        validations
            .append(StoredTasteValidation {
                validation_id: 90,
                preference_id: 42,
                evidence_id: 117,
                origin: StoredValidationOrigin::ExplicitUserAction,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 9_000,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            })
            .expect("user validation");
        drop(validations);

        let mut ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        assert!(matches!(
            ingress.ingest(
                &message("m-1", 17, 117, 8_500, b"deterministic-a"),
                &sender(),
                "cogno",
            ),
            Err(SciRustTasteIngressError::DuplicateTrustedEvidence(117))
        ));
        assert_eq!(ingress.attested_validation_count(), 0);
        assert!(ingress.is_healthy());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn conflicting_validation_id_is_rejected_before_receipt_persistence() {
        let root = root("preflight-conflict");
        let mut validations = PersistentTasteValidationStore::open(&root).expect("validations");
        validations
            .append(StoredTasteValidation {
                validation_id: 17,
                preference_id: 42,
                evidence_id: 999,
                origin: StoredValidationOrigin::ExplicitUserAction,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 8_500,
                subject_kind: ValidationSubjectKind::Unattributed,
                subject_id_sha256: [0u8; 32],
            })
            .expect("conflicting validation");
        drop(validations);

        let mut ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        assert!(matches!(
            ingress.ingest(
                &message("m-1", 17, 117, 8_500, b"deterministic-a"),
                &sender(),
                "cogno",
            ),
            Err(SciRustTasteIngressError::ValidationStore(
                TasteValidationStoreError::DuplicateValidationConflict(17)
            ))
        ));
        assert_eq!(ingress.attested_validation_count(), 0);
        assert!(ingress.is_healthy());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn inconclusive_exchange_does_not_enter_either_store() {
        let root = root("inconclusive");
        let mut value: Value =
            serde_json::from_slice(&message("m-1", 17, 117, 8_500, b"deterministic-a"))
                .expect("fixture");
        value["payload"]["verdict"] = json!("inconclusive");
        let encoded = serde_json::to_vec(&value).expect("encoded");
        let mut ingress = PersistentSciRustTasteIngress::open(&root).expect("ingress");
        assert_eq!(
            ingress
                .ingest(&encoded, &sender(), "cogno")
                .expect("ingest"),
            SciRustTasteIngressOutcome::Inconclusive
        );
        assert_eq!(ingress.attested_validation_count(), 0);
        assert!(ingress.trusted_validations().expect("trusted").is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
