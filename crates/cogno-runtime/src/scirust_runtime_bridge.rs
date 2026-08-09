//! Independent COGNO consumer for authenticated SciRust runtime messages.
//!
//! COGNO deliberately does not add a Git dependency on the SciRust workspace.
//! This module therefore consumes the stable SciRust agent-protocol v1 wire
//! contract directly and re-computes the execution-profile fingerprint before
//! minting any local deterministic-evaluation authority.
//!
//! Sender authenticity is intentionally separate from JSON parsing: callers
//! must supply a host-held transport identity that was authenticated outside the
//! message itself. The serialized sender must then match that identity exactly.

use crate::scientific_exchange::{
    classify_attested_scientific_exchange, HostSciRustExecutionAttestation,
    ScientificExchangeDisposition, ScientificExchangeError, ScientificExchangeOrigin,
    ScientificExchangeRecord, ScientificExchangeVerdict,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SCIRUST_AGENT_SCHEMA_VERSION: u16 = 1;
const SCIRUST_EXECUTION_PROFILE_SCHEMA_VERSION: u16 = 1;
const SCIRUST_EXCHANGE_PAYLOAD_SCHEMA_VERSION: u16 = 1;
const EXECUTION_PROFILE_HASH_DOMAIN: &[u8] = b"scirust.execution-profile.v1\0";
const MAX_SEMANTIC_TEXT_BYTES: usize = 128;

/// Maximum encoded runtime message accepted before JSON allocation/parsing.
pub const MAX_SCIRUST_RUNTIME_MESSAGE_BYTES: usize = 512 * 1024;

/// Runtime identities that the host transport may authenticate for this bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SciRustRuntimeKind {
    RuntimeDiscovery,
    DeterministicKernel,
}

impl SciRustRuntimeKind {
    const fn wire_name(self) -> &'static str {
        match self {
            Self::RuntimeDiscovery => "runtime_discovery",
            Self::DeterministicKernel => "deterministic_kernel",
        }
    }
}

/// Host-held identity established by the transport/runtime boundary rather than
/// by the SciRust JSON message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostAuthenticatedSciRustSender {
    kind: SciRustRuntimeKind,
    id: String,
    _private: (),
}

impl HostAuthenticatedSciRustSender {
    /// Construct a host token after authenticating one RuntimeDiscovery sender.
    pub fn runtime_discovery(id: impl Into<String>) -> Result<Self, SciRustRuntimeBridgeError> {
        Self::new(SciRustRuntimeKind::RuntimeDiscovery, id.into())
    }

    /// Construct a host token after authenticating one DeterministicKernel sender.
    pub fn deterministic_kernel(id: impl Into<String>) -> Result<Self, SciRustRuntimeBridgeError> {
        Self::new(SciRustRuntimeKind::DeterministicKernel, id.into())
    }

    fn new(kind: SciRustRuntimeKind, id: String) -> Result<Self, SciRustRuntimeBridgeError> {
        if id.trim().is_empty() {
            return Err(SciRustRuntimeBridgeError::InvalidHostSender);
        }
        Ok(Self {
            kind,
            id,
            _private: (),
        })
    }

    #[must_use]
    pub const fn kind(&self) -> SciRustRuntimeKind {
        self.kind
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Auditable result of one fully bridged SciRust scientific exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SciRustExchangeBridgeReceipt {
    message_id: String,
    conversation_id: String,
    sender_kind: SciRustRuntimeKind,
    sender_id: String,
    execution_profile_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    disposition: ScientificExchangeDisposition,
}

impl SciRustExchangeBridgeReceipt {
    #[must_use]
    pub fn message_id(&self) -> &str {
        &self.message_id
    }

    #[must_use]
    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    #[must_use]
    pub const fn sender_kind(&self) -> SciRustRuntimeKind {
        self.sender_kind
    }

    #[must_use]
    pub fn sender_id(&self) -> &str {
        &self.sender_id
    }

    #[must_use]
    pub const fn execution_profile_sha256(&self) -> [u8; 32] {
        self.execution_profile_sha256
    }

    #[must_use]
    pub const fn payload_sha256(&self) -> [u8; 32] {
        self.payload_sha256
    }

    #[must_use]
    pub const fn disposition(&self) -> ScientificExchangeDisposition {
        self.disposition
    }
}

/// Fail-closed rejection from the SciRust authenticated runtime bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SciRustRuntimeBridgeError {
    MessageTooLarge,
    MalformedJson,
    UnsupportedMessageSchema(u16),
    InvalidEnvelope,
    InvalidHostSender,
    AuthenticatedSenderMismatch,
    RecipientMismatch,
    MissingExecutionAttestation,
    InvalidExecutionProfile,
    ExecutionProfileDigestMismatch,
    UnsupportedPayloadSchema(u16),
    InvalidPayload,
    Exchange(ScientificExchangeError),
}

/// Consume one authenticated SciRust RuntimeEndpoint experiment result and, only
/// after full v1 envelope/profile verification, classify its embedded scientific
/// exchange through COGNO's deterministic-evaluation boundary.
///
/// `authenticated_sender` must come from the host transport, not from JSON. The
/// payload intentionally contains no origin field: this function assigns
/// `SciRustDeterministicEvaluator` only after the message and execution profile
/// have been verified.
pub fn bridge_authenticated_scirust_exchange(
    encoded_message: &[u8],
    authenticated_sender: &HostAuthenticatedSciRustSender,
    expected_cogno_recipient_id: &str,
) -> Result<SciRustExchangeBridgeReceipt, SciRustRuntimeBridgeError> {
    if encoded_message.len() > MAX_SCIRUST_RUNTIME_MESSAGE_BYTES {
        return Err(SciRustRuntimeBridgeError::MessageTooLarge);
    }
    if expected_cogno_recipient_id.trim().is_empty() {
        return Err(SciRustRuntimeBridgeError::RecipientMismatch);
    }

    let message: WireAgentMessage = serde_json::from_slice(encoded_message)
        .map_err(|_| SciRustRuntimeBridgeError::MalformedJson)?;
    validate_envelope(&message, authenticated_sender, expected_cogno_recipient_id)?;

    let attestation = message
        .execution_attestation
        .as_ref()
        .ok_or(SciRustRuntimeBridgeError::MissingExecutionAttestation)?;
    let execution_profile_sha256 = verify_execution_attestation(attestation)?;

    if message.payload.schema_version != SCIRUST_EXCHANGE_PAYLOAD_SCHEMA_VERSION {
        return Err(SciRustRuntimeBridgeError::UnsupportedPayloadSchema(
            message.payload.schema_version,
        ));
    }

    let verdict = match message.payload.verdict.as_str() {
        "confirmed" => ScientificExchangeVerdict::Confirmed,
        "contradicted" => ScientificExchangeVerdict::Contradicted,
        "inconclusive" => ScientificExchangeVerdict::Inconclusive,
        _ => return Err(SciRustRuntimeBridgeError::InvalidPayload),
    };
    let record = ScientificExchangeRecord {
        observation_id: message.payload.observation_id,
        preference_id: message.payload.preference_id,
        evidence_id: message.payload.evidence_id,
        origin: ScientificExchangeOrigin::SciRustDeterministicEvaluator,
        verdict,
        confidence_bps: message.payload.confidence_bps,
        canonical_payload: message.payload.canonical_payload.clone(),
    };
    let payload_sha256 = record.payload_sha256();
    let host_attestation =
        HostSciRustExecutionAttestation::attest_verified_execution(execution_profile_sha256);
    let disposition = classify_attested_scientific_exchange(&record, host_attestation)
        .map_err(SciRustRuntimeBridgeError::Exchange)?;

    Ok(SciRustExchangeBridgeReceipt {
        message_id: message.message_id,
        conversation_id: message.conversation_id,
        sender_kind: authenticated_sender.kind,
        sender_id: authenticated_sender.id.clone(),
        execution_profile_sha256,
        payload_sha256,
        disposition,
    })
}

fn validate_envelope(
    message: &WireAgentMessage,
    authenticated_sender: &HostAuthenticatedSciRustSender,
    expected_cogno_recipient_id: &str,
) -> Result<(), SciRustRuntimeBridgeError> {
    if message.schema_version != SCIRUST_AGENT_SCHEMA_VERSION {
        return Err(SciRustRuntimeBridgeError::UnsupportedMessageSchema(
            message.schema_version,
        ));
    }
    if message.message_id.trim().is_empty()
        || message.conversation_id.trim().is_empty()
        || message.sender.id.trim().is_empty()
        || message.message_kind != "experiment_result"
        || message.trust_class != "deterministic_kernel_output"
        || message.confidence_bps != 10_000
        || !message.evidence.is_empty()
        || !message.requested_capabilities.is_empty()
    {
        return Err(SciRustRuntimeBridgeError::InvalidEnvelope);
    }
    if message.sender.kind != authenticated_sender.kind.wire_name()
        || message.sender.id != authenticated_sender.id
    {
        return Err(SciRustRuntimeBridgeError::AuthenticatedSenderMismatch);
    }
    if message.recipients.len() != 1
        || message.recipients[0].kind != "cogno_communicator"
        || message.recipients[0].id != expected_cogno_recipient_id
    {
        return Err(SciRustRuntimeBridgeError::RecipientMismatch);
    }
    Ok(())
}

fn verify_execution_attestation(
    attestation: &WireExecutionAttestation,
) -> Result<[u8; 32], SciRustRuntimeBridgeError> {
    validate_execution_profile(&attestation.profile)?;
    let claimed = decode_lower_hex_sha256(&attestation.profile_sha256)
        .ok_or(SciRustRuntimeBridgeError::InvalidExecutionProfile)?;
    let canonical = canonical_execution_profile_bytes(&attestation.profile)?;
    let actual: [u8; 32] = Sha256::digest(canonical).into();
    if actual != claimed {
        return Err(SciRustRuntimeBridgeError::ExecutionProfileDigestMismatch);
    }
    Ok(actual)
}

fn validate_execution_profile(
    profile: &WireExecutionProfile,
) -> Result<(), SciRustRuntimeBridgeError> {
    if profile.schema_version != SCIRUST_EXECUTION_PROFILE_SCHEMA_VERSION
        || backend_tag(&profile.backend).is_none()
        || architecture_tag(&profile.architecture.family).is_none()
        || reproducibility_tag(&profile.reproducibility).is_none()
        || !is_lower_hex_sha256(&profile.capability_profile_sha256)
        || !is_lower_hex_sha256(&profile.topology_profile_sha256)
        || !is_lower_hex_sha256(&profile.model_sha256)
        || !is_lower_hex_sha256(&profile.tokenizer_sha256)
        || !is_valid_semantic_id(&profile.numeric_mode)
        || !is_valid_semantic_id(&profile.kernel_semantic_version)
        || profile
            .sampler_semantic_version
            .as_deref()
            .is_some_and(|value| !is_valid_semantic_id(value))
        || profile
            .architecture
            .name
            .as_deref()
            .is_some_and(|value| !is_valid_architecture_name(value))
        || (profile.architecture.family == "other" && profile.architecture.name.is_none())
    {
        return Err(SciRustRuntimeBridgeError::InvalidExecutionProfile);
    }
    Ok(())
}

fn canonical_execution_profile_bytes(
    profile: &WireExecutionProfile,
) -> Result<Vec<u8>, SciRustRuntimeBridgeError> {
    let backend =
        backend_tag(&profile.backend).ok_or(SciRustRuntimeBridgeError::InvalidExecutionProfile)?;
    let architecture = architecture_tag(&profile.architecture.family)
        .ok_or(SciRustRuntimeBridgeError::InvalidExecutionProfile)?;
    let reproducibility = reproducibility_tag(&profile.reproducibility)
        .ok_or(SciRustRuntimeBridgeError::InvalidExecutionProfile)?;

    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(EXECUTION_PROFILE_HASH_DOMAIN);
    out.extend_from_slice(&profile.schema_version.to_le_bytes());
    out.push(backend);
    out.extend_from_slice(&profile.device_ordinal.to_le_bytes());
    out.push(architecture);
    put_optional_text(&mut out, profile.architecture.name.as_deref())?;
    put_text(&mut out, &profile.capability_profile_sha256)?;
    put_text(&mut out, &profile.topology_profile_sha256)?;
    match profile.memory_budget_bytes {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
        None => out.push(0),
    }
    put_text(&mut out, &profile.numeric_mode)?;
    out.push(reproducibility);
    put_text(&mut out, &profile.kernel_semantic_version)?;
    put_optional_text(&mut out, profile.sampler_semantic_version.as_deref())?;
    put_text(&mut out, &profile.model_sha256)?;
    put_text(&mut out, &profile.tokenizer_sha256)?;
    Ok(out)
}

fn put_text(out: &mut Vec<u8>, value: &str) -> Result<(), SciRustRuntimeBridgeError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SciRustRuntimeBridgeError::InvalidExecutionProfile)?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_optional_text(
    out: &mut Vec<u8>,
    value: Option<&str>,
) -> Result<(), SciRustRuntimeBridgeError> {
    match value {
        Some(value) => {
            out.push(1);
            put_text(out, value)?;
        }
        None => out.push(0),
    }
    Ok(())
}

fn backend_tag(value: &str) -> Option<u8> {
    match value {
        "reference" => Some(0),
        "cpu" => Some(1),
        "wgpu" => Some(2),
        "cuda" => Some(3),
        _ => None,
    }
}

fn architecture_tag(value: &str) -> Option<u8> {
    match value {
        "unknown" => Some(0),
        "x86_64" => Some(1),
        "aarch64" => Some(2),
        "risc_v64" => Some(3),
        "loong_arch64" => Some(4),
        "wasm32" => Some(5),
        "nvidia_gpu" => Some(6),
        "amd_gpu" => Some(7),
        "intel_gpu" => Some(8),
        "apple_gpu" => Some(9),
        "other" => Some(10),
        _ => None,
    }
}

fn reproducibility_tag(value: &str) -> Option<u8> {
    match value {
        "unknown" => Some(0),
        "bit_exact" => Some(1),
        "deterministic" => Some(2),
        "numerically_equivalent" => Some(3),
        "fast_approximate" => Some(4),
        _ => None,
    }
}

fn is_valid_semantic_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_SEMANTIC_TEXT_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+' | b':' | b'/')
        })
}

fn is_valid_architecture_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_SEMANTIC_TEXT_BYTES
        && value.trim() == value
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ')
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_lower_hex_sha256(value: &str) -> Option<[u8; 32]> {
    if !is_lower_hex_sha256(value) {
        return None;
    }
    let bytes = value.as_bytes();
    let mut out = [0_u8; 32];
    for index in 0..32 {
        out[index] = (hex_nibble(bytes[index * 2])? << 4) | hex_nibble(bytes[index * 2 + 1])?;
    }
    Some(out)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct WireAgentIdentity {
    kind: String,
    id: String,
}

#[derive(Debug, Deserialize)]
struct WireAgentMessage {
    schema_version: u16,
    message_id: String,
    conversation_id: String,
    sender: WireAgentIdentity,
    recipients: Vec<WireAgentIdentity>,
    message_kind: String,
    trust_class: String,
    confidence_bps: u16,
    evidence: Vec<Value>,
    payload: WireScientificExchangePayload,
    requested_capabilities: Vec<Value>,
    execution_attestation: Option<WireExecutionAttestation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireScientificExchangePayload {
    schema_version: u16,
    observation_id: u64,
    preference_id: u64,
    evidence_id: u64,
    verdict: String,
    confidence_bps: u16,
    canonical_payload: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct WireExecutionAttestation {
    profile: WireExecutionProfile,
    profile_sha256: String,
}

#[derive(Debug, Deserialize)]
struct WireExecutionProfile {
    schema_version: u16,
    backend: String,
    device_ordinal: u32,
    architecture: WireExecutionArchitecture,
    capability_profile_sha256: String,
    topology_profile_sha256: String,
    memory_budget_bytes: Option<u64>,
    numeric_mode: String,
    reproducibility: String,
    kernel_semantic_version: String,
    sampler_semantic_version: Option<String>,
    model_sha256: String,
    tokenizer_sha256: String,
}

#[derive(Debug, Deserialize)]
struct WireExecutionArchitecture {
    family: String,
    name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StoredValidationOrigin, StoredValidationVerdict};
    use serde_json::json;

    const EXPECTED_PROFILE_SHA256: &str =
        "c984c0151e84300875c2aead5764d018f9ef5d09d218ab8f8f1ea9ab7157bec8";

    fn valid_message() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "message_id": "m-runtime-1",
            "conversation_id": "c-1",
            "parent_message_id": "request-1",
            "sender": {
                "kind": "deterministic_kernel",
                "id": "cuda-decode-local"
            },
            "recipients": [{
                "kind": "cogno_communicator",
                "id": "cogno"
            }],
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
                    "architecture": {
                        "family": "nvidia_gpu",
                        "name": "sm_110"
                    },
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
                "profile_sha256": EXPECTED_PROFILE_SHA256
            }
        }))
        .expect("json fixture")
    }

    #[test]
    fn authenticated_runtime_message_becomes_deterministic_validation() {
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        let receipt = bridge_authenticated_scirust_exchange(&valid_message(), &sender, "cogno")
            .expect("bridge");

        assert_eq!(receipt.message_id(), "m-runtime-1");
        assert_eq!(receipt.conversation_id(), "c-1");
        assert_eq!(
            receipt.sender_kind(),
            SciRustRuntimeKind::DeterministicKernel
        );
        assert_eq!(receipt.sender_id(), "cuda-decode-local");
        assert_eq!(
            receipt.execution_profile_sha256(),
            decode_lower_hex_sha256(EXPECTED_PROFILE_SHA256).expect("fixture digest")
        );
        assert_eq!(
            receipt.disposition(),
            ScientificExchangeDisposition::Validation(crate::StoredTasteValidation {
                validation_id: 17,
                preference_id: 42,
                evidence_id: 117,
                origin: StoredValidationOrigin::DeterministicEvaluation,
                verdict: StoredValidationVerdict::Confirmed,
                confidence_bps: 8_500,
            })
        );
    }

    #[test]
    fn wire_sender_cannot_replace_transport_authenticated_sender() {
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("different-runtime")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&valid_message(), &sender, "cogno"),
            Err(SciRustRuntimeBridgeError::AuthenticatedSenderMismatch)
        );
    }

    #[test]
    fn profile_mutation_with_stale_digest_fails_closed() {
        let mut value: Value = serde_json::from_slice(&valid_message()).expect("fixture");
        value["execution_attestation"]["profile"]["device_ordinal"] = json!(1);
        let encoded = serde_json::to_vec(&value).expect("tampered json");
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&encoded, &sender, "cogno"),
            Err(SciRustRuntimeBridgeError::ExecutionProfileDigestMismatch)
        );
    }

    #[test]
    fn payload_cannot_claim_an_origin() {
        let mut value: Value = serde_json::from_slice(&valid_message()).expect("fixture");
        value["payload"]["origin"] = json!("explicit_user_action");
        let encoded = serde_json::to_vec(&value).expect("tampered json");
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&encoded, &sender, "cogno"),
            Err(SciRustRuntimeBridgeError::MalformedJson)
        );
    }

    #[test]
    fn missing_attestation_fails_closed() {
        let mut value: Value = serde_json::from_slice(&valid_message()).expect("fixture");
        value
            .as_object_mut()
            .expect("object")
            .remove("execution_attestation");
        let encoded = serde_json::to_vec(&value).expect("tampered json");
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&encoded, &sender, "cogno"),
            Err(SciRustRuntimeBridgeError::MissingExecutionAttestation)
        );
    }

    #[test]
    fn wrong_cogno_recipient_fails_closed() {
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&valid_message(), &sender, "other-cogno"),
            Err(SciRustRuntimeBridgeError::RecipientMismatch)
        );
    }

    #[test]
    fn oversized_message_is_rejected_before_parsing() {
        let encoded = vec![b' '; MAX_SCIRUST_RUNTIME_MESSAGE_BYTES + 1];
        let sender = HostAuthenticatedSciRustSender::deterministic_kernel("cuda-decode-local")
            .expect("host sender");
        assert_eq!(
            bridge_authenticated_scirust_exchange(&encoded, &sender, "cogno"),
            Err(SciRustRuntimeBridgeError::MessageTooLarge)
        );
    }
}
