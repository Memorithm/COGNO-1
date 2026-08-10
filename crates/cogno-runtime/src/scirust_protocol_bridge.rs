//! Authenticated SciRust agent-protocol boundary for COGNO deterministic evidence.
//!
//! This module consumes the protocol crate's own authenticated-sender validation
//! and self-checking execution attestation. It never accepts a caller-supplied
//! trust boolean or raw execution digest as authority.

use scirust_agent_protocol::{AgentIdentity, AgentMessage, ProtocolError};

use crate::scientific_exchange::HostSciRustExecutionAttestation;

/// Failure while converting an authenticated SciRust protocol message into
/// COGNO's host-held deterministic-execution authority token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SciRustProtocolBridgeError {
    /// The SciRust protocol rejected sender identity, sender policy, or the
    /// execution attestation itself.
    Protocol(ProtocolError),
    /// The authenticated message carried no execution attestation.
    MissingExecutionAttestation,
    /// The verified protocol digest was not decodable as 32 bytes. This should
    /// be unreachable after protocol validation and therefore fails closed.
    InvalidExecutionProfileDigest,
}

impl From<ProtocolError> for SciRustProtocolBridgeError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

/// Verify one SciRust protocol message against an independently authenticated
/// sender and mint COGNO's host-only execution authority from the exact verified
/// execution-profile fingerprint.
///
/// `AgentMessage::validate_with_authenticated_sender` is authoritative here: it
/// requires the authenticated identity to equal the serialized sender, permits
/// execution attestations only for `RuntimeDiscovery` / `DeterministicKernel`,
/// and recomputes the execution-attestation fingerprint before returning.
pub fn attest_authenticated_scirust_message(
    message: &AgentMessage,
    authenticated_sender: &AgentIdentity,
) -> Result<HostSciRustExecutionAttestation, SciRustProtocolBridgeError> {
    message.validate_with_authenticated_sender(authenticated_sender)?;
    let attestation = message
        .execution_attestation
        .as_ref()
        .ok_or(SciRustProtocolBridgeError::MissingExecutionAttestation)?;
    let digest = decode_sha256(attestation.profile_sha256.as_str())
        .ok_or(SciRustProtocolBridgeError::InvalidExecutionProfileDigest)?;
    Ok(HostSciRustExecutionAttestation::attest_verified_execution(
        digest,
    ))
}

fn decode_sha256(encoded: &str) -> Option<[u8; 32]> {
    if encoded.len() != 64 {
        return None;
    }
    let mut output = [0u8; 32];
    let bytes = encoded.as_bytes();
    for index in 0..32 {
        let high = decode_hex_nibble(bytes[index * 2])?;
        let low = decode_hex_nibble(bytes[index * 2 + 1])?;
        output[index] = (high << 4) | low;
    }
    Some(output)
}

const fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use scirust_agent_protocol::{
        AgentKind, ExecutionArchitecture, ExecutionArchitectureFamily, ExecutionAttestation,
        ExecutionBackendKind, ExecutionProfile, ExecutionReproducibility, MessageKind,
        Sha256Digest, TrustClass, EXECUTION_PROFILE_SCHEMA_VERSION, SCHEMA_VERSION,
    };
    use serde_json::json;

    use super::*;

    fn identity(kind: AgentKind, id: &str) -> AgentIdentity {
        AgentIdentity {
            kind,
            id: id.to_string(),
        }
    }

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn attestation() -> ExecutionAttestation {
        ExecutionAttestation::new(ExecutionProfile {
            schema_version: EXECUTION_PROFILE_SCHEMA_VERSION,
            backend: ExecutionBackendKind::Cuda,
            device_ordinal: 0,
            architecture: ExecutionArchitecture {
                family: ExecutionArchitectureFamily::NvidiaGpu,
                name: Some("sm_110".to_string()),
            },
            capability_profile_sha256: digest(0x11),
            topology_profile_sha256: digest(0x22),
            memory_budget_bytes: Some(1024),
            numeric_mode: "bf16-fp32-accum-v1".to_string(),
            reproducibility: ExecutionReproducibility::NumericallyEquivalent,
            kernel_semantic_version: "route-b-v1".to_string(),
            sampler_semantic_version: Some("sampler-v1".to_string()),
            model_sha256: digest(0x33),
            tokenizer_sha256: digest(0x44),
        })
        .unwrap()
    }

    fn message(sender: AgentIdentity) -> AgentMessage {
        AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: "m-1".to_string(),
            conversation_id: "c-1".to_string(),
            parent_message_id: None,
            sender,
            recipients: vec![identity(AgentKind::CognoObserver, "cogno")],
            message_kind: MessageKind::ExperimentResult,
            trust_class: TrustClass::DeterministicKernelOutput,
            confidence_bps: 10_000,
            evidence: Vec::new(),
            payload: json!({"scientific_exchange": "verified"}),
            requested_capabilities: Vec::new(),
            execution_attestation: Some(attestation()),
        }
    }

    #[test]
    fn authenticated_runtime_message_mints_exact_host_digest() {
        let sender = identity(AgentKind::RuntimeDiscovery, "runtime-0");
        let message = message(sender.clone());
        let expected = decode_sha256(
            message
                .execution_attestation
                .as_ref()
                .unwrap()
                .profile_sha256
                .as_str(),
        )
        .unwrap();
        let host = attest_authenticated_scirust_message(&message, &sender).unwrap();
        assert_eq!(host.execution_profile_sha256(), expected);
    }

    #[test]
    fn self_declared_sender_cannot_mint_host_authority() {
        let sender = identity(AgentKind::RuntimeDiscovery, "runtime-0");
        let message = message(sender);
        let impostor = identity(AgentKind::RuntimeDiscovery, "runtime-1");
        assert!(matches!(
            attest_authenticated_scirust_message(&message, &impostor),
            Err(SciRustProtocolBridgeError::Protocol(
                ProtocolError::AuthenticatedSenderMismatch
            ))
        ));
    }

    #[test]
    fn message_without_execution_attestation_fails_closed() {
        let sender = identity(AgentKind::DeterministicKernel, "kernel-0");
        let mut message = message(sender.clone());
        message.execution_attestation = None;
        assert_eq!(
            attest_authenticated_scirust_message(&message, &sender),
            Err(SciRustProtocolBridgeError::MissingExecutionAttestation)
        );
    }
}
