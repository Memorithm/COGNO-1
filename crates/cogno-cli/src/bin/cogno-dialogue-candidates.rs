//! Derive non-authoritative, quarantined preference candidates from dialogue.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{CandidateOrigin, DialogueCandidateReport};
use serde_json::{json, Value};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-dialogue-candidates: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: cogno-dialogue-candidates ROOT")?;
    let report = DialogueCandidateReport::derive(Path::new(&root))
        .map_err(|error| format!("cannot derive candidates: {error:?}"))?;
    let candidates: Vec<Value> = report
        .candidates
        .iter()
        .map(|candidate| {
            json!({
                "preference_id": candidate.preference_id,
                "evidence_id": candidate.evidence_id,
                "origin": origin_name(candidate.origin),
                "payload_sha256": hex(&candidate.payload_sha256),
                "payload_bytes": candidate.payload_bytes,
                "state": "quarantined",
                "can_activate": false
            })
        })
        .collect();
    let output = json!({
        "schema_version": report.schema_version,
        "records_scanned": report.records_scanned,
        "preference_observations": report.preference_observations,
        "candidates": candidates,
        "authority": report.authority,
        "promotion_rule": "requires_non_model_validation_events"
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize report: {error}"))?
    );
    Ok(())
}

const fn origin_name(origin: CandidateOrigin) -> &'static str {
    match origin {
        CandidateOrigin::HumanObservation => "human_observation",
        CandidateOrigin::ModelInference => "model_inference",
        CandidateOrigin::RuntimeObservation => "runtime_observation",
        CandidateOrigin::DeterministicKernelObservation => "deterministic_kernel_observation",
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_origin_is_named_without_authority() {
        assert_eq!(
            origin_name(CandidateOrigin::ModelInference),
            "model_inference"
        );
    }

    #[test]
    fn digest_hex_is_stable() {
        assert_eq!(hex(&[0x00, 0xab, 0xff]), "00abff");
    }
}
