//! Build and persist a deterministic summary of `dialogue.events`.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::DialogueSnapshot;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
struct SnapshotReport {
    schema_version: u16,
    events: u64,
    payload_bytes: u64,
    sender_counts: SenderCounts,
    message_kind_counts: MessageKindCounts,
    event_log_sha256: String,
    snapshot_path: String,
    authority: &'static str,
}

#[derive(Debug, Serialize)]
struct SenderCounts {
    human: u64,
    external_model: u64,
    sciagent: u64,
    runtime_discovery: u64,
    deterministic_kernel: u64,
}

#[derive(Debug, Serialize)]
struct MessageKindCounts {
    question: u64,
    hypothesis: u64,
    critique: u64,
    counterexample: u64,
    experiment_request: u64,
    experiment_result: u64,
    preference_observation: u64,
    contradiction: u64,
    explanation: u64,
    decision: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-dialogue-snapshot: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: cogno-dialogue-snapshot ROOT")?;
    let snapshot = DialogueSnapshot::derive(Path::new(&root))
        .map_err(|error| format!("cannot derive snapshot: {error:?}"))?;
    let path = snapshot
        .write_atomic(Path::new(&root))
        .map_err(|error| format!("cannot write snapshot: {error:?}"))?;
    let report = report(&snapshot, path.display().to_string());
    let encoded = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot encode report: {error}"))?;
    println!("{encoded}");
    Ok(())
}

fn report(snapshot: &DialogueSnapshot, snapshot_path: String) -> SnapshotReport {
    SnapshotReport {
        schema_version: snapshot.schema_version,
        events: snapshot.events,
        payload_bytes: snapshot.payload_bytes,
        sender_counts: SenderCounts {
            human: snapshot.sender_counts[0],
            external_model: snapshot.sender_counts[1],
            sciagent: snapshot.sender_counts[2],
            runtime_discovery: snapshot.sender_counts[3],
            deterministic_kernel: snapshot.sender_counts[4],
        },
        message_kind_counts: MessageKindCounts {
            question: snapshot.message_kind_counts[0],
            hypothesis: snapshot.message_kind_counts[1],
            critique: snapshot.message_kind_counts[2],
            counterexample: snapshot.message_kind_counts[3],
            experiment_request: snapshot.message_kind_counts[4],
            experiment_result: snapshot.message_kind_counts[5],
            preference_observation: snapshot.message_kind_counts[6],
            contradiction: snapshot.message_kind_counts[7],
            explanation: snapshot.message_kind_counts[8],
            decision: snapshot.message_kind_counts[9],
        },
        event_log_sha256: hex(&snapshot.event_log_sha256),
        snapshot_path,
        authority: "summary_only",
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
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
    fn report_remains_summary_only() {
        let snapshot = DialogueSnapshot {
            schema_version: 1,
            events: 2,
            payload_bytes: 10,
            sender_counts: [0, 0, 1, 1, 0],
            message_kind_counts: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            event_log_sha256: [0xab; 32],
        };
        let report = report(&snapshot, "snapshot.bin".to_string());
        assert_eq!(report.authority, "summary_only");
        assert_eq!(report.sender_counts.sciagent, 1);
        assert_eq!(report.message_kind_counts.experiment_result, 1);
        assert_eq!(report.event_log_sha256.len(), 64);
    }
}
