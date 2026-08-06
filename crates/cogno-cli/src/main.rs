//! # cogno — command-line entry point for COGNO-1.
//!
//! Wires the deterministic pipeline (`cogno-core`), the runtime
//! (`cogno-runtime`) and the model backends (`cogno-model`) behind explicit
//! subcommands. Every external envelope is validated fail-closed before it can
//! reach persistent state.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{
    CognoProposalView, DataClassification, EvidenceId, MemoryBudget, ObservedDialogueKind,
    ObservedSenderClass, ProposalAction, QueueFullPolicy, Reward, RewardParams, SafetyPolicy,
    TrustClass,
};
use cogno_model::{ModelBackend, OwnedProposal, SimBackend};
use cogno_runtime::{PersistentDialogueStore, Pipeline, PipelineParams, Runtime, RuntimeConfig};
use serde::Deserialize;
use std::io::{self, BufRead};
use std::path::Path;

const MAX_OBSERVATION_JSONL_BYTES: usize = 1_048_576;
const OBSERVATION_SCHEMA_VERSION: u16 = 1;
const OBSERVATION_AUTHORITY: &str = "observation_only";
const OBSERVATION_ACTION: &str = "append_to_cogno_journal_after_runtime_sha256";

#[derive(Debug, Deserialize)]
struct ObservationEnvelope {
    schema_version: u16,
    observation_id: String,
    conversation_id: String,
    message_id: String,
    parent_message_id: Option<String>,
    sender_class: String,
    message_kind: String,
    evidence_id: u64,
    canonical_payload: Vec<u8>,
    authority: String,
    permitted_action: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("help");
    let result = match command {
        "phase" => {
            cmd_phase();
            Ok(())
        }
        "doctor" => {
            cmd_doctor();
            Ok(())
        }
        "validate" => {
            cmd_validate(args.get(2));
            Ok(())
        }
        "simulate" => {
            cmd_simulate();
            Ok(())
        }
        "demo-pipeline" => {
            cmd_demo_pipeline();
            Ok(())
        }
        "replay" => {
            cmd_replay();
            Ok(())
        }
        "observe-jsonl" => cmd_observe_jsonl(args.get(2)),
        "help" | "-h" | "--help" | "" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    };

    if let Err(error) = result {
        eprintln!("cogno: {error}");
        std::process::exit(2);
    }
}

fn print_help() {
    println!("cogno — COGNO-1 command line");
    println!();
    println!("Commands:");
    println!("  phase                    Show implementation phase and meta-objective state");
    println!("  doctor                   Validate budget, KV controller, queue and tool gate");
    println!("  validate <n>             Validate scripted proposals against the strict schema");
    println!("  simulate                 Run the deterministic proposal simulator");
    println!("  demo-pipeline            Run a proposal through the full deterministic pipeline");
    println!("  replay                   Replay a scripted journal and derive a profile");
    println!(
        "  observe-jsonl <root>      Persist validated SciAgent observation envelopes from stdin"
    );
    println!("  help                     Show this help");
}

fn cmd_observe_jsonl(root: Option<&String>) -> Result<(), String> {
    let root = root.ok_or("observe-jsonl requires a storage root")?;
    let mut store = PersistentDialogueStore::open(Path::new(root))
        .map_err(|error| format!("cannot open dialogue store: {error:?}"))?;
    let stdin = io::stdin();
    let mut processed = 0usize;
    let mut appended = 0usize;
    let mut duplicates = 0usize;

    for (index, line_result) in stdin.lock().lines().enumerate() {
        let line_number = index + 1;
        let line =
            line_result.map_err(|error| format!("cannot read line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if line.len() > MAX_OBSERVATION_JSONL_BYTES {
            return Err(format!(
                "line {line_number} exceeds {MAX_OBSERVATION_JSONL_BYTES} bytes"
            ));
        }
        let envelope: ObservationEnvelope = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSON on line {line_number}: {error}"))?;
        validate_observation_envelope(&envelope)
            .map_err(|error| format!("invalid envelope on line {line_number}: {error}"))?;

        let outcome = store
            .append(
                envelope.evidence_id,
                parse_sender_class(&envelope.sender_class)?,
                parse_message_kind(&envelope.message_kind)?,
                &envelope.canonical_payload,
            )
            .map_err(|error| format!("cannot persist line {line_number}: {error:?}"))?;
        processed += 1;
        match outcome {
            cogno_core::DialogueObservationOutcome::Appended { .. } => appended += 1,
            cogno_core::DialogueObservationOutcome::Duplicate { .. } => duplicates += 1,
        }
    }

    println!(
        "observe-jsonl: processed={processed} appended={appended} duplicates={duplicates} total={}",
        store.len()
    );
    println!("store_root={}", store.root().display());
    Ok(())
}

fn validate_observation_envelope(envelope: &ObservationEnvelope) -> Result<(), String> {
    if envelope.schema_version != OBSERVATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema version {}",
            envelope.schema_version
        ));
    }
    for (name, value) in [
        ("observation_id", envelope.observation_id.as_str()),
        ("conversation_id", envelope.conversation_id.as_str()),
        ("message_id", envelope.message_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{name} must not be empty"));
        }
    }
    if envelope
        .parent_message_id
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err("parent_message_id must be absent or non-empty".to_string());
    }
    if envelope.authority != OBSERVATION_AUTHORITY {
        return Err("authority must remain observation_only".to_string());
    }
    if envelope.permitted_action != OBSERVATION_ACTION {
        return Err("permitted_action is not authorized".to_string());
    }
    if envelope.canonical_payload.is_empty() {
        return Err("canonical_payload must not be empty".to_string());
    }
    parse_sender_class(&envelope.sender_class)?;
    parse_message_kind(&envelope.message_kind)?;
    Ok(())
}

fn parse_sender_class(value: &str) -> Result<ObservedSenderClass, String> {
    match value {
        "human" => Ok(ObservedSenderClass::Human),
        "external_model" => Ok(ObservedSenderClass::ExternalModel),
        "sciagent" => Ok(ObservedSenderClass::SciAgent),
        "runtime_discovery" => Ok(ObservedSenderClass::RuntimeDiscovery),
        "deterministic_kernel" => Ok(ObservedSenderClass::DeterministicKernel),
        other => Err(format!("unsupported sender_class `{other}`")),
    }
}

fn parse_message_kind(value: &str) -> Result<ObservedDialogueKind, String> {
    match value {
        "question" => Ok(ObservedDialogueKind::Question),
        "hypothesis" => Ok(ObservedDialogueKind::Hypothesis),
        "critique" => Ok(ObservedDialogueKind::Critique),
        "counterexample" => Ok(ObservedDialogueKind::Counterexample),
        "experiment_request" => Ok(ObservedDialogueKind::ExperimentRequest),
        "experiment_result" => Ok(ObservedDialogueKind::ExperimentResult),
        "preference_observation" => Ok(ObservedDialogueKind::PreferenceObservation),
        "contradiction" => Ok(ObservedDialogueKind::Contradiction),
        "explanation" => Ok(ObservedDialogueKind::Explanation),
        "decision" => Ok(ObservedDialogueKind::Decision),
        other => Err(format!("unsupported message_kind `{other}`")),
    }
}

fn cmd_phase() {
    let runtime = runtime_or_die();
    let report = runtime.report();
    println!("Phase: {}", report.phase);
    println!("Meta-objective active: {}", report.meta_active);
    println!("Tools enabled: {}", report.tools_enabled);
    println!("Admissions: {}", report.admissions);
    println!("Rejections: {}", report.rejections);
    println!("Truncations: {}", report.truncations);
}

fn cmd_doctor() {
    let runtime = runtime_or_die();
    let report = runtime.report();
    let ok = !report.tools_enabled && !report.meta_active;
    println!("doctor: budget ok | kv ok | queue ok | tools gated | meta gated");
    println!(
        "admissions={} rejections={} truncations={}",
        report.admissions, report.rejections, report.truncations
    );
    if ok {
        println!("status: MVP-safe (fail-closed)");
    } else {
        println!("status: non-MVP configuration present — review before release");
    }
}

fn cmd_validate(n_arg: Option<&String>) {
    let count: usize = n_arg
        .and_then(|value| value.parse().ok())
        .unwrap_or(3)
        .min(64);
    let mut simulator = script_simulator(count);
    let mut ok = 0usize;
    let mut rejected = 0usize;
    while let Ok(proposal) = simulator.next_proposal() {
        match cogno_core::validate_proposal(&proposal.as_view()) {
            Ok(()) => ok += 1,
            Err(reason) => {
                rejected += 1;
                println!("reject stage=structural reason={reason:?}");
            }
        }
    }
    println!("validate: ok={ok} rejected={rejected}");
}

fn cmd_simulate() {
    let mut simulator = script_simulator(4);
    println!("Phase 1 simulator — scripted proposals");
    while let Ok(proposal) = simulator.next_proposal() {
        let validation = cogno_core::validate_proposal(&proposal.as_view());
        println!(
            "prop action={:?} category={:?} confidence_bps={} validated={}",
            proposal.action,
            proposal.category,
            proposal.confidence_bps,
            validation.is_ok()
        );
    }
    println!("(exhausted -> BackendError::Exhausted, fail-closed)");
}

fn cmd_demo_pipeline() {
    let mut runtime = runtime_or_die();
    let evidence = [EvidenceId::from_u64(1)];
    let proposal = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: cogno_core::RuleCategory::Behavior,
        scope: cogno_core::RuleScope::Session,
        confidence_bps: 5_000,
        evidence_ids: &evidence,
        payload: b"demo",
    };
    let params = PipelineParams {
        policy: SafetyPolicy::MVP,
        reward_params: RewardParams::DEFAULT,
        data_class: DataClassification::Internal,
        trust: TrustClass::AuthenticatedUser,
        reward_signal: Reward::ZERO,
        proposal: Some(proposal),
    };
    let outcome = runtime.run_pipeline(params);
    println!("pipeline: {outcome:?}");
}

fn cmd_replay() {
    use cogno_core::{
        DerivationPolicy, EvidenceOrigin, Fingerprint, InputOrigin, Journal, JournalEvent,
    };
    let mut journal = Journal::default();
    for index in 0..3u8 {
        let mut fingerprint = [0u8; 32];
        fingerprint[0] = index + 1;
        journal.append(JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fingerprint),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin: if index == 2 {
                EvidenceOrigin::FormalValidator
            } else {
                EvidenceOrigin::ExplicitUserApproval
            },
            evidence_id: EvidenceId::from_u64(10 + u64::from(index)),
            category_tag: 7,
            payload: Vec::new(),
        });
    }
    let profile = cogno_core::Profile::derive(&journal, DerivationPolicy::DEFAULT);
    println!("journal events: {}", journal.len());
    println!("derived profile active_count: {}", profile.active_count());
    println!("derived profile hard_count: {}", profile.hard_count());
    println!("replay: deterministic — same journal => same profile");
}

fn runtime_or_die() -> Runtime {
    let budget = MemoryBudget::try_new(
        64 * 1024 * 1024,
        20 * 1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        8 * 1024 * 1024,
        4 * 1024 * 1024,
        1024 * 1024,
        1024 * 1024,
        16 * 1024,
        16 * 1024,
        2048,
        1024,
        8,
        16,
        4,
        64,
    )
    .expect("hardcoded MVP budget must validate");
    let config = RuntimeConfig {
        budget,
        reserved_bytes: 0,
        kv_capacity_tokens: 2048,
        kv_policy: cogno_core::KvCachePolicy::RejectOnOverflow,
        queue_capacity: 64,
        queue_policy: QueueFullPolicy::RejectNewest,
    };
    Runtime::try_new(config).expect("runtime construction must succeed with a valid MVP budget")
}

fn script_simulator(count: usize) -> SimBackend {
    let actions = [
        ProposalAction::ClassifyFeedback,
        ProposalAction::ExtractPreference,
        ProposalAction::RankCandidates,
        ProposalAction::FlagContradiction,
    ];
    let categories = [
        cogno_core::RuleCategory::Style,
        cogno_core::RuleCategory::Behavior,
        cogno_core::RuleCategory::Format,
        cogno_core::RuleCategory::Safety,
    ];
    let mut script = Vec::with_capacity(count);
    for index in 0..count {
        let evidence_ids = if index == 1 {
            vec![EvidenceId::from_u64(1), EvidenceId::from_u64(2)]
        } else {
            vec![EvidenceId::from_u64(1)]
        };
        script.push(OwnedProposal {
            action: actions[index % actions.len()],
            category: categories[index % categories.len()],
            scope: cogno_core::RuleScope::Session,
            confidence_bps: (1_000 + index as u16 * 500).min(10_000),
            evidence_ids,
            payload: vec![index as u8],
        });
    }
    SimBackend::new(script)
}

#[allow(dead_code)]
fn _unused_pipeline_marker(_pipeline: &Pipeline) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> ObservationEnvelope {
        ObservationEnvelope {
            schema_version: 1,
            observation_id: "observation-1".to_string(),
            conversation_id: "conversation-1".to_string(),
            message_id: "message-1".to_string(),
            parent_message_id: None,
            sender_class: "sciagent".to_string(),
            message_kind: "hypothesis".to_string(),
            evidence_id: 1,
            canonical_payload: br#"{"message_id":"message-1"}"#.to_vec(),
            authority: OBSERVATION_AUTHORITY.to_string(),
            permitted_action: OBSERVATION_ACTION.to_string(),
        }
    }

    #[test]
    fn valid_observation_envelope_is_accepted() {
        assert!(validate_observation_envelope(&envelope()).is_ok());
    }

    #[test]
    fn elevated_authority_is_rejected() {
        let mut item = envelope();
        item.authority = "trusted_policy".to_string();
        assert!(validate_observation_envelope(&item).is_err());
    }

    #[test]
    fn unknown_sender_is_rejected() {
        let mut item = envelope();
        item.sender_class = "unknown".to_string();
        assert!(validate_observation_envelope(&item).is_err());
    }
}
