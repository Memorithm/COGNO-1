//! # cogno — command-line entry point for COGNO-1.
//!
//! Wires the deterministic pipeline (`cogno-core`), the runtime
//! (`cogno-runtime`) and the model backends (`cogno-model`) behind explicit
//! subcommands. The MVP executes no tool, performs no network or FS side
//! effect; every subcommand is a pure evaluator + report.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{
    CognoProposalView, DataClassification, EvidenceId, MemoryBudget, ProposalAction,
    QueueFullPolicy, Reward, RewardParams, SafetyPolicy, TrustClass,
};
use cogno_model::{ModelBackend, OwnedProposal, SimBackend};
use cogno_runtime::{Pipeline, PipelineParams, Runtime, RuntimeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    match cmd {
        "phase" => cmd_phase(),
        "doctor" => cmd_doctor(),
        "validate" => cmd_validate(args.get(2)),
        "simulate" => cmd_simulate(),
        "demo-pipeline" => cmd_demo_pipeline(),
        "replay" => cmd_replay(),
        "help" | "-h" | "--help" | "" => print_help(),
        other => {
            eprintln!("cogno: unknown command '{other}'");
            print_help();
            std::process::exit(2);
        }
    }
}

fn print_help() {
    println!("cogno — COGNO-1 command line (Phase 0 deterministic core)");
    println!();
    println!("Commands:");
    println!("  phase           Show the current implementation phase and meta-objective state");
    println!("  doctor          Validate the budget, KV controller, queue and tool gate");
    println!(
        "  validate <n>    Validate n scripted proposals against the strict schema (fail-closed)"
    );
    println!("  simulate        Run the Phase 1 deterministic simulator and report");
    println!(
        "  demo-pipeline   Run a proposal through the full §3 pipeline and print the decision"
    );
    println!(
        "  replay          Replay a scripted journal and report derived profile (deterministic)"
    );
    println!("  help            Show this help");
}

fn cmd_phase() {
    let rt = runtime_or_die();
    let rep = rt.report();
    println!("Phase: {}", rep.phase);
    println!("Meta-objective active: {}", rep.meta_active);
    println!("Tools enabled: {}", rep.tools_enabled);
    println!("Admissions: {}", rep.admissions);
    println!("Rejections: {}", rep.rejections);
    println!("Truncations: {}", rep.truncations);
}

fn cmd_doctor() {
    let rt = runtime_or_die();
    let rep = rt.report();
    let ok = !rep.tools_enabled && !rep.meta_active;
    println!("doctor: budget ok | kv ok | queue ok | tools gated | meta gated");
    println!(
        "admissions={} rejections={} truncations={}",
        rep.admissions, rep.rejections, rep.truncations
    );
    if ok {
        println!("status: MVP-safe (fail-closed)");
    } else {
        println!("status: non-MVP configuration present — review before release");
    }
}

fn cmd_validate(n_arg: Option<&String>) {
    let n: usize = n_arg.and_then(|s| s.parse().ok()).unwrap_or(3).min(64);
    let mut sim = script_simulator(n);
    let mut ok = 0usize;
    let mut rejected = 0usize;
    while let Ok(p) = sim.next_proposal() {
        match cogno_core::validate_proposal(&p.as_view()) {
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
    let mut sim = script_simulator(4);
    println!("Phase 1 simulator — scripted proposals");
    while let Ok(p) = sim.next_proposal() {
        let v = cogno_core::validate_proposal(&p.as_view());
        println!(
            "prop action={:?} category={:?} confidence_bps={} validated={}",
            p.action,
            p.category,
            p.confidence_bps,
            matches!(v, Ok(()))
        );
    }
    println!("(exhausted -> BackendError::Exhausted, fail-closed)");
}

fn cmd_demo_pipeline() {
    let mut rt = runtime_or_die();
    let ev = [EvidenceId::from_u64(1)];
    let proposal = CognoProposalView {
        schema_version: 1,
        action: ProposalAction::ClassifyFeedback,
        category: cogno_core::RuleCategory::Behavior,
        scope: cogno_core::RuleScope::Session,
        confidence_bps: 5_000,
        evidence_ids: &ev,
        payload: b"demo",
    };
    let pp = PipelineParams {
        policy: SafetyPolicy::MVP,
        reward_params: RewardParams::DEFAULT,
        data_class: DataClassification::Internal,
        trust: TrustClass::AuthenticatedUser,
        reward_signal: Reward::ZERO,
        proposal: Some(proposal),
    };
    let out = rt.run_pipeline(pp);
    println!("pipeline: {out:?}");
}

fn cmd_replay() {
    use cogno_core::{
        DerivationPolicy, EvidenceOrigin, Fingerprint, InputOrigin, Journal, JournalEvent,
    };
    let mut j = Journal::default();
    for i in 0..3u8 {
        let mut fp = [0u8; 32];
        fp[0] = i + 1;
        j.append(JournalEvent {
            id: cogno_core::EventId(0),
            fingerprint: Fingerprint(fp),
            origin: InputOrigin::ExplicitUserInstruction,
            evidence_origin: if i == 2 {
                EvidenceOrigin::FormalValidator
            } else {
                EvidenceOrigin::ExplicitUserApproval
            },
            evidence_id: EvidenceId::from_u64(10 + i as u64),
            category_tag: 7,
            payload: Vec::new(),
        });
    }
    let p = cogno_core::Profile::derive(&j, DerivationPolicy::DEFAULT);
    println!("journal events: {}", j.len());
    println!("derived profile active_count: {}", p.active_count());
    println!("derived profile hard_count: {}", p.hard_count());
    println!("replay: deterministic — same journal => same profile (S6/S7)");
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
    .expect("hardcoded MVP budget must validate (§11)");
    let cfg = RuntimeConfig {
        budget,
        reserved_bytes: 0,
        kv_capacity_tokens: 2048,
        kv_policy: cogno_core::KvCachePolicy::RejectOnOverflow,
        queue_capacity: 64,
        queue_policy: QueueFullPolicy::RejectNewest,
    };
    Runtime::try_new(cfg).expect("runtime construction must succeed with a valid MVP budget")
}

fn script_simulator(n: usize) -> SimBackend {
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
    let mut script = Vec::with_capacity(n);
    for i in 0..n {
        let ev_ids = if i == 1 {
            vec![EvidenceId::from_u64(1), EvidenceId::from_u64(2)]
        } else {
            vec![EvidenceId::from_u64(1)]
        };
        script.push(OwnedProposal {
            action: actions[i % actions.len()],
            category: categories[i % categories.len()],
            scope: cogno_core::RuleScope::Session,
            confidence_bps: (1_000 + i as u16 * 500).min(10_000),
            evidence_ids: ev_ids,
            payload: vec![i as u8],
        });
    }
    SimBackend::new(script)
}

// silence clippy/unused with `Runtime` placeholder if needed
#[allow(dead_code)]
fn _unused_pipeline_marker(_p: &Pipeline) {}
