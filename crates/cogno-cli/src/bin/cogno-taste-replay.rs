//! Replay persistent non-model validations into a deterministic taste profile.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{
    ScientificTasteProfile, TasteEvent, TasteEventKind, TasteOrigin, TastePolicy, TasteScope,
    TasteState,
};
use cogno_runtime::{
    PersistentTasteValidationStore, StoredValidationOrigin, StoredValidationVerdict,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Deserialize)]
struct CandidateReport {
    schema_version: u16,
    authority: String,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    preference_id: u64,
    evidence_id: u64,
    state: String,
    can_activate: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-replay: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let candidates_path = args
        .next()
        .ok_or("usage: cogno-taste-replay CANDIDATES_JSON STORE_ROOT")?;
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-replay CANDIDATES_JSON STORE_ROOT")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let candidate_bytes = fs::read(&candidates_path)
        .map_err(|error| format!("cannot read candidates report: {error}"))?;
    let report: CandidateReport = serde_json::from_slice(&candidate_bytes)
        .map_err(|error| format!("invalid candidates report: {error}"))?;
    validate_candidate_report(&report)?;

    let store = PersistentTasteValidationStore::open(Path::new(&store_root))
        .map_err(|error| format!("cannot replay validation store: {error:?}"))?;
    let output = derive_output(&report, &store, &candidate_bytes)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize replay report: {error}"))?
    );
    Ok(())
}

fn validate_candidate_report(report: &CandidateReport) -> Result<(), String> {
    if report.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported candidate schema version {}",
            report.schema_version
        ));
    }
    if report.authority != "quarantine_only" {
        return Err("candidate report authority must be quarantine_only".to_string());
    }
    let mut ids = BTreeSet::new();
    for candidate in &report.candidates {
        if candidate.state != "quarantined" || candidate.can_activate {
            return Err(format!(
                "candidate {} is not strictly quarantined",
                candidate.preference_id
            ));
        }
        if !ids.insert(candidate.preference_id) {
            return Err(format!(
                "duplicate candidate preference_id {}",
                candidate.preference_id
            ));
        }
    }
    Ok(())
}

fn derive_output(
    report: &CandidateReport,
    store: &PersistentTasteValidationStore,
    candidate_bytes: &[u8],
) -> Result<Value, String> {
    let candidate_ids: BTreeSet<u64> = report
        .candidates
        .iter()
        .map(|candidate| candidate.preference_id)
        .collect();
    for validation in store.records() {
        if !candidate_ids.contains(&validation.preference_id) {
            return Err(format!(
                "stored validation {} references unknown preference {}",
                validation.validation_id, validation.preference_id
            ));
        }
    }

    let mut preferences = Vec::with_capacity(report.candidates.len());
    for candidate in &report.candidates {
        let mut events = vec![TasteEvent {
            event_id: 0,
            preference_id: candidate.preference_id,
            scope: TasteScope::Project,
            kind: TasteEventKind::Proposed,
            origin: TasteOrigin::ModelInference,
            confidence_bps: 5_000,
            evidence_ids: vec![candidate.evidence_id],
        }];
        let mut validation_count = 0usize;
        for validation in store.records() {
            if validation.preference_id != candidate.preference_id {
                continue;
            }
            validation_count += 1;
            events.push(TasteEvent {
                event_id: validation.validation_id,
                preference_id: validation.preference_id,
                scope: TasteScope::Project,
                kind: match validation.verdict {
                    StoredValidationVerdict::Confirmed => TasteEventKind::Confirmed,
                    StoredValidationVerdict::Contradicted => TasteEventKind::Contradicted,
                },
                origin: match validation.origin {
                    StoredValidationOrigin::DeterministicEvaluation => {
                        TasteOrigin::DeterministicEvaluation
                    }
                    StoredValidationOrigin::ExplicitUserAction => TasteOrigin::ExplicitUserAction,
                },
                confidence_bps: validation.confidence_bps,
                evidence_ids: vec![validation.evidence_id],
            });
        }
        let profile = ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT)
            .map_err(|error| format!("cannot derive taste profile: {error:?}"))?;
        let taste = profile
            .preferences
            .get(&candidate.preference_id)
            .ok_or("derived profile omitted candidate")?;
        preferences.push(json!({
            "preference_id": candidate.preference_id,
            "state": state_name(taste.state),
            "confidence_bps": taste.confidence_bps,
            "non_model_confirmations": taste.non_model_confirmations,
            "validations": validation_count,
            "can_activate": taste.is_active()
        }));
    }

    let candidate_sha256: [u8; 32] = Sha256::digest(candidate_bytes).into();
    let validation_file = store_root_file(store);
    let validation_sha256: [u8; 32] = Sha256::digest(
        fs::read(&validation_file)
            .map_err(|error| format!("cannot hash validation store: {error}"))?,
    )
    .into();

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "authority": "deterministic_replay_only",
        "candidate_report_sha256": hex_digest(candidate_sha256),
        "validation_store_sha256": hex_digest(validation_sha256),
        "validation_records": store.len(),
        "minimum_non_model_confirmations": TastePolicy::DEFAULT.minimum_non_model_confirmations,
        "activation_threshold_bps": TastePolicy::DEFAULT.activation_threshold_bps,
        "preferences": preferences
    }))
}

fn store_root_file(store: &PersistentTasteValidationStore) -> std::path::PathBuf {
    store.path().to_path_buf()
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("string write");
    }
    output
}

const fn state_name(state: TasteState) -> &'static str {
    match state {
        TasteState::Quarantined => "quarantined",
        TasteState::Candidate => "candidate",
        TasteState::Active => "active",
        TasteState::Conflicted => "conflicted",
        TasteState::Rejected => "rejected",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_hex_is_stable() {
        assert_eq!(hex_digest([0xab; 32]), "ab".repeat(32));
    }

    #[test]
    fn report_requires_quarantine_authority() {
        let report = CandidateReport {
            schema_version: 1,
            authority: "active".to_string(),
            candidates: Vec::new(),
        };
        assert!(validate_candidate_report(&report).is_err());
    }
}
