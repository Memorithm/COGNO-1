//! Promote quarantined preference candidates only after non-model validation.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{
    ScientificTasteProfile, TasteEvent, TasteEventKind, TasteOrigin, TastePolicy, TasteScope,
    TasteSource, TasteState,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const MAX_VALIDATIONS: usize = 16_384;

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

#[derive(Clone, Debug, Deserialize)]
struct Validation {
    schema_version: u16,
    validation_id: u64,
    preference_id: u64,
    evidence_id: u64,
    origin: String,
    verdict: String,
    confidence_bps: u16,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-promote: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let candidates_path = args
        .next()
        .ok_or("usage: cogno-taste-promote CANDIDATES_JSON VALIDATIONS_JSONL")?;
    let validations_path = args
        .next()
        .ok_or("usage: cogno-taste-promote CANDIDATES_JSON VALIDATIONS_JSONL")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let report: CandidateReport = serde_json::from_slice(
        &fs::read(&candidates_path)
            .map_err(|error| format!("cannot read candidates report: {error}"))?,
    )
    .map_err(|error| format!("invalid candidates report: {error}"))?;
    validate_candidate_report(&report)?;
    let validations = read_validations(Path::new(&validations_path))?;
    let output = derive_output(&report, &validations)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize promotion report: {error}"))?
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
    let mut preference_ids = BTreeSet::new();
    for candidate in &report.candidates {
        if candidate.state != "quarantined" || candidate.can_activate {
            return Err(format!(
                "candidate {} is not strictly quarantined",
                candidate.preference_id
            ));
        }
        if !preference_ids.insert(candidate.preference_id) {
            return Err(format!(
                "duplicate candidate preference_id {}",
                candidate.preference_id
            ));
        }
    }
    Ok(())
}

fn read_validations(path: &Path) -> Result<Vec<Validation>, String> {
    let file = fs::File::open(path)
        .map_err(|error| format!("cannot open validations file {}: {error}", path.display()))?;
    let mut validations = Vec::new();
    for (index, line_result) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line_result
            .map_err(|error| format!("cannot read validation line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if validations.len() >= MAX_VALIDATIONS {
            return Err(format!("validation limit {MAX_VALIDATIONS} exceeded"));
        }
        let validation: Validation = serde_json::from_str(&line)
            .map_err(|error| format!("invalid validation line {line_number}: {error}"))?;
        validate_validation(&validation)
            .map_err(|error| format!("invalid validation line {line_number}: {error}"))?;
        validations.push(validation);
    }
    Ok(validations)
}

fn validate_validation(validation: &Validation) -> Result<(), String> {
    if validation.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema version {}",
            validation.schema_version
        ));
    }
    if validation.confidence_bps > 10_000 {
        return Err("confidence_bps exceeds 10000".to_string());
    }
    match validation.origin.as_str() {
        "deterministic_evaluation" | "explicit_user_action" => {}
        "model_inference" => return Err("model inference cannot validate promotion".to_string()),
        other => return Err(format!("unsupported validation origin `{other}`")),
    }
    match validation.verdict.as_str() {
        "confirmed" | "contradicted" => Ok(()),
        other => Err(format!("unsupported validation verdict `{other}`")),
    }
}

fn derive_output(report: &CandidateReport, validations: &[Validation]) -> Result<Value, String> {
    let candidate_ids: BTreeSet<u64> = report
        .candidates
        .iter()
        .map(|candidate| candidate.preference_id)
        .collect();
    let mut seen_validation_ids = BTreeSet::new();
    let mut seen_evidence = BTreeSet::new();
    let mut grouped: BTreeMap<u64, Vec<&Validation>> = BTreeMap::new();

    for validation in validations {
        if !candidate_ids.contains(&validation.preference_id) {
            return Err(format!(
                "validation {} references unknown preference {}",
                validation.validation_id, validation.preference_id
            ));
        }
        if !seen_validation_ids.insert(validation.validation_id) {
            return Err(format!(
                "duplicate validation_id {}",
                validation.validation_id
            ));
        }
        if !seen_evidence.insert((validation.preference_id, validation.evidence_id)) {
            return Err(format!(
                "duplicate evidence {} for preference {}",
                validation.evidence_id, validation.preference_id
            ));
        }
        grouped
            .entry(validation.preference_id)
            .or_default()
            .push(validation);
    }

    let mut results = Vec::with_capacity(report.candidates.len());
    for candidate in &report.candidates {
        let mut events = vec![TasteEvent {
            event_id: 0,
            preference_id: candidate.preference_id,
            scope: TasteScope::Project,
            kind: TasteEventKind::Proposed,
            origin: TasteOrigin::ModelInference,
            confidence_bps: 5_000,
            evidence_ids: vec![candidate.evidence_id],
            source: TasteSource::UNKNOWN,
        }];
        let candidate_validations = grouped
            .get(&candidate.preference_id)
            .cloned()
            .unwrap_or_default();
        for validation in &candidate_validations {
            events.push(TasteEvent {
                event_id: validation.validation_id,
                preference_id: candidate.preference_id,
                scope: TasteScope::Project,
                kind: match validation.verdict.as_str() {
                    "confirmed" => TasteEventKind::Confirmed,
                    "contradicted" => TasteEventKind::Contradicted,
                    _ => unreachable!("validation was checked"),
                },
                origin: match validation.origin.as_str() {
                    "deterministic_evaluation" => TasteOrigin::DeterministicEvaluation,
                    "explicit_user_action" => TasteOrigin::ExplicitUserAction,
                    _ => unreachable!("validation was checked"),
                },
                confidence_bps: validation.confidence_bps,
                evidence_ids: vec![validation.evidence_id],
                source: TasteSource::UNKNOWN,
            });
        }
        let profile = ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT)
            .map_err(|error| format!("cannot derive taste profile: {error:?}"))?;
        let taste = profile
            .preferences
            .get(&candidate.preference_id)
            .ok_or("derived profile omitted candidate")?;
        results.push(json!({
            "preference_id": candidate.preference_id,
            "state": state_name(taste.state),
            "confidence_bps": taste.confidence_bps,
            "non_model_confirmations": taste.non_model_confirmations,
            "validations": candidate_validations.len(),
            "can_activate": taste.is_active()
        }));
    }

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "authority": "deterministic_core_only",
        "minimum_non_model_confirmations": TastePolicy::DEFAULT.minimum_non_model_confirmations,
        "activation_threshold_bps": TastePolicy::DEFAULT.activation_threshold_bps,
        "preferences": results
    }))
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

    fn report() -> CandidateReport {
        CandidateReport {
            schema_version: 1,
            authority: "quarantine_only".to_string(),
            candidates: vec![Candidate {
                preference_id: 7,
                evidence_id: 1,
                state: "quarantined".to_string(),
                can_activate: false,
            }],
        }
    }

    fn validation(id: u64, origin: &str, verdict: &str) -> Validation {
        Validation {
            schema_version: 1,
            validation_id: id,
            preference_id: 7,
            evidence_id: id + 10,
            origin: origin.to_string(),
            verdict: verdict.to_string(),
            confidence_bps: 8_000,
        }
    }

    #[test]
    fn one_confirmation_is_not_enough() {
        let output = derive_output(
            &report(),
            &[validation(1, "deterministic_evaluation", "confirmed")],
        )
        .expect("report");
        assert_eq!(output["preferences"][0]["state"], "candidate");
        assert_eq!(output["preferences"][0]["can_activate"], false);
    }

    #[test]
    fn two_non_model_confirmations_can_activate() {
        let output = derive_output(
            &report(),
            &[
                validation(1, "deterministic_evaluation", "confirmed"),
                validation(2, "explicit_user_action", "confirmed"),
            ],
        )
        .expect("report");
        assert_eq!(output["preferences"][0]["state"], "active");
        assert_eq!(output["preferences"][0]["can_activate"], true);
    }

    #[test]
    fn model_validation_is_rejected() {
        let item = validation(1, "model_inference", "confirmed");
        assert!(validate_validation(&item).is_err());
    }

    #[test]
    fn contradiction_wins() {
        let output = derive_output(
            &report(),
            &[
                validation(1, "deterministic_evaluation", "confirmed"),
                validation(2, "explicit_user_action", "contradicted"),
            ],
        )
        .expect("report");
        assert_eq!(output["preferences"][0]["state"], "conflicted");
        assert_eq!(output["preferences"][0]["can_activate"], false);
    }
}
