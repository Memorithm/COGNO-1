//! Persist non-model taste validations in an append-only runtime store.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{
    PersistentTasteValidationStore, StoredTasteValidation, StoredValidationOrigin,
    StoredValidationVerdict, TasteValidationAppendOutcome,
};
use serde::Deserialize;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const MAX_VALIDATIONS: usize = 16_384;

#[derive(Debug, Deserialize)]
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
        eprintln!("cogno-taste-store: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .ok_or("usage: cogno-taste-store ROOT VALIDATIONS_JSONL")?;
    let validations_path = args
        .next()
        .ok_or("usage: cogno-taste-store ROOT VALIDATIONS_JSONL")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let file = fs::File::open(&validations_path)
        .map_err(|error| format!("cannot open validations file: {error}"))?;
    let mut store = PersistentTasteValidationStore::open(Path::new(&root))
        .map_err(|error| format!("cannot open validation store: {error:?}"))?;
    let mut processed = 0usize;
    let mut appended = 0usize;
    let mut duplicates = 0usize;

    for (index, line_result) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line_result
            .map_err(|error| format!("cannot read validation line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if processed >= MAX_VALIDATIONS {
            return Err(format!("validation limit {MAX_VALIDATIONS} exceeded"));
        }
        let validation: Validation = serde_json::from_str(&line)
            .map_err(|error| format!("invalid validation line {line_number}: {error}"))?;
        let stored = convert(validation)
            .map_err(|error| format!("invalid validation line {line_number}: {error}"))?;
        match store
            .append(stored)
            .map_err(|error| format!("cannot persist validation line {line_number}: {error:?}"))?
        {
            TasteValidationAppendOutcome::Appended => appended += 1,
            TasteValidationAppendOutcome::Duplicate => duplicates += 1,
        }
        processed += 1;
    }

    println!(
        "taste-store: processed={processed} appended={appended} duplicates={duplicates} total={}",
        store.len()
    );
    println!("store_root={root}");
    Ok(())
}

fn convert(validation: Validation) -> Result<StoredTasteValidation, String> {
    if validation.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema version {}",
            validation.schema_version
        ));
    }
    if validation.confidence_bps > 10_000 {
        return Err("confidence_bps exceeds 10000".to_string());
    }
    let origin = match validation.origin.as_str() {
        "deterministic_evaluation" => StoredValidationOrigin::DeterministicEvaluation,
        "explicit_user_action" => StoredValidationOrigin::ExplicitUserAction,
        "model_inference" => return Err("model inference cannot validate promotion".to_string()),
        other => return Err(format!("unsupported validation origin `{other}`")),
    };
    let verdict = match validation.verdict.as_str() {
        "confirmed" => StoredValidationVerdict::Confirmed,
        "contradicted" => StoredValidationVerdict::Contradicted,
        other => return Err(format!("unsupported validation verdict `{other}`")),
    };
    Ok(StoredTasteValidation {
        validation_id: validation.validation_id,
        preference_id: validation.preference_id,
        evidence_id: validation.evidence_id,
        origin,
        verdict,
        confidence_bps: validation.confidence_bps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validation(origin: &str) -> Validation {
        Validation {
            schema_version: 1,
            validation_id: 1,
            preference_id: 7,
            evidence_id: 11,
            origin: origin.to_string(),
            verdict: "confirmed".to_string(),
            confidence_bps: 8_000,
        }
    }

    #[test]
    fn model_validation_is_rejected() {
        assert!(convert(validation("model_inference")).is_err());
    }

    #[test]
    fn deterministic_validation_is_accepted() {
        assert!(convert(validation("deterministic_evaluation")).is_ok());
    }
}
