//! Ingest a verified taste package into the local feedback journal.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{
    current_unix_seconds, load_settings, record_feedback, TasteFeedbackRecord, TastePackage,
    FEEDBACK_SCHEMA_VERSION,
};
use std::path::Path;

/// Imported events live in a reserved high event-id range so they can never
/// collide with host-assigned journal ids (validations, local outcomes).
const IMPORTED_EVENT_BASE: u64 = 1_u64 << 63;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-import: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-import STORE_ROOT PACKAGE_PATH")?;
    let package_path = args
        .next()
        .ok_or("usage: cogno-taste-import STORE_ROOT PACKAGE_PATH")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    // Consent gate (étape 4): a host that forbids imports never ingests one.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let package = TastePackage::load_file(&package_path)
        .map_err(|error| format!("cannot verify {}: {error:?}", package_path))?;
    let events = package
        .import_events_with_consent(&settings)
        .map_err(|error| format!("import denied or failed: {error:?}"))?;

    let mut imported = 0_usize;
    for event in &events {
        let source = &event.source;
        let record = TasteFeedbackRecord {
            schema_version: FEEDBACK_SCHEMA_VERSION,
            event_id: IMPORTED_EVENT_BASE | event.preference_id,
            preference_id: event.preference_id,
            scope: event.scope.scope_tag().to_string(),
            scope_key: event.scope_key.clone(),
            // Imports are quarantine-bound proposals; they can activate only
            // through local non-model confirmations.
            kind: "proposed".to_string(),
            origin: "imported_profile".to_string(),
            source_kind: source.kind.tag().to_string(),
            source_id: source.id.clone(),
            confidence_bps: event.confidence_bps,
            observed_at_unix: current_unix_seconds(),
        };
        record_feedback(Path::new(&store_root), &settings, &record)
            .map_err(|error| format!("cannot record imported preference: {error}"))?;
        imported += 1;
    }
    println!(
        "{}",
        serde_json::json!({
            "event": "imported",
            "digest": package.digest_hex(),
            "records": imported,
        })
    );
    Ok(())
}
