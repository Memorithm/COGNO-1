//! Run one continuous taste feedback iteration over the local journal.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TastePolicy;
use cogno_runtime::{load_settings, run_feedback_iteration};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-loop: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or("usage: cogno-taste-loop STORE_ROOT")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    // Consent gate: learning disabled means no derivation either.
    let settings = load_settings(Path::new(&store_root))
        .map_err(|error| format!("cannot load host consent: {error}"))?;
    let profile = run_feedback_iteration(Path::new(&store_root), &settings, TastePolicy::DEFAULT)
        .map_err(|error| format!("feedback iteration failed: {error}"))?;

    let preferences: Vec<serde_json::Value> = profile
        .preferences
        .values()
        .map(|entry| {
            serde_json::json!({
                "preference_id": entry.preference_id,
                "scope": entry.scope.scope_tag(),
                "scope_key": entry.scope_key,
                "state": format!("{:?}", entry.state),
                "confidence_bps": entry.confidence_bps,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "policy": "default",
            "preferences": preferences,
        })
    );
    Ok(())
}
