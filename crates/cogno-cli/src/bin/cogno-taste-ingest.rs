//! Feed stored validations into the continuous taste feedback journal.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{TasteScope, TasteSettings};
use cogno_runtime::ingest_validation_store;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-ingest: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-ingest STORE_ROOT SCOPE_TAG SCOPE_KEY")?;
    let scope_tag = args
        .next()
        .ok_or("usage: cogno-taste-ingest STORE_ROOT SCOPE_TAG SCOPE_KEY")?;
    let scope_key = args
        .next()
        .ok_or("usage: cogno-taste-ingest STORE_ROOT SCOPE_TAG SCOPE_KEY")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let scope = TasteScope::from_scope_tag(&scope_tag).ok_or(format!(
        "unknown scope tag `{scope_tag}` (user|project|domain|team)"
    ))?;
    // Consent gate lives inside ingest_validation_store (learning_enabled).
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let count = ingest_validation_store(Path::new(&store_root), &settings, scope, &scope_key)
        .map_err(|error| format!("ingest failed: {error}"))?;
    println!(
        "{}",
        serde_json::json!({ "event": "ingested", "records": count })
    );
    Ok(())
}

fn load_settings(store_root: &Path) -> Result<TasteSettings, String> {
    cogno_runtime::load_settings(store_root)
        .map_err(|error| format!("cannot load host consent: {error}"))
}
