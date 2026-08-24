//! Export the derived taste profile as a verified package (consent-gated).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TastePolicy;
use cogno_runtime::{
    load_settings, run_feedback_iteration_tuned, save_profile_snapshot, LoopTuning,
};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-export: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or(usage())?;
    let output_dir = args.next().ok_or(usage())?;
    let mut tuning = LoopTuning::default();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--now" => {
                tuning.now_unix = parse_u64(&mut args, "--now")?;
            }
            "--retention-secs" => {
                tuning.retention_secs = parse_u64(&mut args, "--retention-secs")?;
            }
            "--graduated-decay" => tuning.graduated_decay = true,
            other => return Err(format!("{other}\n{}", usage())),
        }
    }

    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let profile = run_feedback_iteration_tuned(
        Path::new(&store_root),
        &settings,
        TastePolicy::DEFAULT,
        &tuning,
    )
    .map_err(|error| format!("derivation failed: {error}"))?;
    let package = cogno_runtime::TastePackage::from_profile(&profile, TastePolicy::DEFAULT)
        .map_err(|error| match error {
            cogno_runtime::TastePackageError::EmptyProfile => {
                "no Active preference to export yet".to_string()
            }
            other => format!("cannot build package: {other:?}"),
        })?;
    package
        .save_with_consent(Path::new(&output_dir), &settings)
        .map_err(|error| match error {
            cogno_runtime::TastePackageError::ConsentDenied => {
                "export denied by host consent (taste.settings.json)".to_string()
            }
            other => format!("cannot write package: {other:?}"),
        })?;
    // Keep the snapshot in sync with what was exported.
    save_profile_snapshot(
        Path::new(&store_root),
        &settings,
        &profile,
        TastePolicy::DEFAULT,
    )
    .map_err(|error| format!("cannot save snapshot: {error}"))?;
    println!(
        "{}",
        serde_json::json!({
            "event": "exported",
            "digest": package.digest_hex(),
            "entries": package.entries().len(),
        })
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage: cogno-taste-export STORE_ROOT OUTPUT_DIR [--now N] [--retention-secs S] [--graduated-decay]"
}

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    args.next()
        .ok_or(format!("{flag} requires a number"))?
        .parse()
        .map_err(|error| format!("invalid {flag}: {error}"))
}
