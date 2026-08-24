//! Compose verified taste packages into one deterministic merged package.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TastePolicy;
use cogno_runtime::{load_settings, TastePackage, TastePackageError};
use std::path::Path;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-compose: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-compose STORE_ROOT OUTPUT_DIR PACKAGE.md [PACKAGE.md ...]")?;
    let output_dir = args
        .next()
        .ok_or("usage: cogno-taste-compose STORE_ROOT OUTPUT_DIR PACKAGE.md [PACKAGE.md ...]")?;
    let package_paths: Vec<PathBuf> = {
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            return Err(
                "usage: cogno-taste-compose STORE_ROOT OUTPUT_DIR PACKAGE.md [PACKAGE.md ...]"
                    .to_string(),
            );
        }
        rest.into_iter().map(PathBuf::from).collect()
    };
    if package_paths.len() > 256 {
        return Err("too many packages (max 256 per invocation)".to_string());
    }

    // Consent gate (étape 4): composing writes a merged taste.md, which is an
    // export of learned preferences and therefore needs explicit permission.
    let settings = load_settings(Path::new(&store_root))
        .map_err(|error| format!("cannot load host consent: {error}"))?;

    let mut parts = Vec::new();
    for path in &package_paths {
        let package = TastePackage::load_file(path)
            .map_err(|error| format!("cannot verify {}: {error:?}", path.display()))?;
        parts.push(package);
    }
    let first_policy = parts[0].policy();
    let composed = TastePackage::compose(
        &parts.iter().collect::<Vec<_>>(),
        TastePolicy::from(first_policy),
    )
    .map_err(|error| format!("cannot compose packages: {error:?}"))?;
    composed
        .save_with_consent(&output_dir, &settings)
        .map_err(|error| match error {
            TastePackageError::ConsentDenied => {
                "export denied by host consent (taste.settings.json)".to_string()
            }
            other => format!("cannot write composed package: {other:?}"),
        })?;
    println!(
        "{}",
        serde_json::json!({
            "digest": hex(&composed.digest()),
            "entries": composed.entries().len(),
            "parents": composed.parents(),
            "conflicts": composed.conflicts(),
            "policy": serde_json::to_value(first_policy)
                .map_err(|error| format!("cannot encode policy: {error}"))?,
        })
    );
    Ok(())
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
