//! Read-only integrity audit of a taste store.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{audit_store, current_unix_seconds};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-verify: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-verify STORE_ROOT [--retention-secs S]")?;
    let mut retention_secs: u64 = 0;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--retention-secs" => {
                retention_secs = args
                    .next()
                    .ok_or("--retention-secs requires a number")?
                    .parse()
                    .map_err(|error| format!("invalid --retention-secs: {error}"))?;
            }
            other => {
                return Err(format!(
                    "{other}\nusage: cogno-taste-verify STORE_ROOT [--retention-secs S]"
                ))
            }
        }
    }

    let audit = audit_store(
        Path::new(&store_root),
        current_unix_seconds(),
        retention_secs,
    )
    .map_err(|error| format!("audit failed: {error}"))?;
    let report = serde_json::json!({
        "settings_present": audit.settings_present,
        "consent": {
            "learning": audit.learning_enabled,
            "export": audit.export_allowed,
            "import": audit.import_allowed,
        },
        "journal": {
            "records": audit.journal_records,
            "duplicate_ids": audit.journal_duplicate_ids,
            "expired": audit.journal_expired,
            "corrupt": audit.journal_corrupt,
        },
        "snapshot": {
            "valid": audit.snapshot_valid,
            "preferences": audit.snapshot_preferences,
        },
        "accepted_log_digests": audit.accepted_log_digests,
        "package_digest": audit.package_digest,
        "findings": audit.findings,
    });
    println!("{report}");
    // Corrupt journal or hard findings fail the audit.
    if audit.journal_corrupt.is_some() || !audit.findings.is_empty() {
        std::process::exit(3);
    }
    Ok(())
}
