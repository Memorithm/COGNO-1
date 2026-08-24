//! Run one continuous taste feedback iteration over the local journal.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TastePolicy;
use cogno_runtime::{
    load_profile_snapshot, load_settings, run_feedback_iteration_retained, save_profile_snapshot,
    ProfileSnapshot,
};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-loop: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or(usage())?;
    let mut now_unix: Option<u64> = None;
    let mut retention_secs: u64 = 0;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--now" => {
                now_unix = Some(
                    args.next()
                        .ok_or("--now requires a unix seconds value")?
                        .parse()
                        .map_err(|error| format!("invalid --now: {error}"))?,
                );
            }
            "--retention-secs" => {
                retention_secs = args
                    .next()
                    .ok_or("--retention-secs requires a seconds value")?
                    .parse()
                    .map_err(|error| format!("invalid --retention-secs: {error}"))?;
            }
            other => return Err(format!("{other}\n{}", usage())),
        }
    }

    // Consent gate: learning disabled means no derivation either.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let now_unix = now_unix.unwrap_or_else(current_unix_seconds);
    let profile = run_feedback_iteration_retained(
        Path::new(&store_root),
        &settings,
        TastePolicy::DEFAULT,
        now_unix,
        retention_secs,
    )
    .map_err(|error| format!("feedback iteration failed: {error}"))?;

    // Delta against the previous snapshot, then persist the new one.
    let previous: Option<ProfileSnapshot> =
        load_profile_snapshot(Path::new(&store_root)).map_err(|error| error.to_string())?;
    save_profile_snapshot(
        Path::new(&store_root),
        &settings,
        &profile,
        TastePolicy::DEFAULT,
    )
    .map_err(|error| format!("cannot save snapshot: {error}"))?;

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
            "now": now_unix,
            "retention_secs": retention_secs,
            "preferences": preferences,
            "delta": delta_report(previous, &profile),
        })
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage: cogno-taste-loop STORE_ROOT [--now UNIX_SECS] [--retention-secs N]"
}

fn current_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn delta_report(
    previous: Option<ProfileSnapshot>,
    profile: &cogno_core::ScientificTasteProfile,
) -> serde_json::Value {
    let previous = match previous {
        Some(snapshot) => snapshot,
        None => return serde_json::json!({ "changed": [], "added": [] }),
    };
    let mut changed = Vec::new();
    let mut added = Vec::new();
    for entry in profile.preferences.values() {
        match previous
            .preferences
            .iter()
            .find(|snapshot| snapshot.preference_id == entry.preference_id)
        {
            Some(old) => {
                if old.state != cogno_runtime::state_tag_of(entry.state)
                    || old.confidence_bps != entry.confidence_bps
                {
                    changed.push(serde_json::json!({
                        "preference_id": entry.preference_id,
                        "from_state": old.state,
                        "to_state": cogno_runtime::state_tag_of(entry.state),
                        "from_confidence_bps": old.confidence_bps,
                        "to_confidence_bps": entry.confidence_bps,
                    }));
                }
            }
            None => added.push(entry.preference_id),
        }
    }
    serde_json::json!({ "changed": changed, "added": added })
}
