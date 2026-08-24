//! Run one continuous taste feedback iteration over the local journal.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TastePolicy;
use cogno_runtime::{
    current_unix_seconds, load_profile_snapshot, load_settings, run_feedback_iteration_tuned,
    save_profile_snapshot, LoopTuning, ProfileSnapshot,
};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-loop: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or(usage())?;
    let mut tuning = LoopTuning::default();
    let mut now_given = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--now" => {
                tuning.now_unix = args
                    .next()
                    .ok_or("--now requires a unix seconds value")?
                    .parse()
                    .map_err(|error| format!("invalid --now: {error}"))?;
                now_given = true;
            }
            "--retention-secs" => {
                tuning.retention_secs = args
                    .next()
                    .ok_or("--retention-secs requires a seconds value")?
                    .parse()
                    .map_err(|error| format!("invalid --retention-secs: {error}"))?;
            }
            "--graduated-decay" => tuning.graduated_decay = true,
            "--bandit-step-bps" => {
                tuning.bandit_step_bps = args
                    .next()
                    .ok_or("--bandit-step-bps requires a number (<=5000)")?
                    .parse()
                    .map_err(|error| format!("invalid --bandit-step-bps: {error}"))?;
            }
            other => return Err(format!("{other}\n{}", usage())),
        }
    }
    if !now_given && (tuning.retention_secs > 0 || tuning.graduated_decay) {
        tuning.now_unix = current_unix_seconds();
    }

    // Consent gate: learning disabled means no derivation either.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let profile = run_feedback_iteration_tuned(
        Path::new(&store_root),
        &settings,
        TastePolicy::DEFAULT,
        &tuning,
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
            "now": tuning.now_unix,
            "retention_secs": tuning.retention_secs,
            "graduated_decay": tuning.graduated_decay,
            "bandit_step_bps": tuning.bandit_step_bps,
            "preferences": preferences,
            "delta": delta_report(previous, &profile),
        })
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage: cogno-taste-loop STORE_ROOT [--now N] [--retention-secs S] [--graduated-decay] [--bandit-step-bps B]"
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
