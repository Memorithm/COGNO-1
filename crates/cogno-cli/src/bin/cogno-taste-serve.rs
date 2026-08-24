//! Receive taste package pushes over TCP (consent-gated, digest-verified).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TasteSettings;
use cogno_runtime::{load_settings, TastePackage};
use cogno_transport::{
    serve_session, PushOutcome, SessionConfig, SessionEvent, DEFAULT_MAX_REQUESTS,
};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

/// Cross-connection idempotency memory (`<root>/taste.accepted.log`).
const ACCEPTED_LOG: &str = "taste.accepted.log";

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-serve: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    const USAGE: &str =
        "usage: cogno-taste-serve STORE_ROOT BIND_ADDR PORT [--max-pushes N] [--daemon] [--max-connections N]";
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or(USAGE)?;
    let bind_addr = args.next().ok_or(USAGE)?;
    let port: u16 = args
        .next()
        .ok_or(USAGE)?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    let mut max_requests = DEFAULT_MAX_REQUESTS;
    let mut max_connections: usize = 64;
    let mut daemon = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--max-pushes" => {
                max_requests = parse_number(&mut args, "--max-pushes")?;
            }
            "--max-connections" => {
                max_connections = parse_number(&mut args, "--max-connections")?;
            }
            "--daemon" => daemon = true,
            other => return Err(format!("{other}\n{USAGE}")),
        }
    }

    // Consent gate: a host that forbids imports never accepts packages.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    // Shared secret comes from the environment so it never shows in ps(1).
    let token = std::env::var("COGNO_TASTE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let own_package: Option<PathBuf> = PathBuf::from(&store_root).join("taste.md").into();
    let accepted_log_path = Path::new(&store_root).join(ACCEPTED_LOG);

    let listener = TcpListener::bind((bind_addr.as_str(), port))
        .map_err(|error| format!("cannot bind {bind_addr}:{port}: {error}"))?;

    if !daemon {
        return serve_connection(
            &listener,
            &settings,
            token.as_deref(),
            max_requests,
            &own_package,
            accepted_log_path.parent().unwrap_or(Path::new(".")),
        );
    }
    // Daemon mode: bounded accept loop with backoff on errors. Connections
    // are handled sequentially so the idempotency set stays deterministic
    // without locks.
    let mut failures = 0_u32;
    for _ in 0..max_connections {
        match serve_connection(
            &listener,
            &settings,
            token.as_deref(),
            max_requests,
            &own_package,
            accepted_log_path.parent().unwrap_or(Path::new(".")),
        ) {
            Ok(()) => failures = 0,
            Err(message) => {
                eprintln!(
                    "{}",
                    serde_json::json!({ "event": "connection_error", "message": message })
                );
                failures += 1;
                if failures >= 5 {
                    return Err(format!("too many consecutive failures ({message})"));
                }
                std::thread::sleep(std::time::Duration::from_millis(100 * u64::from(failures)));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({ "event": "shutdown", "reason": "connection budget exhausted" })
    );
    Ok(())
}

fn parse_number(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    args.next()
        .ok_or(format!("{flag} requires a number"))?
        .parse()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

fn load_accepted(store_root: &Path) -> BTreeSet<String> {
    let path = store_root.join(ACCEPTED_LOG);
    let Ok(file) = std::fs::File::open(path) else {
        return BTreeSet::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| line.len() == 64 && line.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .collect()
}

fn serve_connection(
    listener: &TcpListener,
    settings: &TasteSettings,
    token: Option<&str>,
    max_requests: usize,
    own_package: &Option<PathBuf>,
    store_dir: &Path,
) -> Result<(), String> {
    let (mut stream, peer) = listener
        .accept()
        .map_err(|error| format!("accept failed: {error}"))?;
    println!(
        "{}",
        serde_json::json!({ "event": "connection", "peer": peer.to_string() })
    );

    let mut seen_digests = load_accepted(store_dir);
    let mut config = SessionConfig {
        settings,
        auth_token: token,
        max_requests,
        seen_digests: &mut seen_digests,
    };
    let own_package = own_package.clone();
    let mut lookup = move |digest: &str| -> Option<TastePackage> {
        let path = own_package.as_ref()?;
        let package = TastePackage::load_file(path).ok()?;
        (package.digest_hex() == digest).then_some(package)
    };
    let events = serve_session(&mut stream, &mut config, Some(&mut lookup))
        .map_err(|error| format!("session failed: {error}"))?;
    let mut accepted_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(store_dir.join(ACCEPTED_LOG))
        .map_err(|error| format!("cannot open accepted log: {error}"))?;
    for event in events {
        match event {
            SessionEvent::Push(PushOutcome::Accepted(digest)) => {
                writeln!(accepted_log, "{digest}")
                    .map_err(|error| format!("cannot record acceptance: {error}"))?;
                println!(
                    "{}",
                    serde_json::json!({ "event": "accepted", "digest": digest })
                );
            }
            SessionEvent::Push(PushOutcome::Duplicate(digest)) => println!(
                "{}",
                serde_json::json!({ "event": "duplicate", "digest": digest })
            ),
            SessionEvent::Push(PushOutcome::Rejected(reason)) => eprintln!(
                "{}",
                serde_json::json!({ "event": "rejected", "reason": reason })
            ),
            SessionEvent::Pull { requested, served } => println!(
                "{}",
                serde_json::json!({ "event": "pull", "digest": requested, "served": served })
            ),
        }
    }
    Ok(())
}
