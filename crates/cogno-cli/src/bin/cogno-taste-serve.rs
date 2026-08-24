//! Receive one taste package push over TCP (consent-gated, digest-verified).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::load_settings;
use cogno_transport::{serve_one_push, PushOutcome};
use std::net::TcpListener;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-serve: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-serve STORE_ROOT BIND_ADDR PORT")?;
    let bind_addr = args
        .next()
        .ok_or("usage: cogno-taste-serve STORE_ROOT BIND_ADDR PORT")?;
    let port: u16 = args
        .next()
        .ok_or("usage: cogno-taste-serve STORE_ROOT BIND_ADDR PORT")?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    // Consent gate: a host that forbids imports never accepts packages.
    let settings = load_settings(Path::new(&store_root))
        .map_err(|error| format!("cannot load host consent: {error}"))?;

    let listener = TcpListener::bind((bind_addr.as_str(), port))
        .map_err(|error| format!("cannot bind {bind_addr}:{port}: {error}"))?;
    let (mut stream, peer) = listener
        .accept()
        .map_err(|error| format!("accept failed: {error}"))?;
    println!(
        "{}",
        serde_json::json!({ "event": "connection", "peer": peer.to_string() })
    );
    match serve_one_push(&mut stream, &settings) {
        Ok(PushOutcome::Accepted(digest)) => {
            println!(
                "{}",
                serde_json::json!({ "event": "accepted", "digest": digest })
            );
            Ok(())
        }
        Ok(PushOutcome::Rejected(reason)) => {
            eprintln!(
                "{}",
                serde_json::json!({ "event": "rejected", "reason": reason })
            );
            std::process::exit(3);
        }
        Err(error) => Err(format!("push failed: {error}")),
    }
}
