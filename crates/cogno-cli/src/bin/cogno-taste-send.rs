//! Push one verified taste package over TCP (consent-gated on the sender).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{load_settings, TastePackage};
use cogno_transport::{push_package, push_package_authorized, PushOutcome};
use std::net::TcpStream;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-send: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-send STORE_ROOT HOST PORT PACKAGE_PATH")?;
    let host = args
        .next()
        .ok_or("usage: cogno-taste-send STORE_ROOT HOST PORT PACKAGE_PATH")?;
    let port: u16 = args
        .next()
        .ok_or("usage: cogno-taste-send STORE_ROOT HOST PORT PACKAGE_PATH")?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    let package_path = args
        .next()
        .ok_or("usage: cogno-taste-send STORE_ROOT HOST PORT PACKAGE_PATH")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    // Consent gate: a host that forbids exports never sends preferences.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let package = TastePackage::load_file(&package_path)
        .map_err(|error| format!("cannot verify {package_path}: {error:?}"))?;
    // Shared secret comes from the environment so it never shows in ps(1).
    let token = std::env::var("COGNO_TASTE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());

    let mut stream = TcpStream::connect((host.as_str(), port))
        .map_err(|error| format!("cannot connect to {host}:{port}: {error}"))?;
    let outcome = match token.as_deref() {
        Some(token) => push_package_authorized(&mut stream, &package, &settings, token),
        None => push_package(&mut stream, &package, &settings),
    };
    match outcome.map_err(|error| format!("send failed: {error}"))? {
        PushOutcome::Accepted(digest) | PushOutcome::Duplicate(digest) => {
            println!(
                "{}",
                serde_json::json!({ "event": "accepted", "digest": digest })
            );
            Ok(())
        }
        PushOutcome::Rejected(reason) => {
            eprintln!(
                "{}",
                serde_json::json!({ "event": "rejected", "reason": reason })
            );
            std::process::exit(3);
        }
    }
}
