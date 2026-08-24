//! Push one verified taste package over TCP (consent-gated on the sender).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::load_settings;
use cogno_transport::{push_package, PushOutcome};
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
    let settings = load_settings(Path::new(&store_root))
        .map_err(|error| format!("cannot load host consent: {error}"))?;
    let package = cogno_runtime::TastePackage::load_file(&package_path)
        .map_err(|error| format!("cannot verify {}: {error:?}", package_path))?;

    let mut stream = TcpStream::connect((host.as_str(), port))
        .map_err(|error| format!("cannot connect to {host}:{port}: {error}"))?;
    match push_package(&mut stream, &package, &settings) {
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
        Err(error) => Err(format!("send failed: {error}")),
    }
}
