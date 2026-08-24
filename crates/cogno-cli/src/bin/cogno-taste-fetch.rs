//! Fetch a verified taste package from a peer by digest (pull mode).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::load_settings;
use cogno_transport::request_package_authorized;
use std::net::TcpStream;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-fetch: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args
        .next()
        .ok_or("usage: cogno-taste-fetch STORE_ROOT HOST PORT DIGEST_HEX")?;
    let host = args
        .next()
        .ok_or("usage: cogno-taste-fetch STORE_ROOT HOST PORT DIGEST_HEX")?;
    let port: u16 = args
        .next()
        .ok_or("usage: cogno-taste-fetch STORE_ROOT HOST PORT DIGEST_HEX")?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    let digest = args
        .next()
        .ok_or("usage: cogno-taste-fetch STORE_ROOT HOST PORT DIGEST_HEX")?;
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("DIGEST_HEX must be 64 hex characters".to_string());
    }

    // Pulling is ingestion: the import consent gate applies.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    let token = std::env::var("COGNO_TASTE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());

    let mut stream = TcpStream::connect((host.as_str(), port))
        .map_err(|error| format!("cannot connect to {host}:{port}: {error}"))?;
    match request_package_authorized(&mut stream, &digest, &settings, token.as_deref())
        .map_err(|error| format!("fetch failed: {error}"))?
    {
        Some(package) => {
            package
                .save_with_consent(Path::new(&store_root), &settings)
                .map_err(|error| format!("cannot store fetched package: {error:?}"))?;
            println!(
                "{}",
                serde_json::json!({ "event": "fetched", "digest": package.digest_hex() })
            );
            Ok(())
        }
        None => Err(format!("peer does not know digest {digest}")),
    }
}
