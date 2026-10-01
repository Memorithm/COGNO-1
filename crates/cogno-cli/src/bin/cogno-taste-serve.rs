//! Receive taste package pushes over TCP (consent-gated, digest-verified).

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_runtime::{load_settings, TastePackage};
use cogno_transport::{
    serve_session, PackagePersister, PushOutcome, SessionConfig, SessionEvent, TransportError,
    DEFAULT_MAX_REQUESTS,
};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const USAGE: &str =
    "usage: cogno-taste-serve STORE_ROOT BIND_ADDR PORT [--max-pushes N] [--io-timeout-ms N]";
const ACCEPTED_LOG: &str = "taste.accepted.log";
const INBOX_DIR: &str = "taste.inbox";
const DEFAULT_IO_TIMEOUT_MS: u64 = 5_000;
const MAX_IO_TIMEOUT_MS: u64 = 300_000;
const MIN_TOKEN_BYTES: usize = 32;
const MAX_TOKEN_BYTES: usize = 128;

fn load_accepted(store_root: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(store_root.join(INBOX_DIR)) else {
        return BTreeSet::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let digest = path.file_stem()?.to_str()?;
            if path.extension()?.to_str()? != "md"
                || digest.len() != 64
                || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return None;
            }
            let package = TastePackage::load_file(&path).ok()?;
            (package.digest_hex() == digest).then(|| digest.to_string())
        })
        .collect()
}

fn required_token(value: Option<String>) -> Result<String, String> {
    let token = value.ok_or("COGNO_TASTE_TOKEN is required for TCP transport")?;
    if !(MIN_TOKEN_BYTES..=MAX_TOKEN_BYTES).contains(&token.len())
        || !token.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(format!(
            "COGNO_TASTE_TOKEN must contain {MIN_TOKEN_BYTES}..={MAX_TOKEN_BYTES} non-whitespace ASCII bytes"
        ));
    }
    Ok(token)
}

#[derive(Debug)]
struct DurableInbox {
    store_root: PathBuf,
}

impl DurableInbox {
    fn append_acceptance(&self, digest: &str) -> Result<(), TransportError> {
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.store_root.join(ACCEPTED_LOG))
            .map_err(|error| TransportError::Io(format!("cannot open acceptance log: {error}")))?;
        writeln!(log, "{digest}")
            .and_then(|()| log.sync_data())
            .map_err(|error| TransportError::Io(format!("cannot sync acceptance log: {error}")))
    }
}

impl PackagePersister for DurableInbox {
    fn persist(&mut self, package: &TastePackage) -> Result<(), TransportError> {
        let digest = package.digest_hex();
        let inbox = self.store_root.join(INBOX_DIR);
        match std::fs::create_dir(&inbox) {
            Ok(()) => {
                #[cfg(unix)]
                File::open(&self.store_root)
                    .and_then(|directory| directory.sync_all())
                    .map_err(|error| {
                        TransportError::Io(format!("cannot sync inbox parent: {error}"))
                    })?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(TransportError::Io(format!("cannot create inbox: {error}"))),
        }
        let target = inbox.join(format!("{digest}.md"));
        if target.exists() {
            let stored = TastePackage::load_file(&target)?;
            if stored.digest_hex() != digest {
                return Err(TransportError::Io(
                    "existing inbox entry does not match its digest".to_string(),
                ));
            }
            return self.append_acceptance(&digest);
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| TransportError::Io(format!("clock before unix epoch: {error}")))?
            .as_nanos();
        let temporary = inbox.join(format!(".{digest}.{}.{}.tmp", std::process::id(), nonce));
        let document = package.render_markdown()?;
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|error| {
                    TransportError::Io(format!("cannot create inbox temporary: {error}"))
                })?;
            file.write_all(document.as_bytes())
                .and_then(|()| file.sync_all())
                .map_err(|error| {
                    TransportError::Io(format!("cannot sync inbox temporary: {error}"))
                })?;
            std::fs::rename(&temporary, &target).map_err(|error| {
                TransportError::Io(format!("cannot publish inbox package: {error}"))
            })?;
            #[cfg(unix)]
            File::open(&inbox)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| TransportError::Io(format!("cannot sync inbox: {error}")))?;
            self.append_acceptance(&digest)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cogno-taste-serve: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let store_root = args.next().ok_or(USAGE)?;
    let bind_addr = args.next().ok_or(USAGE)?;
    let port: u16 = args
        .next()
        .ok_or(USAGE)?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    let mut max_requests = DEFAULT_MAX_REQUESTS;
    let mut io_timeout_ms = DEFAULT_IO_TIMEOUT_MS;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--max-pushes" => {
                max_requests = args
                    .next()
                    .ok_or("--max-pushes requires a number")?
                    .parse()
                    .map_err(|error| format!("invalid --max-pushes: {error}"))?;
            }
            "--io-timeout-ms" => {
                io_timeout_ms = args
                    .next()
                    .ok_or("--io-timeout-ms requires a number")?
                    .parse()
                    .map_err(|error| format!("invalid --io-timeout-ms: {error}"))?;
            }
            other => return Err(format!("{other}\n{USAGE}")),
        }
    }
    if max_requests == 0 {
        return Err("--max-pushes must be greater than zero".to_string());
    }
    if io_timeout_ms == 0 || io_timeout_ms > MAX_IO_TIMEOUT_MS {
        return Err(format!(
            "--io-timeout-ms must be in 1..={MAX_IO_TIMEOUT_MS}"
        ));
    }

    // Consent gate: a host that forbids imports never accepts packages.
    let settings = load_settings(Path::new(&store_root)).map_err(|error| error.to_string())?;
    // Shared secret comes from the environment so it never shows in ps(1).
    let token = required_token(std::env::var("COGNO_TASTE_TOKEN").ok())?;
    // The host's own package is pullable by digest.
    let own_package: Option<PathBuf> = PathBuf::from(&store_root).join("taste.md").into();

    let listener = TcpListener::bind((bind_addr.as_str(), port))
        .map_err(|error| format!("cannot bind {bind_addr}:{port}: {error}"))?;
    let (mut stream, peer) = listener
        .accept()
        .map_err(|error| format!("accept failed: {error}"))?;
    let timeout = Some(Duration::from_millis(io_timeout_ms));
    stream
        .set_read_timeout(timeout)
        .and_then(|()| stream.set_write_timeout(timeout))
        .map_err(|error| format!("cannot configure connection timeout: {error}"))?;
    println!(
        "{}",
        serde_json::json!({ "event": "connection", "peer": peer.to_string() })
    );

    let mut seen_digests = load_accepted(Path::new(&store_root));
    let mut inbox = DurableInbox {
        store_root: PathBuf::from(&store_root),
    };
    let mut config = SessionConfig {
        settings: &settings,
        auth_token: Some(&token),
        max_requests,
        seen_digests: &mut seen_digests,
        package_persister: Some(&mut inbox),
    };
    let mut lookup = move |digest: &str| -> Option<TastePackage> {
        let path = own_package.as_ref()?;
        let package = TastePackage::load_file(path).ok()?;
        (package.digest_hex() == digest).then_some(package)
    };
    let events = serve_session(&mut stream, &mut config, Some(&mut lookup))
        .map_err(|error| format!("session failed: {error}"))?;
    for event in events {
        match event {
            SessionEvent::Push(PushOutcome::Accepted(digest)) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcp_token_is_mandatory_and_bounded() {
        assert!(required_token(None).is_err());
        assert!(required_token(Some("short".to_string())).is_err());
        assert!(required_token(Some("x".repeat(MIN_TOKEN_BYTES))).is_ok());
        assert!(required_token(Some("x".repeat(MAX_TOKEN_BYTES + 1))).is_err());
        assert!(required_token(Some(format!("{} ", "x".repeat(MIN_TOKEN_BYTES)))).is_err());
    }
}
