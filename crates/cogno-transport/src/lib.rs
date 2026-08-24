//! # cogno-transport — host-side taste package exchange for COGNO-1.
//!
//! This crate is the single home of remote transport plumbing (étape 6). It
//! keeps [`cogno_core`] network-free and [`cogno_runtime`] file-bound by
//! owning the wire layer itself:
//!
//! - a **bounded, newline-framed protocol** (`COGNO-TASTE/1`) over any
//!   `Read + Write` byte stream — std only, no async runtime, no framework;
//! - **end-to-end verification on receive**: the full document is read,
//!   then re-parsed and digest-checked by
//!   [`cogno_runtime::TastePackage::parse_markdown`] before it is handed to
//!   the caller. Truncated, oversized or tampered payloads fail closed;
//! - **consent gates on both edges**: sending requires
//!   [`TasteSettings::can_export`], accepting requires
//!   [`TasteSettings::can_import`]. The wire layer refuses to move learned
//!   preferences without the host's explicit permission.
//!
//! The protocol is deliberately minimal so carriers can be swapped (TCP,
//! Unix sockets, in-memory duplex in tests) without touching verification.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TasteSettings;
use cogno_runtime::{TastePackage, TastePackageError};
use std::io::{Read, Write};

/// Wire identifier carried in the request line.
pub const PROTOCOL_ID: &str = "COGNO-TASTE/1";

/// Largest accepted header line, including the trailing `\n`.
pub const MAX_HEADER_BYTES: usize = 256;

/// Largest accepted payload, mirroring the package document bound.
pub const MAX_PAYLOAD_BYTES: usize = cogno_runtime::MAX_PACKAGE_DOCUMENT_BYTES;

/// One push: a verified package travels with its declared size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushRequest {
    /// Declared payload length in bytes (must match the actual body).
    pub payload_len: usize,
}

/// Outcome of a completed push.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    /// Receiver parsed and verified the package; carries its digest hex.
    Accepted(String),
    /// Receiver refused; carries the machine-readable reason code.
    Rejected(String),
}

/// Failure modes of the framing and consent layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// Consent denied on the sending side (export not allowed).
    ExportDeniedByConsent,
    /// Consent denied on the receiving side (import not allowed).
    ImportDeniedByConsent,
    /// Header line was malformed, oversized or unknown.
    MalformedHeader,
    /// Declared length exceeded [`MAX_PAYLOAD_BYTES`] or the stream ended early.
    InvalidLength,
    /// Reading or writing the underlying stream failed.
    Io(String),
    /// The received document failed package verification.
    Package(TastePackageError),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExportDeniedByConsent => write!(f, "export denied by host consent"),
            Self::ImportDeniedByConsent => write!(f, "import denied by host consent"),
            Self::MalformedHeader => write!(f, "malformed protocol header"),
            Self::InvalidLength => write!(f, "invalid or truncated payload length"),
            Self::Io(message) => write!(f, "transport i/o error: {message}"),
            Self::Package(error) => write!(f, "package rejected: {error:?}"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<TastePackageError> for TransportError {
    fn from(error: TastePackageError) -> Self {
        Self::Package(error)
    }
}

/// Send one verified package over `stream` (client edge).
///
/// Consent is enforced here so callers cannot bypass it: an export-denying
/// host never puts preference bytes on the wire.
pub fn push_package<S: Read + Write>(
    stream: &mut S,
    package: &TastePackage,
    settings: &TasteSettings,
) -> Result<PushOutcome, TransportError> {
    if !settings.can_export() {
        return Err(TransportError::ExportDeniedByConsent);
    }
    let document = package.render_markdown()?;
    let header = format!("{} PUSH {}\n", PROTOCOL_ID, document.len());
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(document.as_bytes()))
        .map_err(|error| TransportError::Io(error.to_string()))?;
    let response = read_line_bounded(stream)?;
    if let Some(digest) = response.strip_prefix("OK ") {
        return Ok(PushOutcome::Accepted(digest.trim().to_string()));
    }
    if let Some(reason) = response.strip_prefix("ERR ") {
        return Ok(PushOutcome::Rejected(reason.trim().to_string()));
    }
    Err(TransportError::MalformedHeader)
}

/// Serve exactly one push on `stream` (server edge).
///
/// The whole payload is consumed first (so a truncated or hostile sender can
/// never leave the connection half-read), then consent is checked, then the
/// document is fully verified before acceptance. Any refusal answers `ERR`
/// on the wire instead of silently dropping the connection.
pub fn serve_one_push<S: Read + Write>(
    stream: &mut S,
    settings: &TasteSettings,
) -> Result<PushOutcome, TransportError> {
    let header = read_line_bounded(stream)?;
    let mut parts = header.split_whitespace();
    let protocol = parts.next().unwrap_or("");
    let verb = parts.next().unwrap_or("");
    let declared: usize = parts.next().and_then(|len| len.parse().ok()).unwrap_or(0);
    if protocol != PROTOCOL_ID || verb != "PUSH" || parts.next().is_some() {
        return Err(TransportError::MalformedHeader);
    }
    if declared == 0 || declared > MAX_PAYLOAD_BYTES {
        return Err(TransportError::InvalidLength);
    }
    let mut payload = vec![0_u8; declared];
    stream
        .read_exact(&mut payload)
        .map_err(|error| TransportError::Io(error.to_string()))?;
    if !settings.can_import() {
        answer(stream, "ERR import_denied_by_consent")?;
        return Ok(PushOutcome::Rejected(
            "import_denied_by_consent".to_string(),
        ));
    }
    match TastePackage::parse_markdown(&payload) {
        Ok(package) => {
            answer(stream, &format!("OK {}", package.digest_hex()))?;
            Ok(PushOutcome::Accepted(package.digest_hex()))
        }
        Err(error) => {
            answer(stream, "ERR verification_failed")?;
            Err(TransportError::Package(error))
        }
    }
}

fn answer<S: Write>(stream: &mut S, line: &str) -> Result<(), TransportError> {
    stream
        .write_all(line.as_bytes())
        .and_then(|()| stream.write_all(b"\n"))
        .map_err(|error| TransportError::Io(error.to_string()))
}

/// Read one `\n`-terminated line with a hard bound; no line may exceed
/// [`MAX_HEADER_BYTES`].
fn read_line_bounded<S: Read>(stream: &mut S) -> Result<String, TransportError> {
    let mut line = Vec::new();
    let mut byte = [0_u8; 1];
    loop {
        let read = stream
            .read(&mut byte)
            .map_err(|error| TransportError::Io(error.to_string()))?;
        if read == 0 {
            return Err(TransportError::MalformedHeader);
        }
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
        if line.len() > MAX_HEADER_BYTES {
            return Err(TransportError::MalformedHeader);
        }
    }
    String::from_utf8(line).map_err(|_| TransportError::MalformedHeader)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package() -> TastePackage {
        let events: Vec<cogno_core::TasteEvent> = vec![
            event(1, cogno_core::TasteEventKind::Proposed, "claude-x"),
            event(2, cogno_core::TasteEventKind::Confirmed, "scirust@gen7"),
            event(3, cogno_core::TasteEventKind::Confirmed, "scirust@gen7"),
        ];
        let profile =
            cogno_core::ScientificTasteProfile::derive(&events, cogno_core::TastePolicy::DEFAULT)
                .expect("profile");
        TastePackage::from_profile(&profile, cogno_core::TastePolicy::DEFAULT).expect("package")
    }

    fn event(event_id: u64, kind: cogno_core::TasteEventKind, id: &str) -> cogno_core::TasteEvent {
        cogno_core::TasteEvent {
            event_id,
            preference_id: 7,
            scope: cogno_core::TasteScope::Project,
            scope_key: "project:cogno-1".to_string(),
            kind,
            origin: if matches!(kind, cogno_core::TasteEventKind::Proposed) {
                cogno_core::TasteOrigin::ModelInference
            } else {
                cogno_core::TasteOrigin::DeterministicEvaluation
            },
            confidence_bps: 5_000,
            evidence_ids: vec![event_id],
            source: cogno_core::TasteSource::try_new(
                if matches!(kind, cogno_core::TasteEventKind::Proposed) {
                    cogno_core::TasteSourceKind::AssistanceModel
                } else {
                    cogno_core::TasteSourceKind::DeterministicKernel
                },
                id,
            )
            .expect("source"),
        }
    }

    fn granted() -> TasteSettings {
        TasteSettings {
            learning_enabled: true,
            export_allowed: true,
            import_allowed: true,
        }
    }

    #[test]
    fn push_round_trips_over_a_duplex_pair() {
        let (client_side, server_side) = std::os::unix::net::UnixStream::pair().expect("pair");
        let mut client = client_side;
        let package = package();
        let expected = package.digest_hex();
        let server_job =
            std::thread::spawn(move || serve_one_push(&mut { server_side }, &granted()));
        let outcome = push_package(&mut client, &package, &granted()).expect("push");
        assert_eq!(outcome, PushOutcome::Accepted(expected.clone()));
        assert_eq!(
            server_job.join().expect("join"),
            Ok(PushOutcome::Accepted(expected))
        );
    }

    #[test]
    fn consent_denies_both_edges_before_any_bytes_move_meaningfully() {
        let package = package();
        let local_only = TasteSettings::default();
        // Sender refuses without export consent.
        let (mut a, mut b) = std::os::unix::net::UnixStream::pair().expect("pair");
        assert_eq!(
            push_package(&mut a, &package, &local_only),
            Err(TransportError::ExportDeniedByConsent)
        );
        // Receiver accepts the bytes on the wire but refuses to hand them over.
        let header = format!("{PROTOCOL_ID} PUSH 1\n");
        a.write_all(header.as_bytes()).expect("header");
        a.write_all(b"x").expect("body");
        assert_eq!(
            serve_one_push(&mut b, &local_only),
            Ok(PushOutcome::Rejected(
                "import_denied_by_consent".to_string()
            ))
        );
    }

    #[test]
    fn tampered_payloads_fail_verification_and_are_answered_with_err() {
        let (mut client, mut server) = std::os::unix::net::UnixStream::pair().expect("pair");
        let package = package();
        let mut document = package.render_markdown().expect("document");
        let marker = "\"confidence_bps\":";
        let flip = document.find(marker).expect("marker") + marker.len();
        let digit = document.as_bytes()[flip];
        let flipped = if digit == b'8' { b'9' } else { b'8' };
        document.replace_range(flip..flip + 1, &(flipped as char).to_string());
        // A hostile sender writes its own frame; no consent, no verification.
        {
            use std::io::Write as _;
            let header = format!("{} PUSH {}\n", PROTOCOL_ID, document.len());
            client.write_all(header.as_bytes()).expect("header");
            client.write_all(document.as_bytes()).expect("body");
        }
        // The receiver verifies end-to-end and refuses.
        assert!(matches!(
            serve_one_push(&mut server, &granted()),
            Err(TransportError::Package(_))
        ));
        // ...and still answers ERR so the peer learns of the refusal.
        drop(server);
        let mut answer = String::new();
        std::io::Read::read_to_string(&mut client, &mut answer).expect("answer");
        assert_eq!(answer, "ERR verification_failed\n");
    }

    #[test]
    fn oversized_declared_lengths_fail_closed() {
        let (mut a, mut b) = std::os::unix::net::UnixStream::pair().expect("pair");
        let header = format!("{} PUSH {}\n", PROTOCOL_ID, MAX_PAYLOAD_BYTES + 1);
        a.write_all(header.as_bytes()).expect("write");
        assert_eq!(
            serve_one_push(&mut b, &granted()),
            Err(TransportError::InvalidLength)
        );
        // Unknown protocol id.
        let header = format!("OTHER/1 PUSH {}\n", 8);
        a.write_all(header.as_bytes()).expect("write");
        assert_eq!(
            serve_one_push(&mut b, &granted()),
            Err(TransportError::MalformedHeader)
        );
    }
}
