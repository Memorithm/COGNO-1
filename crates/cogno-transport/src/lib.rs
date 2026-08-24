//! # cogno-transport — host-side taste package exchange for COGNO-1.
//!
//! This crate is the single home of remote transport plumbing (étape 6). It
//! keeps [`cogno_core`] network-free and [`cogno_runtime`] file-bound by
//! owning the wire layer itself:
//!
//! - a **bounded, newline-framed protocol** (`COGNO-TASTE/1`) over any
//!   `Read + Write` byte stream — std only, no async runtime, no framework;
//! - **end-to-end verification on receive**: every document is fully read,
//!   then re-parsed and digest-checked by
//!   [`cogno_runtime::TastePackage::parse_markdown`] before it is handed to
//!   the caller. Truncated, oversized or tampered payloads fail closed;
//! - **consent gates on both edges**: sending requires
//!   [`TasteSettings::can_export`], accepting requires
//!   [`TasteSettings::can_import`];
//! - **hardened sessions**: optional shared-secret authentication
//!   (constant-time compared), multiple requests per connection under a hard
//!   budget, idempotent re-push detection, pull mode for fetching a package
//!   by digest, and sender-side retry across reconnects.
//!
//! ## Wire summary
//!
//! ```text
//! COGNO-TASTE/1 PUSH <len> [AUTH <token>] \n <payload>   ->  OK|DUP|ERR ...
//! COGNO-TASTE/1 GET <digest> [AUTH <token>] \n           ->  PKG <len> \n <payload> | ERR not_found
//! ```

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::TasteSettings;
use cogno_runtime::{TastePackage, TastePackageError};

/// Callback resolving a digest to a locally-held package (pull mode).
pub type PackageLookup<'a> = dyn FnMut(&str) -> Option<TastePackage> + 'a;
use std::collections::BTreeSet;
use std::io::{Read, Write};

/// Wire identifier carried in the request line.
pub const PROTOCOL_ID: &str = "COGNO-TASTE/1";

/// Largest accepted header line, including the trailing `\n`.
pub const MAX_HEADER_BYTES: usize = 256;

/// Largest accepted payload, mirroring the package document bound.
pub const MAX_PAYLOAD_BYTES: usize = cogno_runtime::MAX_PACKAGE_DOCUMENT_BYTES;

/// Default per-connection request budget.
pub const DEFAULT_MAX_REQUESTS: usize = 8;

/// Outcome of a completed push.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    /// Receiver parsed and verified the package; carries its digest hex.
    Accepted(String),
    /// Receiver already had this exact digest; nothing was double-counted.
    Duplicate(String),
    /// Receiver refused; carries the machine-readable reason code.
    Rejected(String),
}

/// One event observed by the server during a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    /// A push was processed.
    Push(PushOutcome),
    /// A pull was answered (`served = false` means digest unknown locally).
    Pull { requested: String, served: bool },
}

/// Server-side configuration for one connection.
#[derive(Debug)]
pub struct SessionConfig<'a> {
    /// Consent gates for this host.
    pub settings: &'a TasteSettings,
    /// Shared secret required on every request; `None` disables auth.
    pub auth_token: Option<&'a str>,
    /// Hard budget of requests served before the connection must be closed.
    pub max_requests: usize,
    /// Idempotency memory: digests already accepted (per host, caller-owned).
    pub seen_digests: &'a mut BTreeSet<String>,
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
    /// The peer supplied a missing or wrong authentication token.
    AuthenticationFailed,
    /// A pulled package did not match its requested digest.
    DigestMismatch,
    /// The peer answered `ERR <reason>` to our request.
    PeerRejected(String),
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
            Self::AuthenticationFailed => write!(f, "authentication failed"),
            Self::DigestMismatch => write!(f, "pulled package digest mismatch"),
            Self::PeerRejected(reason) => write!(f, "peer rejected request: {reason}"),
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

/// Send one verified package (client edge, no authentication).
pub fn push_package<S: Read + Write>(
    stream: &mut S,
    package: &TastePackage,
    settings: &TasteSettings,
) -> Result<PushOutcome, TransportError> {
    match send_request(stream, Request::Push(package), settings, None)? {
        Response::Status(outcome) => Ok(outcome),
        Response::Package(_) => Err(TransportError::MalformedHeader),
    }
}

/// Send one verified package with a shared-secret token (client edge).
pub fn push_package_authorized<S: Read + Write>(
    stream: &mut S,
    package: &TastePackage,
    settings: &TasteSettings,
    token: &str,
) -> Result<PushOutcome, TransportError> {
    match send_request(stream, Request::Push(package), settings, Some(token))? {
        Response::Status(outcome) => Ok(outcome),
        Response::Package(_) => Err(TransportError::MalformedHeader),
    }
}

/// Fetch a verified package by digest (client edge). Returns `Ok(None)` when
/// the peer does not know the digest.
pub fn request_package<S: Read + Write>(
    stream: &mut S,
    digest_hex: &str,
    settings: &TasteSettings,
) -> Result<Option<TastePackage>, TransportError> {
    match request_package_authorized(stream, digest_hex, settings, None)? {
        Some(package) => Ok(Some(package)),
        None => Ok(None),
    }
}

/// Same as [`request_package`] with a shared-secret token.
pub fn request_package_authorized<S: Read + Write>(
    stream: &mut S,
    digest_hex: &str,
    settings: &TasteSettings,
    token: Option<&str>,
) -> Result<Option<TastePackage>, TransportError> {
    // Pulling is a form of ingestion: an import-denying host never fetches.
    if !settings.can_import() {
        return Err(TransportError::ImportDeniedByConsent);
    }
    let response = send_request(stream, Request::Get(digest_hex), settings, token)?;
    match response {
        Response::Package(document) => {
            let package = TastePackage::parse_markdown(&document)?;
            // Never trust an unverified mapping between request and answer.
            if package.digest_hex() != digest_hex.to_ascii_lowercase() {
                return Err(TransportError::DigestMismatch);
            }
            Ok(Some(package))
        }
        Response::Status(outcome) => match outcome {
            PushOutcome::Rejected(reason) if reason == "not_found" => Ok(None),
            PushOutcome::Rejected(reason) if reason == "auth_failed" => {
                Err(TransportError::AuthenticationFailed)
            }
            PushOutcome::Rejected(reason) => Err(TransportError::PeerRejected(reason)),
            _ => Err(TransportError::MalformedHeader),
        },
    }
}

enum Request<'p> {
    Push(&'p TastePackage),
    Get(&'p str),
}

enum Response {
    Status(PushOutcome),
    Package(Vec<u8>),
}

fn send_request<S: Read + Write>(
    stream: &mut S,
    request: Request<'_>,
    settings: &TasteSettings,
    token: Option<&str>,
) -> Result<Response, TransportError> {
    let mut header = String::new();
    match request {
        Request::Push(package) => {
            // Consent is enforced here so callers cannot bypass it: an
            // export-denying host never puts preference bytes on the wire.
            if !settings.can_export() {
                return Err(TransportError::ExportDeniedByConsent);
            }
            let document = package.render_markdown()?;
            header.push_str(&format!("{} PUSH {}", PROTOCOL_ID, document.len()));
            append_auth(&mut header, token);
            header.push('\n');
            stream
                .write_all(header.as_bytes())
                .and_then(|()| stream.write_all(document.as_bytes()))
                .map_err(|error| TransportError::Io(error.to_string()))?;
        }
        Request::Get(digest) => {
            if !is_digest_hex(digest) {
                return Err(TransportError::MalformedHeader);
            }
            header.push_str(&format!(
                "{PROTOCOL_ID} GET {}",
                digest.to_ascii_lowercase()
            ));
            append_auth(&mut header, token);
            header.push('\n');
            stream
                .write_all(header.as_bytes())
                .map_err(|error| TransportError::Io(error.to_string()))?;
        }
    }
    read_response(stream)
}

fn append_auth(header: &mut String, token: Option<&str>) {
    if let Some(token) = token.filter(|t| !t.is_empty() && t.len() <= MAX_HEADER_BYTES / 2) {
        header.push_str(&format!(" AUTH {token}"));
    }
}

fn read_response<S: Read>(stream: &mut S) -> Result<Response, TransportError> {
    let line = read_line_bounded(stream)?;
    if let Some(rest) = line.strip_prefix("PKG ") {
        let declared: usize = rest
            .trim()
            .parse()
            .map_err(|_| TransportError::InvalidLength)?;
        if declared == 0 || declared > MAX_PAYLOAD_BYTES {
            return Err(TransportError::InvalidLength);
        }
        let mut payload = vec![0_u8; declared];
        stream
            .read_exact(&mut payload)
            .map_err(|error| TransportError::Io(error.to_string()))?;
        return Ok(Response::Package(payload));
    }
    let outcome = if let Some(digest) = line.strip_prefix("OK ") {
        PushOutcome::Accepted(digest.trim().to_string())
    } else if let Some(digest) = line.strip_prefix("DUP ") {
        PushOutcome::Duplicate(digest.trim().to_string())
    } else if let Some(reason) = line.strip_prefix("ERR ") {
        PushOutcome::Rejected(reason.trim().to_string())
    } else {
        return Err(TransportError::MalformedHeader);
    };
    Ok(Response::Status(outcome))
}

/// Push with sender-side retry: reconnects via `connect` after transport i/o
/// failures only. Consent refusals, peer rejections and verification errors
/// are final and returned immediately.
pub fn push_with_retry<S, F, E>(
    mut connect: F,
    attempts: usize,
    package: &TastePackage,
    settings: &TasteSettings,
    token: Option<&str>,
) -> Result<PushOutcome, TransportError>
where
    S: Read + Write,
    F: FnMut() -> Result<S, E>,
    E: std::fmt::Display,
{
    if !settings.can_export() {
        return Err(TransportError::ExportDeniedByConsent);
    }
    let mut last_error = TransportError::ExportDeniedByConsent;
    for _ in 0..attempts.max(1) {
        let mut stream = connect().map_err(|error| TransportError::Io(error.to_string()))?;
        match send_request(&mut stream, Request::Push(package), settings, token) {
            Ok(Response::Status(outcome)) => return Ok(outcome),
            Ok(Response::Package(_)) => return Err(TransportError::MalformedHeader),
            Err(error @ TransportError::Io(_)) => last_error = error,
            Err(other) => return Err(other),
        }
    }
    Err(last_error)
}

/// Serve one full request on `stream`, answering on the wire before returning.
///
/// The whole payload is consumed first (so a truncated or hostile sender can
/// never leave the connection half-read), then authentication, consent and
/// finally full package verification decide the outcome. Refusals are always
/// answered with `ERR` instead of silently dropping the connection.
pub fn serve_request<S: Read + Write>(
    stream: &mut S,
    config: &mut SessionConfig<'_>,
    lookup: Option<&mut PackageLookup<'_>>,
) -> Result<SessionEvent, TransportError> {
    let header = read_line_bounded(stream)?;
    let mut parts = header.split_whitespace();
    let protocol = parts.next().unwrap_or("");
    let verb = parts.next().unwrap_or("");
    if protocol != PROTOCOL_ID {
        return Err(TransportError::MalformedHeader);
    }
    // Collect remaining fields, extracting an optional trailing AUTH pair so
    // verb-specific parsing below sees only its own fields.
    let mut fields: Vec<&str> = parts.collect();
    let provided = extract_auth_token(&mut fields)?;
    // Authentication happens before anything else; a wrong token never
    // reaches consent or verification paths. A rejected PUSH still has its
    // whole frame drained first, so the stream stays framed for the next
    // request instead of desynchronising.
    if !authenticate(config.auth_token, provided) {
        if verb == "PUSH" {
            drain_frame(stream, fields.first().copied())?;
        }
        answer(stream, "ERR auth_failed")?;
        return Ok(SessionEvent::Push(PushOutcome::Rejected(
            "auth_failed".to_string(),
        )));
    }
    match verb {
        "PUSH" => serve_push(stream, config, &fields).map(SessionEvent::Push),
        "GET" => {
            let Some(lookup) = lookup else {
                answer(stream, "ERR pulls_disabled")?;
                return Ok(SessionEvent::Pull {
                    requested: String::new(),
                    served: false,
                });
            };
            if fields.len() != 1 || !is_digest_hex(fields[0]) {
                return Err(TransportError::MalformedHeader);
            }
            let digest = fields[0];
            match lookup(&digest.to_ascii_lowercase()) {
                Some(package) => {
                    let document = package.render_markdown()?;
                    answer(stream, &format!("PKG {}", document.len()))?;
                    stream
                        .write_all(document.as_bytes())
                        .map_err(|error| TransportError::Io(error.to_string()))?;
                    Ok(SessionEvent::Pull {
                        requested: digest.to_ascii_lowercase(),
                        served: true,
                    })
                }
                None => {
                    answer(stream, "ERR not_found")?;
                    Ok(SessionEvent::Pull {
                        requested: digest.to_ascii_lowercase(),
                        served: false,
                    })
                }
            }
        }
        _ => Err(TransportError::MalformedHeader),
    }
}

/// Read and discard one full PUSH frame based on its declared length. Fails
/// closed (connection dies) when the header carries no usable length: an
/// unframed stream cannot be resynchronised safely.
fn drain_frame<S: Read>(stream: &mut S, declared: Option<&str>) -> Result<(), TransportError> {
    let declared: usize = declared.and_then(|len| len.parse().ok()).unwrap_or(0);
    if declared == 0 || declared > MAX_PAYLOAD_BYTES {
        return Err(TransportError::InvalidLength);
    }
    let mut sink = vec![0_u8; declared];
    stream
        .read_exact(&mut sink)
        .map_err(|error| TransportError::Io(error.to_string()))
}

fn extract_auth_token<'a>(fields: &mut Vec<&'a str>) -> Result<Option<&'a str>, TransportError> {
    let Some(position) = fields.iter().position(|field| *field == "AUTH") else {
        return Ok(None);
    };
    if fields.len() != position + 2 {
        return Err(TransportError::MalformedHeader);
    }
    // Remove the pair so verb parsing only sees its own fields.
    let token = fields[position + 1];
    fields.truncate(position);
    Ok(Some(token))
}

fn authenticate(expected: Option<&str>, provided: Option<&str>) -> bool {
    match expected {
        None => true,
        Some(expected) => {
            matches!(provided, Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()))
        }
    }
}

fn serve_push<S: Read + Write>(
    stream: &mut S,
    config: &mut SessionConfig<'_>,
    fields: &[&str],
) -> Result<PushOutcome, TransportError> {
    let declared: usize = fields.first().and_then(|len| len.parse().ok()).unwrap_or(0);
    if fields.len() != 1 || declared == 0 || declared > MAX_PAYLOAD_BYTES {
        return Err(TransportError::InvalidLength);
    }
    // Consume the entire frame first: half-read connections are worse than
    // honest refusals.
    let mut payload = vec![0_u8; declared];
    stream
        .read_exact(&mut payload)
        .map_err(|error| TransportError::Io(error.to_string()))?;
    if !config.settings.can_import() {
        answer(stream, "ERR import_denied_by_consent")?;
        return Ok(PushOutcome::Rejected(
            "import_denied_by_consent".to_string(),
        ));
    }
    match TastePackage::parse_markdown(&payload) {
        Ok(package) => {
            let digest = package.digest_hex();
            // Idempotency: an already-accepted digest is acknowledged without
            // counting twice anywhere downstream.
            if !config.seen_digests.insert(digest.clone()) {
                answer(stream, &format!("DUP {digest}"))?;
                return Ok(PushOutcome::Duplicate(digest));
            }
            answer(stream, &format!("OK {digest}"))?;
            Ok(PushOutcome::Accepted(digest))
        }
        Err(error) => {
            answer(stream, "ERR verification_failed")?;
            Err(TransportError::Package(error))
        }
    }
}

/// Serve a whole connection: repeated requests under a hard budget, until
/// the peer closes the stream or the budget is exhausted (answered with
/// `ERR too_many_requests`). Pull support requires a `lookup`.
pub fn serve_session<S: Read + Write>(
    stream: &mut S,
    config: &mut SessionConfig<'_>,
    mut lookup: Option<&mut PackageLookup<'_>>,
) -> Result<Vec<SessionEvent>, TransportError> {
    let mut events = Vec::new();
    for _ in 0..config.max_requests.max(1) {
        match serve_request(stream, config, lookup.as_deref_mut()) {
            Ok(event) => events.push(event),
            Err(TransportError::MalformedHeader) if peek_eof(stream) => break,
            Err(error) => return Err(error),
        }
    }
    if events.len() >= config.max_requests.max(1) {
        // Budget exhausted: tell the peer why the next request would fail.
        let _ = answer(stream, "ERR too_many_requests");
    }
    Ok(events)
}

fn peek_eof<S: Read>(stream: &mut S) -> bool {
    let mut byte = [0_u8; 1];
    matches!(stream.read(&mut byte), Ok(0))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0_u8;
    for (a, b) in left.iter().zip(right.iter()) {
        difference |= a ^ b;
    }
    difference == 0
}

fn is_digest_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
    use std::collections::BTreeMap;

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

    fn config<'a>(
        settings: &'a TasteSettings,
        seen: &'a mut BTreeSet<String>,
    ) -> SessionConfig<'a> {
        SessionConfig {
            settings,
            auth_token: None,
            max_requests: DEFAULT_MAX_REQUESTS,
            seen_digests: seen,
        }
    }

    #[test]
    fn push_round_trips_over_a_duplex_pair() {
        let (mut client, server) = std::os::unix::net::UnixStream::pair().expect("pair");
        let pkg = package();
        let expected = pkg.digest_hex();
        let server_job = std::thread::spawn(move || {
            let settings = granted();
            let mut seen = BTreeSet::new();
            let mut cfg = config(&settings, &mut seen);
            serve_session(&mut { server }, &mut cfg, None)
        });
        let outcome = push_package(&mut client, &pkg, &granted()).expect("push");
        assert_eq!(outcome, PushOutcome::Accepted(expected.clone()));
        drop(client);
        let events = server_job.join().expect("join").expect("serve");
        assert_eq!(
            events,
            vec![SessionEvent::Push(PushOutcome::Accepted(expected))]
        );
    }

    #[test]
    fn duplicate_pushes_are_acknowledged_not_recounted() {
        let (mut client, server) = std::os::unix::net::UnixStream::pair().expect("pair");
        let pkg = package();
        let expected = pkg.digest_hex();
        let server_job = std::thread::spawn(move || {
            let settings = granted();
            let mut seen = BTreeSet::new();
            let mut cfg = config(&settings, &mut seen);
            serve_session(&mut { server }, &mut cfg, None)
        });
        let first = push_package(&mut client, &pkg, &granted()).expect("push 1");
        let second = push_package(&mut client, &pkg, &granted()).expect("push 2");
        assert_eq!(first, PushOutcome::Accepted(expected.clone()));
        assert_eq!(second, PushOutcome::Duplicate(expected));
        drop(client);
        let _ = server_job.join().expect("join");
    }

    #[test]
    fn consent_denies_both_edges() {
        let pkg = package();
        let local_only = TasteSettings {
            learning_enabled: false,
            ..TasteSettings::default()
        };
        // Sender refuses without export consent.
        let (mut a, mut b) = std::os::unix::net::UnixStream::pair().expect("pair");
        assert_eq!(
            push_package(&mut a, &pkg, &local_only),
            Err(TransportError::ExportDeniedByConsent)
        );
        // Receiver consumes the frame then refuses the import.
        a.write_all(format!("{PROTOCOL_ID} PUSH 1\n").as_bytes())
            .expect("header");
        a.write_all(b"x").expect("body");
        let mut seen = BTreeSet::new();
        let mut cfg = config(&local_only, &mut seen);
        assert_eq!(
            serve_request(&mut b, &mut cfg, None),
            Ok(SessionEvent::Push(PushOutcome::Rejected(
                "import_denied_by_consent".to_string()
            )))
        );
        // Pulling is import-gated too.
        assert_eq!(
            request_package(&mut a, &"0".repeat(64), &local_only),
            Err(TransportError::ImportDeniedByConsent)
        );
    }

    #[test]
    fn authentication_is_enforced_before_consent_and_verification() {
        let (mut client, server) = std::os::unix::net::UnixStream::pair().expect("pair");
        let pkg = package();
        let digest = pkg.digest_hex();
        let server_job = std::thread::spawn(move || {
            let settings = granted();
            let mut seen = BTreeSet::new();
            let mut cfg = SessionConfig {
                settings: &settings,
                auth_token: Some("s3cret-token"),
                max_requests: DEFAULT_MAX_REQUESTS,
                seen_digests: &mut seen,
            };
            serve_session(&mut { server }, &mut cfg, None)
        });
        // Wrong token -> rejected on the wire, nothing verified.
        let wrong = push_package_authorized(&mut client, &pkg, &granted(), "wrong").expect("push");
        assert_eq!(wrong, PushOutcome::Rejected("auth_failed".to_string()));
        // Right token -> accepted.
        let right =
            push_package_authorized(&mut client, &pkg, &granted(), "s3cret-token").expect("push");
        assert_eq!(right, PushOutcome::Accepted(digest));
        drop(client);
        let _ = server_job.join().expect("join");
    }

    #[test]
    fn pull_returns_the_verified_requested_package_or_not_found() {
        let (mut client, server) = std::os::unix::net::UnixStream::pair().expect("pair");
        let pkg = package();
        let digest = pkg.digest_hex();
        let missing = "f".repeat(64);
        let mut library: BTreeMap<String, TastePackage> = BTreeMap::new();
        library.insert(digest.clone(), pkg);

        let server_job = std::thread::spawn(move || {
            let settings = granted();
            let mut seen = BTreeSet::new();
            let mut cfg = config(&settings, &mut seen);
            serve_session(
                &mut { server },
                &mut cfg,
                Some(&mut |digest| library.get(digest).cloned()),
            )
        });
        let fetched = request_package(&mut client, &digest, &granted())
            .expect("pull")
            .expect("found");
        assert_eq!(fetched.digest_hex(), digest);
        let unknown = request_package(&mut client, &missing, &granted()).expect("pull miss");
        assert_eq!(unknown, None);
        drop(client);
        let events = server_job.join().expect("join").expect("serve");
        assert_eq!(
            events,
            vec![
                SessionEvent::Pull {
                    requested: digest,
                    served: true
                },
                SessionEvent::Pull {
                    requested: missing.to_ascii_lowercase(),
                    served: false
                },
            ]
        );
    }

    /// A stream that swallows writes and answers EOF on reads.
    struct DeadStream;

    impl Read for DeadStream {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Ok(0)
        }
    }

    impl Write for DeadStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn retry_survives_transient_failures_but_not_final_refusals() {
        use std::cell::Cell;
        let pkg = package();
        let attempts_left = Cell::new(2_u32);
        let outcome = push_with_retry(
            || {
                let left = attempts_left.get();
                attempts_left.set(left.saturating_sub(1));
                if left == 0 {
                    Ok(DeadStream)
                } else {
                    Err("transient connect failure")
                }
            },
            4,
            &pkg,
            &granted(),
            None,
        );
        // The sink accepts writes but never answers -> i/o path until the
        // attempt budget is exhausted.
        assert!(matches!(outcome, Err(TransportError::Io(_))));
        // Consent refusal is final and immediate even with attempts left.
        let local_only = TasteSettings {
            learning_enabled: false,
            ..TasteSettings::default()
        };
        assert_eq!(
            push_with_retry(
                || -> Result<DeadStream, &'static str> { Ok(DeadStream) },
                9,
                &pkg,
                &local_only,
                None
            ),
            Err(TransportError::ExportDeniedByConsent)
        );
    }

    #[test]
    fn oversized_lengths_unknown_protocols_and_tampering_fail_closed() {
        let (mut a, mut b) = std::os::unix::net::UnixStream::pair().expect("pair");
        a.write_all(format!("{PROTOCOL_ID} PUSH {}\n", MAX_PAYLOAD_BYTES + 1).as_bytes())
            .expect("write");
        let settings = granted();
        let mut seen = BTreeSet::new();
        let mut cfg = config(&settings, &mut seen);
        assert_eq!(
            serve_request(&mut b, &mut cfg, None),
            Err(TransportError::InvalidLength)
        );
        // Unknown protocol id.
        a.write_all(b"OTHER/1 PUSH 8\n").expect("write");
        assert_eq!(
            serve_request(&mut b, &mut cfg, None),
            Err(TransportError::MalformedHeader)
        );
        // Tampered payload fails verification and answers ERR on the wire.
        let (mut c, mut d) = std::os::unix::net::UnixStream::pair().expect("pair");
        let mut document = package().render_markdown().expect("document");
        let marker = "\"confidence_bps\":";
        let flip = document.find(marker).expect("marker") + marker.len();
        let digit = document.as_bytes()[flip];
        let flipped = if digit == b'8' { b'9' } else { b'8' };
        document.replace_range(flip..flip + 1, &(flipped as char).to_string());
        c.write_all(format!("{PROTOCOL_ID} PUSH {}\n", document.len()).as_bytes())
            .expect("header");
        c.write_all(document.as_bytes()).expect("body");
        let result = serve_request(&mut d, &mut cfg, None);
        drop(c);
        assert!(matches!(result, Err(TransportError::Package(_))));
    }
}
