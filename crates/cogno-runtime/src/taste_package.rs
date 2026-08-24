//! Portable taste package — the `taste.md` artifact (étape 3).
//!
//! A package carries ONLY derived, `Active` preferences plus their full
//! attribution, wrapped in a human-readable markdown shell around one
//! canonical JSON block. Guarantees:
//!
//! - **Deterministic**: same inputs ⇒ identical bytes. No timestamps, no
//!   hostnames, sorted everything, fixed field order.
//! - **Bounded**: entry/source/parent counts, id lengths and total size are
//!   capped so hostile packages cannot inflate memory (S5).
//! - **Integrity-checked**: SHA-256 over the canonical block, verified on
//!   every parse; tampered markdown fails closed.
//! - **Never authoritative**: importing a package yields `Proposed` events
//!   with `ImportedProfile` origin. Imported data starts quarantined and can
//!   never activate on its own (zero local non-model confirmations, S4/S6).

use cogno_core::{
    ScientificTasteProfile, TasteEvent, TasteEventKind, TasteOrigin, TastePolicy, TasteScope,
    TasteSettings, TasteSource, TasteSourceKind, TasteState, MAX_TASTE_CONFIDENCE_BPS,
    MAX_TASTE_SOURCE_ID_BYTES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Wire/format identifier baked into every package body.
pub const PACKAGE_FORMAT: &str = "cogno-taste-package/1";
/// Maximum preferences a single package may carry.
pub const MAX_PACKAGE_PREFERENCES: usize = 256;
/// Maximum contributors recorded per preference.
pub const MAX_PACKAGE_SOURCES_PER_ENTRY: usize = 16;
/// Maximum recorded ancestry digests (compose lineage).
pub const MAX_PACKAGE_PARENTS: usize = 16;
/// Hard size cap for a parsed or rendered package document.
pub const MAX_PACKAGE_DOCUMENT_BYTES: usize = 256 * 1024;
const DIGEST_HEX_LEN: usize = 64;

/// Attribution record inside a package entry.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageSource {
    pub kind: PackageSourceKind,
    #[serde(default)]
    pub id: String,
}

/// Serializable mirror of [`TasteSourceKind`] with stable snake-case tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PackageSourceKind {
    AssistanceModel,
    Human,
    DeterministicKernel,
    ExternalAgent,
    Unknown,
}

impl From<TasteSourceKind> for PackageSourceKind {
    fn from(kind: TasteSourceKind) -> Self {
        match kind {
            TasteSourceKind::AssistanceModel => Self::AssistanceModel,
            TasteSourceKind::Human => Self::Human,
            TasteSourceKind::DeterministicKernel => Self::DeterministicKernel,
            TasteSourceKind::ExternalAgent => Self::ExternalAgent,
            TasteSourceKind::Unknown => Self::Unknown,
        }
    }
}

impl From<PackageSourceKind> for TasteSourceKind {
    fn from(kind: PackageSourceKind) -> Self {
        match kind {
            PackageSourceKind::AssistanceModel => TasteSourceKind::AssistanceModel,
            PackageSourceKind::Human => TasteSourceKind::Human,
            PackageSourceKind::DeterministicKernel => TasteSourceKind::DeterministicKernel,
            PackageSourceKind::ExternalAgent => TasteSourceKind::ExternalAgent,
            PackageSourceKind::Unknown => TasteSourceKind::Unknown,
        }
    }
}

/// One exported `Active` preference with its provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageEntry {
    pub preference_id: u64,
    pub scope: PackageScope,
    /// Bounded identity of the scope instance (étape 4): a preference is
    /// only meaningful inside the scope it was learned in.
    pub scope_key: String,
    pub confidence_bps: u16,
    /// Sorted, deduplicated contributor set (attribution, never authority).
    pub sources: Vec<PackageSource>,
}

/// Serializable mirror of [`TasteScope`] with stable lowercase tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PackageScope {
    User,
    Project,
    Domain,
    Team,
}

impl From<TasteScope> for PackageScope {
    fn from(scope: TasteScope) -> Self {
        match scope {
            TasteScope::User => Self::User,
            TasteScope::Project => Self::Project,
            TasteScope::Domain => Self::Domain,
            TasteScope::Team => Self::Team,
        }
    }
}

impl From<PackageScope> for TasteScope {
    fn from(scope: PackageScope) -> Self {
        match scope {
            PackageScope::User => TasteScope::User,
            PackageScope::Project => TasteScope::Project,
            PackageScope::Domain => TasteScope::Domain,
            PackageScope::Team => TasteScope::Team,
        }
    }
}

/// Snapshot of the policy that governed derivation (compose compatibility).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackagePolicy {
    pub activation_threshold_bps: u16,
    pub minimum_non_model_confirmations: u16,
    pub acceptance_gain_bps: u16,
    pub confirmation_gain_bps: u16,
    pub edit_gain_bps: u16,
    pub rejection_loss_bps: u16,
    pub contradiction_loss_bps: u16,
}

impl From<TastePolicy> for PackagePolicy {
    fn from(policy: TastePolicy) -> Self {
        Self {
            activation_threshold_bps: policy.activation_threshold_bps,
            minimum_non_model_confirmations: policy.minimum_non_model_confirmations,
            acceptance_gain_bps: policy.acceptance_gain_bps,
            confirmation_gain_bps: policy.confirmation_gain_bps,
            edit_gain_bps: policy.edit_gain_bps,
            rejection_loss_bps: policy.rejection_loss_bps,
            contradiction_loss_bps: policy.contradiction_loss_bps,
        }
    }
}

impl From<PackagePolicy> for TastePolicy {
    fn from(policy: PackagePolicy) -> Self {
        Self {
            activation_threshold_bps: policy.activation_threshold_bps,
            minimum_non_model_confirmations: policy.minimum_non_model_confirmations,
            acceptance_gain_bps: policy.acceptance_gain_bps,
            confirmation_gain_bps: policy.confirmation_gain_bps,
            edit_gain_bps: policy.edit_gain_bps,
            rejection_loss_bps: policy.rejection_loss_bps,
            contradiction_loss_bps: policy.contradiction_loss_bps,
        }
    }
}

/// Canonical machine-readable body. Field order IS the wire order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageBody {
    format: String,
    policy: PackagePolicy,
    /// Ancestry: digests of packages composed into this one (sorted, unique).
    #[serde(default)]
    parents: Vec<String>,
    /// Preference ids whose values differed across composed parents
    /// (deterministically resolved, kept visible for auditability).
    #[serde(default)]
    conflicts: Vec<u64>,
    /// Sorted by `preference_id`, strictly increasing (no duplicates).
    entries: Vec<PackageEntry>,
}

/// A verified taste package: parsed, bounds-checked, digest-checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TastePackage {
    body: PackageBody,
    digest: [u8; 32],
}

/// Failure modes for package construction, rendering or parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TastePackageError {
    /// Derived profile carried no `Active` preference to export.
    EmptyProfile,
    /// Profile carried more `Active` preferences than the package bound.
    TooManyEntries,
    /// A source set exceeded [`MAX_PACKAGE_SOURCES_PER_ENTRY`].
    TooManySources,
    /// An id violated the bounded printable rule shared with the core.
    InvalidSourceId,
    /// Policy snapshot inconsistent (threshold above 10000 bps).
    InvalidPolicy,
    /// Compose arguments disagree on the governing policy.
    PolicyMismatch,
    /// Compose ancestry would exceed [`MAX_PACKAGE_PARENTS`].
    TooManyParents,
    /// Document exceeded [`MAX_PACKAGE_DOCUMENT_BYTES`] (or was empty).
    InvalidDocumentSize,
    /// Markdown shell malformed (headers/fence misplaced).
    MalformedDocument,
    /// Canonical block was unreadable, out of bounds or non-canonical.
    InvalidBody,
    /// Recomputed digest differs from the declared one.
    DigestMismatch,
    /// Host consent (`taste.settings.json`) denies this export/import (étape 4).
    ConsentDenied,
    /// Filesystem failure while saving or loading.
    Io(String),
}

impl TastePackage {
    /// Export every `Active` preference of a derived profile. Attribution is
    /// copied verbatim; nothing else (quarantined/candidate/conflicted/
    /// rejected state, counts) ever leaves the host.
    pub fn from_profile(
        profile: &ScientificTasteProfile,
        policy: TastePolicy,
    ) -> Result<Self, TastePackageError> {
        if policy.activation_threshold_bps > MAX_TASTE_CONFIDENCE_BPS
            || policy.minimum_non_model_confirmations == 0
        {
            return Err(TastePackageError::InvalidPolicy);
        }
        let mut entries = Vec::new();
        for entry in profile.preferences.values() {
            if !matches!(entry.state, TasteState::Active) {
                continue;
            }
            if entries.len() == MAX_PACKAGE_PREFERENCES {
                return Err(TastePackageError::TooManyEntries);
            }
            if entry.sources.len() > MAX_PACKAGE_SOURCES_PER_ENTRY {
                return Err(TastePackageError::TooManySources);
            }
            let mut sources: Vec<PackageSource> = entry
                .sources
                .iter()
                .map(|source| PackageSource {
                    kind: PackageSourceKind::from(source.kind),
                    id: source.id.clone(),
                })
                .collect();
            sources.sort();
            sources.dedup();
            for source in &sources {
                validate_source(source)?;
            }
            entries.push(PackageEntry {
                preference_id: entry.preference_id,
                scope: PackageScope::from(entry.scope),
                scope_key: entry.scope_key.clone(),
                confidence_bps: entry.confidence_bps,
                sources,
            });
        }
        if entries.is_empty() {
            return Err(TastePackageError::EmptyProfile);
        }
        Ok(Self::from_parts(PackageBody {
            format: PACKAGE_FORMAT.to_string(),
            policy: PackagePolicy::from(policy),
            parents: Vec::new(),
            conflicts: Vec::new(),
            entries,
        }))
    }

    fn from_parts(body: PackageBody) -> Self {
        let canonical =
            serde_json::to_vec(&body).expect("canonical body serialization cannot fail");
        let digest: [u8; 32] = Sha256::digest(&canonical).into();
        Self { body, digest }
    }

    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    #[must_use]
    pub fn digest_hex(&self) -> String {
        hex(&self.digest)
    }

    #[must_use]
    pub fn entries(&self) -> &[PackageEntry] {
        &self.body.entries
    }

    #[must_use]
    pub fn conflicts(&self) -> &[u64] {
        &self.body.conflicts
    }

    #[must_use]
    pub fn parents(&self) -> &[String] {
        &self.body.parents
    }

    #[must_use]
    pub const fn policy(&self) -> PackagePolicy {
        self.body.policy
    }

    /// Render the deterministic `taste.md` document: markdown shell around
    /// the canonical JSON block, digest pinned in the header.
    pub fn render_markdown(&self) -> Result<String, TastePackageError> {
        let canonical =
            serde_json::to_vec(&self.body).map_err(|_| TastePackageError::InvalidBody)?;
        if canonical.len() > MAX_PACKAGE_DOCUMENT_BYTES {
            return Err(TastePackageError::InvalidDocumentSize);
        }
        let canonical_str =
            std::str::from_utf8(&canonical).map_err(|_| TastePackageError::InvalidBody)?;
        let mut document = String::with_capacity(canonical.len() + 512);
        document.push_str("# COGNO Taste Package\n\n");
        document.push_str("- format: ");
        document.push_str(PACKAGE_FORMAT);
        document.push_str("\n- entries: ");
        document.push_str(&self.body.entries.len().to_string());
        document.push_str("\n- digest: sha256:");
        document.push_str(&self.digest_hex());
        document.push_str("\n\n");
        document.push_str("The fenced block below is the canonical artifact; edit it never.\n\n");
        document.push_str("```json\n");
        document.push_str(canonical_str);
        document.push('\n');
        document.push_str("```\n\n");
        document.push_str("## Preferences\n\n");
        for entry in &self.body.entries {
            document.push_str(&render_entry_line(entry));
        }
        if document.len() > MAX_PACKAGE_DOCUMENT_BYTES {
            return Err(TastePackageError::InvalidDocumentSize);
        }
        Ok(document)
    }

    /// Save atomically (temp file + rename) to `<root>/taste.md`.
    pub fn save(&self, root: impl AsRef<Path>) -> Result<(), TastePackageError> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|error| TastePackageError::Io(error.to_string()))?;
        let document = self.render_markdown()?;
        let path = root.join("taste.md");
        let tmp = root.join(".taste.md.tmp");
        fs::write(&tmp, document).map_err(|error| TastePackageError::Io(error.to_string()))?;
        fs::rename(&tmp, &path).map_err(|error| TastePackageError::Io(error.to_string()))?;
        Ok(())
    }

    /// Consent-gated export (étape 4): writes `<root>/taste.md` only when the
    /// host settings allow it. The consent check lives here so callers cannot
    /// forget it.
    pub fn save_with_consent(
        &self,
        root: impl AsRef<Path>,
        settings: &TasteSettings,
    ) -> Result<(), TastePackageError> {
        if !settings.can_export() {
            return Err(TastePackageError::ConsentDenied);
        }
        self.save(root)
    }

    /// Consent-gated import (étape 4): yields quarantine-bound proposal
    /// events only when the host settings allow ingestion. Imports still can
    /// never activate alone — consent permits reading, not trust.
    pub fn import_events_with_consent(
        &self,
        settings: &TasteSettings,
    ) -> Result<Vec<TasteEvent>, TastePackageError> {
        if !settings.can_import() {
            return Err(TastePackageError::ConsentDenied);
        }
        Ok(self.import_events())
    }

    /// Load and fully verify `<root>/taste.md`.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, TastePackageError> {
        let path = root.as_ref().join("taste.md");
        let bytes = fs::read(&path).map_err(|error| TastePackageError::Io(error.to_string()))?;
        Self::parse_markdown(&bytes)
    }

    /// Strict parser: bounded read, exact markdown shell, canonical body,
    /// recomputed digest. Any deviation fails closed.
    pub fn parse_markdown(document: &[u8]) -> Result<Self, TastePackageError> {
        if document.is_empty() || document.len() > MAX_PACKAGE_DOCUMENT_BYTES {
            return Err(TastePackageError::InvalidDocumentSize);
        }
        let text =
            std::str::from_utf8(document).map_err(|_| TastePackageError::MalformedDocument)?;
        let digest_hex = parse_header_digest(text)?;
        let canonical = extract_canonical_block(text)?;
        let body: PackageBody =
            serde_json::from_slice(canonical).map_err(|_| TastePackageError::InvalidBody)?;
        validate_body(&body)?;
        // Canonicality check: re-encoding must reproduce the exact bytes, so
        // hand-edited key order or whitespace changes are rejected.
        let reencoded = serde_json::to_vec(&body).map_err(|_| TastePackageError::InvalidBody)?;
        if reencoded != canonical {
            return Err(TastePackageError::InvalidBody);
        }
        let digest: [u8; 32] = Sha256::digest(canonical).into();
        if hex(&digest) != digest_hex {
            return Err(TastePackageError::DigestMismatch);
        }
        Ok(Self { body, digest })
    }

    /// Re-import as quarantine-bound proposal events. Every event carries
    /// `ImportedProfile` origin and the original source attribution; none can
    /// activate alone (imported proposals contribute zero local non-model
    /// confirmations).
    #[must_use]
    pub fn import_events(&self) -> Vec<TasteEvent> {
        self.body
            .entries
            .iter()
            .map(|entry| TasteEvent {
                event_id: entry.preference_id,
                preference_id: entry.preference_id,
                scope: TasteScope::from(entry.scope),
                scope_key: entry.scope_key.clone(),
                kind: TasteEventKind::Proposed,
                origin: TasteOrigin::ImportedProfile,
                confidence_bps: entry.confidence_bps,
                evidence_ids: vec![entry.preference_id],
                source: TasteSource {
                    kind: entry
                        .sources
                        .first()
                        .map_or(TasteSourceKind::Unknown, |s| TasteSourceKind::from(s.kind)),
                    id: entry
                        .sources
                        .first()
                        .map_or(String::new(), |s| s.id.clone()),
                },
            })
            .collect()
    }
}

fn render_entry_line(entry: &PackageEntry) -> String {
    let scope = match entry.scope {
        PackageScope::User => "user",
        PackageScope::Project => "project",
        PackageScope::Domain => "domain",
        PackageScope::Team => "team",
    };
    let mut line = format!(
        "- #{} {scope}[{}] confidence={}bps sources=",
        entry.preference_id, entry.scope_key, entry.confidence_bps
    );
    if entry.sources.is_empty() {
        line.push_str("none");
    } else {
        for (index, source) in entry.sources.iter().enumerate() {
            if index > 0 {
                line.push(',');
            }
            line.push_str(source_kind_label(source.kind));
            line.push(':');
            if source.id.is_empty() {
                line.push('-');
            } else {
                line.push_str(&source.id);
            }
        }
    }
    line.push('\n');
    line
}

const fn source_kind_label(kind: PackageSourceKind) -> &'static str {
    match kind {
        PackageSourceKind::AssistanceModel => "assistance-model",
        PackageSourceKind::Human => "human",
        PackageSourceKind::DeterministicKernel => "deterministic-kernel",
        PackageSourceKind::ExternalAgent => "external-agent",
        PackageSourceKind::Unknown => "unknown",
    }
}

fn validate_source(source: &PackageSource) -> Result<(), TastePackageError> {
    if source.id.len() > MAX_TASTE_SOURCE_ID_BYTES
        || !source.id.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
    {
        return Err(TastePackageError::InvalidSourceId);
    }
    // Non-empty ids are required for attributed kinds; Unknown may be blank.
    if source.id.is_empty() && source.kind != PackageSourceKind::Unknown {
        return Err(TastePackageError::InvalidSourceId);
    }
    Ok(())
}

fn validate_body(body: &PackageBody) -> Result<(), TastePackageError> {
    if body.format != PACKAGE_FORMAT {
        return Err(TastePackageError::InvalidBody);
    }
    if body.policy.activation_threshold_bps > MAX_TASTE_CONFIDENCE_BPS
        || body.policy.minimum_non_model_confirmations == 0
    {
        return Err(TastePackageError::InvalidPolicy);
    }
    if body.parents.len() > MAX_PACKAGE_PARENTS || body.entries.len() > MAX_PACKAGE_PREFERENCES {
        return Err(TastePackageError::InvalidBody);
    }
    let mut previous_parent = String::new();
    for parent in &body.parents {
        if parent.len() != DIGEST_HEX_LEN || !parent.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(TastePackageError::InvalidBody);
        }
        if parent.as_str() <= previous_parent.as_str() && !previous_parent.is_empty() {
            return Err(TastePackageError::InvalidBody);
        }
        previous_parent.clone_from(parent);
    }
    let mut previous_id = 0_u64;
    let mut first = true;
    for entry in &body.entries {
        if entry.confidence_bps > MAX_TASTE_CONFIDENCE_BPS {
            return Err(TastePackageError::InvalidBody);
        }
        if entry.sources.len() > MAX_PACKAGE_SOURCES_PER_ENTRY {
            return Err(TastePackageError::InvalidBody);
        }
        if TasteScope::from(entry.scope)
            .validate_key(&entry.scope_key)
            .is_err()
        {
            return Err(TastePackageError::InvalidBody);
        }
        if !first && entry.preference_id <= previous_id {
            return Err(TastePackageError::InvalidBody);
        }
        first = false;
        previous_id = entry.preference_id;
        let mut sorted = entry.sources.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != entry.sources.len() || sorted != entry.sources {
            return Err(TastePackageError::InvalidBody);
        }
        for source in &entry.sources {
            validate_source(source)?;
        }
    }
    let mut previous_conflict = 0_u64;
    let mut first_conflict = true;
    for conflict in &body.conflicts {
        if !first_conflict && *conflict <= previous_conflict {
            return Err(TastePackageError::InvalidBody);
        }
        first_conflict = false;
        previous_conflict = *conflict;
    }
    Ok(())
}

/// Extract `- digest: sha256:<hex>` from the header section.
fn parse_header_digest(text: &str) -> Result<String, TastePackageError> {
    let header_end = text
        .find("```")
        .ok_or(TastePackageError::MalformedDocument)?;
    for line in text[..header_end].lines() {
        if let Some(rest) = line.strip_prefix("- digest: sha256:") {
            if rest.len() == DIGEST_HEX_LEN && rest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Ok(rest.to_string());
            }
            return Err(TastePackageError::MalformedDocument);
        }
    }
    Err(TastePackageError::MalformedDocument)
}

/// Extract exactly the bytes between the opening ```json fence line and the
/// closing ``` fence line.
fn extract_canonical_block(text: &str) -> Result<&[u8], TastePackageError> {
    let opener = "\n```json\n";
    let start = text
        .find(opener)
        .ok_or(TastePackageError::MalformedDocument)?
        + opener.len();
    let rest = &text[start..];
    let end = rest
        .find("\n```\n")
        .ok_or(TastePackageError::MalformedDocument)?;
    Ok(&rest.as_bytes()[..end])
}

fn hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(DIGEST_HEX_LEN);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("string write");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{TasteArbitration, TasteArbitrationAction, TasteArbitrationAuthority};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-package-{tag}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn source(kind: TasteSourceKind, id: &str) -> TasteSource {
        TasteSource::try_new(kind, id).expect("source")
    }

    fn event(
        event_id: u64,
        kind: TasteEventKind,
        origin: TasteOrigin,
        src: TasteSource,
    ) -> TasteEvent {
        TasteEvent {
            event_id,
            preference_id: 7,
            scope: TasteScope::Project,
            scope_key: "project:cogno-1".to_string(),
            kind,
            origin,
            confidence_bps: 5_000,
            evidence_ids: vec![event_id],
            source: src,
        }
    }

    /// Derive a profile with one Active preference (#7) contributed by a
    /// model proposal + two kernel confirmations.
    fn active_profile() -> ScientificTasteProfile {
        let events = [
            event(
                1,
                TasteEventKind::Proposed,
                TasteOrigin::ModelInference,
                source(TasteSourceKind::AssistanceModel, "claude-x"),
            ),
            event(
                2,
                TasteEventKind::Confirmed,
                TasteOrigin::DeterministicEvaluation,
                source(TasteSourceKind::DeterministicKernel, "scirust@gen7"),
            ),
            event(
                3,
                TasteEventKind::Confirmed,
                TasteOrigin::DeterministicEvaluation,
                source(TasteSourceKind::DeterministicKernel, "scirust@gen7"),
            ),
        ];
        ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT).expect("profile")
    }

    #[test]
    fn export_only_includes_active_entries_and_round_trips() {
        let profile = active_profile();
        assert_eq!(
            profile.preferences[&7].state,
            TasteState::Active,
            "fixture must activate"
        );
        let package = TastePackage::from_profile(&profile, TastePolicy::DEFAULT).expect("package");
        assert_eq!(package.entries().len(), 1);
        assert_eq!(package.entries()[0].preference_id, 7);
        assert_eq!(package.entries()[0].sources.len(), 2);

        let document = package.render_markdown().expect("markdown");
        let parsed = TastePackage::parse_markdown(document.as_bytes()).expect("parse");
        assert_eq!(parsed, package);
        assert_eq!(parsed.import_events().len(), 1);
        let imported = &parsed.import_events()[0];
        assert_eq!(imported.origin, TasteOrigin::ImportedProfile);
        assert_eq!(imported.kind, TasteEventKind::Proposed);
    }

    #[test]
    fn scope_identity_survives_the_round_trip() {
        let profile = active_profile();
        let package = TastePackage::from_profile(&profile, TastePolicy::DEFAULT).expect("package");
        assert_eq!(package.entries()[0].scope_key, "project:cogno-1");
        let document = package.render_markdown().expect("markdown");
        assert!(document.contains("- #7 project[project:cogno-1] "));
        let imported = TastePackage::parse_markdown(document.as_bytes())
            .expect("parse")
            .import_events();
        assert_eq!(imported[0].scope_key, "project:cogno-1");

        // A tampered scope key breaks the digest like any other edit.
        let tampered = document.replace(
            "\"scope_key\":\"project:cogno-1\"",
            "\"scope_key\":\"project:other\"",
        );
        assert_ne!(tampered, document);
        assert_eq!(
            TastePackage::parse_markdown(tampered.as_bytes()),
            Err(TastePackageError::DigestMismatch)
        );
    }

    #[test]
    fn consent_gates_block_export_and_import() {
        let package =
            TastePackage::from_profile(&active_profile(), TastePolicy::DEFAULT).expect("package");
        let root = temp_root("consent");
        // Defaults: learning on, export/import off — nothing leaves or enters.
        let defaults = TasteSettings::default();
        assert_eq!(
            package.save_with_consent(&root, &defaults),
            Err(TastePackageError::ConsentDenied)
        );
        assert_eq!(
            package.import_events_with_consent(&defaults),
            Err(TastePackageError::ConsentDenied)
        );
        assert!(!root.join("taste.md").exists());
        // Granting export lets the write through.
        let exporter = TasteSettings {
            export_allowed: true,
            ..TasteSettings::default()
        };
        package
            .save_with_consent(&root, &exporter)
            .expect("consented export");
        assert!(root.join("taste.md").exists());
        // Import stays denied even with export granted; granting import lets
        // the proposal events through (still quarantine-bound).
        assert_eq!(
            package.import_events_with_consent(&exporter),
            Err(TastePackageError::ConsentDenied)
        );
        let importer = TasteSettings {
            import_allowed: true,
            ..TasteSettings::default()
        };
        let events = package
            .import_events_with_consent(&importer)
            .expect("consented import");
        assert!(!events.is_empty());
        assert!(events.iter().all(|event| {
            matches!(event.kind, TasteEventKind::Proposed)
                && matches!(event.origin, TasteOrigin::ImportedProfile)
        }));
    }

    #[test]
    fn rendering_is_deterministic() {
        let package =
            TastePackage::from_profile(&active_profile(), TastePolicy::DEFAULT).expect("package");
        let first = package.render_markdown().expect("first");
        let second = package.render_markdown().expect("second");
        assert_eq!(first, second);
        assert!(first.contains("- digest: sha256:"));
        assert!(first.contains("\"format\":\"cogno-taste-package/1\""));
    }

    #[test]
    fn quarantined_and_conflicted_never_export() {
        let events = [
            event(
                1,
                TasteEventKind::Proposed,
                TasteOrigin::ModelInference,
                source(TasteSourceKind::AssistanceModel, "claude-x"),
            ),
            event(
                2,
                TasteEventKind::Contradicted,
                TasteOrigin::ExplicitUserAction,
                source(TasteSourceKind::Human, "host-op"),
            ),
        ];
        let profile =
            ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT).expect("conflicted");
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);
        assert_eq!(
            TastePackage::from_profile(&profile, TastePolicy::DEFAULT),
            Err(TastePackageError::EmptyProfile)
        );

        // Human arbitration activates the conflict resolution path but the
        // entry stays non-Active unless gates pass — Reject never exports.
        let mut arbitrated =
            ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT).expect("profile");
        arbitrated
            .arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Reject,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                },
            )
            .expect("reject");
        assert_eq!(
            TastePackage::from_profile(&arbitrated, TastePolicy::DEFAULT),
            Err(TastePackageError::EmptyProfile)
        );
    }

    #[test]
    fn tampered_documents_fail_closed() {
        let package =
            TastePackage::from_profile(&active_profile(), TastePolicy::DEFAULT).expect("package");
        let document = package.render_markdown().expect("markdown");

        // Flip a confidence value inside the canonical block: the edit stays
        // canonical JSON, so the header digest is what breaks.
        let tampered = document.replace("\"confidence_bps\":8000", "\"confidence_bps\":8001");
        assert_ne!(tampered, document, "fixture must hit the field");
        assert_eq!(
            TastePackage::parse_markdown(tampered.as_bytes()),
            Err(TastePackageError::DigestMismatch),
            "digest binds every byte of the canonical body"
        );

        // Non-canonical whitespace is rejected even with a matching digest.
        let pretty = document.replace("{\"format\"", "{\n  \"format\"");
        assert_eq!(
            TastePackage::parse_markdown(pretty.as_bytes()),
            Err(TastePackageError::InvalidBody)
        );

        // Truncated document is rejected outright.
        assert_eq!(
            TastePackage::parse_markdown(&document.as_bytes()[..64]),
            Err(TastePackageError::MalformedDocument)
        );
        assert_eq!(
            TastePackage::parse_markdown(b""),
            Err(TastePackageError::InvalidDocumentSize)
        );
    }

    #[test]
    fn save_and_load_round_trip_through_filesystem() {
        let root = temp_root("fs");
        let package =
            TastePackage::from_profile(&active_profile(), TastePolicy::DEFAULT).expect("package");
        package.save(&root).expect("save");
        let loaded = TastePackage::load(&root).expect("load");
        assert_eq!(loaded, package);
        assert!(root.join("taste.md").exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn oversized_or_invalid_sources_are_refused() {
        let long_id = "a".repeat(MAX_TASTE_SOURCE_ID_BYTES + 1);
        let entry = PackageEntry {
            preference_id: 1,
            scope: PackageScope::User,
            scope_key: "user:alice".to_string(),
            confidence_bps: 9_000,
            sources: vec![PackageSource {
                kind: PackageSourceKind::Human,
                id: long_id,
            }],
        };
        let body = PackageBody {
            format: PACKAGE_FORMAT.to_string(),
            policy: PackagePolicy::from(TastePolicy::DEFAULT),
            parents: Vec::new(),
            conflicts: Vec::new(),
            entries: vec![entry],
        };
        let canonical = serde_json::to_vec(&body).expect("body");
        let forged = format!(
            "# COGNO Taste Package\n\n- format: {PACKAGE_FORMAT}\n- entries: 1\n- digest: sha256:{}\n\n```json\n{}\n```\n",
            Sha256::digest(&canonical)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            std::str::from_utf8(&canonical).expect("utf8")
        );
        assert_eq!(
            TastePackage::parse_markdown(forged.as_bytes()),
            Err(TastePackageError::InvalidSourceId)
        );
    }

    #[test]
    fn imported_events_alone_cannot_activate() {
        let package =
            TastePackage::from_profile(&active_profile(), TastePolicy::DEFAULT).expect("package");
        let events = package.import_events();
        assert!(!events.is_empty());
        let replayed =
            ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT).expect("imported derive");
        for entry in replayed.preferences.values() {
            assert_eq!(
                entry.state,
                TasteState::Quarantined,
                "imports stay quarantined: no local confirmation yet"
            );
        }
    }
}
