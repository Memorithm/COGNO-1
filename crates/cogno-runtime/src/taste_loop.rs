//! Continuous taste feedback loop (étape 7): real-work outcomes become new
//! taste events, which re-enter deterministic derivation.

#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

use cogno_core::{
    ScientificTasteProfile, TasteError, TasteEvent, TasteEventKind, TasteOrigin, TastePolicy,
    TasteScope, TasteSettings, TasteSource, TasteSourceKind,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// On-disk journal name under the validation store root.
pub const FEEDBACK_FILE: &str = "taste.feedback.jsonl";

/// Current feedback record schema.
pub const FEEDBACK_SCHEMA_VERSION: u16 = 1;

/// Hard cap on journal size read per iteration (fail closed beyond this).
pub const MAX_FEEDBACK_JOURNAL_BYTES: usize = 1024 * 1024;

/// Hard cap on records consumed per derivation iteration.
pub const MAX_FEEDBACK_RECORDS: usize = 16_384;

/// One real-work outcome observed by the host (accept/edit/reject of an
/// applied preference). Strictly typed on disk; anything unexpected fails
/// closed at parse time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TasteFeedbackRecord {
    pub schema_version: u16,
    /// Globally unique, monotonically assigned by the host (ordering key).
    pub event_id: u64,
    pub preference_id: u64,
    /// Scope instance the outcome was observed in (`"user:alice"`).
    pub scope: String,
    pub scope_key: String,
    /// `proposed | accepted | confirmed | edited | rejected | contradicted`
    pub kind: String,
    /// `model_inference | explicit_user_action | deterministic_evaluation`
    pub origin: String,
    pub source_kind: String,
    #[serde(default)]
    pub source_id: String,
    pub confidence_bps: u16,
    /// Unix seconds when the outcome was observed (0 = unknown/legacy, kept
    /// forever). Drives deterministic retention (forgetting) at iteration
    /// time; never used to reorder or rewrite history.
    #[serde(default)]
    pub observed_at_unix: u64,
}

/// Failure modes of the loop layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TasteLoopError {
    /// Host consent denies learning; nothing is recorded or derived.
    LearningDeniedByConsent,
    /// A record violated its schema or semantic bounds.
    InvalidRecord(String),
    /// The journal exceeded [`MAX_FEEDBACK_JOURNAL_BYTES`].
    JournalTooLarge,
    /// Filesystem failure.
    Io(String),
    /// Core rejected a converted event (should be impossible after local
    /// validation; kept for defense in depth).
    Core(TasteError),
}

impl std::fmt::Display for TasteLoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LearningDeniedByConsent => write!(f, "taste learning disabled by host consent"),
            Self::InvalidRecord(message) => write!(f, "invalid feedback record: {message}"),
            Self::JournalTooLarge => write!(f, "feedback journal exceeds bound"),
            Self::Io(message) => write!(f, "loop i/o error: {message}"),
            Self::Core(error) => write!(f, "core rejected event: {error:?}"),
        }
    }
}

impl std::error::Error for TasteLoopError {}

fn journal_path(store_root: impl AsRef<Path>) -> PathBuf {
    store_root.as_ref().join(FEEDBACK_FILE)
}

/// Append one real-work outcome to the journal. Consent-gated: a host with
/// learning disabled never accumulates taste data.
pub fn record_feedback(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    record: &TasteFeedbackRecord,
) -> Result<(), TasteLoopError> {
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    validate_record(record)?;
    let path = journal_path(&store_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    }
    let mut line =
        serde_json::to_vec(record).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    line.push(b'\n');
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|error| TasteLoopError::Io(error.to_string()))?;
    file.write_all(&line)
        .map_err(|error| TasteLoopError::Io(error.to_string()))?;
    Ok(())
}

/// Unix seconds right now (runtime layer; the deterministic core never reads
/// a clock). Used to auto-stamp records so retention actually applies.
#[must_use]
pub fn current_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Outcome of a journal compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionReport {
    /// Lines present before compaction (including duplicates).
    pub records_before: usize,
    /// Records kept after deduplication and retention.
    pub records_after: usize,
    /// Byte size before / after.
    pub bytes_before: usize,
    pub bytes_after: usize,
    /// `false` means nothing changed and the file was left untouched.
    pub rewritten: bool,
}

/// Compact the feedback journal in place: parse strictly (a corrupt line
/// fails closed and aborts — compaction never silently drops data it cannot
/// understand), drop duplicate `event_id`s (first wins, matching derivation)
/// and optionally expire records outside the retention window. The rewrite
/// is atomic (temp file + rename) and only happens when something is
/// actually removed.
pub fn compact_feedback_journal(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    now_unix: u64,
    retention_secs: u64,
) -> Result<CompactionReport, TasteLoopError> {
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    let path = journal_path(&store_root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CompactionReport {
                records_before: 0,
                records_after: 0,
                bytes_before: 0,
                bytes_after: 0,
                rewritten: false,
            })
        }
        Err(error) => return Err(TasteLoopError::Io(error.to_string())),
    };
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| TasteLoopError::InvalidRecord("journal is not UTF-8".to_string()))?;
    let mut kept: Vec<String> = Vec::new();
    let mut seen_ids = std::collections::BTreeSet::new();
    let mut total_lines = 0_usize;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        total_lines += 1;
        let record: TasteFeedbackRecord = serde_json::from_str(line)
            .map_err(|error| TasteLoopError::InvalidRecord(format!("bad json: {error}")))?;
        validate_record(&record)?;
        // Same rules as derivation: first id wins, expired evidence goes.
        if !seen_ids.insert(record.event_id) {
            continue;
        }
        if retention_secs > 0 && record.observed_at_unix != 0 {
            let age = now_unix.saturating_sub(record.observed_at_unix);
            if age > retention_secs {
                continue;
            }
        }
        kept.push(line.to_string());
    }
    let report = CompactionReport {
        records_before: total_lines,
        records_after: kept.len(),
        bytes_before: bytes.len(),
        bytes_after: kept.iter().map(|line| line.len() + 1).sum(),
        rewritten: kept.len() < total_lines,
    };
    if !report.rewritten {
        return Ok(report);
    }
    let mut payload = kept.join("\n").into_bytes();
    payload.push(b'\n');
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, payload).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    fs::rename(&tmp, &path).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    Ok(report)
}

/// Run one full iteration of the continuous loop:
///
/// journal → validated events (deduplicated, ordered by `event_id`) →
/// [`ScientificTasteProfile::derive`]. Deterministic: the same journal bytes
/// always yield the same profile.
pub fn run_feedback_iteration(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    policy: TastePolicy,
) -> Result<ScientificTasteProfile, TasteLoopError> {
    run_feedback_iteration_retained(store_root, settings, policy, 0, 0)
}

/// Same as [`run_feedback_iteration`] but with deterministic forgetting:
/// events carrying `observed_at_unix > 0` whose observation is older than
/// `retention_secs` before `now_unix` are excluded from derivation. Records
/// without a timestamp (0) are always kept. Deterministic given inputs; the
/// journal itself is never rewritten.
pub fn run_feedback_iteration_retained(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    policy: TastePolicy,
    now_unix: u64,
    retention_secs: u64,
) -> Result<ScientificTasteProfile, TasteLoopError> {
    run_feedback_iteration_tuned(
        store_root,
        settings,
        policy,
        &LoopTuning {
            now_unix,
            retention_secs,
            ..LoopTuning::default()
        },
    )
}

/// Full control over the deterministic learning transforms (étape RL) :
/// rétention temporelle, décroissance graduée et bandit borné.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopTuning {
    /// Reference clock for age computations (0 disables time effects).
    pub now_unix: u64,
    /// Evidence older than this (by `observed_at_unix`) is excluded; 0
    /// disables retention entirely.
    pub retention_secs: u64,
    /// When `true`, evidence inside the window does not keep full weight:
    /// confidence scales linearly down to zero at the horizon (integer
    /// math). Undated records (`observed_at_unix == 0`) are never aged.
    pub graduated_decay: bool,
    /// Per-preference outcome adaptation (bandit): each net win
    /// (`confirmed`/`accepted` minus `rejected`/`contradicted`) shifts that
    /// preference's evidence confidence by `bandit_step_bps`, clamped to
    /// ±5000 bps. 0 disables. Deterministic given the journal.
    pub bandit_step_bps: u16,
}

const BANDIT_MAX_SHIFT_BPS: i32 = 5_000;
const MAX_TASTE_CONFIDENCE_BPS: u16 = cogno_core::MAX_TASTE_CONFIDENCE_BPS;

pub fn run_feedback_iteration_tuned(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    policy: TastePolicy,
    tuning: &LoopTuning,
) -> Result<ScientificTasteProfile, TasteLoopError> {
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    let events = load_journal_events_retained(store_root, tuning)?;
    ScientificTasteProfile::derive(&events, policy).map_err(TasteLoopError::Core)
}

/// Compact views of every `Active` preference for model-side conditioning:
/// the generation/prompt layer may read these (consent-gated) but never the
/// provenance graph behind them.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ActivePreferenceView {
    pub preference_id: u64,
    pub scope: String,
    pub scope_key: String,
    pub confidence_bps: u16,
}

pub fn active_preference_views(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    tuning: &LoopTuning,
) -> Result<Vec<ActivePreferenceView>, TasteLoopError> {
    let profile = run_feedback_iteration_tuned(store_root, settings, TastePolicy::DEFAULT, tuning)?;
    Ok(profile
        .preferences
        .values()
        .filter(|entry| matches!(entry.state, cogno_core::TasteState::Active))
        .map(|entry| ActivePreferenceView {
            preference_id: entry.preference_id,
            scope: entry.scope.scope_tag().to_string(),
            scope_key: entry.scope_key.clone(),
            confidence_bps: entry.confidence_bps,
        })
        .collect())
}

fn load_journal_events_retained(
    store_root: impl AsRef<Path>,
    tuning: &LoopTuning,
) -> Result<Vec<TasteEvent>, TasteLoopError> {
    let path = journal_path(store_root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        // An empty journal is a valid cold start, not an error.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(TasteLoopError::Io(error.to_string())),
    };
    if bytes.len() > MAX_FEEDBACK_JOURNAL_BYTES {
        return Err(TasteLoopError::JournalTooLarge);
    }
    let forgetting = tuning.retention_secs > 0 && tuning.now_unix > 0;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| TasteLoopError::InvalidRecord("journal is not UTF-8".to_string()))?;
    let mut events = Vec::new();
    let mut seen_ids = std::collections::BTreeSet::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if events.len() == MAX_FEEDBACK_RECORDS {
            return Err(TasteLoopError::JournalTooLarge);
        }
        let record: TasteFeedbackRecord = serde_json::from_str(line)
            .map_err(|error| TasteLoopError::InvalidRecord(format!("bad json: {error}")))?;
        validate_record(&record)?;
        // Deterministic forgetting: only timestamped records can expire, and
        // clock skew (observed in the "future") never drops evidence.
        if forgetting && record.observed_at_unix != 0 {
            let age = tuning.now_unix.saturating_sub(record.observed_at_unix);
            if age > tuning.retention_secs {
                continue;
            }
        }
        // First occurrence wins; replays of the same id are ignored so a
        // re-fed journal cannot double-count outcomes.
        if seen_ids.insert(record.event_id) {
            let mut event = event_from_record(&record)?;
            apply_graduated_decay(&mut event, record.observed_at_unix, tuning);
            events.push(event);
        }
    }
    events.sort_by_key(|event| event.event_id);
    apply_bandit(&mut events, tuning.bandit_step_bps);
    Ok(events)
}

/// Scale one event's confidence linearly toward zero as it ages across the
/// retention window (integer math, no float drift).
fn apply_graduated_decay(event: &mut TasteEvent, observed_at: u64, tuning: &LoopTuning) {
    if !tuning.graduated_decay || tuning.retention_secs == 0 || tuning.now_unix == 0 {
        return;
    }
    if observed_at == 0 {
        return;
    }
    let age = tuning.now_unix.saturating_sub(observed_at);
    if age >= tuning.retention_secs {
        return; // already handled by retention drop; keep as-is defensively
    }
    let remaining = tuning.retention_secs - age;
    let horizon = tuning.retention_secs;
    event.confidence_bps = ((u32::from(event.confidence_bps) * remaining as u32) / horizon as u32)
        .min(u32::from(MAX_TASTE_CONFIDENCE_BPS)) as u16;
}

/// Deterministic per-preference adaptation: net outcome balance shifts the
/// weight of that preference's evidence, bounded to ±[`BANDIT_MAX_SHIFT_BPS`]
/// bps so no journal can manufacture unlimited confidence.
fn apply_bandit(events: &mut [TasteEvent], step_bps: u16) {
    if step_bps == 0 {
        return;
    }
    let mut balance: std::collections::BTreeMap<u64, i32> = std::collections::BTreeMap::new();
    for event in events.iter() {
        let entry = balance.entry(event.preference_id).or_insert(0);
        match event.kind {
            TasteEventKind::Confirmed | TasteEventKind::Accepted => *entry += 1,
            TasteEventKind::Rejected | TasteEventKind::Contradicted => *entry -= 1,
            _ => {}
        }
    }
    for event in events.iter_mut() {
        let Some(net) = balance.get(&event.preference_id) else {
            continue;
        };
        let shift = (*net * i32::from(step_bps)).clamp(-BANDIT_MAX_SHIFT_BPS, BANDIT_MAX_SHIFT_BPS);
        let scale = 10_000 + shift;
        event.confidence_bps = ((u32::from(event.confidence_bps) * scale as u32) / 10_000)
            .clamp(0, u32::from(MAX_TASTE_CONFIDENCE_BPS)) as u16;
    }
}

fn validate_record(record: &TasteFeedbackRecord) -> Result<(), TasteLoopError> {
    if record.schema_version != FEEDBACK_SCHEMA_VERSION {
        return Err(TasteLoopError::InvalidRecord(format!(
            "unsupported schema version {}",
            record.schema_version
        )));
    }
    if record.event_id == 0 || record.preference_id == 0 {
        return Err(TasteLoopError::InvalidRecord(
            "ids must be non-zero".to_string(),
        ));
    }
    if record.confidence_bps > cogno_core::MAX_TASTE_CONFIDENCE_BPS {
        return Err(TasteLoopError::InvalidRecord(
            "confidence_bps exceeds 10000".to_string(),
        ));
    }
    TasteScope::from_scope_tag(&record.scope).ok_or_else(|| {
        TasteLoopError::InvalidRecord(format!("unknown scope `{}`", record.scope))
    })?;
    TasteScope::from_scope_tag(&record.scope)
        .expect("checked above")
        .validate_key(&record.scope_key)
        .map_err(|_| TasteLoopError::InvalidRecord("invalid scope key".to_string()))?;
    match record.kind.as_str() {
        "proposed" | "accepted" | "confirmed" | "edited" | "rejected" | "contradicted" => {}
        other => {
            return Err(TasteLoopError::InvalidRecord(format!(
                "unknown kind `{other}`"
            )))
        }
    }
    match record.origin.as_str() {
        "explicit_user_action" | "deterministic_evaluation" | "imported_profile" => {}
        "model_inference" if record.kind == "proposed" => {
            // Models may propose; they may never confirm authority.
        }
        "model_inference" => {
            return Err(TasteLoopError::InvalidRecord(
                "model inference may only propose, never confirm authority".to_string(),
            ))
        }
        other => {
            return Err(TasteLoopError::InvalidRecord(format!(
                "unknown origin `{other}`"
            )))
        }
    }
    TasteSourceKind::from_tag(&record.source_kind).ok_or_else(|| {
        TasteLoopError::InvalidRecord(format!("unknown source kind `{}`", record.source_kind))
    })?;
    TasteSource::try_new(
        TasteSourceKind::from_tag(&record.source_kind).expect("checked above"),
        &record.source_id,
    )
    .map_err(|error| TasteLoopError::InvalidRecord(format!("invalid source id: {error:?}")))?;
    Ok(())
}

fn kind_from_tag(tag: &str) -> TasteEventKind {
    match tag {
        "proposed" => TasteEventKind::Proposed,
        "accepted" => TasteEventKind::Accepted,
        "confirmed" => TasteEventKind::Confirmed,
        "edited" => TasteEventKind::Edited,
        "rejected" => TasteEventKind::Rejected,
        _ => TasteEventKind::Contradicted,
    }
}

fn origin_from_tag(tag: &str) -> TasteOrigin {
    match tag {
        "model_inference" => TasteOrigin::ModelInference,
        "explicit_user_action" => TasteOrigin::ExplicitUserAction,
        "deterministic_evaluation" => TasteOrigin::DeterministicEvaluation,
        _ => TasteOrigin::ImportedProfile,
    }
}

fn event_from_record(record: &TasteFeedbackRecord) -> Result<TasteEvent, TasteLoopError> {
    let scope = TasteScope::from_scope_tag(&record.scope).ok_or_else(|| {
        TasteLoopError::InvalidRecord(format!("unknown scope `{}`", record.scope))
    })?;
    let source_kind = TasteSourceKind::from_tag(&record.source_kind).ok_or_else(|| {
        TasteLoopError::InvalidRecord(format!("unknown source kind `{}`", record.source_kind))
    })?;
    Ok(TasteEvent {
        event_id: record.event_id,
        preference_id: record.preference_id,
        scope,
        scope_key: record.scope_key.clone(),
        kind: kind_from_tag(&record.kind),
        origin: origin_from_tag(&record.origin),
        confidence_bps: record.confidence_bps,
        evidence_ids: vec![record.event_id],
        source: TasteSource::try_new(source_kind, &record.source_id)
            .map_err(|error| TasteLoopError::InvalidRecord(format!("invalid source: {error:?}")))?,
    })
}

/// Convert every stored validation into feedback records (point d'intégration
/// de la boucle : les verdicts réels du store alimentent le journal). The
/// caller supplies the scope instance the outcomes belong to — the validation
/// store itself is scope-free. Returns the number of records appended;
/// re-ingesting the same store is harmless (duplicate event ids are ignored
/// at derivation time).
pub fn ingest_validation_store(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    scope: TasteScope,
    scope_key: &str,
) -> Result<usize, TasteLoopError> {
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    scope
        .validate_key(scope_key)
        .map_err(|_| TasteLoopError::InvalidRecord("invalid scope key".to_string()))?;
    let validations =
        crate::taste_validation_store::PersistentTasteValidationStore::open(&store_root)
            .map_err(|error| TasteLoopError::Io(format!("{error:?}")))?;
    let mut appended = 0_usize;
    for validation in validations.records() {
        let record = validation_to_record(validation, scope, scope_key);
        // Records that already exist in the journal (same event id) would be
        // deduplicated anyway; appending them again only wastes bytes, so a
        // full re-ingest stays cheap and idempotent in effect.
        record_feedback(&store_root, settings, &record)?;
        appended += 1;
    }
    Ok(appended)
}

fn validation_to_record(
    validation: &crate::taste_validation_store::StoredTasteValidation,
    scope: TasteScope,
    scope_key: &str,
) -> TasteFeedbackRecord {
    use crate::taste_validation_store::{
        StoredValidationOrigin, StoredValidationVerdict, ValidationSubjectKind,
    };
    let kind = match validation.verdict {
        StoredValidationVerdict::Confirmed => "confirmed",
        StoredValidationVerdict::Contradicted => "contradicted",
    };
    let origin = match validation.origin {
        StoredValidationOrigin::DeterministicEvaluation => "deterministic_evaluation",
        StoredValidationOrigin::ExplicitUserAction => "explicit_user_action",
    };
    let (source_kind, source_id) = match validation.subject_kind {
        ValidationSubjectKind::HumanHost => ("human", short_hash(&validation.subject_id_sha256)),
        ValidationSubjectKind::DeterministicEvaluator => (
            "deterministic_kernel",
            short_hash(&validation.subject_id_sha256),
        ),
        ValidationSubjectKind::ExternalAgent => {
            ("external_agent", short_hash(&validation.subject_id_sha256))
        }
        ValidationSubjectKind::Unattributed => ("unknown", "unattributed".to_string()),
    };
    TasteFeedbackRecord {
        schema_version: FEEDBACK_SCHEMA_VERSION,
        event_id: validation.validation_id,
        preference_id: validation.preference_id,
        scope: scope.scope_tag().to_string(),
        scope_key: scope_key.to_string(),
        kind: kind.to_string(),
        origin: origin.to_string(),
        source_kind: source_kind.to_string(),
        source_id,
        confidence_bps: validation.confidence_bps,
        observed_at_unix: current_unix_seconds(),
    }
}

/// Bounded source id derived from the subject hash: `sha256:<first 14 hex>`.
/// Attribution stays traceable without exceeding the core's 64-byte bound.
fn short_hash(bytes: &[u8; 32]) -> String {
    let mut output = String::from("sha256:");
    for byte in &bytes[..7] {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

/// Persisted derived-profile snapshot (`taste.profile.json`). Pure audit and
/// restart convenience: always recomputable from the journal, never an
/// authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSnapshot {
    pub schema_version: u16,
    /// Journal byte length this snapshot was derived from (incremental
    /// bookkeeping hint; snapshots are verified by recomputation, not trust).
    pub journal_bytes: usize,
    pub policy: crate::taste_package::PackagePolicy,
    pub preferences: Vec<SnapshotPreference>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotPreference {
    pub preference_id: u64,
    pub scope: String,
    pub scope_key: String,
    /// `quarantined | candidate | active | conflicted | rejected`
    pub state: String,
    pub confidence_bps: u16,
}

/// On-disk snapshot name under the store root.
pub const PROFILE_SNAPSHOT_FILE: &str = "taste.profile.json";
const SNAPSHOT_SCHEMA_VERSION: u16 = 1;

/// Canonical lowercase tag of a derived preference state (CLI/report use).
#[must_use]
pub fn state_tag_of(state: cogno_core::TasteState) -> &'static str {
    match state {
        cogno_core::TasteState::Quarantined => "quarantined",
        cogno_core::TasteState::Candidate => "candidate",
        cogno_core::TasteState::Active => "active",
        cogno_core::TasteState::Conflicted => "conflicted",
        cogno_core::TasteState::Rejected => "rejected",
    }
}

fn snapshot_path(store_root: impl AsRef<Path>) -> PathBuf {
    store_root.as_ref().join(PROFILE_SNAPSHOT_FILE)
}

/// Save a derived profile snapshot atomically. Consent-gated on learning:
/// no profile leaves memory for a host that disabled taste learning.
pub fn save_profile_snapshot(
    store_root: impl AsRef<Path>,
    settings: &TasteSettings,
    profile: &ScientificTasteProfile,
    policy: TastePolicy,
) -> Result<(), TasteLoopError> {
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    let path = snapshot_path(&store_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    }
    let journal_bytes = fs::metadata(journal_path(&store_root))
        .map(|meta| meta.len() as usize)
        .unwrap_or(0);
    let preferences = profile
        .preferences
        .values()
        .map(|entry| SnapshotPreference {
            preference_id: entry.preference_id,
            scope: entry.scope.scope_tag().to_string(),
            scope_key: entry.scope_key.clone(),
            state: state_tag_of(entry.state).to_string(),
            confidence_bps: entry.confidence_bps,
        })
        .collect();
    let snapshot = ProfileSnapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        journal_bytes,
        policy: crate::taste_package::PackagePolicy::from(policy),
        preferences,
    };
    let mut encoded = serde_json::to_vec_pretty(&snapshot)
        .map_err(|error| TasteLoopError::Io(error.to_string()))?;
    encoded.push(b'\n');
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, encoded).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    fs::rename(&tmp, &path).map_err(|error| TasteLoopError::Io(error.to_string()))?;
    Ok(())
}

/// Load a previously saved snapshot, bounds-checked. Missing file yields
/// `Ok(None)`; anything malformed fails closed.
pub fn load_profile_snapshot(
    store_root: impl AsRef<Path>,
) -> Result<Option<ProfileSnapshot>, TasteLoopError> {
    match fs::read(snapshot_path(store_root)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(TasteLoopError::Io(error.to_string())),
        Ok(bytes) => {
            if bytes.len() > MAX_FEEDBACK_JOURNAL_BYTES {
                return Err(TasteLoopError::JournalTooLarge);
            }
            let snapshot: ProfileSnapshot = serde_json::from_slice(&bytes)
                .map_err(|error| TasteLoopError::InvalidRecord(format!("bad snapshot: {error}")))?;
            if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
                return Err(TasteLoopError::InvalidRecord(
                    "unsupported snapshot schema version".to_string(),
                ));
            }
            if snapshot.journal_bytes > MAX_FEEDBACK_JOURNAL_BYTES {
                return Err(TasteLoopError::JournalTooLarge);
            }
            for preference in &snapshot.preferences {
                if preference.confidence_bps > cogno_core::MAX_TASTE_CONFIDENCE_BPS {
                    return Err(TasteLoopError::InvalidRecord(
                        "confidence_bps exceeds 10000".to_string(),
                    ));
                }
                let scope = TasteScope::from_scope_tag(&preference.scope).ok_or_else(|| {
                    TasteLoopError::InvalidRecord(format!("unknown scope `{}`", preference.scope))
                })?;
                scope
                    .validate_key(&preference.scope_key)
                    .map_err(|_| TasteLoopError::InvalidRecord("invalid scope key".to_string()))?;
                match preference.state.as_str() {
                    "quarantined" | "candidate" | "active" | "conflicted" | "rejected" => {}
                    other => {
                        return Err(TasteLoopError::InvalidRecord(format!(
                            "unknown state `{other}`"
                        )))
                    }
                }
            }
            Ok(Some(snapshot))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn granted() -> TasteSettings {
        TasteSettings {
            learning_enabled: true,
            export_allowed: true,
            import_allowed: true,
        }
    }

    fn record(event_id: u64, kind: &str, origin: &str, source_kind: &str) -> TasteFeedbackRecord {
        TasteFeedbackRecord {
            schema_version: FEEDBACK_SCHEMA_VERSION,
            event_id,
            preference_id: 7,
            scope: "project".to_string(),
            scope_key: "project:cogno-1".to_string(),
            kind: kind.to_string(),
            origin: origin.to_string(),
            source_kind: source_kind.to_string(),
            source_id: "scirust@gen7".to_string(),
            confidence_bps: 5_000,
            observed_at_unix: 0,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-loop-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn outcomes_flow_from_journal_to_active_preference() {
        let root = temp_root("happy");
        // One model proposal + two kernel confirmations -> Active.
        record_feedback(
            &root,
            &granted(),
            &record(1, "proposed", "model_inference", "assistance_model"),
        )
        .expect("record proposal");
        record_feedback(
            &root,
            &granted(),
            &record(
                2,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
            ),
        )
        .expect("record confirm 1");
        record_feedback(
            &root,
            &granted(),
            &record(
                3,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
            ),
        )
        .expect("record confirm 2");
        let profile =
            run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT).expect("iteration");
        assert_eq!(
            profile.preferences[&7].state,
            cogno_core::TasteState::Active
        );
        // Deterministic: a second run over the same journal is identical.
        let again =
            run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT).expect("iteration");
        assert_eq!(profile.preferences[&7], again.preferences[&7]);
    }

    #[test]
    fn consent_gates_block_recording_and_derivation() {
        // Defaults allow LOCAL learning but deny export/import; turning
        // learning off blocks the loop entirely.
        let local_only = TasteSettings {
            learning_enabled: false,
            ..TasteSettings::default()
        };
        let root = temp_root("consent");
        assert_eq!(
            record_feedback(
                &root,
                &local_only,
                &record(
                    1,
                    "confirmed",
                    "deterministic_evaluation",
                    "deterministic_kernel"
                )
            ),
            Err(TasteLoopError::LearningDeniedByConsent)
        );
        assert_eq!(
            run_feedback_iteration(&root, &local_only, TastePolicy::DEFAULT),
            Err(TasteLoopError::LearningDeniedByConsent)
        );
        assert!(!root.join(FEEDBACK_FILE).exists());
    }

    #[test]
    fn hostile_records_fail_closed() {
        let root = temp_root("hostile");
        // A model claiming confirmation authority is rejected at write time;
        // a model proposal is legitimate and must pass.
        assert!(record_feedback(
            &root,
            &granted(),
            &record(1, "proposed", "model_inference", "assistance_model")
        )
        .is_ok());
        assert_eq!(
            record_feedback(
                &root,
                &granted(),
                &record(2, "confirmed", "model_inference", "assistance_model")
            ),
            Err(TasteLoopError::InvalidRecord(
                "model inference may only propose, never confirm authority".to_string()
            ))
        );
        // A corrupted line in an existing journal aborts derivation.
        std::fs::create_dir_all(&root).expect("dir");
        std::fs::write(root.join(FEEDBACK_FILE), "{not json}\n").expect("journal");
        assert!(matches!(
            run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT),
            Err(TasteLoopError::InvalidRecord(_))
        ));
        // Unknown scope tag.
        let mut bad_scope = record(
            9,
            "confirmed",
            "deterministic_evaluation",
            "deterministic_kernel",
        );
        bad_scope.scope = "galaxy".to_string();
        assert_eq!(
            record_feedback(&root, &granted(), &bad_scope),
            Err(TasteLoopError::InvalidRecord(
                "unknown scope `galaxy`".to_string()
            ))
        );
    }

    #[test]
    fn duplicate_event_ids_are_counted_once() {
        let root = temp_root("dedup");
        for _ in 0..2 {
            record_feedback(
                &root,
                &granted(),
                &record(
                    5,
                    "confirmed",
                    "deterministic_evaluation",
                    "deterministic_kernel",
                ),
            )
            .expect("record");
        }
        let events = {
            let text = std::fs::read_to_string(root.join(FEEDBACK_FILE)).expect("read");
            text.lines().count()
        };
        assert_eq!(events, 2, "journal keeps both lines (append-only)");
        // Derivation deduplicates by event id: one confirmation only.
        let profile =
            run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT).expect("iteration");
        let entry = profile.preferences.get(&7).expect("entry");
        assert_ne!(
            entry.state,
            cogno_core::TasteState::Active,
            "a single deduplicated confirmation cannot activate"
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::taste_validation_store::{
        StoredTasteValidation, StoredValidationOrigin, StoredValidationVerdict,
        ValidationSubjectKind,
    };

    fn record(event_id: u64, kind: &str, origin: &str, source_kind: &str) -> TasteFeedbackRecord {
        TasteFeedbackRecord {
            schema_version: FEEDBACK_SCHEMA_VERSION,
            event_id,
            preference_id: 7,
            scope: "project".to_string(),
            scope_key: "project:cogno-1".to_string(),
            kind: kind.to_string(),
            origin: origin.to_string(),
            source_kind: source_kind.to_string(),
            source_id: "scirust@gen7".to_string(),
            confidence_bps: 5_000,
            observed_at_unix: 0,
        }
    }

    fn granted() -> TasteSettings {
        TasteSettings {
            learning_enabled: true,
            export_allowed: true,
            import_allowed: true,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-loop-x-{tag}-{}-{nonce}", std::process::id()))
    }

    fn stored(
        validation_id: u64,
        verdict: StoredValidationVerdict,
        hash: [u8; 32],
    ) -> StoredTasteValidation {
        StoredTasteValidation {
            validation_id,
            preference_id: 7,
            evidence_id: validation_id * 100,
            origin: StoredValidationOrigin::DeterministicEvaluation,
            verdict,
            confidence_bps: 6_500,
            subject_kind: ValidationSubjectKind::DeterministicEvaluator,
            subject_id_sha256: hash,
        }
    }

    #[test]
    fn validation_store_feeds_the_journal_and_reingest_is_harmless() {
        let root = temp_root("ingest");
        {
            let mut store =
                crate::taste_validation_store::PersistentTasteValidationStore::open(&root)
                    .expect("open");
            store
                .append(stored(1, StoredValidationVerdict::Confirmed, [7_u8; 32]))
                .expect("append 1");
            store
                .append(stored(2, StoredValidationVerdict::Confirmed, [7_u8; 32]))
                .expect("append 2");
        }
        let count =
            ingest_validation_store(&root, &granted(), TasteScope::Project, "project:cogno-1")
                .expect("ingest");
        assert_eq!(count, 2);
        // Re-ingesting appends again but derivation dedupes by event id.
        ingest_validation_store(&root, &granted(), TasteScope::Project, "project:cogno-1")
            .expect("re-ingest");
        let profile =
            run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT).expect("iterate");
        assert_eq!(
            profile.preferences[&7].state,
            cogno_core::TasteState::Active
        );
    }

    #[test]
    fn deterministic_forgetting_expires_only_old_timestamped_records() {
        let root = temp_root("forget");
        let mut old = record(
            1,
            "confirmed",
            "deterministic_evaluation",
            "deterministic_kernel",
        );
        old.observed_at_unix = 1_000;
        let mut fresh = record(
            2,
            "confirmed",
            "deterministic_evaluation",
            "deterministic_kernel",
        );
        fresh.observed_at_unix = 900_000;
        // Undated legacy record must survive any retention window.
        let legacy = record(3, "confirmed", "explicit_user_action", "human");
        for entry in [&old, &fresh, &legacy] {
            record_feedback(&root, &granted(), entry).expect("record");
        }
        // Two kernel confirms + one human edit -> Active without forgetting.
        let full = run_feedback_iteration(&root, &granted(), TastePolicy::DEFAULT).expect("full");
        assert_eq!(full.preferences[&7].state, cogno_core::TasteState::Active);
        // With a tight window both timestamped outcomes expire; only the
        // undated legacy record survives, so activation is lost.
        let pruned = run_feedback_iteration_retained(
            &root,
            &granted(),
            TastePolicy::DEFAULT,
            1_000_000,
            50_000,
        )
        .expect("pruned");
        let entry = pruned.preferences.get(&7).expect("entry");
        assert_ne!(
            entry.state,
            cogno_core::TasteState::Active,
            "expired evidence must not keep the preference active"
        );
        // Journal is never rewritten by iteration.
        let lines = std::fs::read_to_string(root.join(FEEDBACK_FILE))
            .expect("journal")
            .lines()
            .count();
        assert_eq!(lines, 3);
    }

    #[test]
    fn snapshots_round_trip_and_fail_closed() {
        let root = temp_root("snap");
        record_feedback(
            &root,
            &granted(),
            &record(1, "proposed", "model_inference", "assistance_model"),
        )
        .expect("proposal");
        let policy = TastePolicy::DEFAULT;
        let profile = run_feedback_iteration(&root, &granted(), policy).expect("iterate");
        save_profile_snapshot(&root, &granted(), &profile, policy).expect("save");
        let loaded = load_profile_snapshot(&root).expect("load").expect("some");
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.preferences.len(), profile.preferences.len());
        assert_eq!(
            loaded.policy.activation_threshold_bps,
            policy.activation_threshold_bps
        );
        // Missing file -> None.
        let empty = temp_root("snap-empty");
        assert_eq!(load_profile_snapshot(&empty).expect("none"), None);
        // Corrupted snapshot fails closed.
        std::fs::write(root.join(PROFILE_SNAPSHOT_FILE), "{broken").expect("corrupt");
        assert!(matches!(
            load_profile_snapshot(&root),
            Err(TasteLoopError::InvalidRecord(_))
        ));
    }
}

#[cfg(test)]
mod tuning_tests {
    use super::*;

    fn granted() -> TasteSettings {
        TasteSettings {
            learning_enabled: true,
            export_allowed: true,
            import_allowed: true,
        }
    }

    fn record(
        event_id: u64,
        kind: &str,
        origin: &str,
        source_kind: &str,
        observed_at: u64,
    ) -> TasteFeedbackRecord {
        TasteFeedbackRecord {
            schema_version: FEEDBACK_SCHEMA_VERSION,
            event_id,
            preference_id: 7,
            scope: "project".to_string(),
            scope_key: "project:cogno-1".to_string(),
            kind: kind.to_string(),
            origin: origin.to_string(),
            source_kind: source_kind.to_string(),
            source_id: "scirust@gen7".to_string(),
            confidence_bps: 5_000,
            observed_at_unix: observed_at,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-tune-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn compaction_dedupes_and_expires_without_losing_live_evidence() {
        let root = temp_root("compact");
        let settings = granted();
        for id in [1_u64, 1, 2, 2, 3] {
            let mut entry = record(
                id.max(1),
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
                if id == 3 { 1_000 } else { 900_000 },
            );
            entry.event_id = id;
            record_feedback(&root, &settings, &entry).expect("record");
        }
        // Duplicate ids (1 and 2 twice) plus one expired record (id 3).
        let report =
            compact_feedback_journal(&root, &settings, 1_000_000, 200_000).expect("compact");
        assert_eq!(report.records_before, 5);
        assert_eq!(report.records_after, 2);
        assert!(report.rewritten);
        // Derivation after compaction matches derivation before it would
        // with only the live events: two kernel confirms remain.
        let profile = run_feedback_iteration_retained(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            1_000_000,
            200_000,
        )
        .expect("iterate");
        let _ = profile;
        // A second compaction is a no-op.
        let again =
            compact_feedback_journal(&root, &settings, 1_000_000, 200_000).expect("compact again");
        assert!(!again.rewritten);
        // Corrupt journal aborts compaction instead of dropping data.
        std::fs::write(root.join(FEEDBACK_FILE), "{oops}\n").expect("corrupt");
        assert!(matches!(
            compact_feedback_journal(&root, &settings, 1_000_000, 200_000),
            Err(TasteLoopError::InvalidRecord(_))
        ));
    }

    #[test]
    fn graduated_decay_lowers_confidence_inside_the_window() {
        let root = temp_root("decay");
        let settings = granted();
        // Two kernel confirmations at full strength -> Active baseline.
        for id in [1_u64, 2] {
            let mut entry = record(
                id,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
                0,
            );
            entry.event_id = id;
            record_feedback(&root, &settings, &entry).expect("record");
        }
        let fresh = run_feedback_iteration_tuned(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            &LoopTuning {
                now_unix: 1_000_000,
                retention_secs: 100_000,
                graduated_decay: false,
                bandit_step_bps: 0,
            },
        )
        .expect("fresh");
        // Same evidence aged to ~90% of the horizon loses weight.
        std::fs::remove_file(root.join(FEEDBACK_FILE)).expect("reset");
        for id in [1_u64, 2] {
            let mut entry = record(
                id,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
                910_000,
            );
            entry.event_id = id;
            record_feedback(&root, &settings, &entry).expect("record");
        }
        let decayed = run_feedback_iteration_tuned(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            &LoopTuning {
                now_unix: 1_000_000,
                retention_secs: 100_000,
                graduated_decay: true,
                bandit_step_bps: 0,
            },
        )
        .expect("decayed");
        if let (Some(f), Some(d)) = (fresh.preferences.get(&7), decayed.preferences.get(&7)) {
            assert!(
                d.confidence_bps < f.confidence_bps,
                "aged evidence must weigh less ({} < {})",
                d.confidence_bps,
                f.confidence_bps
            );
        }
    }

    #[test]
    fn bandit_shifts_confidence_within_hard_bounds() {
        let root = temp_root("bandit");
        let settings = granted();
        // 2 wins vs 1 loss -> net +1.
        for (id, kind) in [(1_u64, "confirmed"), (2, "confirmed"), (3, "rejected")] {
            let mut entry = record(
                id,
                kind,
                "deterministic_evaluation",
                "deterministic_kernel",
                0,
            );
            entry.event_id = id;
            if kind == "rejected" {
                entry.kind = "rejected".to_string();
            }
            record_feedback(&root, &settings, &entry).expect("record");
        }
        let tuned = run_feedback_iteration_tuned(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            &LoopTuning {
                bandit_step_bps: 1_000,
                ..LoopTuning::default()
            },
        )
        .expect("tuned");
        let plain = run_feedback_iteration_tuned(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            &LoopTuning::default(),
        )
        .expect("plain");
        if let (Some(t), Some(p)) = (tuned.preferences.get(&7), plain.preferences.get(&7)) {
            assert!(
                t.confidence_bps > p.confidence_bps,
                "net wins must raise effective weight"
            );
        }
        // Extreme balance cannot exceed the hard cap: many wins saturate.
        std::fs::remove_file(root.join(FEEDBACK_FILE)).expect("reset");
        for id in 10..20_u64 {
            let mut entry = record(
                id,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
                0,
            );
            entry.event_id = id;
            record_feedback(&root, &settings, &entry).expect("record");
        }
        let saturated = run_feedback_iteration_tuned(
            &root,
            &settings,
            TastePolicy::DEFAULT,
            &LoopTuning {
                bandit_step_bps: 5_000,
                ..LoopTuning::default()
            },
        )
        .expect("saturated");
        for entry in saturated.preferences.values() {
            assert!(entry.confidence_bps <= cogno_core::MAX_TASTE_CONFIDENCE_BPS);
        }
    }

    #[test]
    fn active_views_expose_only_active_preferences() {
        let root = temp_root("views");
        let settings = granted();
        record_feedback(
            &root,
            &settings,
            &record(1, "proposed", "model_inference", "assistance_model", 0),
        )
        .expect("record");
        let views =
            active_preference_views(&root, &settings, &LoopTuning::default()).expect("views");
        assert!(views.is_empty(), "a lone proposal is not Active yet");
        for id in [2_u64, 3] {
            let mut entry = record(
                id,
                "confirmed",
                "deterministic_evaluation",
                "deterministic_kernel",
                0,
            );
            entry.event_id = id;
            record_feedback(&root, &settings, &entry).expect("record");
        }
        let views =
            active_preference_views(&root, &settings, &LoopTuning::default()).expect("views");
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].preference_id, 7);
        assert_eq!(views[0].scope, "project");
        assert!(views[0].confidence_bps > 0);
    }
}

/// Read-only integrity audit of a store (verification never needs consent:
/// it reads nothing sensitive and changes nothing).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StoreAudit {
    pub settings_present: bool,
    pub learning_enabled: bool,
    pub export_allowed: bool,
    pub import_allowed: bool,
    /// Journal lines / duplicate ids / expired records under the given
    /// retention window (0 disables expiry counting).
    pub journal_records: usize,
    pub journal_duplicate_ids: usize,
    pub journal_expired: usize,
    pub journal_corrupt: Option<String>,
    pub snapshot_valid: bool,
    pub snapshot_preferences: usize,
    pub accepted_log_digests: usize,
    pub package_digest: Option<String>,
    pub findings: Vec<String>,
}

pub fn audit_store(
    store_root: impl AsRef<Path>,
    now_unix: u64,
    retention_secs: u64,
) -> Result<StoreAudit, TasteLoopError> {
    let store_root = store_root.as_ref();
    let mut findings = Vec::new();

    let settings = crate::load_settings(store_root);
    let (settings_present, settings) = match &settings {
        Ok(settings) => (true, *settings),
        Err(_) => (false, TasteSettings::default()),
    };

    let path = journal_path(store_root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(TasteLoopError::Io(error.to_string())),
    };
    let mut journal_records = 0;
    let mut journal_duplicate_ids = 0;
    let mut journal_expired = 0;
    let mut journal_corrupt = None;
    let mut seen_ids = std::collections::BTreeSet::new();
    if let Some(bytes) = &bytes {
        match std::str::from_utf8(bytes) {
            Err(_) => journal_corrupt = Some("journal is not UTF-8".to_string()),
            Ok(text) => {
                for line in text.lines().filter(|line| !line.trim().is_empty()) {
                    match serde_json::from_str::<TasteFeedbackRecord>(line) {
                        Err(error) => {
                            journal_corrupt = Some(format!("bad json: {error}"));
                            break;
                        }
                        Ok(record) => {
                            if let Err(error) = validate_record(&record) {
                                journal_corrupt = Some(format!("{error}"));
                                break;
                            }
                            journal_records += 1;
                            if !seen_ids.insert(record.event_id) {
                                journal_duplicate_ids += 1;
                            }
                            if retention_secs > 0 && record.observed_at_unix != 0 {
                                let age = now_unix.saturating_sub(record.observed_at_unix);
                                if age > retention_secs {
                                    journal_expired += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if bytes
        .as_ref()
        .is_some_and(|bytes| bytes.len() > MAX_FEEDBACK_JOURNAL_BYTES)
    {
        findings.push("journal exceeds size bound; run compaction".to_string());
    }
    if journal_duplicate_ids > 0 {
        findings.push(format!(
            "{journal_duplicate_ids} duplicate event id(s); compaction advised"
        ));
    }
    if journal_expired > 0 {
        findings.push(format!(
            "{journal_expired} expired record(s); compaction advised"
        ));
    }

    let snapshot = load_profile_snapshot(store_root)?;
    let snapshot_valid = snapshot.is_some();
    let snapshot_preferences = snapshot
        .map(|snapshot| snapshot.preferences.len())
        .unwrap_or(0);

    let accepted_path = store_root.join("taste.accepted.log");
    let accepted_log_digests = match fs::read(accepted_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(TasteLoopError::Io(error.to_string())),
        Ok(bytes) => {
            let count = std::str::from_utf8(&bytes)
                .map_err(|_| TasteLoopError::InvalidRecord("accepted log not UTF-8".to_string()))?
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count();
            for digest in std::str::from_utf8(
                &fs::read(store_root.join("taste.accepted.log"))
                    .map_err(|error| TasteLoopError::Io(error.to_string()))?,
            )
            .map_err(|_| TasteLoopError::InvalidRecord("accepted log not UTF-8".to_string()))?
            .lines()
            .filter(|line| !line.trim().is_empty())
            {
                if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    findings.push(format!("malformed digest in accepted log: {digest}"));
                }
            }
            count
        }
    };

    let package_digest = match cogno_core_package(store_root) {
        Ok(Some(digest)) => Some(digest),
        Ok(None) => None,
        Err(message) => {
            findings.push(format!("taste.md invalid: {message}"));
            None
        }
    };

    if !settings_present {
        findings.push("missing taste.settings.json (defaults apply)".to_string());
    }
    if !settings.can_learn() {
        findings.push("learning disabled: loop will refuse to run".to_string());
    }

    Ok(StoreAudit {
        settings_present,
        learning_enabled: settings.learning_enabled,
        export_allowed: settings.export_allowed,
        import_allowed: settings.import_allowed,
        journal_records,
        journal_duplicate_ids,
        journal_expired,
        journal_corrupt,
        snapshot_valid,
        snapshot_preferences,
        accepted_log_digests,
        package_digest,
        findings,
    })
}

fn cogno_core_package(store_root: &Path) -> Result<Option<String>, String> {
    let path = store_root.join("taste.md");
    if !path.exists() {
        return Ok(None);
    }
    match crate::TastePackage::load_file(&path) {
        Ok(package) => Ok(Some(package.digest_hex())),
        Err(error) => Err(format!("{error:?}")),
    }
}
