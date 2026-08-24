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
    if !settings.can_learn() {
        return Err(TasteLoopError::LearningDeniedByConsent);
    }
    let events = load_journal_events(store_root)?;
    ScientificTasteProfile::derive(&events, policy).map_err(TasteLoopError::Core)
}

/// Read and validate the whole journal into canonical event order.
fn load_journal_events(store_root: impl AsRef<Path>) -> Result<Vec<TasteEvent>, TasteLoopError> {
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
        // First occurrence wins; replays of the same id are ignored so a
        // re-fed journal cannot double-count outcomes.
        if seen_ids.insert(record.event_id) {
            events.push(event_from_record(&record)?);
        }
    }
    events.sort_by_key(|event| event.event_id);
    Ok(events)
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
