//! End-to-end loop integration: validation store → feedback journal →
//! derived profile → snapshot delta. Covers the wiring between layers that
//! unit tests exercise only in isolation.

#![forbid(unsafe_code)]
#![deny(warnings)]

use cogno_core::{TastePolicy, TasteScope, TasteSettings, TasteState};
use cogno_runtime::taste_validation_store::{
    PersistentTasteValidationStore, StoredTasteValidation, StoredValidationOrigin,
    StoredValidationVerdict, ValidationSubjectKind,
};
use cogno_runtime::{
    ingest_validation_store, load_profile_snapshot, record_feedback,
    run_feedback_iteration_retained, save_profile_snapshot, TasteFeedbackRecord,
};

fn granted() -> TasteSettings {
    TasteSettings {
        learning_enabled: true,
        export_allowed: true,
        import_allowed: true,
    }
}

fn temp_root(tag: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!("cogno-chain-{tag}-{}-{nonce}", std::process::id()))
}

#[test]
fn full_chain_from_validations_to_active_snapshot_with_delta() {
    let root = temp_root("chain");
    // 1) Real outcomes land in the validation store.
    {
        let mut store = PersistentTasteValidationStore::open(&root).expect("open");
        for id in 1..=2_u64 {
            store
                .append(StoredTasteValidation {
                    validation_id: id,
                    preference_id: 7,
                    evidence_id: id * 100,
                    origin: StoredValidationOrigin::DeterministicEvaluation,
                    verdict: StoredValidationVerdict::Confirmed,
                    confidence_bps: 6_500,
                    subject_kind: ValidationSubjectKind::DeterministicEvaluator,
                    subject_id_sha256: [11_u8; 32],
                })
                .expect("append");
        }
    }
    let settings = granted();
    let policy = TastePolicy::DEFAULT;
    // 2) Ingestion feeds the journal; derivation activates.
    let count = ingest_validation_store(&root, &settings, TasteScope::Project, "project:cogno-1")
        .expect("ingest");
    assert_eq!(count, 2);
    let profile = run_feedback_iteration_retained(&root, &settings, policy, 0, 0).expect("iterate");
    assert_eq!(profile.preferences[&7].state, TasteState::Active);

    // 3) First iteration has no previous snapshot; second reports no change.
    save_profile_snapshot(&root, &settings, &profile, policy).expect("save");
    let first = load_profile_snapshot(&root).expect("load").expect("some");
    assert_eq!(first.preferences.len(), 1);
    assert_eq!(first.preferences[0].state, "active");

    // 4) A new human edit shifts confidence -> next snapshot sees the delta
    // through a fresh derivation over the extended journal.
    record_feedback(
        &root,
        &settings,
        &TasteFeedbackRecord {
            schema_version: 1,
            event_id: 500,
            preference_id: 7,
            scope: "project".to_string(),
            scope_key: "project:cogno-1".to_string(),
            kind: "edited".to_string(),
            origin: "explicit_user_action".to_string(),
            source_kind: "human".to_string(),
            source_id: "alice".to_string(),
            confidence_bps: 5_000,
            observed_at_unix: 0,
        },
    )
    .expect("record edit");
    let updated =
        run_feedback_iteration_retained(&root, &settings, policy, 0, 0).expect("re-iterate");
    assert!(
        updated.preferences[&7].confidence_bps >= profile.preferences[&7].confidence_bps,
        "a confirming edit must not lower confidence"
    );
    save_profile_snapshot(&root, &settings, &updated, policy).expect("save 2");
    let second = load_profile_snapshot(&root).expect("load").expect("some");
    assert!(second.journal_bytes > first.journal_bytes);

    // 5) Forgetting the kernel confirmations (timestamped) drops activation.
    let mut timestamped = TasteFeedbackRecord {
        schema_version: 1,
        event_id: 501,
        preference_id: 8,
        scope: "project".to_string(),
        scope_key: "project:cogno-1".to_string(),
        kind: "confirmed".to_string(),
        origin: "deterministic_evaluation".to_string(),
        source_kind: "deterministic_kernel".to_string(),
        source_id: "scirust@gen7".to_string(),
        confidence_bps: 6_000,
        observed_at_unix: 10_000,
    };
    record_feedback(&root, &settings, &timestamped).expect("record");
    timestamped.event_id = 502;
    record_feedback(&root, &settings, &timestamped).expect("record");
    let pruned = run_feedback_iteration_retained(&root, &settings, policy, 100_000, 20_000)
        .expect("retained iterate");
    // Preference 7 evidence was undated so it survives; preference 8's two
    // timestamped confirms expired -> no active state for it.
    assert!(pruned
        .preferences
        .get(&8)
        .is_none_or(|entry| entry.state != TasteState::Active));
}
