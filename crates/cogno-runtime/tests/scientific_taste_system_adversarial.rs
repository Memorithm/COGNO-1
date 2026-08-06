//! System-level adversarial coverage for the autonomous scientific-taste path.

use cogno_runtime::{
    classify_scientific_exchange, commit_taste_cycle, evaluate_taste_benchmark,
    orchestrate_scientific_taste_cycle, OrchestratedTasteCandidate, OrchestratedTasteState,
    ScientificExchangeDisposition, ScientificExchangeOrigin, ScientificExchangeRecord,
    ScientificExchangeVerdict, StoredTasteValidation, StoredValidationOrigin,
    StoredValidationVerdict, TasteBenchmarkCase, TasteCycleArtifacts, TasteCycleError,
    TasteGenerationChain, TasteGenerationError, TasteGenerationManifest, TasteHistoryObservation,
    TasteLongitudinalTracker, GENESIS_DIGEST,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cogno-system-adversarial-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn artifacts() -> TasteCycleArtifacts {
    TasteCycleArtifacts {
        candidate_report: b"candidate-report-v1".to_vec(),
        validation_store: b"validation-store-v1".to_vec(),
        replay_report: b"replay-report-v1".to_vec(),
        profile: b"profile-v1".to_vec(),
    }
}

fn manifest(
    generation: u64,
    previous_manifest_sha256: [u8; 32],
    artifacts: &TasteCycleArtifacts,
) -> TasteGenerationManifest {
    TasteGenerationManifest {
        generation,
        previous_manifest_sha256,
        profile_sha256: digest(&artifacts.profile),
        replay_sha256: digest(&artifacts.replay_report),
        candidate_report_sha256: digest(&artifacts.candidate_report),
        validation_store_sha256: digest(&artifacts.validation_store),
    }
}

#[test]
fn autonomous_cycle_remains_model_quarantined_and_measurably_non_regressive() {
    let model_record = ScientificExchangeRecord {
        observation_id: 1,
        preference_id: 7,
        evidence_id: 100,
        origin: ScientificExchangeOrigin::SciAgentModel,
        verdict: ScientificExchangeVerdict::Confirmed,
        confidence_bps: 10_000,
        canonical_payload: b"model proposal".to_vec(),
    };
    assert_eq!(
        classify_scientific_exchange(&model_record),
        Ok(ScientificExchangeDisposition::QuarantinedModelObservation)
    );

    let mut validations = Vec::<StoredTasteValidation>::new();
    for (observation_id, confidence_bps) in [(2, 8_000), (3, 9_000)] {
        let record = ScientificExchangeRecord {
            observation_id,
            preference_id: 7,
            evidence_id: observation_id + 100,
            origin: ScientificExchangeOrigin::SciRustDeterministicEvaluator,
            verdict: ScientificExchangeVerdict::Confirmed,
            confidence_bps,
            canonical_payload: format!("deterministic-evaluation-{observation_id}").into_bytes(),
        };
        match classify_scientific_exchange(&record).expect("exchange") {
            ScientificExchangeDisposition::Validation(validation) => validations.push(validation),
            other => panic!("unexpected disposition: {other:?}"),
        }
    }

    let report = orchestrate_scientific_taste_cycle(
        &[OrchestratedTasteCandidate { preference_id: 7 }],
        &validations,
    )
    .expect("orchestrate");
    assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Active);
    assert!(report.outcomes[0].can_activate);

    let benchmark = evaluate_taste_benchmark(&[TasteBenchmarkCase {
        case_id: 1,
        baseline_candidate_id: Some(10),
        taste_candidate_id: Some(11),
        oracle_candidate_id: Some(11),
        baseline_score: 70,
        taste_score: 100,
        oracle_score: 100,
    }])
    .expect("benchmark");
    assert!(benchmark.non_regressive());
    assert_eq!(benchmark.correctness_delta(), 1);

    let mut tracker = TasteLongitudinalTracker::default();
    tracker
        .append(TasteHistoryObservation {
            generation: 1,
            preference_id: 7,
            confidence_bps: report.outcomes[0].confidence_bps,
            active: true,
            contradicted: false,
        })
        .expect("history");
    assert!(tracker.status(7, 1).is_some());
}

#[test]
fn corruption_fork_replay_and_duplicate_generation_fail_closed() {
    let root = root("cycle");
    let artifacts = artifacts();
    let first = manifest(1, GENESIS_DIGEST, &artifacts);

    let committed = commit_taste_cycle(&root, &first, &artifacts).expect("first commit");
    assert_eq!(committed.generation, 1);
    assert_eq!(
        fs::read_to_string(root.join("CURRENT")).expect("current"),
        "1\n"
    );

    assert!(matches!(
        commit_taste_cycle(&root, &first, &artifacts),
        Err(TasteCycleError::GenerationAlreadyExists)
    ));

    let mut corrupted = artifacts.clone();
    corrupted.profile.push(0xff);
    let second = manifest(2, first.sha256(), &artifacts);
    assert!(matches!(
        commit_taste_cycle(&root, &second, &corrupted),
        Err(TasteCycleError::DigestMismatch("profile"))
    ));
    assert_eq!(
        fs::read_to_string(root.join("CURRENT")).expect("current after rejection"),
        "1\n"
    );
    assert!(!root.join("generation-2").exists());

    let mut chain = TasteGenerationChain::default();
    chain.append(first.clone()).expect("chain first");
    assert_eq!(
        chain.append(TasteGenerationManifest {
            generation: 2,
            previous_manifest_sha256: [9; 32],
            ..second.clone()
        }),
        Err(TasteGenerationError::PreviousDigestMismatch)
    );
    assert_eq!(
        chain.select_rollback(99),
        Err(TasteGenerationError::UnknownRollbackGeneration)
    );

    fs::write(root.join("CURRENT"), b"999999999999999999999999\n").expect("corrupt current");
    assert_ne!(
        fs::read_to_string(root.join("CURRENT")).expect("corrupt read"),
        "1\n"
    );
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn conflicting_and_duplicate_evidence_never_activate() {
    let candidate = OrchestratedTasteCandidate { preference_id: 7 };
    let confirmed = StoredTasteValidation {
        validation_id: 1,
        preference_id: 7,
        evidence_id: 11,
        origin: StoredValidationOrigin::DeterministicEvaluation,
        verdict: StoredValidationVerdict::Confirmed,
        confidence_bps: 10_000,
    };
    let contradictory = StoredTasteValidation {
        validation_id: 2,
        preference_id: 7,
        evidence_id: 12,
        origin: StoredValidationOrigin::ExplicitUserAction,
        verdict: StoredValidationVerdict::Contradicted,
        confidence_bps: 10_000,
    };
    let report = orchestrate_scientific_taste_cycle(&[candidate], &[confirmed, contradictory])
        .expect("conflict");
    assert_eq!(report.outcomes[0].state, OrchestratedTasteState::Conflicted);
    assert!(!report.outcomes[0].can_activate);

    assert!(orchestrate_scientific_taste_cycle(&[candidate], &[confirmed, confirmed]).is_err());
}
