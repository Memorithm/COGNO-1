//! End-to-end deterministic autonomous scientific-taste evaluation.

use cogno_runtime::{
    run_autonomous_scientific_taste, OrchestratedTasteCandidate, OrchestratedTasteState,
    ScientificExchangeOrigin, ScientificExchangeRecord, ScientificExchangeVerdict,
    TasteBenchmarkCase, TasteDriftState, TasteLongitudinalTracker,
};

fn exchange(
    observation_id: u64,
    origin: ScientificExchangeOrigin,
    confidence_bps: u16,
) -> ScientificExchangeRecord {
    ScientificExchangeRecord {
        observation_id,
        preference_id: 42,
        evidence_id: observation_id + 1_000,
        origin,
        verdict: ScientificExchangeVerdict::Confirmed,
        confidence_bps,
        canonical_payload: format!("exchange-{observation_id}").into_bytes(),
    }
}

#[test]
fn two_generations_are_deterministic_and_track_strengthening() {
    let candidate = OrchestratedTasteCandidate { preference_id: 42 };
    let benchmark = TasteBenchmarkCase {
        case_id: 1,
        baseline_candidate_id: Some(1),
        taste_candidate_id: Some(2),
        oracle_candidate_id: Some(2),
        baseline_score: 70,
        taste_score: 100,
        oracle_score: 100,
    };
    let mut longitudinal = TasteLongitudinalTracker::default();

    let first = run_autonomous_scientific_taste(
        1,
        &[candidate],
        &[
            exchange(1, ScientificExchangeOrigin::SciAgentModel, 10_000),
            exchange(
                2,
                ScientificExchangeOrigin::SciRustDeterministicEvaluator,
                7_000,
            ),
            exchange(3, ScientificExchangeOrigin::ExplicitUserAction, 7_000),
        ],
        &[benchmark],
        &mut longitudinal,
    )
    .expect("first generation");

    assert_eq!(first.quarantined_model_observations, 1);
    assert_eq!(first.runtime_observations, 1);
    assert_eq!(first.accepted_validations, 1);
    assert_eq!(first.cycle.outcomes[0].confirmations, 1);
    assert_eq!(first.cycle.outcomes[0].confidence_bps, 7_000);
    assert_eq!(
        first.cycle.outcomes[0].state,
        OrchestratedTasteState::Candidate
    );
    assert!(first.benchmark.non_regressive());

    let second = run_autonomous_scientific_taste(
        2,
        &[candidate],
        &[
            exchange(
                4,
                ScientificExchangeOrigin::SciRustDeterministicEvaluator,
                9_000,
            ),
            exchange(5, ScientificExchangeOrigin::ExplicitUserAction, 9_000),
        ],
        &[benchmark],
        &mut longitudinal,
    )
    .expect("second generation");

    assert_eq!(second.runtime_observations, 1);
    assert_eq!(second.accepted_validations, 1);
    assert_eq!(second.cycle.outcomes[0].confirmations, 1);
    assert_eq!(second.cycle.outcomes[0].confidence_bps, 9_000);
    assert_eq!(
        second.cycle.outcomes[0].state,
        OrchestratedTasteState::Candidate
    );
    assert_eq!(
        longitudinal.status(42, 2).map(|status| status.drift),
        Some(TasteDriftState::Strengthening)
    );
}
