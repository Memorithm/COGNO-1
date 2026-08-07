//! Public API coverage for bounded post-adversarial scientific-taste campaigns.

use cogno_runtime::{
    run_scientific_taste_campaign, OrchestratedTasteCandidate, ScientificExchangeOrigin,
    ScientificExchangeRecord, ScientificExchangeVerdict, TasteCampaignGeneration,
    TasteDriftState, TasteLongitudinalTracker,
};

fn exchange(
    observation_id: u64,
    preference_id: u64,
    origin: ScientificExchangeOrigin,
    confidence_bps: u16,
) -> ScientificExchangeRecord {
    ScientificExchangeRecord {
        observation_id,
        preference_id,
        evidence_id: observation_id + 10_000,
        origin,
        verdict: ScientificExchangeVerdict::Confirmed,
        confidence_bps,
        canonical_payload: format!("integration-campaign-{observation_id}").into_bytes(),
    }
}

#[test]
fn campaign_is_deterministic_and_model_non_authoritative() {
    let candidate = [OrchestratedTasteCandidate { preference_id: 42 }];
    let generation_one = [
        exchange(1, 42, ScientificExchangeOrigin::SciAgentModel, 10_000),
        exchange(
            2,
            42,
            ScientificExchangeOrigin::SciRustDeterministicEvaluator,
            7_500,
        ),
        exchange(
            3,
            42,
            ScientificExchangeOrigin::ExplicitUserAction,
            7_500,
        ),
    ];
    let generation_two = [
        exchange(4, 42, ScientificExchangeOrigin::SciAgentModel, 10_000),
        exchange(
            5,
            42,
            ScientificExchangeOrigin::SciRustDeterministicEvaluator,
            8_500,
        ),
        exchange(
            6,
            42,
            ScientificExchangeOrigin::ExplicitUserAction,
            8_500,
        ),
    ];
    let campaign = [
        TasteCampaignGeneration {
            generation: 10,
            candidates: &candidate,
            exchange_records: &generation_one,
            benchmark_cases: &[],
        },
        TasteCampaignGeneration {
            generation: 11,
            candidates: &candidate,
            exchange_records: &generation_two,
            benchmark_cases: &[],
        },
    ];

    let mut first_tracker = TasteLongitudinalTracker::default();
    let first = run_scientific_taste_campaign(&campaign, &mut first_tracker).expect("first run");
    let mut second_tracker = TasteLongitudinalTracker::default();
    let second =
        run_scientific_taste_campaign(&campaign, &mut second_tracker).expect("second run");

    assert_eq!(first, second);
    assert_eq!(first_tracker, second_tracker);
    assert_eq!(first.autonomous_reports.len(), 2);
    assert_eq!(first.autonomous_reports[0].quarantined_model_observations, 1);
    assert_eq!(first.autonomous_reports[1].quarantined_model_observations, 1);
    assert_eq!(first.autonomous_reports[0].accepted_validations, 2);
    assert_eq!(first.autonomous_reports[1].accepted_validations, 2);
    assert_eq!(first.longitudinal_statuses.len(), 1);
    assert_eq!(
        first.longitudinal_statuses[0].drift,
        TasteDriftState::Strengthening
    );
}
