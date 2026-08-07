//! Public integration coverage for the post-campaign review gate.

use cogno_runtime::{
    review_scientific_taste_campaign, AutonomousTasteReport, OrchestratedTasteOutcome,
    OrchestratedTasteState, ScientificTasteCycleReport, TasteBenchmarkReport, TasteCampaignReport,
    TasteCampaignReviewAuthority, TasteCampaignReviewDisposition, TasteCampaignReviewPolicy,
    TasteDriftState, TasteLongitudinalStatus,
};

fn autonomous(generation: u64, quarantined_model_observations: usize) -> AutonomousTasteReport {
    AutonomousTasteReport {
        generation,
        quarantined_model_observations,
        runtime_observations: 0,
        inconclusive_observations: 0,
        accepted_validations: 2,
        cycle: ScientificTasteCycleReport {
            candidates_scanned: 1,
            validations_scanned: 2,
            outcomes: vec![OrchestratedTasteOutcome {
                preference_id: 42,
                state: OrchestratedTasteState::Active,
                confidence_bps: 8_200,
                confirmations: 2,
                validations: 2,
                can_activate: true,
            }],
        },
        benchmark: TasteBenchmarkReport {
            cases: 2,
            baseline_correct: 1,
            taste_correct: 2,
            baseline_abstentions: 0,
            taste_abstentions: 0,
            baseline_total_regret: 20,
            taste_total_regret: 0,
            improved_cases: 1,
            regressed_cases: 0,
            unchanged_cases: 1,
        },
    }
}

#[test]
fn campaign_review_is_deterministic_and_model_non_authoritative() {
    let campaign = TasteCampaignReport {
        first_generation: 11,
        last_generation: 12,
        generations: 2,
        autonomous_reports: vec![autonomous(11, 3), autonomous(12, 4)],
        longitudinal_statuses: vec![TasteLongitudinalStatus {
            preference_id: 42,
            first_generation: 11,
            last_generation: 12,
            observations: 2,
            latest_confidence_bps: 8_200,
            drift: TasteDriftState::Stable,
            eligible_for_supersession: false,
        }],
    };

    let first = review_scientific_taste_campaign(
        &campaign,
        TasteCampaignReviewPolicy::default(),
    )
    .expect("first review");
    let second = review_scientific_taste_campaign(
        &campaign,
        TasteCampaignReviewPolicy::default(),
    )
    .expect("second review");

    assert_eq!(first, second);
    assert_eq!(
        first.disposition,
        TasteCampaignReviewDisposition::EligibleForControlledRestartReview
    );
    assert_eq!(first.authority, TasteCampaignReviewAuthority::ReviewOnly);
    assert_eq!(first.quarantined_model_observations, 7);
    assert_eq!(first.preferences.len(), 1);
    assert!(first.preferences[0].eligible);
}
