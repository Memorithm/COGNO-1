//! Deterministic review gate for completed scientific-taste campaigns.
//!
//! The gate consumes an immutable [`TasteCampaignReport`] and derives whether
//! its results are eligible for a *controlled restart review*. Eligibility is
//! deliberately weaker than runtime authority: this module cannot persist,
//! install, replace, or hot-swap a verified taste profile.

use crate::taste_campaign::TasteCampaignReport;
use crate::taste_longitudinal::TasteDriftState;
use crate::taste_orchestrator::OrchestratedTasteState;
use std::collections::{BTreeMap, BTreeSet};

/// Minimum number of successful generations required before review eligibility.
pub const MIN_CAMPAIGN_REVIEW_GENERATIONS: usize = 2;
/// Minimum latest deterministic confidence required by the default review gate.
pub const MIN_CAMPAIGN_REVIEW_CONFIDENCE_BPS: u16 = 7_000;

/// Configuration for deterministic campaign review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteCampaignReviewPolicy {
    pub minimum_generations: usize,
    pub minimum_confidence_bps: u16,
    pub maximum_regressed_cases: usize,
}

impl Default for TasteCampaignReviewPolicy {
    fn default() -> Self {
        Self {
            minimum_generations: MIN_CAMPAIGN_REVIEW_GENERATIONS,
            minimum_confidence_bps: MIN_CAMPAIGN_REVIEW_CONFIDENCE_BPS,
            maximum_regressed_cases: 0,
        }
    }
}

/// Non-authoritative result of the campaign review gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteCampaignReviewDisposition {
    Hold,
    EligibleForControlledRestartReview,
}

/// Why a preference is held back by the review gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TasteCampaignReviewBlocker {
    MissingFinalOutcome,
    FinalOutcomeNotActive,
    ConfidenceBelowThreshold,
    Contradicted,
    Weakening,
    Stale,
}

/// Review result for one longitudinally observed preference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TastePreferenceReview {
    pub preference_id: u64,
    pub latest_confidence_bps: u16,
    pub blockers: Vec<TasteCampaignReviewBlocker>,
    pub eligible: bool,
}

/// Deterministic aggregate campaign review.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCampaignReviewReport {
    pub first_generation: u64,
    pub last_generation: u64,
    pub generations: usize,
    pub benchmark_cases: usize,
    pub baseline_total_regret: u128,
    pub taste_total_regret: u128,
    pub improved_cases: usize,
    pub regressed_cases: usize,
    pub quarantined_model_observations: usize,
    pub preferences: Vec<TastePreferenceReview>,
    pub disposition: TasteCampaignReviewDisposition,
    pub authority: TasteCampaignReviewAuthority,
}

/// Explicit authority marker for review output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteCampaignReviewAuthority {
    ReviewOnly,
}

/// Invalid or internally inconsistent campaign review input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteCampaignReviewError {
    InvalidPolicy,
    EmptyCampaignReport,
    GenerationCountMismatch,
    InvalidGenerationRange,
    NonMonotonicGeneration { previous: u64, current: u64 },
    DuplicateLongitudinalPreference(u64),
}

/// Review one completed campaign without mutating any runtime or persisted state.
pub fn review_scientific_taste_campaign(
    campaign: &TasteCampaignReport,
    policy: TasteCampaignReviewPolicy,
) -> Result<TasteCampaignReviewReport, TasteCampaignReviewError> {
    validate_policy(policy)?;
    validate_campaign_shape(campaign)?;

    let mut benchmark_cases = 0usize;
    let mut baseline_total_regret = 0u128;
    let mut taste_total_regret = 0u128;
    let mut improved_cases = 0usize;
    let mut regressed_cases = 0usize;
    let mut quarantined_model_observations = 0usize;

    for autonomous in &campaign.autonomous_reports {
        benchmark_cases = benchmark_cases.saturating_add(autonomous.benchmark.cases);
        baseline_total_regret = baseline_total_regret
            .saturating_add(autonomous.benchmark.baseline_total_regret);
        taste_total_regret =
            taste_total_regret.saturating_add(autonomous.benchmark.taste_total_regret);
        improved_cases = improved_cases.saturating_add(autonomous.benchmark.improved_cases);
        regressed_cases = regressed_cases.saturating_add(autonomous.benchmark.regressed_cases);
        quarantined_model_observations = quarantined_model_observations
            .saturating_add(autonomous.quarantined_model_observations);
    }

    let final_outcomes = campaign
        .autonomous_reports
        .last()
        .expect("validated non-empty campaign")
        .cycle
        .outcomes
        .iter()
        .map(|outcome| (outcome.preference_id, *outcome))
        .collect::<BTreeMap<_, _>>();

    let mut seen_preferences = BTreeSet::new();
    let mut preferences = Vec::with_capacity(campaign.longitudinal_statuses.len());
    for status in &campaign.longitudinal_statuses {
        if !seen_preferences.insert(status.preference_id) {
            return Err(TasteCampaignReviewError::DuplicateLongitudinalPreference(
                status.preference_id,
            ));
        }

        let mut blockers = Vec::new();
        match final_outcomes.get(&status.preference_id) {
            Some(outcome) if outcome.state == OrchestratedTasteState::Active => {}
            Some(_) => blockers.push(TasteCampaignReviewBlocker::FinalOutcomeNotActive),
            None => blockers.push(TasteCampaignReviewBlocker::MissingFinalOutcome),
        }

        if status.latest_confidence_bps < policy.minimum_confidence_bps {
            blockers.push(TasteCampaignReviewBlocker::ConfidenceBelowThreshold);
        }
        match status.drift {
            TasteDriftState::Contradicted => {
                blockers.push(TasteCampaignReviewBlocker::Contradicted)
            }
            TasteDriftState::Weakening => blockers.push(TasteCampaignReviewBlocker::Weakening),
            TasteDriftState::Stale => blockers.push(TasteCampaignReviewBlocker::Stale),
            TasteDriftState::New | TasteDriftState::Stable | TasteDriftState::Strengthening => {}
        }

        preferences.push(TastePreferenceReview {
            preference_id: status.preference_id,
            latest_confidence_bps: status.latest_confidence_bps,
            eligible: blockers.is_empty(),
            blockers,
        });
    }

    let benchmark_is_non_regressive = taste_total_regret <= baseline_total_regret
        && regressed_cases <= policy.maximum_regressed_cases;
    let enough_generations = campaign.generations >= policy.minimum_generations;
    let preferences_are_eligible = !preferences.is_empty()
        && preferences.iter().all(|preference| preference.eligible);

    let disposition = if benchmark_is_non_regressive
        && enough_generations
        && preferences_are_eligible
    {
        TasteCampaignReviewDisposition::EligibleForControlledRestartReview
    } else {
        TasteCampaignReviewDisposition::Hold
    };

    Ok(TasteCampaignReviewReport {
        first_generation: campaign.first_generation,
        last_generation: campaign.last_generation,
        generations: campaign.generations,
        benchmark_cases,
        baseline_total_regret,
        taste_total_regret,
        improved_cases,
        regressed_cases,
        quarantined_model_observations,
        preferences,
        disposition,
        authority: TasteCampaignReviewAuthority::ReviewOnly,
    })
}

fn validate_policy(policy: TasteCampaignReviewPolicy) -> Result<(), TasteCampaignReviewError> {
    if policy.minimum_generations == 0 || policy.minimum_confidence_bps > 10_000 {
        return Err(TasteCampaignReviewError::InvalidPolicy);
    }
    Ok(())
}

fn validate_campaign_shape(campaign: &TasteCampaignReport) -> Result<(), TasteCampaignReviewError> {
    if campaign.generations == 0 || campaign.autonomous_reports.is_empty() {
        return Err(TasteCampaignReviewError::EmptyCampaignReport);
    }
    if campaign.generations != campaign.autonomous_reports.len() {
        return Err(TasteCampaignReviewError::GenerationCountMismatch);
    }
    if campaign.first_generation == 0
        || campaign.last_generation < campaign.first_generation
        || campaign.autonomous_reports.first().map(|report| report.generation)
            != Some(campaign.first_generation)
        || campaign.autonomous_reports.last().map(|report| report.generation)
            != Some(campaign.last_generation)
    {
        return Err(TasteCampaignReviewError::InvalidGenerationRange);
    }
    for pair in campaign.autonomous_reports.windows(2) {
        if pair[1].generation <= pair[0].generation {
            return Err(TasteCampaignReviewError::NonMonotonicGeneration {
                previous: pair[0].generation,
                current: pair[1].generation,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taste_autonomy::AutonomousTasteReport;
    use crate::taste_benchmark::TasteBenchmarkReport;
    use crate::taste_longitudinal::TasteLongitudinalStatus;
    use crate::taste_orchestrator::{OrchestratedTasteOutcome, ScientificTasteCycleReport};

    fn autonomous(generation: u64, regret: u128, regressed_cases: usize) -> AutonomousTasteReport {
        AutonomousTasteReport {
            generation,
            quarantined_model_observations: 1,
            runtime_observations: 0,
            inconclusive_observations: 0,
            accepted_validations: 2,
            cycle: ScientificTasteCycleReport {
                candidates_scanned: 1,
                validations_scanned: 2,
                outcomes: vec![OrchestratedTasteOutcome {
                    preference_id: 7,
                    state: OrchestratedTasteState::Active,
                    confidence_bps: 8_000,
                    confirmations: 2,
                    validations: 2,
                    can_activate: true,
                }],
            },
            benchmark: TasteBenchmarkReport {
                cases: 1,
                baseline_correct: 0,
                taste_correct: 1,
                baseline_abstentions: 0,
                taste_abstentions: 0,
                baseline_total_regret: 10,
                taste_total_regret: regret,
                improved_cases: usize::from(regret < 10),
                regressed_cases,
                unchanged_cases: 0,
            },
        }
    }

    fn campaign(drift: TasteDriftState) -> TasteCampaignReport {
        TasteCampaignReport {
            first_generation: 1,
            last_generation: 2,
            generations: 2,
            autonomous_reports: vec![autonomous(1, 0, 0), autonomous(2, 0, 0)],
            longitudinal_statuses: vec![TasteLongitudinalStatus {
                preference_id: 7,
                first_generation: 1,
                last_generation: 2,
                observations: 2,
                latest_confidence_bps: 8_000,
                drift,
                eligible_for_supersession: false,
            }],
        }
    }

    #[test]
    fn stable_non_regressive_campaign_is_review_eligible() {
        let report = review_scientific_taste_campaign(
            &campaign(TasteDriftState::Stable),
            TasteCampaignReviewPolicy::default(),
        )
        .expect("review");
        assert_eq!(
            report.disposition,
            TasteCampaignReviewDisposition::EligibleForControlledRestartReview
        );
        assert_eq!(report.authority, TasteCampaignReviewAuthority::ReviewOnly);
        assert_eq!(report.quarantined_model_observations, 2);
    }

    #[test]
    fn contradiction_forces_hold() {
        let report = review_scientific_taste_campaign(
            &campaign(TasteDriftState::Contradicted),
            TasteCampaignReviewPolicy::default(),
        )
        .expect("review");
        assert_eq!(report.disposition, TasteCampaignReviewDisposition::Hold);
        assert_eq!(
            report.preferences[0].blockers,
            vec![TasteCampaignReviewBlocker::Contradicted]
        );
    }

    #[test]
    fn benchmark_regression_forces_hold() {
        let mut input = campaign(TasteDriftState::Stable);
        input.autonomous_reports[1] = autonomous(2, 20, 1);
        let report = review_scientific_taste_campaign(
            &input,
            TasteCampaignReviewPolicy::default(),
        )
        .expect("review");
        assert_eq!(report.disposition, TasteCampaignReviewDisposition::Hold);
        assert_eq!(report.regressed_cases, 1);
    }

    #[test]
    fn malformed_generation_sequence_fails_closed() {
        let mut input = campaign(TasteDriftState::Stable);
        input.autonomous_reports[1].generation = 1;
        assert_eq!(
            review_scientific_taste_campaign(&input, TasteCampaignReviewPolicy::default()),
            Err(TasteCampaignReviewError::InvalidGenerationRange)
        );
    }
}
