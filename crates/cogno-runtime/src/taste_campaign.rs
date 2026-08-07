//! Bounded post-adversarial scientific-taste campaign coordination.
//!
//! A campaign evaluates multiple autonomous generations as one deterministic
//! transaction over a cloned longitudinal tracker. If any generation fails,
//! no partial longitudinal state is committed. The campaign is report-only: it
//! cannot write, replace, or install a verified runtime taste profile.

use crate::scientific_exchange::ScientificExchangeRecord;
use crate::taste_autonomy::{
    run_autonomous_scientific_taste, AutonomousTasteError, AutonomousTasteReport,
};
use crate::taste_benchmark::TasteBenchmarkCase;
use crate::taste_longitudinal::{TasteLongitudinalStatus, TasteLongitudinalTracker};
use crate::taste_orchestrator::OrchestratedTasteCandidate;
use std::collections::BTreeSet;

/// Maximum number of autonomous generations evaluated by one campaign.
pub const MAX_TASTE_CAMPAIGN_GENERATIONS: usize = 64;

/// Borrowed inputs for one campaign generation.
#[derive(Clone, Copy, Debug)]
pub struct TasteCampaignGeneration<'a> {
    pub generation: u64,
    pub candidates: &'a [OrchestratedTasteCandidate],
    pub exchange_records: &'a [ScientificExchangeRecord],
    pub benchmark_cases: &'a [TasteBenchmarkCase],
}

/// Deterministic report produced after all campaign generations succeed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteCampaignReport {
    pub first_generation: u64,
    pub last_generation: u64,
    pub generations: usize,
    pub autonomous_reports: Vec<AutonomousTasteReport>,
    pub longitudinal_statuses: Vec<TasteLongitudinalStatus>,
}

/// Failure while evaluating a bounded campaign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteCampaignError {
    EmptyCampaign,
    TooManyGenerations,
    NonMonotonicGeneration {
        previous: u64,
        current: u64,
    },
    Autonomous(AutonomousTasteError),
}

/// Evaluate a bounded sequence of autonomous scientific-taste generations.
///
/// The supplied longitudinal tracker is updated only after every generation
/// succeeds. This gives campaign-level fail-closed behavior without changing
/// the runtime profile authority boundary.
pub fn run_scientific_taste_campaign(
    generations: &[TasteCampaignGeneration<'_>],
    longitudinal: &mut TasteLongitudinalTracker,
) -> Result<TasteCampaignReport, TasteCampaignError> {
    let first = generations
        .first()
        .ok_or(TasteCampaignError::EmptyCampaign)?;
    if generations.len() > MAX_TASTE_CAMPAIGN_GENERATIONS {
        return Err(TasteCampaignError::TooManyGenerations);
    }

    for pair in generations.windows(2) {
        if pair[1].generation <= pair[0].generation {
            return Err(TasteCampaignError::NonMonotonicGeneration {
                previous: pair[0].generation,
                current: pair[1].generation,
            });
        }
    }

    let mut working_longitudinal = longitudinal.clone();
    let mut autonomous_reports = Vec::with_capacity(generations.len());
    let mut preference_ids = BTreeSet::new();

    for generation in generations {
        for candidate in generation.candidates {
            preference_ids.insert(candidate.preference_id);
        }
        let report = run_autonomous_scientific_taste(
            generation.generation,
            generation.candidates,
            generation.exchange_records,
            generation.benchmark_cases,
            &mut working_longitudinal,
        )
        .map_err(TasteCampaignError::Autonomous)?;
        autonomous_reports.push(report);
    }

    let last_generation = generations
        .last()
        .expect("non-empty campaign checked above")
        .generation;
    let longitudinal_statuses = preference_ids
        .into_iter()
        .filter_map(|preference_id| working_longitudinal.status(preference_id, last_generation))
        .collect();

    *longitudinal = working_longitudinal;

    Ok(TasteCampaignReport {
        first_generation: first.generation,
        last_generation,
        generations: generations.len(),
        autonomous_reports,
        longitudinal_statuses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_exchange::{ScientificExchangeOrigin, ScientificExchangeVerdict};
    use crate::taste_longitudinal::TasteDriftState;

    fn exchange(
        observation_id: u64,
        preference_id: u64,
        origin: ScientificExchangeOrigin,
        confidence_bps: u16,
    ) -> ScientificExchangeRecord {
        ScientificExchangeRecord {
            observation_id,
            preference_id,
            evidence_id: observation_id + 1_000,
            origin,
            verdict: ScientificExchangeVerdict::Confirmed,
            confidence_bps,
            canonical_payload: format!("campaign-{observation_id}").into_bytes(),
        }
    }

    #[test]
    fn campaign_commits_only_after_all_generations_succeed() {
        let candidate = [OrchestratedTasteCandidate { preference_id: 7 }];
        let first_records = [
            exchange(
                1,
                7,
                ScientificExchangeOrigin::SciRustDeterministicEvaluator,
                8_000,
            ),
            exchange(
                2,
                7,
                ScientificExchangeOrigin::ExplicitUserAction,
                8_000,
            ),
        ];
        let second_records = [
            exchange(
                3,
                7,
                ScientificExchangeOrigin::SciRustDeterministicEvaluator,
                9_000,
            ),
            exchange(
                4,
                7,
                ScientificExchangeOrigin::ExplicitUserAction,
                9_000,
            ),
        ];
        let generations = [
            TasteCampaignGeneration {
                generation: 1,
                candidates: &candidate,
                exchange_records: &first_records,
                benchmark_cases: &[],
            },
            TasteCampaignGeneration {
                generation: 2,
                candidates: &candidate,
                exchange_records: &second_records,
                benchmark_cases: &[],
            },
        ];
        let mut tracker = TasteLongitudinalTracker::default();

        let report = run_scientific_taste_campaign(&generations, &mut tracker).expect("campaign");

        assert_eq!(report.generations, 2);
        assert_eq!(report.first_generation, 1);
        assert_eq!(report.last_generation, 2);
        assert_eq!(report.autonomous_reports.len(), 2);
        assert_eq!(report.longitudinal_statuses.len(), 1);
        assert_eq!(report.longitudinal_statuses[0].drift, TasteDriftState::Strengthening);
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn campaign_error_leaves_longitudinal_tracker_unchanged() {
        let candidate = [OrchestratedTasteCandidate { preference_id: 7 }];
        let valid_records = [
            exchange(
                10,
                7,
                ScientificExchangeOrigin::SciRustDeterministicEvaluator,
                8_000,
            ),
            exchange(
                11,
                7,
                ScientificExchangeOrigin::ExplicitUserAction,
                8_000,
            ),
        ];
        let invalid_records = [exchange(
            12,
            7,
            ScientificExchangeOrigin::SciRustDeterministicEvaluator,
            10_001,
        )];
        let generations = [
            TasteCampaignGeneration {
                generation: 1,
                candidates: &candidate,
                exchange_records: &valid_records,
                benchmark_cases: &[],
            },
            TasteCampaignGeneration {
                generation: 2,
                candidates: &candidate,
                exchange_records: &invalid_records,
                benchmark_cases: &[],
            },
        ];
        let mut tracker = TasteLongitudinalTracker::default();
        let before = tracker.clone();

        assert!(run_scientific_taste_campaign(&generations, &mut tracker).is_err());
        assert_eq!(tracker, before);
    }

    #[test]
    fn campaign_rejects_non_monotonic_generations_before_mutation() {
        let candidate = [OrchestratedTasteCandidate { preference_id: 7 }];
        let generations = [
            TasteCampaignGeneration {
                generation: 2,
                candidates: &candidate,
                exchange_records: &[],
                benchmark_cases: &[],
            },
            TasteCampaignGeneration {
                generation: 2,
                candidates: &candidate,
                exchange_records: &[],
                benchmark_cases: &[],
            },
        ];
        let mut tracker = TasteLongitudinalTracker::default();

        assert_eq!(
            run_scientific_taste_campaign(&generations, &mut tracker),
            Err(TasteCampaignError::NonMonotonicGeneration {
                previous: 2,
                current: 2,
            })
        );
        assert!(tracker.is_empty());
    }
}
