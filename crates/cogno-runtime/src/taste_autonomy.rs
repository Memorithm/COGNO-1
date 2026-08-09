//! High-level deterministic scientific-taste evaluation pass.
//!
//! This is the coordination layer above the SciAgent/SciRust trust boundary,
//! promotion orchestrator, benchmark metrics, and longitudinal history. It does
//! not write a live runtime profile and therefore cannot bypass transactional
//! generation persistence or restart-time profile sealing.

use crate::scientific_exchange::{
    classify_scientific_exchange, ScientificExchangeDisposition, ScientificExchangeError,
    ScientificExchangeRecord,
};
use crate::taste_benchmark::{
    evaluate_taste_benchmark, TasteBenchmarkCase, TasteBenchmarkError, TasteBenchmarkReport,
};
use crate::taste_longitudinal::{
    TasteHistoryObservation, TasteLongitudinalError, TasteLongitudinalTracker,
};
use crate::taste_orchestrator::{
    orchestrate_scientific_taste_cycle, OrchestratedTasteCandidate, OrchestratedTasteState,
    ScientificTasteCycleReport, ScientificTasteOrchestratorError,
};
use crate::taste_validation_store::StoredTasteValidation;

/// Maximum exchange records handled by one autonomous pass.
pub const MAX_AUTONOMOUS_EXCHANGE_RECORDS: usize = 16_384;

/// Complete deterministic result of one autonomous evaluation pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutonomousTasteReport {
    pub generation: u64,
    pub quarantined_model_observations: usize,
    pub runtime_observations: usize,
    pub inconclusive_observations: usize,
    pub accepted_validations: usize,
    pub cycle: ScientificTasteCycleReport,
    pub benchmark: TasteBenchmarkReport,
}

/// Failure while coordinating an autonomous evaluation pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutonomousTasteError {
    ZeroGeneration,
    TooManyExchangeRecords,
    Exchange(ScientificExchangeError),
    Orchestrator(ScientificTasteOrchestratorError),
    Benchmark(TasteBenchmarkError),
    Longitudinal(TasteLongitudinalError),
}

/// Run a deterministic, model-non-authoritative scientific-taste pass.
pub fn run_autonomous_scientific_taste(
    generation: u64,
    candidates: &[OrchestratedTasteCandidate],
    exchange_records: &[ScientificExchangeRecord],
    benchmark_cases: &[TasteBenchmarkCase],
    longitudinal: &mut TasteLongitudinalTracker,
) -> Result<AutonomousTasteReport, AutonomousTasteError> {
    if generation == 0 {
        return Err(AutonomousTasteError::ZeroGeneration);
    }
    if exchange_records.len() > MAX_AUTONOMOUS_EXCHANGE_RECORDS {
        return Err(AutonomousTasteError::TooManyExchangeRecords);
    }

    let mut quarantined_model_observations = 0usize;
    let mut runtime_observations = 0usize;
    let mut inconclusive_observations = 0usize;
    let mut validations = Vec::<StoredTasteValidation>::new();

    for record in exchange_records {
        match classify_scientific_exchange(record).map_err(AutonomousTasteError::Exchange)? {
            ScientificExchangeDisposition::QuarantinedModelObservation => {
                quarantined_model_observations += 1;
            }
            ScientificExchangeDisposition::RuntimeObservationOnly => runtime_observations += 1,
            ScientificExchangeDisposition::Inconclusive => inconclusive_observations += 1,
            ScientificExchangeDisposition::Validation(validation) => validations.push(validation),
        }
    }

    let cycle = orchestrate_scientific_taste_cycle(candidates, &validations)
        .map_err(AutonomousTasteError::Orchestrator)?;
    let benchmark =
        evaluate_taste_benchmark(benchmark_cases).map_err(AutonomousTasteError::Benchmark)?;

    for outcome in &cycle.outcomes {
        longitudinal
            .append(TasteHistoryObservation {
                generation,
                preference_id: outcome.preference_id,
                confidence_bps: outcome.confidence_bps,
                active: matches!(outcome.state, OrchestratedTasteState::Active),
                contradicted: matches!(outcome.state, OrchestratedTasteState::Conflicted),
            })
            .map_err(AutonomousTasteError::Longitudinal)?;
    }

    Ok(AutonomousTasteReport {
        generation,
        quarantined_model_observations,
        runtime_observations,
        inconclusive_observations,
        accepted_validations: validations.len(),
        cycle,
        benchmark,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_exchange::{ScientificExchangeOrigin, ScientificExchangeVerdict};

    fn exchange(observation_id: u64, origin: ScientificExchangeOrigin) -> ScientificExchangeRecord {
        ScientificExchangeRecord {
            observation_id,
            preference_id: 7,
            evidence_id: observation_id + 100,
            origin,
            verdict: ScientificExchangeVerdict::Confirmed,
            confidence_bps: 8_000,
            canonical_payload: format!("record-{observation_id}").into_bytes(),
        }
    }

    #[test]
    fn autonomous_pass_quarantines_model_and_promotes_only_non_model_evidence() {
        let mut history = TasteLongitudinalTracker::default();
        let report = run_autonomous_scientific_taste(
            1,
            &[OrchestratedTasteCandidate { preference_id: 7 }],
            &[
                exchange(1, ScientificExchangeOrigin::SciAgentModel),
                exchange(2, ScientificExchangeOrigin::SciRustDeterministicEvaluator),
                exchange(3, ScientificExchangeOrigin::ExplicitUserAction),
            ],
            &[],
            &mut history,
        )
        .expect("autonomous pass");
        assert_eq!(report.quarantined_model_observations, 1);
        // The unauthenticated batch path must not promote a caller-supplied
        // SciRust deterministic origin. Only the explicit user action is a
        // validation here; attested runtime evidence uses the dedicated boundary.
        assert_eq!(report.runtime_observations, 1);
        assert_eq!(report.accepted_validations, 1);
        assert_eq!(
            report.cycle.outcomes[0].state,
            OrchestratedTasteState::Active
        );
        assert_eq!(history.len(), 1);
    }
}
