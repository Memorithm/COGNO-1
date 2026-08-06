//! Deterministic benchmark metrics for scientific-taste decisions.
//!
//! The benchmark is intentionally model-free. Callers provide stable decision
//! outcomes and an oracle identifier derived from deterministic evaluation.
//! Metrics are integer-only so repeated runs are bit-for-bit reproducible.

/// Maximum benchmark cases accepted in one report.
pub const MAX_BENCHMARK_CASES: usize = 65_536;

/// One evaluated decision case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteBenchmarkCase {
    pub case_id: u64,
    pub baseline_candidate_id: Option<u64>,
    pub taste_candidate_id: Option<u64>,
    pub oracle_candidate_id: Option<u64>,
    pub baseline_score: i64,
    pub taste_score: i64,
    pub oracle_score: i64,
}

/// Aggregate deterministic benchmark report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TasteBenchmarkReport {
    pub cases: usize,
    pub baseline_correct: usize,
    pub taste_correct: usize,
    pub baseline_abstentions: usize,
    pub taste_abstentions: usize,
    pub baseline_total_regret: u128,
    pub taste_total_regret: u128,
    pub improved_cases: usize,
    pub regressed_cases: usize,
    pub unchanged_cases: usize,
}

/// Invalid benchmark corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteBenchmarkError {
    TooManyCases,
    DuplicateCase(u64),
}

/// Evaluate deterministic quality and regret for a benchmark corpus.
pub fn evaluate_taste_benchmark(
    cases: &[TasteBenchmarkCase],
) -> Result<TasteBenchmarkReport, TasteBenchmarkError> {
    if cases.len() > MAX_BENCHMARK_CASES {
        return Err(TasteBenchmarkError::TooManyCases);
    }

    let mut sorted = cases.to_vec();
    sorted.sort_by_key(|case| case.case_id);
    for pair in sorted.windows(2) {
        if pair[0].case_id == pair[1].case_id {
            return Err(TasteBenchmarkError::DuplicateCase(pair[0].case_id));
        }
    }

    let mut report = TasteBenchmarkReport {
        cases: sorted.len(),
        ..TasteBenchmarkReport::default()
    };

    for case in sorted {
        if case.baseline_candidate_id == case.oracle_candidate_id {
            report.baseline_correct += 1;
        }
        if case.taste_candidate_id == case.oracle_candidate_id {
            report.taste_correct += 1;
        }
        if case.baseline_candidate_id.is_none() {
            report.baseline_abstentions += 1;
        }
        if case.taste_candidate_id.is_none() {
            report.taste_abstentions += 1;
        }

        let baseline_regret = regret(case.oracle_score, case.baseline_score);
        let taste_regret = regret(case.oracle_score, case.taste_score);
        report.baseline_total_regret = report.baseline_total_regret.saturating_add(baseline_regret);
        report.taste_total_regret = report.taste_total_regret.saturating_add(taste_regret);

        match taste_regret.cmp(&baseline_regret) {
            std::cmp::Ordering::Less => report.improved_cases += 1,
            std::cmp::Ordering::Greater => report.regressed_cases += 1,
            std::cmp::Ordering::Equal => report.unchanged_cases += 1,
        }
    }

    Ok(report)
}

fn regret(oracle_score: i64, observed_score: i64) -> u128 {
    if observed_score >= oracle_score {
        0
    } else {
        u128::from(oracle_score.abs_diff(observed_score))
    }
}

impl TasteBenchmarkReport {
    /// Signed change in exact-oracle selections.
    #[must_use]
    pub fn correctness_delta(&self) -> i64 {
        i64::try_from(self.taste_correct).unwrap_or(i64::MAX)
            - i64::try_from(self.baseline_correct).unwrap_or(i64::MAX)
    }

    /// True only when taste does not worsen aggregate regret.
    #[must_use]
    pub const fn non_regressive(&self) -> bool {
        self.taste_total_regret <= self.baseline_total_regret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_reports_improvement_without_floats() {
        let report = evaluate_taste_benchmark(&[
            TasteBenchmarkCase {
                case_id: 2,
                baseline_candidate_id: Some(1),
                taste_candidate_id: Some(2),
                oracle_candidate_id: Some(2),
                baseline_score: 70,
                taste_score: 100,
                oracle_score: 100,
            },
            TasteBenchmarkCase {
                case_id: 1,
                baseline_candidate_id: Some(3),
                taste_candidate_id: Some(3),
                oracle_candidate_id: Some(3),
                baseline_score: 50,
                taste_score: 50,
                oracle_score: 50,
            },
        ])
        .expect("benchmark");
        assert_eq!(report.baseline_correct, 1);
        assert_eq!(report.taste_correct, 2);
        assert_eq!(report.correctness_delta(), 1);
        assert_eq!(report.improved_cases, 1);
        assert!(report.non_regressive());
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let case = TasteBenchmarkCase {
            case_id: 1,
            baseline_candidate_id: None,
            taste_candidate_id: None,
            oracle_candidate_id: None,
            baseline_score: 0,
            taste_score: 0,
            oracle_score: 0,
        };
        assert_eq!(
            evaluate_taste_benchmark(&[case, case]),
            Err(TasteBenchmarkError::DuplicateCase(1))
        );
    }
}
