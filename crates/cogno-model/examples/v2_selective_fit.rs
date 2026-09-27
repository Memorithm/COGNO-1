//! Validation-selected abstention is descriptive, never a guaranteed error bound.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::{BTreeMap, BTreeSet};
const GRID: [f64; 11] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];
fn accepts(r: &Observation, t: Option<f64>) -> bool {
    t.is_some_and(|t| r.probability.max(1.0 - r.probability) >= t)
}
fn fit(rows: &[Observation], max_error: f64) -> Option<f64> {
    let mut best = None;
    let mut best_count = 0;
    for t in GRID {
        let selected: Vec<_> = rows
            .iter()
            .filter(|r| r.split == "validation" && accepts(r, Some(t)))
            .collect();
        if !selected.is_empty()
            && selected.len() > best_count
            && mean(selected.iter().map(|r| f64::from(r.prediction != r.target))) <= max_error
        {
            best = Some(t);
            best_count = selected.len();
        }
    }
    best
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 7 {
        return Err(format!("usage: {USAGE} MAX_VALIDATION_EMPIRICAL_ERROR"));
    }
    let max_error = a[6].parse::<f64>().map_err(|e| e.to_string())?;
    if !max_error.is_finite() || !(0.0..=1.0).contains(&max_error) {
        return Err("error bound must be finite in [0,1]".into());
    }
    let e = load(&a[..6])?;
    let threshold = fit(&e.rows, max_error);
    header(&e);
    println!("# threshold={}; max_validation_empirical_error={max_error}; no_test_error_guarantee; unique_accepted_means_at_least_one_seed",threshold.map_or("ABSTAIN_ALL".into(),|t|t.to_string()));
    println!("split,group,unique_sources,observations,accepted_observations,unique_accepted,coverage,accepted_error");
    for split in ["validation", "test"] {
        let mut groups: BTreeMap<&str, Vec<&Observation>> = BTreeMap::new();
        for r in e.rows.iter().filter(|r| r.split == split) {
            groups.entry(&r.group).or_default().push(r);
        }
        for (group, rows) in groups {
            let accepted: Vec<_> = rows
                .iter()
                .copied()
                .filter(|r| accepts(r, threshold))
                .collect();
            let unique: BTreeSet<_> = accepted.iter().map(|r| &r.source).collect();
            let risk = if accepted.is_empty() {
                "NA".into()
            } else {
                format!(
                    "{:.12}",
                    mean(accepted.iter().map(|r| f64::from(r.prediction != r.target)))
                )
            };
            println!(
                "{split},{group},{},{},{},{},{:.12},{risk}",
                rows.len() / e.seeds.len(),
                rows.len(),
                accepted.len(),
                unique.len(),
                accepted.len() as f64 / rows.len() as f64
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(split: &str, target: usize) -> Observation {
        Observation {
            seed: 1,
            source: "s".into(),
            split: split.into(),
            group: "g".into(),
            target,
            prediction: 1,
            probability: 0.9,
            bytes: 2,
        }
    }
    #[test]
    fn test_labels_cannot_select_threshold() {
        let mut rows = vec![row("validation", 1), row("test", 0)];
        assert_eq!(fit(&rows, 0.0), Some(0.5));
        rows[1].target = 1;
        assert_eq!(fit(&rows, 0.0), Some(0.5));
    }
    #[test]
    fn no_admissible_threshold_abstains_instead_of_claiming_zero_error() {
        let rows = vec![row("validation", 0)];
        assert_eq!(fit(&rows, 0.0), None);
        assert!(!accepts(&rows[0], None));
    }
}
