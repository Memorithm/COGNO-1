//! Source-bound diagnostic rows retain all seeds and avoid best-seed reporting.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::BTreeMap;
#[derive(Debug)]
struct Summary<'a> {
    first: &'a Observation,
    seeds: usize,
    errors: usize,
    disagreement: bool,
    mean_nll: f64,
    mean_probability: f64,
    min_probability: f64,
    max_probability: f64,
}
fn summarize<'a>(rows: &[&'a Observation]) -> Summary<'a> {
    let first = rows[0];
    let errors = rows.iter().filter(|r| r.prediction != r.target).count();
    Summary {
        first,
        seeds: rows.len(),
        errors,
        disagreement: rows.iter().any(|r| r.prediction != first.prediction),
        mean_nll: mean(rows.iter().map(|r| nll(r.target, r.probability))),
        mean_probability: mean(rows.iter().map(|r| r.probability)),
        min_probability: rows
            .iter()
            .map(|r| r.probability)
            .fold(f64::INFINITY, f64::min),
        max_probability: rows
            .iter()
            .map(|r| r.probability)
            .fold(f64::NEG_INFINITY, f64::max),
    }
}
fn main() -> Result<(), String> {
    let e = load(&std::env::args().skip(1).collect::<Vec<_>>())?;
    header(&e);
    println!("# diagnostic_only; test_errors_must_not_be_reused_as_fresh_holdout; sorted_by_errors_then_nll_then_hash");
    let mut sources: BTreeMap<&str, Vec<&Observation>> = BTreeMap::new();
    for r in &e.rows {
        sources.entry(&r.source).or_default().push(r);
    }
    let mut summaries: Vec<_> = sources.values().map(|r| summarize(r)).collect();
    summaries.sort_by(|a, b| {
        b.errors
            .cmp(&a.errors)
            .then_with(|| b.mean_nll.total_cmp(&a.mean_nll))
            .then_with(|| a.first.source.cmp(&b.first.source))
    });
    println!("split,group,source_sha256,target,source_bytes,seeds,errors,hard_prediction_disagreement,mean_nll,mean_p_compile,min_p_compile,max_p_compile");
    for s in summaries {
        println!(
            "{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12}",
            s.first.split,
            s.first.group,
            s.first.source,
            s.first.target,
            s.first.bytes,
            s.seeds,
            s.errors,
            s.disagreement,
            s.mean_nll,
            s.mean_probability,
            s.min_probability,
            s.max_probability
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_failures_and_disagreement() {
        let a = Observation {
            seed: 1,
            source: "s".into(),
            split: "test".into(),
            group: "g".into(),
            target: 1,
            prediction: 0,
            probability: 0.1,
            bytes: 20,
        };
        let mut b = a.clone();
        b.seed = 7;
        b.prediction = 1;
        b.probability = 0.8;
        let s = summarize(&[&a, &b]);
        assert_eq!(s.errors, 1);
        assert!(s.disagreement);
        assert_eq!(s.seeds, 2);
        assert_eq!(s.min_probability, 0.1);
        assert_eq!(s.max_probability, 0.8);
        assert!((s.mean_probability - 0.45).abs() < 1e-12);
    }
}
