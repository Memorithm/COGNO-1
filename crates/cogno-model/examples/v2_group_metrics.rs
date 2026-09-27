//! Equal-group summaries keep seed repetition separate from source counts.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug)]
struct Metrics {
    accuracy: f64,
    nll: f64,
    brier: f64,
}
fn metrics(rows: &[&Observation]) -> Metrics {
    Metrics {
        accuracy: mean(rows.iter().map(|r| f64::from(r.prediction == r.target))),
        nll: mean(rows.iter().map(|r| nll(r.target, r.probability))),
        brier: mean(
            rows.iter()
                .map(|r| (r.probability - r.target as f64).powi(2)),
        ),
    }
}
fn main() -> Result<(), String> {
    let e = load(&std::env::args().skip(1).collect::<Vec<_>>())?;
    header(&e);
    println!("split,scope,group,unique_sources,seeds,accuracy,nll,brier");
    for split in ["train", "validation", "test"] {
        let mut groups: BTreeMap<&str, Vec<&Observation>> = BTreeMap::new();
        for r in e.rows.iter().filter(|r| r.split == split) {
            groups.entry(&r.group).or_default().push(r);
        }
        let mut summaries = Vec::new();
        for (group, rows) in &groups {
            let m = metrics(rows);
            summaries.push(m);
            println!(
                "{split},group,{group},{},{},{:.12},{:.12},{:.12}",
                rows.len() / e.seeds.len(),
                e.seeds.len(),
                m.accuracy,
                m.nll,
                m.brier
            );
        }
        println!(
            "{split},equal_group,ALL,{},{},{:.12},{:.12},{:.12}",
            groups.values().map(|r| r.len()).sum::<usize>() / e.seeds.len(),
            e.seeds.len(),
            mean(summaries.iter().map(|m| m.accuracy)),
            mean(summaries.iter().map(|m| m.nll)),
            mean(summaries.iter().map(|m| m.brier))
        );
        let all: Vec<_> = e.rows.iter().filter(|r| r.split == split).collect();
        let m = metrics(&all);
        println!(
            "{split},source_weighted,ALL,{},{},{:.12},{:.12},{:.12}",
            all.len() / e.seeds.len(),
            e.seeds.len(),
            m.accuracy,
            m.nll,
            m.brier
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seed_replication_does_not_change_metrics() {
        let r = Observation {
            seed: 1,
            source: "s".into(),
            split: "test".into(),
            group: "g".into(),
            target: 1,
            prediction: 0,
            probability: 0.25,
            bytes: 20,
        };
        let a = metrics(&[&r]);
        let b = metrics(&[&r, &r, &r]);
        assert_eq!(a.accuracy, b.accuracy);
        assert_eq!(a.brier, b.brier);
        assert!((a.nll - b.nll).abs() < 1e-12);
    }
}
