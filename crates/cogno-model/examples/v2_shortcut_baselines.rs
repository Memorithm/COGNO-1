//! Fixed source-length bins expose simple shortcuts, fitted on unique train sources.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::BTreeMap;
fn bin(bytes: usize) -> usize {
    match bytes {
        0..=63 => 0,
        64..=127 => 1,
        128..=255 => 2,
        256..=511 => 3,
        _ => 4,
    }
}
#[derive(Debug, PartialEq)]
struct Baselines {
    prevalence: f64,
    bins: [f64; 5],
    train_sources: usize,
}
fn fit(rows: &[Observation]) -> Result<Baselines, String> {
    let unique: BTreeMap<_, _> = rows
        .iter()
        .filter(|r| r.split == "train")
        .map(|r| (&r.source, r))
        .collect();
    if unique.is_empty() {
        return Err("training sources required".into());
    }
    let mut counts = [[0usize; 2]; 5];
    let mut labels = [0usize; 2];
    for r in unique.values() {
        labels[r.target] += 1;
        counts[bin(r.bytes)][r.target] += 1;
    }
    let prevalence = (labels[1] + 1) as f64 / (unique.len() + 2) as f64;
    // Empty bins back off to training prevalence, not a holdout-derived choice.
    let bins = counts.map(|c| {
        if c[0] + c[1] == 0 {
            prevalence
        } else {
            (c[1] + 1) as f64 / (c[0] + c[1] + 2) as f64
        }
    });
    Ok(Baselines {
        prevalence,
        bins,
        train_sources: unique.len(),
    })
}
fn main() -> Result<(), String> {
    let e = load(&std::env::args().skip(1).collect::<Vec<_>>())?;
    let baseline = fit(&e.rows)?;
    header(&e);
    println!("# fitted_unique_train_sources={}; fixed_byte_bins=0..63|64..127|128..255|256..511|512+; laplace_alpha=1; ties_predict_zero",baseline.train_sources);
    println!("split,method,unique_sources,seeds,accuracy,nll,brier");
    for split in ["train", "validation", "test"] {
        let rows: Vec<_> = e.rows.iter().filter(|r| r.split == split).collect();
        for method in ["uniform", "train_prevalence", "train_length_bins", "model"] {
            let p = |r: &Observation| match method {
                "uniform" => 0.5,
                "train_prevalence" => baseline.prevalence,
                "train_length_bins" => baseline.bins[bin(r.bytes)],
                _ => r.probability,
            };
            let predicted = |r: &Observation| {
                if method == "model" {
                    r.prediction
                } else {
                    usize::from(p(r) > 0.5)
                }
            };
            println!(
                "{split},{method},{},{},{:.12},{:.12},{:.12}",
                rows.len() / e.seeds.len(),
                e.seeds.len(),
                mean(rows.iter().map(|r| f64::from(predicted(r) == r.target))),
                mean(rows.iter().map(|r| nll(r.target, p(r)))),
                mean(rows.iter().map(|r| (p(r) - r.target as f64).powi(2)))
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(source: &str, split: &str, target: usize, bytes: usize) -> Observation {
        Observation {
            seed: 1,
            source: source.into(),
            split: split.into(),
            group: "g".into(),
            target,
            prediction: target,
            probability: 0.5,
            bytes,
        }
    }
    #[test]
    fn baseline_fit_ignores_holdout_and_seed_duplicates() {
        let mut rows = vec![
            row("a", "train", 1, 10),
            row("b", "train", 0, 100),
            row("c", "test", 1, 10),
        ];
        let b = fit(&rows).unwrap();
        rows[2].target = 0;
        rows[2].bytes = 1000;
        rows.push(rows[0].clone());
        assert_eq!(fit(&rows).unwrap(), b);
        assert_eq!(b.prevalence, 0.5);
        assert_eq!(b.bins[4], 0.5);
        assert_eq!(b.train_sources, 2);
    }
    #[test]
    fn bin_boundaries_are_fixed() {
        assert_eq!(
            [
                bin(63),
                bin(64),
                bin(127),
                bin(128),
                bin(255),
                bin(256),
                bin(511),
                bin(512)
            ],
            [0, 1, 1, 2, 2, 3, 3, 4]
        );
    }
}
