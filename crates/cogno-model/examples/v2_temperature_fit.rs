//! A bounded, shared-across-seeds temperature is selected using validation only.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
const GRID: [f64; 9] = [1.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0, 8.0];
fn calibrate(p: f64, t: f64) -> f64 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    let logit = (p / (1.0 - p)).ln() / t;
    1.0 / (1.0 + (-logit).exp())
}
fn fit(rows: &[Observation]) -> Result<f64, String> {
    let validation: Vec<_> = rows.iter().filter(|r| r.split == "validation").collect();
    if validation.is_empty() {
        return Err("validation required".into());
    }
    let mut best = (f64::INFINITY, 1.0);
    for t in GRID {
        let score = mean(
            validation
                .iter()
                .map(|r| nll(r.target, calibrate(r.probability, t))),
        );
        if score < best.0 {
            best = (score, t);
        }
    }
    Ok(best.1)
}
fn main() -> Result<(), String> {
    let e = load(&std::env::args().skip(1).collect::<Vec<_>>())?;
    let t = fit(&e.rows)?;
    header(&e);
    println!("# validation_nll_selection; shared_temperature={t}; fixed_grid=1|0.25|0.5|0.75|1.5|2|3|4|8; exact_ties_grid_order");
    println!(
        "split,unique_sources,seeds,temperature,raw_nll,calibrated_nll,raw_brier,calibrated_brier"
    );
    for split in ["train", "validation", "test"] {
        let rows: Vec<_> = e.rows.iter().filter(|r| r.split == split).collect();
        println!(
            "{split},{},{},{t},{:.12},{:.12},{:.12},{:.12}",
            rows.len() / e.seeds.len(),
            e.seeds.len(),
            mean(rows.iter().map(|r| nll(r.target, r.probability))),
            mean(
                rows.iter()
                    .map(|r| nll(r.target, calibrate(r.probability, t)))
            ),
            mean(
                rows.iter()
                    .map(|r| (r.probability - r.target as f64).powi(2))
            ),
            mean(
                rows.iter()
                    .map(|r| (calibrate(r.probability, t) - r.target as f64).powi(2))
            )
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn observation(split: &str, target: usize) -> Observation {
        Observation {
            seed: 1,
            source: "s".into(),
            split: split.into(),
            group: "g".into(),
            target,
            prediction: 1,
            probability: 0.99,
            bytes: 20,
        }
    }
    #[test]
    fn fitting_cannot_use_test_labels() {
        let mut rows = vec![observation("validation", 0), observation("test", 1)];
        let t = fit(&rows).unwrap();
        rows[1].target = 0;
        rows[1].probability = 0.001;
        assert_eq!(fit(&rows).unwrap(), t);
        assert_eq!(t, 8.0);
    }
    #[test]
    fn identity_symmetry_and_bounded_extremes() {
        for p in [0.01, 0.5, 0.9] {
            assert!((calibrate(p, 1.0) - p).abs() < 1e-12);
            assert!((calibrate(p, 2.0) + calibrate(1.0 - p, 2.0) - 1.0).abs() < 1e-12);
        }
        for p in [0.0, 1.0] {
            assert!(calibrate(p, 0.25).is_finite());
        }
    }
}
