//! Descriptive selective-prediction curve on fixed, label-independent thresholds.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;

fn observations(args: &[String]) -> Result<Vec<(usize, Option<f64>)>, String> {
    let split = match args[5].as_str() {
        "train" => CorpusSplit::Train,
        "validation" => CorpusSplit::Validation,
        "test" => CorpusSplit::Test,
        _ => return Err("unknown split".into()),
    };
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1])
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let model = load_checkpoint(&bytes, parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("{e:?}"))?;
    if model.heads().config().num_classes != 2 {
        return Err("binary classifier required".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[3]).map_err(|e| e.to_string())?,
        parse_hash(&args[4]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    corpus
        .records()
        .iter()
        .filter(|r| r.split == split)
        .map(|r| {
            match model.tokenizer().encode(&r.source) {
                Ok(_) => (),
                Err(cogno_model::bpe_tokenizer::BpeError::Capacity) => return Ok((r.label, None)),
                Err(e) => return Err(format!("{e:?}")),
            }
            let p = model.classify(&r.source).map_err(|e| format!("{e:?}"))?;
            if p.len() != 2
                || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || (p[0] + p[1] - 1.0).abs() > 1e-5
            {
                return Err("invalid binary probabilities".into());
            }
            Ok((r.label, Some(f64::from(p[1]))))
        })
        .collect()
}

// Frozen grid; the CLI deliberately offers no data-dependent threshold selector.
const THRESHOLDS: [f64; 8] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0];
#[derive(Debug, PartialEq)]
struct Point {
    total: usize,
    capacity: usize,
    abstained: usize,
    retained: usize,
    correct: usize,
}
impl Point {
    fn coverage(&self) -> f64 {
        self.retained as f64 / self.total as f64
    }
    fn risk(&self) -> Option<f64> {
        if self.retained == 0 {
            None
        } else {
            Some((self.retained - self.correct) as f64 / self.retained as f64)
        }
    }
}
fn curve(rows: &[(usize, Option<f64>)]) -> Result<Vec<Point>, String> {
    if rows.is_empty() {
        return Err("empty observations".into());
    }
    for &(label, p) in rows {
        if label > 1 || p.is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p)) {
            return Err("invalid binary observation".into());
        }
    }
    Ok(THRESHOLDS
        .iter()
        .map(|&threshold| {
            let mut point = Point {
                total: rows.len(),
                capacity: 0,
                abstained: 0,
                retained: 0,
                correct: 0,
            };
            for &(label, p) in rows {
                match p {
                    None => point.capacity += 1,
                    Some(p) if p.max(1.0 - p) < threshold => point.abstained += 1,
                    Some(p) => {
                        point.retained += 1;
                        point.correct += usize::from(usize::from(p > 0.5) == label);
                    }
                }
            }
            point
        })
        .collect())
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_selective_report CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA train|validation|test".into());
    }
    let rows = observations(&a)?;
    let points = curve(&rows)?;
    println!("checkpoint_sha256,corpus_sha256,split,threshold,total,capacity_refused,confidence_abstained,retained,correct,coverage_of_total,risk_on_retained,accuracy_on_retained");
    for (threshold, p) in THRESHOLDS.iter().zip(points) {
        let scores = p
            .risk()
            .map(|r| format!("{r:.12},{:.12}", 1.0 - r))
            .unwrap_or_else(|| "NA,NA".into());
        println!(
            "{},{},{},{threshold:.2},{},{},{},{},{},{:.12},{scores}",
            a[2],
            a[4],
            a[5],
            p.total,
            p.capacity,
            p.abstained,
            p.retained,
            p.correct,
            p.coverage()
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refusal_and_abstention_denominators() {
        let points = curve(&[(1, Some(0.9)), (0, Some(0.7)), (0, None), (0, Some(0.5))]).unwrap();
        assert_eq!(
            points[0],
            Point {
                total: 4,
                capacity: 1,
                abstained: 0,
                retained: 3,
                correct: 2
            }
        );
        assert_eq!(points[0].coverage(), 0.75);
        assert_eq!(points[0].risk(), Some(1.0 / 3.0));
        assert_eq!(
            points[4],
            Point {
                total: 4,
                capacity: 1,
                abstained: 2,
                retained: 1,
                correct: 1
            }
        );
        assert_eq!(points[4].risk(), Some(0.0));
        assert_eq!(points[4].coverage(), 0.25);
    }
    #[test]
    fn no_retained_is_undefined_risk_not_perfect_score() {
        let points = curve(&[(0, None), (1, Some(0.51))]).unwrap();
        assert_eq!(points[7].risk(), None);
        assert_eq!(points[7].coverage(), 0.0);
        let all_capacity = curve(&[(1, None)]).unwrap();
        assert!(all_capacity.iter().all(|p| p.risk().is_none()));
    }
    #[test]
    fn coverage_is_label_independent_and_monotone() {
        let original = [(0, Some(0.1)), (1, Some(0.6)), (1, Some(0.99)), (0, None)];
        let reversed: Vec<_> = original.iter().map(|&(y, p)| (1 - y, p)).collect();
        let a = curve(&original).unwrap();
        let b = curve(&reversed).unwrap();
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.retained, y.retained);
            assert_eq!(x.abstained, y.abstained);
            assert_eq!(x.capacity, y.capacity);
        }
        assert!(a.windows(2).all(|p| p[1].retained <= p[0].retained));
    }
    #[test]
    fn invalid_inputs_and_exact_certainty() {
        for r in [
            vec![],
            vec![(2, None)],
            vec![(0, Some(f64::NAN))],
            vec![(1, Some(-0.1))],
            vec![(0, Some(f64::INFINITY))],
        ] {
            assert!(curve(&r).is_err());
        }
        let p = curve(&[(1, Some(1.0)), (1, Some(0.0))]).unwrap();
        assert_eq!(p[7].retained, 2);
        assert_eq!(p[7].risk(), Some(0.5));
    }
}
