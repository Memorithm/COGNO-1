//! Frozen binary calibration; fixed probability bins, never changes the checkpoint.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;

fn observations(args: &[String]) -> Result<Vec<(usize, f64)>, String> {
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
            let p = model.classify(&r.source).map_err(|e| format!("{e:?}"))?;
            if p.len() != 2
                || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || (p[0] + p[1] - 1.0).abs() > 1e-5
            {
                return Err("invalid binary probabilities".into());
            }
            Ok((r.label, f64::from(p[1])))
        })
        .collect()
}
fn validate(rows: &[(usize, f64)]) -> Result<(), String> {
    if rows.is_empty() {
        return Err("empty observations".into());
    }
    if rows
        .iter()
        .any(|&(y, p)| y > 1 || !p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        return Err("invalid binary observation".into());
    }
    Ok(())
}

const BINS: usize = 10;
const CLIP: f64 = 1e-7;
#[derive(Debug, Default, Clone, Copy)]
struct Bin {
    count: usize,
    probability: f64,
    positives: usize,
}
fn calibration(rows: &[(usize, f64)]) -> Result<(f64, f64, f64, [Bin; BINS]), String> {
    validate(rows)?;
    let mut bins = [Bin::default(); BINS];
    let (mut nll, mut brier) = (0.0, 0.0);
    for &(y, p) in rows {
        let q = p.clamp(CLIP, 1.0 - CLIP);
        nll -= if y == 1 { q.ln() } else { (1.0 - q).ln() };
        brier += (p - y as f64).powi(2);
        let bin = &mut bins[((p * BINS as f64).floor() as usize).min(BINS - 1)];
        bin.count += 1;
        bin.probability += p;
        bin.positives += y;
    }
    let n = rows.len() as f64;
    let ece = bins
        .iter()
        .map(|b| (b.probability - b.positives as f64).abs())
        .sum::<f64>()
        / n;
    Ok((nll / n, brier / n, ece, bins))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_calibration_report CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA train|validation|test".into());
    }
    let rows = observations(&a)?;
    let (nll, brier, ece, bins) = calibration(&rows)?;
    println!("checkpoint_sha256,corpus_sha256,split,count,nll_clip_1e_7,brier,positive_probability_ece_10");
    println!(
        "{},{},{},{},{nll:.12},{brier:.12},{ece:.12}",
        a[2],
        a[4],
        a[5],
        rows.len()
    );
    println!(
        "bin,left_inclusive,right_exclusive_except_last,count,mean_p_compile,positive_fraction"
    );
    for (i, b) in bins.iter().enumerate() {
        let values = if b.count == 0 {
            "NA,NA".into()
        } else {
            format!(
                "{:.12},{:.12}",
                b.probability / b.count as f64,
                b.positives as f64 / b.count as f64
            )
        };
        println!(
            "{i},{:.1},{:.1},{},{values}",
            i as f64 / 10.0,
            (i + 1) as f64 / 10.0,
            b.count
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibrated_fixture() {
        let (n, b, e, _) = calibration(&[(0, 0.25), (0, 0.25), (0, 0.25), (1, 0.25)]).unwrap();
        assert!((n - (-0.75_f64 * 0.75_f64.ln() - 0.25_f64 * 0.25_f64.ln())).abs() < 1e-12);
        assert!((b - 0.1875).abs() < 1e-12);
        assert_eq!(e, 0.0);
    }
    #[test]
    fn boundaries_and_empty_bins() {
        let (n, b, e, bins) = calibration(&[(0, 0.0), (1, 1.0), (1, 0.1)]).unwrap();
        assert!(n.is_finite());
        assert!((b - 0.27).abs() < 1e-12);
        assert!((e - 0.3).abs() < 1e-12);
        assert_eq!(bins[0].count, 1);
        assert_eq!(bins[1].count, 1);
        assert_eq!(bins[9].count, 1);
        assert_eq!(bins[5].count, 0);
    }
    #[test]
    fn invalid_inputs_refused() {
        for rows in [
            vec![],
            vec![(2, 0.5)],
            vec![(1, f64::NAN)],
            vec![(0, f64::INFINITY)],
            vec![(0, -0.1)],
            vec![(0, 1.1)],
        ] {
            assert!(calibration(&rows).is_err());
        }
    }
    #[test]
    fn wrong_certain_predictions_are_finite_and_penalized() {
        let (n, b, e, _) = calibration(&[(1, 0.0), (0, 1.0)]).unwrap();
        assert!(n > 16.0 && n < 17.0);
        assert_eq!(b, 1.0);
        assert_eq!(e, 1.0);
    }
}
