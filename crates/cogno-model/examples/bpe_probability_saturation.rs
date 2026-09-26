#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    bpe_cognitive::BpeCognitiveModel,
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;
fn model(path: &str, hash: &str) -> Result<BpeCognitiveModel, String> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut b)
        .map_err(|e| e.to_string())?;
    let m = load_checkpoint(&b, parse_hash(hash).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("{e:?}"))?;
    if m.heads().config().num_classes != 2 {
        return Err("binary classifier required".into());
    }
    Ok(m)
}
fn corpus(path: &str, hash: &str, split: &str) -> Result<(RustCorpus, CorpusSplit), String> {
    let h = parse_hash(hash).map_err(|e| format!("{e:?}"))?;
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let s = match split {
        "train" => CorpusSplit::Train,
        "validation" => CorpusSplit::Validation,
        "test" | "external" => CorpusSplit::Test,
        _ => return Err("split: train|validation|test|external".into()),
    };
    let c = if split == "external" {
        RustCorpus::read_test_only(f, h)
    } else {
        RustCorpus::read(f, h)
    }
    .map_err(|e| format!("{e:?}"))?;
    Ok((c, s))
}
fn probability(m: &BpeCognitiveModel, source: &[u8]) -> Result<f64, String> {
    let p = m.classify(source).map_err(|e| format!("{e:?}"))?;
    if p.len() != 2
        || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || (p[0] + p[1] - 1.0).abs() > 1e-5
    {
        return Err("invalid probabilities".into());
    }
    Ok(f64::from(p[1]))
}

#[derive(Debug, PartialEq)]
struct Summary {
    n: usize,
    low: usize,
    high: usize,
    wrong_extreme: usize,
    mean_entropy: f64,
    min: f64,
    max: f64,
}
fn summary(rows: &[(usize, f64)]) -> Result<Summary, String> {
    if rows.is_empty() {
        return Err("empty rows".into());
    }
    let mut s = Summary {
        n: rows.len(),
        low: 0,
        high: 0,
        wrong_extreme: 0,
        mean_entropy: 0.0,
        min: 1.0,
        max: 0.0,
    };
    for &(y, p) in rows {
        if y > 1 || !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err("invalid row".into());
        }
        let low = p <= 1e-6;
        let high = p >= 1.0 - 1e-6;
        s.low += usize::from(low);
        s.high += usize::from(high);
        s.wrong_extreme += usize::from((low && y == 1) || (high && y == 0));
        s.min = s.min.min(p);
        s.max = s.max.max(p);
        for q in [p, 1.0 - p] {
            if q > 0.0 {
                s.mean_entropy -= q * q.ln();
            }
        }
    }
    s.mean_entropy /= s.n as f64;
    Ok(s)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_probability_saturation CHECKPOINT SHA CORPUS SHA SPLIT".into());
    }
    let m = model(&a[1], &a[2])?;
    let (c, s) = corpus(&a[3], &a[4], &a[5])?;
    let rows = c
        .records()
        .iter()
        .filter(|r| r.split == s)
        .map(|r| Ok((r.label, probability(&m, &r.source)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let z = summary(&rows)?;
    println!("checkpoint_sha256,corpus_sha256,split,n,p_le_1e_6,p_ge_1_minus_1e_6,wrong_extreme,mean_entropy_nats,min_p,max_p");
    println!(
        "{},{},{},{},{},{},{},{:.12},{:.12},{:.12}",
        a[2], a[4], a[5], z.n, z.low, z.high, z.wrong_extreme, z.mean_entropy, z.min, z.max
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoints() {
        let s = summary(&[(1, 0.0), (0, 1.0)]).unwrap();
        assert_eq!((s.low, s.high, s.wrong_extreme), (1, 1, 2));
        assert_eq!(s.mean_entropy, 0.0);
    }
    #[test]
    fn uncertain() {
        let s = summary(&[(1, 0.5), (0, 0.5)]).unwrap();
        assert!((s.mean_entropy - std::f64::consts::LN_2).abs() < 1e-12);
        assert_eq!((s.low, s.high), (0, 0));
    }
    #[test]
    fn threshold_and_invalid() {
        assert_eq!(
            summary(&[(0, 1e-6), (1, 1.0 - 1e-6)])
                .unwrap()
                .wrong_extreme,
            0
        );
        for rows in [vec![], vec![(3, 0.5)], vec![(0, f64::NAN)], vec![(0, -0.1)]] {
            assert!(summary(&rows).is_err());
        }
    }
}
