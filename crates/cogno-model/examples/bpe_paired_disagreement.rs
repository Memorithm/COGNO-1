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

fn paired(a: &[String]) -> Result<Vec<(bool, bool)>, String> {
    let x = model(&a[1], &a[2])?;
    let y = model(&a[3], &a[4])?;
    let (c, s) = corpus(&a[5], &a[6], &a[7])?;
    c.records()
        .iter()
        .filter(|r| r.split == s)
        .map(|r| {
            Ok((
                usize::from(probability(&x, &r.source)? > 0.5) == r.label,
                usize::from(probability(&y, &r.source)? > 0.5) == r.label,
            ))
        })
        .collect()
}

fn exact_p(b: usize, c: usize) -> Result<f64, String> {
    let n = b.checked_add(c).ok_or("overflow")?;
    if n > 4096 {
        return Err("too many rows".into());
    }
    if n == 0 {
        return Ok(1.0);
    }
    // Log-sum-exp prevents underflow of the initial 2^-n at large n.
    let k = b.min(c);
    let mut log_term = -(n as f64) * std::f64::consts::LN_2;
    let mut log_sum = log_term;
    for i in 1..=k {
        log_term += ((n - i + 1) as f64).ln() - (i as f64).ln();
        let hi = log_sum.max(log_term);
        log_sum = hi + ((log_sum - hi).exp() + (log_term - hi).exp()).ln();
    }
    Ok((2.0 * log_sum.exp()).min(1.0))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err("usage: bpe_paired_disagreement A SHA B SHA CORPUS SHA SPLIT".into());
    }
    let rows = paired(&a)?;
    let mut n = [0; 4];
    for &(x, y) in &rows {
        n[usize::from(x) * 2 + usize::from(y)] += 1;
    }
    let p = exact_p(n[2], n[1])?;
    println!("a_sha256,b_sha256,corpus_sha256,split,n,both_wrong,b_only_correct,a_only_correct,both_correct,exact_two_sided_p");
    println!(
        "{},{},{},{},{},{},{},{},{},{p:.12}",
        a[2],
        a[4],
        a[6],
        a[7],
        rows.len(),
        n[0],
        n[1],
        n[2],
        n[3]
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_small_cases() {
        assert_eq!(exact_p(0, 0).unwrap(), 1.0);
        assert!((exact_p(0, 5).unwrap() - 0.0625).abs() < 1e-12);
        assert_eq!(exact_p(3, 3).unwrap(), 1.0);
    }
    #[test]
    fn symmetric_large() {
        assert!((exact_p(2048, 2048).unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(exact_p(1, 20), exact_p(20, 1));
        assert!(exact_p(4097, 0).is_err());
    }
}
