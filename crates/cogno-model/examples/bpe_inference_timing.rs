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

use std::{hint::black_box, time::Instant};
fn budget(rows: usize, iterations: usize) -> Result<usize, String> {
    if rows == 0 || !(1..=100).contains(&iterations) {
        return Err("nonempty rows and 1..100 iterations required".into());
    }
    let n = rows.checked_mul(iterations).ok_or("overflow")?;
    if n > 32768 {
        return Err("at most 32768 timed calls".into());
    }
    Ok(n)
}
fn quantile(sorted: &[u128], percent: usize) -> u128 {
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err(
            "usage: bpe_inference_timing CHECKPOINT SHA CORPUS SHA SPLIT ITERATIONS".into(),
        );
    }
    let iterations = a[6].parse::<usize>().map_err(|e| e.to_string())?;
    let m = model(&a[1], &a[2])?;
    let (c, s) = corpus(&a[3], &a[4], &a[5])?;
    let rows: Vec<_> = c.records().iter().filter(|r| r.split == s).collect();
    let calls = budget(rows.len(), iterations)?;
    let reference = rows
        .iter()
        .map(|r| probability(&m, &r.source))
        .collect::<Result<Vec<_>, _>>()?;
    let mut times = Vec::with_capacity(calls);
    for _ in 0..iterations {
        for (r, &expected) in rows.iter().zip(&reference) {
            let start = Instant::now();
            let observed = black_box(probability(black_box(&m), black_box(&r.source))?);
            let elapsed = start.elapsed().as_nanos();
            if observed.to_bits() != expected.to_bits() {
                return Err("inference parity failure".into());
            }
            times.push(elapsed);
        }
    }
    times.sort_unstable();
    let mean = times.iter().map(|&x| x as f64).sum::<f64>() / calls as f64;
    println!("checkpoint_sha256,corpus_sha256,split,rows,iterations,calls,p50_ns,p95_ns,p99_ns,mean_ns,exact_parity");
    println!(
        "{},{},{},{},{iterations},{calls},{},{},{},{mean:.3},true",
        a[2],
        a[4],
        a[5],
        rows.len(),
        quantile(&times, 50),
        quantile(&times, 95),
        quantile(&times, 99)
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds() {
        assert_eq!(budget(16, 100).unwrap(), 1600);
        for (r, i) in [(0, 1), (1, 0), (1, 101), (4096, 100), (usize::MAX, 2)] {
            assert!(budget(r, i).is_err());
        }
    }
    #[test]
    fn nearest_rank() {
        let x: Vec<u128> = (1..=100).collect();
        assert_eq!(quantile(&x, 50), 50);
        assert_eq!(quantile(&x, 95), 95);
        assert_eq!(quantile(&[7], 99), 7);
    }
}
