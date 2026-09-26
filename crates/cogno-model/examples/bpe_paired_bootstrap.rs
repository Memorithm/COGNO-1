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

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn interval(rows: &[(bool, bool)], seed: u64) -> Result<(f64, f64, f64), String> {
    if rows.is_empty() || rows.len() > 4096 {
        return Err("expected 1..4096 paired rows".into());
    }
    let n = rows.len();
    let mut state = seed;
    let mut samples = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sum = 0i32;
        for _ in 0..n {
            let bound = n as u64;
            let threshold = bound.wrapping_neg() % bound;
            let mut chosen = None;
            for _ in 0..16 {
                let z = next(&mut state);
                if z >= threshold {
                    chosen = Some((z % bound) as usize);
                    break;
                }
            }
            let (x, y) = rows[chosen.ok_or("sampling limit")?];
            sum += i32::from(y) - i32::from(x);
        }
        samples.push(sum as f64 / n as f64);
    }
    samples.sort_by(f64::total_cmp);
    let mean = rows
        .iter()
        .map(|&(x, y)| i32::from(y) - i32::from(x))
        .sum::<i32>() as f64
        / n as f64;
    Ok((mean, samples[49], samples[1949]))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 9 {
        return Err("usage: bpe_paired_bootstrap A SHA B SHA CORPUS SHA SPLIT SEED".into());
    }
    let seed = a[8].parse::<u64>().map_err(|e| e.to_string())?;
    let rows = paired(&a)?;
    let (d, l, u) = interval(&rows, seed)?;
    println!("a_sha256,b_sha256,corpus_sha256,split,n,seed,resamples,b_minus_a,percentile_025,percentile_975");
    println!(
        "{},{},{},{},{},{seed},2000,{d:.12},{l:.12},{u:.12}",
        a[2],
        a[4],
        a[6],
        a[7],
        rows.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoints() {
        assert_eq!(interval(&[(false, true); 10], 0).unwrap(), (1.0, 1.0, 1.0));
        assert_eq!(
            interval(&[(true, false); 10], 0).unwrap(),
            (-1.0, -1.0, -1.0)
        );
        assert_eq!(interval(&[(true, true); 10], 0).unwrap(), (0.0, 0.0, 0.0));
    }
    #[test]
    fn deterministic_and_bounded() {
        let r = [(true, false), (false, true), (true, true), (false, false)];
        let a = interval(&r, 42).unwrap();
        assert_eq!(a, interval(&r, 42).unwrap());
        assert_eq!(a.0, 0.0);
        assert!(a.1 >= -1.0 && a.2 <= 1.0 && a.1 <= a.2);
        assert!(interval(&[], 1).is_err());
        assert!(interval(&vec![(true, true); 4097], 1).is_err());
    }
}
