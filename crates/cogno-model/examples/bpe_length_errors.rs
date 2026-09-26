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

use cogno_model::bpe_tokenizer::BpeError;
const LIMITS: [usize; 6] = [32, 64, 128, 256, 512, usize::MAX];
fn bin(n: usize) -> usize {
    LIMITS.iter().position(|&x| n <= x).unwrap()
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_length_errors CHECKPOINT SHA CORPUS SHA SPLIT".into());
    }
    let m = model(&a[1], &a[2])?;
    let (c, s) = corpus(&a[3], &a[4], &a[5])?;
    let mut bins = [[0usize; 4]; 6];
    for r in c.records().iter().filter(|r| r.split == s) {
        let required = m
            .tokenizer()
            .required_tokens(&r.source)
            .map_err(|e| format!("{e:?}"))?;
        let b = &mut bins[bin(required)];
        b[0] += 1;
        match m.tokenizer().encode(&r.source) {
            Ok(_) => {
                let p = probability(&m, &r.source)?;
                b[1] += 1;
                b[2] += usize::from(usize::from(p > 0.5) == r.label);
            }
            Err(BpeError::Capacity) => b[3] += 1,
            Err(e) => return Err(format!("{e:?}")),
        }
    }
    println!(
        "checkpoint_sha256,corpus_sha256,split\n{},{},{}",
        a[2], a[4], a[5]
    );
    println!("tokens_inclusive,total,accepted,correct,capacity_refused,accepted_accuracy");
    for (i, b) in bins.iter().enumerate() {
        let label = if i == 5 {
            "513+".into()
        } else {
            format!(
                "{}-{}",
                if i == 0 { 0 } else { LIMITS[i - 1] + 1 },
                LIMITS[i]
            )
        };
        let acc = if b[1] == 0 {
            "NA".into()
        } else {
            format!("{:.12}", b[2] as f64 / b[1] as f64)
        };
        println!("{label},{},{},{},{},{acc}", b[0], b[1], b[2], b[3]);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries() {
        for (i, &n) in LIMITS[..5].iter().enumerate() {
            assert_eq!(bin(n), i);
            assert_eq!(bin(n + 1), i + 1);
        }
        assert_eq!(bin(0), 0);
        assert_eq!(bin(usize::MAX), 5);
    }
}
