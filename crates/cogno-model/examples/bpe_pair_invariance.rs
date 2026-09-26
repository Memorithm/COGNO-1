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

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
type SourcePair = ([u8; 32], [u8; 32]);
fn pairs(bytes: &[u8]) -> Result<Vec<SourcePair>, String> {
    if bytes.len() > 270336 {
        return Err("manifest too large".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err("LF-terminated manifest required".into());
    }
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 2 {
            return Err("expected original_sha TAB variant_sha".into());
        }
        let x = parse_hash(f[0]).map_err(|e| format!("{e:?}"))?;
        let y = parse_hash(f[1]).map_err(|e| format!("{e:?}"))?;
        let key = if x < y { (x, y) } else { (y, x) };
        if x == y || !seen.insert(key) {
            return Err("self-pair or duplicate pair".into());
        }
        rows.push((x, y));
        if rows.len() > 2048 {
            return Err("at most 2048 pairs".into());
        }
    }
    if rows.is_empty() {
        return Err("empty manifest".into());
    }
    Ok(rows)
}
fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err(
            "usage: bpe_pair_invariance CHECKPOINT SHA CORPUS SHA SPLIT PAIRS_TSV PAIRS_SHA".into(),
        );
    }
    let m = model(&a[1], &a[2])?;
    let (c, s) = corpus(&a[3], &a[4], &a[5])?;
    let mut bytes = Vec::new();
    std::fs::File::open(&a[6])
        .map_err(|e| e.to_string())?
        .take(270337)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    if digest != parse_hash(&a[7]).map_err(|e| format!("{e:?}"))? {
        return Err("pair manifest hash mismatch".into());
    }
    let list = pairs(&bytes)?;
    let records: BTreeMap<_, _> = c
        .records()
        .iter()
        .filter(|r| r.split == s)
        .map(|r| (r.source_hash, r))
        .collect();
    let mut out = Vec::new();
    for (x, y) in list {
        let r = records
            .get(&x)
            .ok_or("original absent from selected split")?;
        let t = records
            .get(&y)
            .ok_or("variant absent from selected split")?;
        if r.label != t.label {
            return Err("pair compiler labels differ".into());
        }
        let p = probability(&m, &r.source)?;
        let q = probability(&m, &t.source)?;
        out.push(format!(
            "{},{},{},{p:.12},{q:.12},{:.12},{}",
            hex(&x),
            hex(&y),
            r.label,
            (q - p).abs(),
            usize::from((p > 0.5) != (q > 0.5))
        ));
    }
    println!(
        "checkpoint_sha256,corpus_sha256,split,pairs_sha256\n{},{},{},{}",
        a[2], a[4], a[5], a[7]
    );
    println!("original_sha256,variant_sha256,target,original_p,variant_p,absolute_probability_shift,prediction_flip");
    for row in out {
        println!("{row}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn valid_and_duplicate() {
        let x = "01".repeat(32);
        let y = "02".repeat(32);
        let row = format!("{x}\t{y}\n");
        assert_eq!(pairs(row.as_bytes()).unwrap().len(), 1);
        assert!(pairs(format!("{row}{y}\t{x}\n").as_bytes()).is_err());
        assert!(pairs(format!("{x}\t{x}\n").as_bytes()).is_err());
    }
    #[test]
    fn malformed() {
        for b in [&b""[..], b"x\ty\n", b"x\ty\r\n", b"x\ty"] {
            assert!(pairs(b).is_err());
        }
        assert!(pairs(&vec![b'a'; 270337]).is_err());
    }
}
