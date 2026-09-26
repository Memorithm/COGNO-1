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

use std::collections::BTreeMap;
fn counts(rows: &[(usize, f64)]) -> Result<[usize; 4], String> {
    let mut n = [0; 4];
    for &(y, p) in rows {
        if y > 1 || !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err("invalid row".into());
        }
        n[y * 2 + usize::from(p > 0.5)] += 1;
    }
    Ok(n)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_project_confusion CHECKPOINT SHA CORPUS SHA SPLIT".into());
    }
    let m = model(&a[1], &a[2])?;
    let (c, s) = corpus(&a[3], &a[4], &a[5])?;
    let mut groups = BTreeMap::<String, Vec<(usize, f64)>>::new();
    for r in c.records().iter().filter(|r| r.split == s) {
        groups
            .entry(r.project.clone())
            .or_default()
            .push((r.label, probability(&m, &r.source)?));
    }
    let mut out = Vec::new();
    for (project, rows) in groups {
        let n = counts(&rows)?;
        let recall0 = if n[0] + n[1] == 0 {
            "NA".into()
        } else {
            format!("{:.12}", n[0] as f64 / (n[0] + n[1]) as f64)
        };
        let recall1 = if n[2] + n[3] == 0 {
            "NA".into()
        } else {
            format!("{:.12}", n[3] as f64 / (n[2] + n[3]) as f64)
        };
        out.push(format!(
            "{project}\t{}\t{}\t{}\t{}\t{}\t{recall0}\t{recall1}",
            rows.len(),
            n[0],
            n[1],
            n[2],
            n[3]
        ));
    }
    println!(
        "checkpoint_sha256\tcorpus_sha256\tsplit\n{}\t{}\t{}",
        a[2], a[4], a[5]
    );
    println!("project\tn\ttn\tfp\tfn\ttp\trecall_fail\trecall_pass");
    for l in out {
        println!("{l}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_cells_and_tie() {
        assert_eq!(
            counts(&[(0, 0.5), (0, 0.8), (1, 0.2), (1, 0.9)]).unwrap(),
            [1, 1, 1, 1]
        );
    }
    #[test]
    fn invalid() {
        assert!(counts(&[(2, 0.1)]).is_err());
        assert!(counts(&[(0, f64::NAN)]).is_err());
        assert!(counts(&[(0, 1.1)]).is_err());
    }
    #[test]
    fn empty_group_is_zero() {
        assert_eq!(counts(&[]).unwrap(), [0; 4]);
    }
}
