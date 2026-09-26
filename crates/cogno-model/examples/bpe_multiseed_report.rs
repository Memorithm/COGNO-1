//! Explicit all-seed binary evaluation; reports every seed, never selects a winner.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;

fn observations(args: &[String]) -> Result<ObservationSet, String> {
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
    let rows = corpus
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
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ObservationSet {
        rows,
        config: model.heads().config(),
        tokenizer: model.tokenizer().fingerprint(),
    })
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

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
struct ObservationSet {
    rows: Vec<(usize, f64)>,
    config: cogno_scirust::SequenceCognitiveConfig,
    tokenizer: [u8; 32],
}
#[derive(Debug)]
struct Entry {
    seed: u64,
    path: String,
    hash: String,
}
fn manifest(bytes: &[u8]) -> Result<Vec<Entry>, String> {
    if bytes.len() > 65536 {
        return Err("manifest exceeds 64 KiB".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut seeds = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    let mut result = Vec::new();
    for line in text.lines() {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 3 || f[1].is_empty() {
            return Err("expected seed TAB checkpoint_path TAB SHA256".into());
        }
        let seed = f[0].parse::<u64>().map_err(|e| e.to_string())?;
        if f[0] != seed.to_string() {
            return Err("noncanonical seed".into());
        }
        let hash = parse_hash(f[2]).map_err(|e| format!("{e:?}"))?;
        if !seeds.insert(seed) || !hashes.insert(hash) {
            return Err("duplicate seed or checkpoint digest".into());
        }
        result.push(Entry {
            seed,
            path: f[1].into(),
            hash: f[2].into(),
        });
        if result.len() > 64 {
            return Err("at most 64 seeds".into());
        }
    }
    if result.len() < 2 {
        return Err("at least two seeds required".into());
    }
    result.sort_by_key(|r| r.seed);
    Ok(result)
}
#[derive(Debug)]
struct Metrics {
    counts: [usize; 4],
    precision: [f64; 2],
    recall: [f64; 2],
    f1: [f64; 2],
    accuracy: f64,
    macro_f1: f64,
}
fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}
fn metrics(rows: &[(usize, f64)]) -> Result<Metrics, String> {
    validate(rows)?;
    let mut c = [0; 4];
    for &(y, p) in rows {
        c[y * 2 + usize::from(p > 0.5)] += 1;
    }
    if c[0] + c[1] == 0 || c[2] + c[3] == 0 {
        return Err("both classes required for per-class evaluation".into());
    }
    let precision = [ratio(c[0], c[0] + c[2]), ratio(c[3], c[1] + c[3])];
    let recall = [ratio(c[0], c[0] + c[1]), ratio(c[3], c[2] + c[3])];
    let f1 = [
        ratio(2 * c[0], 2 * c[0] + c[1] + c[2]),
        ratio(2 * c[3], 2 * c[3] + c[1] + c[2]),
    ];
    Ok(Metrics {
        counts: c,
        precision,
        recall,
        f1,
        accuracy: ratio(c[0] + c[3], rows.len()),
        macro_f1: (f1[0] + f1[1]) / 2.0,
    })
}
fn mean_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (
        mean,
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64,
    )
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_multiseed_report SEEDS_TSV SEEDS_SHA CORPUS CORPUS_SHA train|validation|test".into());
    }
    let mut bytes = Vec::new();
    std::fs::File::open(&a[1])
        .map_err(|e| e.to_string())?
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let hash: [u8; 32] = Sha256::digest(&bytes).into();
    if hash != parse_hash(&a[2]).map_err(|e| format!("{e:?}"))? {
        return Err("manifest digest mismatch".into());
    }
    let entries = manifest(&bytes)?;
    let parent = std::path::Path::new(&a[1])
        .parent()
        .ok_or("manifest parent missing")?;
    let mut outputs = Vec::new();
    let mut reference = None;
    let mut accuracies = Vec::new();
    let mut macro_f1s = Vec::new();
    for entry in &entries {
        let path = parent
            .join(&entry.path)
            .to_str()
            .ok_or("non-UTF8 path")?
            .to_owned();
        let data = observations(&[
            a[0].clone(),
            path,
            entry.hash.clone(),
            a[3].clone(),
            a[4].clone(),
            a[5].clone(),
        ])?;
        let mut config = data.config;
        if config.encoder.seed != entry.seed {
            return Err("manifest seed differs from checkpoint encoder seed".into());
        }
        config.encoder.seed = 0;
        let identity = (config, data.tokenizer);
        if let Some(r) = reference {
            if r != identity {
                return Err("model configuration or tokenizer differs across seeds".into());
            }
        } else {
            reference = Some(identity);
        }
        let m = metrics(&data.rows)?;
        accuracies.push(m.accuracy);
        macro_f1s.push(m.macro_f1);
        outputs.push(format!(
            "{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            entry.seed,
            entry.hash,
            data.rows.len(),
            m.counts[0],
            m.counts[1],
            m.counts[2],
            m.counts[3],
            m.precision[0],
            m.recall[0],
            m.f1[0],
            m.precision[1],
            m.recall[1],
            m.f1[1],
            m.accuracy,
            m.macro_f1
        ));
    }
    let (am, av) = mean_variance(&accuracies);
    let (fm, fv) = mean_variance(&macro_f1s);
    println!("manifest_sha256,corpus_sha256,split,seeds,mean_accuracy,population_variance_accuracy,mean_macro_f1,population_variance_macro_f1");
    println!(
        "{},{},{},{},{am:.12},{av:.12},{fm:.12},{fv:.12}",
        a[2],
        a[4],
        a[5],
        entries.len()
    );
    println!("seed,checkpoint_sha256,count,tn,fp,fn,tp,precision_fail,recall_fail,f1_fail,precision_pass,recall_pass,f1_pass,accuracy,macro_f1");
    for row in outputs {
        println!("{row}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn confusion_and_absent_predicted_class() {
        let m = metrics(&[(0, 0.1), (0, 0.1), (1, 0.1), (1, 0.1)]).unwrap();
        assert_eq!(m.counts, [2, 0, 2, 0]);
        assert_eq!(m.precision, [0.5, 0.0]);
        assert_eq!(m.recall, [1.0, 0.0]);
        assert!((m.macro_f1 - 1.0 / 3.0).abs() < 1e-12);
    }
    #[test]
    fn known_confusion_and_variance() {
        let m = metrics(&[(0, 0.1), (0, 0.7), (1, 0.8), (1, 0.9)]).unwrap();
        assert_eq!(m.counts, [1, 1, 0, 2]);
        assert_eq!(m.accuracy, 0.75);
        assert_eq!(mean_variance(&[0.0, 0.5, 1.0]), (0.5, 1.0 / 6.0));
    }
    #[test]
    fn invalid_or_one_class_observations_rejected() {
        for rows in [vec![], vec![(1, 0.1)], vec![(0, f64::NAN)], vec![(2, 0.5)]] {
            assert!(metrics(&rows).is_err());
        }
    }
    #[test]
    fn manifest_requires_unique_seeds_and_checkpoints() {
        let h = "01".repeat(32);
        let j = "02".repeat(32);
        let e = manifest(format!("7\ta\t{h}\n1\tb\t{j}").as_bytes()).unwrap();
        assert_eq!(e[0].seed, 1);
        for s in [
            format!("1\ta\t{h}"),
            format!("1\ta\t{h}\n1\tb\t{j}"),
            format!("1\ta\t{h}\n2\tb\t{h}"),
            format!("01\ta\t{h}\n2\tb\t{j}"),
        ] {
            assert!(manifest(s.as_bytes()).is_err());
        }
        assert!(manifest(&vec![b'a'; 65537]).is_err());
    }
}
