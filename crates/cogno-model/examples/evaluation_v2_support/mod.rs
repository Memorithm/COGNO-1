//! Strict, bounded offline evaluation of already observed compiler-acceptance data.
#![allow(dead_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

pub const USAGE: &str = "PREDICTIONS SHA CORPUS SHA GROUPS_TSV SHA";
#[derive(Clone, Debug)]
pub struct Observation {
    pub seed: u64,
    pub source: String,
    pub split: String,
    pub group: String,
    pub target: usize,
    pub prediction: usize,
    pub probability: f64,
    pub bytes: usize,
}
#[derive(Debug)]
pub struct Evidence {
    pub rows: Vec<Observation>,
    pub seeds: BTreeSet<u64>,
    pub arm: String,
    pub bindings: [String; 3],
}
pub fn split_name(s: CorpusSplit) -> &'static str {
    match s {
        CorpusSplit::Train => "train",
        CorpusSplit::Validation => "validation",
        CorpusSplit::Test => "test",
    }
}
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
}
pub fn read_pinned(path: &str, expected: &str, limit: usize) -> Result<Vec<u8>, String> {
    let hash = parse_hash(expected).map_err(|e| format!("hash: {e:?}"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err("input exceeds byte bound".into());
    }
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != hash {
        return Err("input hash mismatch".into());
    }
    Ok(bytes)
}
fn text(bytes: &[u8]) -> Result<&str, String> {
    let s = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    if !s.ends_with('\n') || s.contains('\r') {
        return Err("expected LF-terminated text".into());
    }
    Ok(s)
}
fn integer(s: &str) -> Result<u64, String> {
    let n = s.parse::<u64>().map_err(|e| e.to_string())?;
    if s != n.to_string() {
        return Err("noncanonical integer".into());
    }
    Ok(n)
}
pub fn groups(bytes: &[u8], corpus: &RustCorpus) -> Result<BTreeMap<String, String>, String> {
    let mut lines = text(bytes)?.lines();
    if lines.next() != Some("project\tgroup\tsplit") {
        return Err("invalid group header".into());
    }
    let projects: BTreeMap<_, _> = corpus
        .records()
        .iter()
        .map(|r| (r.project.clone(), split_name(r.split)))
        .collect();
    let mut result = BTreeMap::new();
    let mut group_splits = BTreeMap::new();
    for line in lines {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 3
            || !identifier(f[0])
            || !identifier(f[1])
            || projects.get(f[0]) != Some(&f[2])
        {
            return Err("group project/split mismatch".into());
        }
        if result.insert(f[0].into(), f[1].into()).is_some() {
            return Err("duplicate group project".into());
        }
        if let Some(old) = group_splits.insert(f[1], f[2]) {
            if old != f[2] {
                return Err("group crosses splits".into());
            }
        }
    }
    if result.len() != projects.len() {
        return Err("incomplete group map".into());
    }
    Ok(result)
}
pub fn predictions(
    bytes: &[u8],
    corpus: &RustCorpus,
    groups: &BTreeMap<String, String>,
) -> Result<Evidence, String> {
    let mut lines = text(bytes)?.lines();
    if lines.next() != Some("arm,seed,split,source_sha256,target,prediction,p_compile,tokens") {
        return Err("invalid prediction header".into());
    }
    let records: BTreeMap<_, _> = corpus
        .records()
        .iter()
        .map(|r| (hex(&r.source_hash), r))
        .collect();
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut arm = String::new();
    for line in lines {
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 8 || !identifier(f[0]) {
            return Err("invalid prediction row".into());
        }
        if arm.is_empty() {
            arm = f[0].into();
        }
        if arm != f[0] {
            return Err("one arm required per file".into());
        }
        let seed = integer(f[1])?;
        let record = records.get(f[3]).ok_or("unknown source")?;
        let target = integer(f[4])?;
        let predicted = integer(f[5])?;
        let p = f[6].parse::<f64>().map_err(|e| e.to_string())?;
        let tokens = integer(f[7])?;
        if target != record.label as u64
            || f[2] != split_name(record.split)
            || predicted > 1
            || !(2..=512).contains(&tokens)
            || !p.is_finite()
            || !(0.0..=1.0).contains(&p)
        {
            return Err("invalid observation or corpus binding".into());
        }
        // Stored f32 probabilities can round across the exact argmax tie.
        if (p - 0.5).abs() > 1e-7 && predicted != u64::from(p > 0.5) {
            return Err("prediction/probability mismatch".into());
        }
        if !seen.insert((seed, f[3].to_string())) {
            return Err("duplicate seed/source".into());
        }
        seeds.insert(seed);
        if seeds.len() > 64 || rows.len() >= 262144 {
            return Err("prediction capacity".into());
        }
        rows.push(Observation {
            seed,
            source: f[3].into(),
            split: f[2].into(),
            group: groups.get(&record.project).ok_or("missing group")?.clone(),
            target: target as usize,
            prediction: predicted as usize,
            probability: p,
            bytes: record.source.len(),
        });
    }
    if seeds.is_empty() || rows.len() != seeds.len() * records.len() {
        return Err("incomplete seed/source Cartesian product".into());
    }
    Ok(Evidence {
        rows,
        seeds,
        arm,
        bindings: Default::default(),
    })
}
pub fn load(args: &[String]) -> Result<Evidence, String> {
    if args.len() != 6 {
        return Err(USAGE.into());
    }
    let p = read_pinned(&args[0], &args[1], 64 * 1024 * 1024)?;
    let c = read_pinned(&args[2], &args[3], 4 * 1024 * 1024)?;
    let corpus = RustCorpus::parse(&c, parse_hash(&args[3]).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("corpus: {e:?}"))?;
    let g = read_pinned(&args[4], &args[5], 1024 * 1024)?;
    let mut evidence = predictions(&p, &corpus, &groups(&g, &corpus)?)?;
    evidence.bindings = [args[1].clone(), args[3].clone(), args[5].clone()];
    Ok(evidence)
}
pub fn header(e: &Evidence) {
    println!(
        "# arm={} predictions_sha256={} corpus_sha256={} groups_sha256={}",
        e.arm, e.bindings[0], e.bindings[1], e.bindings[2]
    );
    println!("# groups_are_user_supplied_not_authenticated; observed_test_not_fresh; classifier_only; no_promotion");
}
pub fn nll(target: usize, p: f64) -> f64 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    if target == 1 {
        -p.ln()
    } else {
        -(1.0 - p).ln()
    }
}
pub fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    sum / n as f64
}
pub fn paired<'a>(
    a: &'a Evidence,
    b: &'a Evidence,
    split: &str,
) -> Result<Vec<(&'a Observation, &'a Observation)>, String> {
    if a.bindings[1..] != b.bindings[1..] || a.seeds != b.seeds {
        return Err("paired evidence requires identical corpus, groups and seeds".into());
    }
    let indexed: BTreeMap<_, _> = b
        .rows
        .iter()
        .filter(|r| r.split == split)
        .map(|r| ((r.seed, &r.source), r))
        .collect();
    let result = a
        .rows
        .iter()
        .filter(|r| r.split == split)
        .map(|r| {
            indexed
                .get(&(r.seed, &r.source))
                .map(|s| (r, *s))
                .ok_or_else(|| "unpaired source".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if result.is_empty() {
        return Err("empty paired split".into());
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (RustCorpus, BTreeMap<String, String>, String) {
        let mut wire = String::from("CRUST001\n");
        let mut predictions =
            String::from("arm,seed,split,source_sha256,target,prediction,p_compile,tokens\n");
        for (i, split) in ["train", "validation", "test"].iter().enumerate() {
            for label in 0..2 {
                let source = format!("fn source_{i}_{label}() {{}}");
                let hash = hex(&Sha256::digest(source.as_bytes()));
                wire.push_str(&format!(
                    "{split}\tp{i}_{label}\t{label}\t{hash}\t{}\n",
                    hex(source.as_bytes())
                ));
                predictions.push_str(&format!(
                    "a,1,{split},{hash},{label},{label},{},3\n",
                    if label == 1 { 0.9 } else { 0.1 }
                ));
            }
        }
        let corpus =
            RustCorpus::parse(wire.as_bytes(), Sha256::digest(wire.as_bytes()).into()).unwrap();
        let groups = corpus
            .records()
            .iter()
            .map(|r| (r.project.clone(), split_name(r.split).to_string()))
            .collect();
        (corpus, groups, predictions)
    }
    #[test]
    fn rejects_missing_duplicate_wrong_probability_and_label() {
        let (c, g, p) = fixture();
        assert_eq!(predictions(p.as_bytes(), &c, &g).unwrap().rows.len(), 6);
        let missing = p.lines().take(6).collect::<Vec<_>>().join("\n") + "\n";
        assert!(predictions(missing.as_bytes(), &c, &g).is_err());
        let duplicate = p.clone() + p.lines().nth(1).unwrap() + "\n";
        assert!(predictions(duplicate.as_bytes(), &c, &g).is_err());
        assert!(predictions(p.replacen(",0,0,0.1,", ",0,1,0.1,", 1).as_bytes(), &c, &g).is_err());
        assert!(predictions(p.replacen(",0,0,0.1,", ",1,1,0.9,", 1).as_bytes(), &c, &g).is_err());
        assert!(predictions(p.replacen("0.1", "NaN", 1).as_bytes(), &c, &g).is_err());
    }
    #[test]
    fn groups_reject_split_leak_and_omissions() {
        let (c, _, _) = fixture();
        let mut t = String::from("project\tgroup\tsplit\n");
        for r in c.records() {
            t.push_str(&format!(
                "{}\t{}\t{}\n",
                r.project,
                split_name(r.split),
                split_name(r.split)
            ));
        }
        assert!(groups(t.as_bytes(), &c).is_ok());
        assert!(groups(
            t.replacen("\tvalidation\tvalidation", "\ttrain\tvalidation", 1)
                .as_bytes(),
            &c
        )
        .is_err());
        assert!(groups(b"project\tgroup\tsplit\n", &c).is_err());
    }
}
