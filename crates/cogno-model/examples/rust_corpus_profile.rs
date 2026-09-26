//! Deterministic bounded corpus composition report; never trains a model.
#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq, Eq)]
struct Profile {
    rows: usize,
    labels: [usize; 2],
    projects: usize,
    bytes: usize,
    min: usize,
    median: usize,
    p95: usize,
    max: usize,
}
fn profile(records: &[&RustRecord]) -> Option<Profile> {
    if records.is_empty() {
        return None;
    }
    let mut lengths: Vec<_> = records.iter().map(|r| r.source.len()).collect();
    lengths.sort_unstable();
    let mut labels = [0; 2];
    let mut projects = BTreeSet::new();
    for r in records {
        labels[r.label] += 1;
        projects.insert(&r.project);
    }
    Some(Profile {
        rows: records.len(),
        labels,
        projects: projects.len(),
        bytes: lengths.iter().sum(),
        min: lengths[0],
        median: lengths[(lengths.len() - 1) / 2],
        p95: lengths[(lengths.len() * 95).div_ceil(100) - 1],
        max: *lengths.last().unwrap(),
    })
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 || !matches!(a[1].as_str(), "full" | "test-only") {
        return Err("usage: rust_corpus_profile full|test-only CORPUS EXPECTED_SHA256".into());
    }
    let file = std::fs::File::open(&a[2]).map_err(|e| e.to_string())?;
    let hash = parse_hash(&a[3]).map_err(|e| format!("{e:?}"))?;
    let corpus = if a[1] == "full" {
        RustCorpus::read(file, hash)
    } else {
        RustCorpus::read_test_only(file, hash)
    }
    .map_err(|e| format!("{e:?}"))?;
    let mut splits = BTreeMap::<CorpusSplit, Vec<&RustRecord>>::new();
    for r in corpus.records() {
        splits.entry(r.split).or_default().push(r);
    }
    println!("corpus_sha256,split,rows,label_0,label_1,projects,source_bytes,min_bytes,median_bytes,p95_bytes,max_bytes");
    for (split, records) in splits {
        let p = profile(&records).ok_or("empty split")?;
        let split = match split {
            CorpusSplit::Train => "train",
            CorpusSplit::Validation => "validation",
            CorpusSplit::Test => "test",
        };
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            a[3],
            split,
            p.rows,
            p.labels[0],
            p.labels[1],
            p.projects,
            p.bytes,
            p.min,
            p.median,
            p.p95,
            p.max
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(length: usize, label: usize, project: &str) -> RustRecord {
        RustRecord {
            split: CorpusSplit::Train,
            project: project.into(),
            label,
            source: vec![b'x'; length],
            source_hash: [0; 32],
        }
    }
    #[test]
    fn quantiles_classes_and_unique_projects_are_exact() {
        let rows = [
            row(10, 0, "a"),
            row(2, 1, "b"),
            row(4, 1, "b"),
            row(6, 0, "c"),
        ];
        let refs = rows.iter().collect::<Vec<_>>();
        assert_eq!(
            profile(&refs),
            Some(Profile {
                rows: 4,
                labels: [2, 2],
                projects: 3,
                bytes: 22,
                min: 2,
                median: 4,
                p95: 10,
                max: 10
            })
        );
        let mut reversed = refs;
        reversed.reverse();
        assert_eq!(
            profile(&reversed),
            profile(&rows.iter().collect::<Vec<_>>())
        );
    }
    #[test]
    fn singleton_and_empty_are_explicit() {
        assert_eq!(profile(&[]), None);
        let r = row(3, 1, "one");
        let p = profile(&[&r]).unwrap();
        assert_eq!((p.min, p.median, p.p95, p.max), (3, 3, 3, 3));
    }
}
