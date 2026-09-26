//! Cross-corpus identity audit; no normalization or near-duplicate claim.
#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord};
use std::collections::BTreeMap;
#[derive(Debug, PartialEq, Eq)]
struct Finding {
    kind: &'static str,
    key: String,
    left: CorpusSplit,
    right: CorpusSplit,
    left_rows: usize,
    right_rows: usize,
    label_conflict: bool,
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn overlap(left: &[RustRecord], right: &[RustRecord]) -> Vec<Finding> {
    let sources: BTreeMap<_, _> = left.iter().map(|r| (r.source_hash, r)).collect();
    let mut out = Vec::new();
    for r in right {
        if let Some(l) = sources.get(&r.source_hash) {
            out.push(Finding {
                kind: "source",
                key: hex(&r.source_hash),
                left: l.split,
                right: r.split,
                left_rows: 1,
                right_rows: 1,
                label_conflict: l.label != r.label,
            });
        }
    }
    fn projects(records: &[RustRecord]) -> BTreeMap<&str, (CorpusSplit, usize)> {
        let mut map = BTreeMap::new();
        for r in records {
            map.entry(r.project.as_str()).or_insert((r.split, 0)).1 += 1;
        }
        map
    }
    let left_projects = projects(left);
    for (key, (split, count)) in projects(right) {
        if let Some(&(left_split, left_count)) = left_projects.get(key) {
            out.push(Finding {
                kind: "project",
                key: key.into(),
                left: left_split,
                right: split,
                left_rows: left_count,
                right_rows: count,
                label_conflict: false,
            });
        }
    }
    out.sort_by(|a, b| (a.kind, &a.key).cmp(&(b.kind, &b.key)));
    out
}
fn split(s: CorpusSplit) -> &'static str {
    match s {
        CorpusSplit::Train => "train",
        CorpusSplit::Validation => "validation",
        CorpusSplit::Test => "test",
    }
}
fn read(mode: &str, path: &str, expected: &str) -> Result<RustCorpus, String> {
    if !matches!(mode, "full" | "test-only") {
        return Err("mode must be full or test-only".into());
    }
    let hash = parse_hash(expected).map_err(|e| format!("{e:?}"))?;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if mode == "full" {
        RustCorpus::read(file, hash)
    } else {
        RustCorpus::read_test_only(file, hash)
    }
    .map_err(|e| format!("{e:?}"))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: rust_corpus_overlap LEFT_MODE LEFT_CORPUS LEFT_SHA RIGHT_MODE RIGHT_CORPUS RIGHT_SHA".into());
    }
    let left = read(&a[1], &a[2], &a[3])?;
    let right = read(&a[4], &a[5], &a[6])?;
    let findings = overlap(left.records(), right.records());
    println!("left_corpus_sha256,right_corpus_sha256,kind,identity,left_split,right_split,left_rows,right_rows,label_conflict");
    for f in &findings {
        println!(
            "{},{},{},{},{},{},{},{},{}",
            a[3],
            a[6],
            f.kind,
            f.key,
            split(f.left),
            split(f.right),
            f.left_rows,
            f.right_rows,
            f.label_conflict
        );
    }
    eprintln!(
        "left_rows={} right_rows={} shared_sources={} shared_projects={}",
        left.records().len(),
        right.records().len(),
        findings.iter().filter(|f| f.kind == "source").count(),
        findings.iter().filter(|f| f.kind == "project").count()
    );
    if findings.is_empty() {
        Ok(())
    } else {
        Err(
            "cross-corpus overlap detected; inspect CSV before treating evaluation as independent"
                .into(),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(split: CorpusSplit, project: &str, hash: u8, label: usize) -> RustRecord {
        RustRecord {
            split,
            project: project.into(),
            source_hash: [hash; 32],
            label,
            source: vec![hash],
        }
    }
    #[test]
    fn exact_duplicate_and_label_conflict_are_reported() {
        let out = overlap(
            &[row(CorpusSplit::Train, "left", 1, 0)],
            &[row(CorpusSplit::Test, "right", 1, 1)],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, "source");
        assert!(out[0].label_conflict);
        assert_eq!(
            (out[0].left, out[0].right),
            (CorpusSplit::Train, CorpusSplit::Test)
        );
    }
    #[test]
    fn shared_project_detected_without_identical_sources() {
        let out = overlap(
            &[
                row(CorpusSplit::Train, "p", 1, 0),
                row(CorpusSplit::Train, "p", 2, 1),
            ],
            &[row(CorpusSplit::Test, "p", 3, 0)],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, "project");
        assert_eq!((out[0].left_rows, out[0].right_rows), (2, 1));
    }
    #[test]
    fn disjoint_identities_are_clean_and_order_is_stable() {
        let left = [
            row(CorpusSplit::Train, "a", 1, 0),
            row(CorpusSplit::Train, "b", 2, 0),
        ];
        assert!(overlap(&left, &[row(CorpusSplit::Test, "c", 3, 1)]).is_empty());
        assert_eq!(
            overlap(&left, &left),
            overlap(&left, &[left[1].clone(), left[0].clone()])
        );
    }
}
