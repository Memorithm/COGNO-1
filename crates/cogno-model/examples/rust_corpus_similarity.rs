#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord};
#[cfg(test)]
use sha2::{Digest, Sha256};
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn split_name(s: CorpusSplit) -> &'static str {
    match s {
        CorpusSplit::Train => "train",
        CorpusSplit::Validation => "validation",
        CorpusSplit::Test => "test",
    }
}
fn read(path: &str, hash: &str) -> Result<RustCorpus, String> {
    RustCorpus::read(
        std::fs::File::open(path).map_err(|e| e.to_string())?,
        parse_hash(hash).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))
}

use std::collections::BTreeSet;
fn grams(source: &[u8]) -> Result<BTreeSet<Vec<u8>>, String> {
    let bytes: Vec<_> = source
        .iter()
        .copied()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if bytes.len() > 2048 {
        return Err("similarity audit supports <=2048 normalized bytes per source".into());
    }
    if bytes.len() < 3 {
        return Ok(BTreeSet::from([bytes]));
    }
    Ok(bytes.windows(3).map(|x| x.to_vec()).collect())
}
fn similarity(a: &BTreeSet<Vec<u8>>, b: &BTreeSet<Vec<u8>>) -> (usize, usize) {
    let intersection = a.intersection(b).count();
    (intersection, a.len() + b.len() - intersection)
}
fn audit(
    rows: &[RustRecord],
    threshold: usize,
) -> Result<Vec<(usize, usize, usize, usize)>, String> {
    if rows.len() > 512 || threshold > 1_000_000 {
        return Err("audit limit: 512 rows; threshold 0..1000000 ppm".into());
    }
    let sets = rows
        .iter()
        .map(|r| grams(&r.source))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Vec::new();
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            if rows[i].split == rows[j].split {
                continue;
            }
            let (n, d) = similarity(&sets[i], &sets[j]);
            if n * 1_000_000 >= threshold * d {
                out.push((i, j, n, d));
            }
        }
    }
    Ok(out)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: rust_corpus_similarity CORPUS SHA MIN_JACCARD_PPM".into());
    }
    let c = read(&a[1], &a[2])?;
    let rows = c.records();
    let pairs = audit(rows, a[3].parse().map_err(|_| "invalid threshold")?)?;
    println!("left_sha256,right_sha256,left_split,right_split,intersection,union");
    for (i, j, n, d) in pairs {
        println!(
            "{},{},{},{},{n},{d}",
            hex(&rows[i].source_hash),
            hex(&rows[j].source_hash),
            split_name(rows[i].split),
            split_name(rows[j].split)
        );
    }
    Ok(())
}

#[cfg(test)]
fn fixture(prefix: &str) -> Vec<RustRecord> {
    let mut rows = Vec::new();
    for split in [
        CorpusSplit::Train,
        CorpusSplit::Validation,
        CorpusSplit::Test,
    ] {
        for label in 0..2 {
            let source = format!("{prefix} {} {label}", split_name(split)).into_bytes();
            rows.push(RustRecord {
                split,
                project: format!("{prefix}/{}", split_name(split)),
                label,
                source_hash: Sha256::digest(&source).into(),
                source,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whitespace_and_literal_changes_are_only_heuristics() {
        assert_eq!(grams(b"a b c").unwrap(), grams(b"abc").unwrap());
        assert_eq!(
            similarity(&grams(b"abc").unwrap(), &grams(b"abd").unwrap()),
            (0, 2)
        );
        assert!(grams(&vec![b'a'; 2049]).is_err());
    }
    #[test]
    fn only_cross_split_pairs_and_exact_threshold() {
        let mut r = fixture("p");
        r[0].source = b"abc".to_vec();
        r[1].source = b"abc".to_vec();
        r[2].source = b"a b c".to_vec();
        let p = audit(&r, 1_000_000).unwrap();
        assert!(p.contains(&(0, 2, 1, 1)));
        assert!(!p.iter().any(|x| x.0 == 0 && x.1 == 1));
        assert!(audit(&r, 1_000_001).is_err());
        assert!(audit(&vec![r[0].clone(); 513], 0).is_err());
    }
}
