#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord};
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
fn wire(rows: &[&RustRecord]) -> Vec<u8> {
    let mut out = String::from("CRUST001\n");
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            split_name(r.split),
            r.project,
            r.label,
            hex(&r.source_hash),
            hex(&r.source)
        ));
    }
    out.into_bytes()
}
fn admit(rows: &[&RustRecord]) -> Result<Vec<u8>, String> {
    let w = wire(rows);
    RustCorpus::parse(&w, Sha256::digest(&w).into()).map_err(|e| format!("{e:?}"))?;
    Ok(w)
}

fn select(rows: &[RustRecord], per_class: usize, seed: u64) -> Result<Vec<&RustRecord>, String> {
    if !(1..=2048).contains(&per_class) {
        return Err("per-class count must be 1..2048".into());
    }
    let mut chosen = std::collections::BTreeSet::new();
    for label in 0..2 {
        let mut candidates: Vec<_> = rows
            .iter()
            .filter(|r| r.split == CorpusSplit::Train && r.label == label)
            .map(|r| {
                let mut h = Sha256::new();
                h.update(seed.to_le_bytes());
                h.update(r.source_hash);
                (<[u8; 32]>::from(h.finalize()), r.source_hash)
            })
            .collect();
        candidates.sort();
        if candidates.len() < per_class {
            return Err("insufficient class rows".into());
        }
        chosen.extend(candidates.into_iter().take(per_class).map(|x| x.1));
    }
    Ok(rows
        .iter()
        .filter(|r| r.split != CorpusSplit::Train || chosen.contains(&r.source_hash))
        .collect())
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: rust_train_subsample CORPUS SHA PER_CLASS SEED NEW_OUTPUT".into());
    }
    let c = read(&a[1], &a[2])?;
    let w = admit(&select(
        c.records(),
        a[3].parse().map_err(|_| "invalid count")?,
        a[4].parse().map_err(|_| "invalid seed")?,
    )?)?;
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[5])
        .map_err(|e| e.to_string())?;
    f.write_all(&w).map_err(|e| e.to_string())?;
    println!("{}", hex(&Sha256::digest(&w)));
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
    fn balanced_and_preserves_heldout() {
        let r = [fixture("a"), fixture("b")].concat();
        let got = select(&r, 1, 42).unwrap();
        for label in 0..2 {
            assert_eq!(
                got.iter()
                    .filter(|r| r.split == CorpusSplit::Train && r.label == label)
                    .count(),
                1
            );
        }
        assert_eq!(
            got.iter()
                .filter(|r| r.split != CorpusSplit::Train)
                .copied()
                .collect::<Vec<_>>(),
            r.iter()
                .filter(|r| r.split != CorpusSplit::Train)
                .collect::<Vec<_>>()
        );
        assert!(admit(&got).is_ok());
        assert_eq!(got, select(&r, 1, 42).unwrap());
    }
    #[test]
    fn stable_selection_under_input_permutation() {
        let r = [fixture("a"), fixture("b")].concat();
        let mut rev = r.clone();
        rev.reverse();
        let hashes = |r: &[RustRecord]| {
            select(r, 1, 7)
                .unwrap()
                .iter()
                .filter(|x| x.split == CorpusSplit::Train)
                .map(|x| x.source_hash)
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(hashes(&r), hashes(&rev));
        assert!(select(&r, 0, 0).is_err());
        assert!(select(&r, 3, 0).is_err());
    }
}
