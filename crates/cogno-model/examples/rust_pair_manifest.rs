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

use std::collections::{BTreeMap, BTreeSet};
fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
}
fn validate(rows: &[RustRecord], bytes: &[u8], expected: [u8; 32]) -> Result<Vec<String>, String> {
    if bytes.len() > 262144 {
        return Err("manifest too large".into());
    }
    if <[u8; 32]>::from(Sha256::digest(bytes)) != expected {
        return Err("manifest hash mismatch".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "manifest UTF-8")?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err("manifest requires LF lines".into());
    }
    let by_hash: BTreeMap<_, _> = rows.iter().map(|r| (r.source_hash, r)).collect();
    let mut seen = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut family_splits = BTreeMap::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 4 || !valid_id(f[0]) || !valid_id(f[1]) || !ids.insert(f[0]) {
            return Err("invalid pair id/family/columns".into());
        }
        let mut pair = Vec::new();
        for hash in &f[2..] {
            let h = parse_hash(hash).map_err(|_| "invalid hash")?;
            if !seen.insert(h) {
                return Err("source reused".into());
            }
            pair.push(*by_hash.get(&h).ok_or("unknown source")?);
        }
        if pair[0].label != 0
            || pair[1].label != 1
            || pair[0].split != pair[1].split
            || pair[0].project != pair[1].project
        {
            return Err("pair must share split/project and have labels 0,1".into());
        }
        let split = pair[0].split;
        if family_splits
            .insert(f[1], split)
            .is_some_and(|old| old != split)
        {
            return Err("family crosses splits".into());
        }
        out.push(format!(
            "{}\t{}\t{}\t{}\t{}",
            f[0],
            f[1],
            split_name(split),
            hex(&pair[0].source_hash),
            hex(&pair[1].source_hash)
        ));
    }
    if seen.len() != rows.len() {
        return Err("manifest must cover every corpus source exactly once".into());
    }
    Ok(out)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: rust_pair_manifest CORPUS SHA MANIFEST MANIFEST_SHA".into());
    }
    let c = read(&a[1], &a[2])?;
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(&a[3])
        .map_err(|e| e.to_string())?
        .take(262145)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let out = validate(
        c.records(),
        &bytes,
        parse_hash(&a[4]).map_err(|_| "invalid manifest hash")?,
    )?;
    println!("pair_id\tfamily\tsplit\tfail_sha256\tpass_sha256");
    for row in out {
        println!("{row}");
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
    fn manifest(r: &[RustRecord]) -> Vec<u8> {
        r.chunks_exact(2)
            .enumerate()
            .map(|(i, p)| {
                format!(
                    "pair{i}\tfamily{i}\t{}\t{}\n",
                    hex(&p[0].source_hash),
                    hex(&p[1].source_hash)
                )
            })
            .collect::<String>()
            .into_bytes()
    }
    fn check(r: &[RustRecord], b: &[u8]) -> Result<Vec<String>, String> {
        validate(r, b, Sha256::digest(b).into())
    }
    #[test]
    fn exact_coverage_and_ordered_labels() {
        let r = fixture("p");
        let m = manifest(&r);
        assert_eq!(check(&r, &m).unwrap().len(), 3);
        assert!(check(&r, &m[..m.iter().position(|&x| x == b'\n').unwrap() + 1]).is_err());
        let mut bad = r.clone();
        bad[0].label = 1;
        assert!(check(&bad, &m).is_err());
        assert!(validate(&r, &m, [0; 32]).is_err());
    }
    #[test]
    fn rejects_family_leakage_and_source_reuse() {
        let r = fixture("p");
        let m = String::from_utf8(manifest(&r))
            .unwrap()
            .replace("family1", "family0");
        assert!(check(&r, m.as_bytes()).is_err());
        let repeated = [manifest(&r), manifest(&r)].concat();
        assert!(check(&r, &repeated).is_err());
    }
}
