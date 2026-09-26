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

fn union<'a>(left: &'a RustCorpus, right: &'a RustCorpus) -> Result<Vec<u8>, String> {
    let rows: Vec<_> = left.records().iter().chain(right.records()).collect();
    admit(&rows)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: rust_corpus_union LEFT SHA RIGHT SHA NEW_OUTPUT".into());
    }
    let w = union(&read(&a[1], &a[2])?, &read(&a[3], &a[4])?)?;
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
    fn corpus(rows: &[RustRecord]) -> RustCorpus {
        let w = wire(&rows.iter().collect::<Vec<_>>());
        RustCorpus::parse(&w, Sha256::digest(&w).into()).unwrap()
    }
    #[test]
    fn union_revalidates_duplicates_and_project_leakage() {
        let a = fixture("a");
        let mut b = fixture("b");
        assert!(union(&corpus(&a), &corpus(&a)).is_err());
        b[2].project = a[0].project.clone();
        assert!(union(&corpus(&a), &corpus(&b)).is_err());
    }
    #[test]
    fn disjoint_union_preserves_sources() {
        let a = fixture("a");
        let b = fixture("b");
        let w = union(&corpus(&a), &corpus(&b)).unwrap();
        let c = RustCorpus::parse(&w, Sha256::digest(&w).into()).unwrap();
        assert_eq!(c.records(), [a, b].concat());
    }
}
