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

use std::io::Write;
use std::path::Path;
fn exclusive(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
fn export(rows: &[RustRecord], corpus_hash: [u8; 32], dir: &Path) -> Result<[u8; 32], String> {
    std::fs::create_dir(dir).map_err(|e| e.to_string())?;
    let mut sorted: Vec<_> = rows.iter().collect();
    sorted.sort_by_key(|r| r.source_hash);
    let mut manifest = String::from("source_sha256\tfile\tsplit\tproject\tlabel\tbytes\n");
    for r in sorted {
        let hash = hex(&r.source_hash);
        let name = format!("{hash}.rs");
        exclusive(&dir.join(&name), &r.source)?;
        manifest.push_str(&format!(
            "{hash}\t{name}\t{}\t{}\t{}\t{}\n",
            split_name(r.split),
            r.project,
            r.label,
            r.source.len()
        ));
    }
    exclusive(&dir.join("manifest.tsv"), manifest.as_bytes())?;
    let manifest_hash: [u8; 32] = Sha256::digest(manifest.as_bytes()).into();
    exclusive(
        &dir.join("COMPLETE"),
        format!(
            "corpus_sha256={}\nmanifest_sha256={}\n",
            hex(&corpus_hash),
            hex(&manifest_hash)
        )
        .as_bytes(),
    )?;
    Ok(manifest_hash)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: rust_source_export CORPUS SHA NEW_DIRECTORY".into());
    }
    let c = read(&a[1], &a[2])?;
    println!(
        "manifest_sha256={}",
        hex(&export(c.records(), c.hash(), Path::new(&a[3]))?)
    );
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
    fn temp() -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "cogno-source-export-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }
    #[test]
    fn byte_exact_export_completion_and_existing_refusal() {
        let r = fixture("p");
        let p = temp();
        let h = export(&r, [9; 32], &p).unwrap();
        for row in &r {
            assert_eq!(
                std::fs::read(p.join(format!("{}.rs", hex(&row.source_hash)))).unwrap(),
                row.source
            );
        }
        let manifest = std::fs::read(p.join("manifest.tsv")).unwrap();
        assert_eq!(h, <[u8; 32]>::from(Sha256::digest(manifest)));
        assert!(std::fs::read_to_string(p.join("COMPLETE"))
            .unwrap()
            .contains(&hex(&h)));
        assert!(export(&r, [9; 32], &p).is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn duplicate_failure_has_no_completion_marker() {
        let r = fixture("p");
        let p = temp();
        assert!(export(&[r[0].clone(), r[0].clone()], [0; 32], &p).is_err());
        assert!(!p.join("COMPLETE").exists());
        std::fs::remove_dir_all(p).unwrap();
    }
}
