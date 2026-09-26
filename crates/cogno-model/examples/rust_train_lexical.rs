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
#[derive(Debug, PartialEq, Eq)]
struct Lexical {
    bytes: usize,
    lines: usize,
    words: usize,
    distinct_words: usize,
    non_ascii_bytes: usize,
    brace_peak: usize,
    unmatched_closing: usize,
    unclosed: usize,
}
fn lexical(source: &[u8]) -> Lexical {
    let words: Vec<_> = source
        .split(|b| !b.is_ascii_alphanumeric() && *b != b'_')
        .filter(|w| !w.is_empty())
        .collect();
    let mut depth = 0usize;
    let mut peak = 0;
    let mut unmatched = 0;
    for &b in source {
        if b == b'{' {
            depth += 1;
            peak = peak.max(depth);
        } else if b == b'}' {
            if depth == 0 {
                unmatched += 1
            } else {
                depth -= 1;
            }
        }
    }
    Lexical {
        bytes: source.len(),
        lines: source.iter().filter(|&&b| b == b'\n').count()
            + usize::from(!source.is_empty() && source.last() != Some(&b'\n')),
        words: words.len(),
        distinct_words: words.iter().collect::<BTreeSet<_>>().len(),
        non_ascii_bytes: source.iter().filter(|b| !b.is_ascii()).count(),
        brace_peak: peak,
        unmatched_closing: unmatched,
        unclosed: depth,
    }
}
fn training_stats(rows: &[RustRecord]) -> Vec<(&RustRecord, Lexical)> {
    rows.iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .map(|r| (r, lexical(&r.source)))
        .collect()
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: rust_train_lexical CORPUS SHA".into());
    }
    let c = read(&a[1], &a[2])?;
    println!("source_sha256,split,label,bytes,lines,ascii_word_runs,distinct_ascii_word_runs,non_ascii_bytes,raw_brace_peak,raw_unmatched_closing,raw_unclosed");
    for (r, x) in training_stats(c.records()) {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            hex(&r.source_hash),
            split_name(r.split),
            r.label,
            x.bytes,
            x.lines,
            x.words,
            x.distinct_words,
            x.non_ascii_bytes,
            x.brace_peak,
            x.unmatched_closing,
            x.unclosed
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
    fn exact_byte_heuristics_including_unbalanced_braces() {
        let x = lexical(b"a a\n{{} } } {");
        assert_eq!(x.words, 2);
        assert_eq!(x.distinct_words, 1);
        assert_eq!(x.lines, 2);
        assert_eq!(x.brace_peak, 2);
        assert_eq!(x.unmatched_closing, 1);
        assert_eq!(x.unclosed, 1);
        assert_eq!(lexical("é".as_bytes()).non_ascii_bytes, 2);
        assert_eq!(lexical(b"x\n").lines, 1);
        assert_eq!(lexical(b"").lines, 0);
    }
    #[test]
    fn training_only_and_literals_not_parsed() {
        let r = fixture("p");
        assert_eq!(training_stats(&r).len(), 2);
        assert!(training_stats(&r)
            .iter()
            .all(|(r, _)| r.split == CorpusSplit::Train));
        assert_eq!(lexical(b"let x = \"{\";").unclosed, 1);
    }
}
