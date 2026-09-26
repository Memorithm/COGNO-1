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

use cogno_model::bpe_tokenizer::{BpeError, BpeTokenizer};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, PartialEq, Eq)]
struct Activation {
    source: usize,
    counts: Option<BTreeMap<u16, usize>>,
}
fn activations(
    rows: &[RustRecord],
    t: &BpeTokenizer,
) -> Result<(Vec<Activation>, BTreeSet<u16>), BpeError> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, r) in rows.iter().enumerate() {
        let counts = match t.encode(&r.source) {
            Ok(ids) => {
                let mut counts = BTreeMap::new();
                for &id in &ids[1..ids.len() - 1] {
                    *counts.entry(id).or_insert(0) += 1;
                }
                if r.split == CorpusSplit::Train {
                    seen.extend(counts.keys().copied());
                }
                Some(counts)
            }
            Err(BpeError::Capacity) => None,
            Err(e) => return Err(e),
        };
        out.push(Activation { source: i, counts });
    }
    Ok((out, seen))
}
fn train(rows: &[RustRecord], vocab: usize, context: usize) -> Result<BpeTokenizer, BpeError> {
    BpeTokenizer::train(
        &rows
            .iter()
            .filter(|r| r.split == CorpusSplit::Train)
            .map(|r| r.source.as_slice())
            .collect::<Vec<_>>(),
        vocab,
        context,
    )
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: rust_source_tokens CORPUS SHA VOCAB CONTEXT".into());
    }
    let c = read(&a[1], &a[2])?;
    let t = train(
        c.records(),
        a[3].parse().map_err(|_| "invalid vocab")?,
        a[4].parse().map_err(|_| "invalid context")?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let (out, seen) = activations(c.records(), &t).map_err(|e| format!("{e:?}"))?;
    eprintln!(
        "corpus_sha256={} tokenizer_sha256={}",
        a[2],
        hex(&t.fingerprint())
    );
    println!("source_sha256,split,status,token_id,occurrences,unseen_in_admitted_train");
    for activation in out {
        let r = &c.records()[activation.source];
        match activation.counts {
            None => println!("{},{},refused,,,", hex(&r.source_hash), split_name(r.split)),
            Some(counts) => {
                for (id, count) in counts {
                    println!(
                        "{},{},accepted,{id},{count},{}",
                        hex(&r.source_hash),
                        split_name(r.split),
                        usize::from(!seen.contains(&id))
                    );
                }
            }
        }
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
    fn sparse_counts_and_heldout_novelty() {
        let mut r = fixture("p");
        r[0].source = b"abab".to_vec();
        r[1].source = b"ab".to_vec();
        r[2].source = b"abz".to_vec();
        let t = BpeTokenizer::from_merges(64, &[(97, 98)]).unwrap();
        let (a, seen) = activations(&r, &t).unwrap();
        assert_eq!(a[0].counts.as_ref().unwrap().get(&259), Some(&2));
        assert!(seen.contains(&259));
        assert!(!seen.contains(&122));
        assert!(a
            .iter()
            .filter_map(|a| a.counts.as_ref())
            .all(|c| c.keys().all(|id| !(256..259).contains(id))));
    }
    #[test]
    fn refused_train_does_not_create_activation_and_no_eval_training() {
        let mut r = fixture("p");
        r[0].source = b"zzzz".to_vec();
        r[1].source = b"x".to_vec();
        let t = BpeTokenizer::from_merges(3, &[]).unwrap();
        let (a, seen) = activations(&r, &t).unwrap();
        assert!(a[0].counts.is_none());
        assert_eq!(seen, BTreeSet::from([120]));
        let before = train(&r, 270, 64).unwrap();
        r[2].source = b"zzzzzzzzzzzzzz".to_vec();
        assert_eq!(before, train(&r, 270, 64).unwrap());
    }
}
