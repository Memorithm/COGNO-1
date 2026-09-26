//! Vocabulary activation on context-admitted records; excludes framing tokens.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_tokenizer::{BpeError, BpeTokenizer},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord},
};
#[derive(Debug, PartialEq, Eq)]
struct Usage {
    counts: Vec<[usize; 3]>,
    accepted: [usize; 3],
    refused: [usize; 3],
}
fn index(split: CorpusSplit) -> usize {
    match split {
        CorpusSplit::Train => 0,
        CorpusSplit::Validation => 1,
        CorpusSplit::Test => 2,
    }
}
fn utilization(records: &[RustRecord], tokenizer: &BpeTokenizer) -> Result<Usage, BpeError> {
    let mut usage = Usage {
        counts: vec![[0; 3]; tokenizer.vocab_size()],
        accepted: [0; 3],
        refused: [0; 3],
    };
    for r in records {
        let split = index(r.split);
        match tokenizer.encode(&r.source) {
            Ok(ids) => {
                usage.accepted[split] += 1;
                for &id in &ids[1..ids.len() - 1] {
                    usage.counts[usize::from(id)][split] += 1;
                }
            }
            Err(BpeError::Capacity) => {
                usage.refused[split] += 1;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(usage)
}
fn train(records: &[RustRecord], vocab: usize, context: usize) -> Result<BpeTokenizer, BpeError> {
    BpeTokenizer::train(
        &records
            .iter()
            .filter(|r| r.split == CorpusSplit::Train)
            .map(|r| r.source.as_slice())
            .collect::<Vec<_>>(),
        vocab,
        context,
    )
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: rust_bpe_utilization FULL_CORPUS EXPECTED_SHA256 VOCAB CONTEXT".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&a[1]).map_err(|e| e.to_string())?,
        parse_hash(&a[2]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let tokenizer = train(
        corpus.records(),
        a[3].parse().map_err(|_| "invalid vocabulary")?,
        a[4].parse().map_err(|_| "invalid context")?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let usage = utilization(corpus.records(), &tokenizer).map_err(|e| format!("{e:?}"))?;
    let fingerprint = hex(&tokenizer.fingerprint());
    println!("corpus_sha256,tokenizer_sha256,context,token_id,kind,train_occurrences,validation_occurrences,test_occurrences");
    for (id, counts) in usage.counts.iter().enumerate() {
        if (256..259).contains(&id) {
            continue;
        }
        println!(
            "{},{},{},{},{},{},{},{}",
            a[2],
            fingerprint,
            tokenizer.max_tokens(),
            id,
            if id < 256 { "byte" } else { "merge" },
            counts[0],
            counts[1],
            counts[2]
        );
    }
    for (s, name) in ["train", "validation", "test"].iter().enumerate() {
        let raw: usize = usage.counts[..256].iter().map(|c| c[s]).sum();
        let merged: usize = usage.counts[259..].iter().map(|c| c[s]).sum();
        let active_merges = usage.counts[259..].iter().filter(|c| c[s] > 0).count();
        let unseen_occurrences: usize = usage
            .counts
            .iter()
            .filter(|c| c[0] == 0)
            .map(|c| c[s])
            .sum();
        eprintln!("split={name} accepted={} refused={} byte_occurrences={raw} merge_occurrences={merged} active_merge_types={active_merges} total_merge_types={} unseen_in_admitted_train_occurrences={unseen_occurrences}",usage.accepted[s],usage.refused[s],tokenizer.vocab_size()-259);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(split: CorpusSplit, source: &[u8]) -> RustRecord {
        RustRecord {
            split,
            project: "p".into(),
            label: 0,
            source: source.into(),
            source_hash: [0; 32],
        }
    }
    #[test]
    fn counts_actual_merges_bytes_and_excludes_framing() {
        let tokenizer = BpeTokenizer::from_merges(20, &[(97, 98)]).unwrap();
        let usage = utilization(
            &[
                row(CorpusSplit::Train, b"abab"),
                row(CorpusSplit::Test, b"abz"),
            ],
            &tokenizer,
        )
        .unwrap();
        assert_eq!(usage.counts[259], [2, 0, 1]);
        assert_eq!(usage.counts[122], [0, 0, 1]);
        assert!(usage.counts[256..259].iter().all(|c| *c == [0; 3]));
        assert_eq!(
            usage
                .counts
                .iter()
                .map(|c| c.iter().sum::<usize>())
                .sum::<usize>(),
            4
        );
    }
    #[test]
    fn refusals_do_not_silently_contribute_truncated_counts() {
        let tokenizer = BpeTokenizer::from_merges(3, &[]).unwrap();
        let usage = utilization(
            &[
                row(CorpusSplit::Train, b"abc"),
                row(CorpusSplit::Validation, b"x"),
            ],
            &tokenizer,
        )
        .unwrap();
        assert_eq!(usage.accepted, [0, 1, 0]);
        assert_eq!(usage.refused, [1, 0, 0]);
        assert!(usage.counts.iter().all(|c| c[0] == 0));
        assert_eq!(usage.counts[120], [0, 1, 0]);
    }
    #[test]
    fn held_out_rows_cannot_train_merges() {
        let rows = [
            row(CorpusSplit::Train, b"abababab"),
            row(CorpusSplit::Test, b"zzzzzzzz"),
        ];
        assert_eq!(
            train(&rows, 270, 64).unwrap(),
            train(&rows[..1], 270, 64).unwrap()
        );
    }
}
