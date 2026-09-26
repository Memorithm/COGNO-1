//! Train-only BPE context diagnostics; no model fitting or truncation.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_tokenizer::{BpeError, BpeTokenizer},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord},
};
fn train(records: &[RustRecord], vocab: usize, context: usize) -> Result<BpeTokenizer, BpeError> {
    let inputs: Vec<_> = records
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .map(|r| r.source.as_slice())
        .collect();
    BpeTokenizer::train(&inputs, vocab, context)
}
fn measure(tokenizer: &BpeTokenizer, source: &[u8]) -> Result<(usize, bool), BpeError> {
    let required = tokenizer.required_tokens(source)?;
    match tokenizer.encode(source) {
        Ok(ids) if ids.len() == required => Ok((required, true)),
        Err(BpeError::Capacity) if required > tokenizer.max_tokens() => Ok((required, false)),
        Err(e) => Err(e),
        _ => Err(BpeError::InvalidArtifact),
    }
}
fn hex(hash: &[u8]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}
fn report(
    corpus: &RustCorpus,
    tokenizer: &BpeTokenizer,
    training_hash: &str,
) -> Result<Vec<String>, String> {
    corpus
        .records()
        .iter()
        .map(|r| {
            let (required, accepted) =
                measure(tokenizer, &r.source).map_err(|e| format!("{e:?}"))?;
            let split = match r.split {
                CorpusSplit::Train => "train",
                CorpusSplit::Validation => "validation",
                CorpusSplit::Test => "test",
            };
            Ok(format!(
                "{},{},{},{},{},{},{},{},{},{},{}",
                training_hash,
                hex(&corpus.hash()),
                hex(&tokenizer.fingerprint()),
                tokenizer.vocab_size(),
                tokenizer.max_tokens(),
                hex(&r.source_hash),
                split,
                r.source.len(),
                required,
                if accepted { "accepted" } else { "capacity" },
                r.label
            ))
        })
        .collect()
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 && a.len() != 7 {
        return Err(
            "usage: rust_bpe_coverage FULL_CORPUS SHA VOCAB CONTEXT [TEST_ONLY_CORPUS SHA]".into(),
        );
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
    let mut rows = report(&corpus, &tokenizer, &a[2])?;
    if a.len() == 7 {
        let external = RustCorpus::read_test_only(
            std::fs::File::open(&a[5]).map_err(|e| e.to_string())?,
            parse_hash(&a[6]).map_err(|e| format!("{e:?}"))?,
        )
        .map_err(|e| format!("{e:?}"))?;
        rows.extend(report(&external, &tokenizer, &a[2])?);
    }
    println!("training_corpus_sha256,corpus_sha256,tokenizer_sha256,vocab,context,source_sha256,split,source_bytes,required_tokens,status,label");
    for row in rows {
        println!("{row}");
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
    fn evaluation_content_never_changes_training() {
        let mut rows = vec![
            row(CorpusSplit::Train, b"abababab"),
            row(CorpusSplit::Validation, b"xyxyxyxy"),
            row(CorpusSplit::Test, b"zzzz"),
        ];
        let before = train(&rows, 270, 32).unwrap();
        rows[1].source = b"totally changed validation".to_vec();
        rows[2].source = b"changed test".to_vec();
        assert_eq!(
            before.fingerprint(),
            train(&rows, 270, 32).unwrap().fingerprint()
        );
    }
    #[test]
    fn refusal_reports_complete_length_without_truncation() {
        let tokenizer = BpeTokenizer::from_merges(3, &[]).unwrap();
        assert_eq!(measure(&tokenizer, b"a"), Ok((3, true)));
        assert_eq!(measure(&tokenizer, b"abcdef"), Ok((8, false)));
        assert_eq!(
            measure(&tokenizer, &vec![0; 16_385]),
            Err(BpeError::Capacity)
        );
    }
    #[test]
    fn missing_train_is_rejected() {
        assert_eq!(
            train(&[row(CorpusSplit::Test, b"abc")], 259, 3),
            Err(BpeError::EmptyTraining)
        );
    }
}
