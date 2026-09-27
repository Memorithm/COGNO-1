//! Fit a standalone CBPE0001 artifact from an admitted train-only corpus view.
#![forbid(unsafe_code)]
use cogno_model::bpe_tokenizer::BpeTokenizer;
use cogno_model::rust_corpus::{parse_hash, RustCorpus};
use std::io::Write;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: bpe_train_corpus CORPUS SHA256 VOCAB CONTEXT NEW_ARTIFACT".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[1]).map_err(|e| e.to_string())?,
        parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let train = corpus.training_view().map_err(|e| format!("{e:?}"))?;
    let tokenizer = BpeTokenizer::train_from_view(
        &train,
        args[3].parse().map_err(|_| "invalid vocab")?,
        args[4].parse().map_err(|_| "invalid context")?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])
        .map_err(|e| e.to_string())?;
    output
        .write_all(&tokenizer.to_bytes())
        .and_then(|()| output.sync_all())
        .map_err(|e| e.to_string())?;
    let hash: String = tokenizer
        .fingerprint()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "corpus_sha256={} train_records={} vocabulary={} context={} tokenizer_sha256={hash}",
        args[2],
        train.records().len(),
        tokenizer.vocab_size(),
        tokenizer.max_tokens()
    );
    Ok(())
}
