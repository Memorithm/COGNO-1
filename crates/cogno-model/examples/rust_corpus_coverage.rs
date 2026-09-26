//! Corpus coverage diagnostic; learns merges only from the train partition.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_tokenizer::BpeTokenizer,
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
    ByteTokenizer,
};
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: rust_corpus_coverage CORPUS EXPECTED_SHA256".into());
    }
    let hash = parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?;
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[1]).map_err(|e| e.to_string())?,
        hash,
    )
    .map_err(|e| format!("{e:?}"))?;
    let train: Vec<_> = corpus
        .records()
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .map(|r| r.source.as_slice())
        .collect();
    let bpe = BpeTokenizer::train(&train, 384, 128).map_err(|e| format!("{e:?}"))?;
    let byte = ByteTokenizer::try_new(128).map_err(|e| format!("{e:?}"))?;
    eprintln!(
        "bpe_vocab={} tokenizer_sha256={:02x?}",
        bpe.vocab_size(),
        bpe.fingerprint()
    );
    println!("split,examples,source_bytes,byte_tokens,bpe_tokens,byte_rejected,bpe_rejected");
    for (split, name) in [
        (CorpusSplit::Train, "train"),
        (CorpusSplit::Validation, "validation"),
        (CorpusSplit::Test, "test"),
    ] {
        let (mut n, mut bytes, mut bt, mut pt, mut br, mut pr) = (0, 0, 0, 0, 0, 0);
        for r in corpus.records().iter().filter(|r| r.split == split) {
            n += 1;
            bytes += r.source.len();
            match byte.encode(&r.source) {
                Ok(ids) => bt += ids.len(),
                Err(_) => br += 1,
            }
            match bpe.encode(&r.source) {
                Ok(ids) => {
                    if bpe.decode(&ids).map_err(|e| format!("{e:?}"))? != r.source {
                        return Err("lossless check".into());
                    }
                    pt += ids.len();
                }
                Err(_) => pr += 1,
            }
        }
        println!("{name},{n},{bytes},{bt},{pt},{br},{pr}");
    }
    Ok(())
}
