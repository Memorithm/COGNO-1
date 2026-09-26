//! Frozen checkpoint evaluation on an external test-only panel; report capacity refusals.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    bpe_tokenizer::BpeError,
    rust_corpus::{parse_hash, RustCorpus},
};
use std::io::Read;
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: bpe_external_eval CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA".into());
    }
    let mh = parse_hash(&a[2]).map_err(|e| format!("{e:?}"))?;
    let ch = parse_hash(&a[4]).map_err(|e| format!("{e:?}"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(&a[1])
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let model = load_checkpoint(&bytes, mh).map_err(|e| format!("{e:?}"))?;
    if model.heads().config().num_classes != 2 {
        return Err("binary classifier required".into());
    }
    let corpus =
        RustCorpus::read_test_only(std::fs::File::open(&a[3]).map_err(|e| e.to_string())?, ch)
            .map_err(|e| format!("{e:?}"))?;
    let mut rows = Vec::new();
    for r in corpus.records() {
        let hash = r
            .source_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        match model.tokenizer().encode(&r.source) {
            Ok(ids) => {
                let p = model.classify(&r.source).map_err(|e| format!("{e:?}"))?;
                rows.push(format!(
                    "{hash},{},accepted,{},{},{}",
                    r.label,
                    usize::from(p[1] > p[0]),
                    p[1],
                    ids.len()
                ));
            }
            Err(BpeError::Capacity) => rows.push(format!("{hash},{},capacity,,,", r.label)),
            Err(e) => return Err(format!("{e:?}")),
        }
    }
    println!("source_sha256,target,status,prediction,p_compile,tokens");
    for row in rows {
        println!("{row}");
    }
    Ok(())
}
