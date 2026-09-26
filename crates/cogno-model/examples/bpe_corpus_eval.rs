//! Read-only checkpoint evaluation in a fresh process. Never trains or executes corpus code.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bpe_corpus_eval CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA train|validation|test".into());
    }
    let split = match a[5].as_str() {
        "train" => CorpusSplit::Train,
        "validation" => CorpusSplit::Validation,
        "test" => CorpusSplit::Test,
        _ => return Err("unknown split".into()),
    };
    let model_hash = parse_hash(&a[2]).map_err(|e| format!("{e:?}"))?;
    let corpus_hash = parse_hash(&a[4]).map_err(|e| format!("{e:?}"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(&a[1])
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let model = load_checkpoint(&bytes, model_hash).map_err(|e| format!("{e:?}"))?;
    if model.heads().config().num_classes != 2 {
        return Err("binary classification required".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&a[3]).map_err(|e| e.to_string())?,
        corpus_hash,
    )
    .map_err(|e| format!("{e:?}"))?;
    // Compute all requested results before emitting any CSV; never silently skip overlength inputs.
    let mut rows = Vec::new();
    for r in corpus.records().iter().filter(|r| r.split == split) {
        let p = model.classify(&r.source).map_err(|e| format!("{e:?}"))?;
        let hash = r
            .source_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        rows.push(format!(
            "{},{},{},{},{},{}",
            a[5],
            r.project,
            hash,
            r.label,
            usize::from(p[1] > p[0]),
            p[1]
        ));
    }
    println!("split,project,source_sha256,target,prediction,p_compile");
    for row in rows {
        println!("{row}");
    }
    Ok(())
}
