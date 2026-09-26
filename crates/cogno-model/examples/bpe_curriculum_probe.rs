//! Preregistered research comparison: full BPE versus partial-merge training exposure.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{checkpoint_hash, encode_checkpoint, load_checkpoint},
    bpe_cognitive::BpeCognitiveModel,
    bpe_tokenizer::BpeTokenizer,
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
    training_order::epoch_order,
};
use cogno_scirust::*;
use std::collections::BTreeSet;
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn prefix(arm: &str, epoch: usize, merges: usize) -> usize {
    match arm {
        "full" => merges,
        "byte_mix" => {
            if epoch.is_multiple_of(2) {
                merges
            } else {
                0
            }
        }
        "half_mix" => {
            if epoch.is_multiple_of(2) {
                merges
            } else {
                merges / 2
            }
        }
        "cycle_mix" => [0, merges / 2, merges][epoch % 3],
        _ => unreachable!("fixed internal arms"),
    }
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: bpe_curriculum_probe CORPUS CORPUS_SHA NEW_OUTPUT_DIR".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[1]).map_err(|e| e.to_string())?,
        parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let rows = corpus.records();
    let train: Vec<_> = rows
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .collect();
    // Fixed campaign budget, not a general-purpose unbounded training entry point.
    if train.len() != 64
        || rows.len() != 96
        || rows
            .iter()
            .filter(|r| r.split == CorpusSplit::Validation)
            .count()
            != 16
    {
        return Err("fixed campaign requires 64/16/16 rows".into());
    }
    let sources: Vec<_> = train.iter().map(|r| r.source.as_slice()).collect();
    let tokenizer = BpeTokenizer::train(&sources, 384, 512).map_err(|e| format!("{e:?}"))?;
    let merges = tokenizer.vocab_size() - 259;
    // Every training representation admitted before any optimizer; no truncation.
    for source in &sources {
        for count in [0, merges / 2, merges] {
            tokenizer
                .encode_with_merge_prefix(source, count)
                .map_err(|e| format!("training capacity: {e:?}"))?;
        }
    }
    let out = std::path::Path::new(&args[3]);
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut inventory =
        String::from("arm,seed,file,sha256,parameters,tokenizer_sha256,corpus_sha256\n");
    let mut predictions =
        String::from("arm,seed,split,source_sha256,target,prediction,p_compile,tokens\n");
    let mut exposure = String::from(
        "arm,seed,split,emitted_tokens,unseen_in_train_occurrences,train_active_ids\n",
    );
    for arm in ["full", "byte_mix", "half_mix", "cycle_mix"] {
        for seed in [1, 7, 42] {
            let mut model = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: tokenizer.vocab_size(),
                    max_tokens: 512,
                    embedding_dim: 8,
                    hidden_dim: 16,
                    seed,
                },
                num_classes: 2,
                num_rules: 1,
                classification_seed: 1,
                preference_seed: 2,
                symbolic_seed: 3,
                contradiction_seed: 4,
            })
            .map_err(|e| format!("{e:?}"))?;
            let mut optimizer =
                SequenceCognitiveAdamW::try_new(0.003, &model).map_err(|e| format!("{e:?}"))?;
            let mut active = BTreeSet::new();
            for epoch in 0..24 {
                let count = prefix(arm, epoch, merges);
                for idx in
                    epoch_order(train.len(), seed, epoch as u64).map_err(|e| format!("{e:?}"))?
                {
                    let tokens = tokenizer
                        .encode_with_merge_prefix(&train[idx].source, count)
                        .map_err(|e| format!("{e:?}"))?;
                    active.extend(tokens[1..tokens.len() - 1].iter().copied());
                    model
                        .train_classification_step(
                            &mut optimizer,
                            CognitiveClassification {
                                token_ids: &tokens,
                                target_class: train[idx].label,
                            },
                        )
                        .map_err(|e| format!("{e:?}"))?;
                }
            }
            let model =
                BpeCognitiveModel::from_heads(tokenizer.clone(), model, tokenizer.fingerprint(), 2)
                    .map_err(|e| format!("{e:?}"))?;
            let bytes = encode_checkpoint(&model);
            let hash = checkpoint_hash(&bytes);
            let file = format!("{arm}-seed-{seed}.cbpc");
            std::fs::write(out.join(&file), &bytes).map_err(|e| e.to_string())?;
            let restored = load_checkpoint(
                &std::fs::read(out.join(&file)).map_err(|e| e.to_string())?,
                hash,
            )
            .map_err(|e| format!("{e:?}"))?;
            if restored != model {
                return Err("checkpoint mismatch".into());
            }
            inventory.push_str(&format!(
                "{arm},{seed},{file},{},{},{},{}\n",
                hex(&hash),
                model.heads().parameter_count(),
                hex(&tokenizer.fingerprint()),
                hex(&corpus.hash())
            ));
            for (split, name) in [
                (CorpusSplit::Train, "train"),
                (CorpusSplit::Validation, "validation"),
                (CorpusSplit::Test, "test"),
            ] {
                let (mut total, mut unseen) = (0, 0);
                for r in rows.iter().filter(|r| r.split == split) {
                    let ids = tokenizer
                        .encode(&r.source)
                        .map_err(|e| format!("evaluation capacity: {e:?}"))?;
                    let p = restored.classify(&r.source).map_err(|e| format!("{e:?}"))?;
                    let payload = &ids[1..ids.len() - 1];
                    total += payload.len();
                    unseen += payload.iter().filter(|id| !active.contains(id)).count();
                    predictions.push_str(&format!(
                        "{arm},{seed},{name},{},{},{},{},{}\n",
                        hex(&r.source_hash),
                        r.label,
                        usize::from(p[1] > p[0]),
                        p[1],
                        ids.len()
                    ));
                }
                exposure.push_str(&format!(
                    "{arm},{seed},{name},{total},{unseen},{}\n",
                    active.len()
                ));
            }
            eprintln!("completed arm={arm} seed={seed}");
        }
    }
    for (name, data) in [
        ("checkpoints.csv", inventory),
        ("predictions.csv", predictions),
        ("exposure.csv", exposure),
    ] {
        std::fs::write(out.join(name), data).map_err(|e| e.to_string())?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_schedules_have_declared_exposures() {
        for m in [0, 1, 125] {
            assert!((0usize..24).all(|e| prefix("full", e, m) == m));
            assert_eq!(
                (0usize..24)
                    .filter(|&e| !e.is_multiple_of(2) && prefix("byte_mix", e, m) == 0)
                    .count(),
                12
            );
            for e in 0usize..24 {
                assert_eq!(prefix("cycle_mix", e, m), [0, m / 2, m][e % 3]);
                assert_eq!(
                    prefix("half_mix", e, m),
                    if e.is_multiple_of(2) { m } else { m / 2 }
                );
            }
        }
    }
}
