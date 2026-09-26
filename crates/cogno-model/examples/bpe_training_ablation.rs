//! Fixed research ablation; never activates a production checkpoint.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{checkpoint_hash, encode_checkpoint, load_checkpoint},
    bpe_cognitive::BpeCognitiveModel,
    bpe_tokenizer::BpeTokenizer,
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
    training_order::epoch_order,
};
use cogno_scirust::*;
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: bpe_training_ablation CORPUS CORPUS_SHA NEW_OUTPUT_DIR".into());
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
    let sources: Vec<_> = train.iter().map(|r| r.source.as_slice()).collect();
    let out = std::path::Path::new(&args[3]);
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut inventory =
        String::from("context,order,seed,file,sha256,parameters,tokenizer_sha256,corpus_sha256\n");
    let mut predictions = String::from(
        "context,order,seed,split,source_sha256,target,prediction,p_compile,tokens,parameters\n",
    );
    for context in [128, 256] {
        let bpe = BpeTokenizer::train(&sources, 384, context).map_err(|e| format!("{e:?}"))?;
        let encoded: Vec<_> = train
            .iter()
            .map(|r| bpe.encode(&r.source).map_err(|e| format!("{e:?}")))
            .collect::<Result<_, _>>()?;
        let a = bpe.encode(b"a").map_err(|e| format!("{e:?}"))?;
        let b = bpe.encode(b"b").map_err(|e| format!("{e:?}"))?;
        let pair = bpe.encode_pair(b"a", b"b").map_err(|e| format!("{e:?}"))?;
        let candidates: [&[u16]; 2] = [&a, &b];
        for order in ["ordered", "shuffled"] {
            for seed in [1, 7, 42] {
                let mut model = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
                    encoder: SequenceEncoderConfig {
                        vocab_size: bpe.vocab_size(),
                        max_tokens: context,
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
                for epoch in 0..24 {
                    let indices = if order == "shuffled" {
                        epoch_order(train.len(), seed, epoch).map_err(|e| format!("{e:?}"))?
                    } else {
                        (0..train.len()).collect()
                    };
                    for idx in indices {
                        let r = train[idx];
                        let tokens = &encoded[idx];
                        let batch = SequenceCognitiveBatch {
                            classification: CognitiveClassification {
                                token_ids: tokens,
                                target_class: r.label,
                            },
                            preference: CognitivePreference {
                                preferred: &a,
                                dispreferred: &b,
                                margin: 1.0,
                            },
                            symbolic: CognitiveSymbolic {
                                token_ids: &a,
                                targets: &[1.0],
                            },
                            contradiction: CognitiveContradiction {
                                pair_token_ids: &pair,
                                contradicts: false,
                            },
                            retrieval: CognitiveRetrieval {
                                query: &a,
                                candidates: &candidates,
                                positive_idx: 0,
                                temperature: 0.2,
                            },
                        };
                        model
                            .train_joint_step(
                                &mut optimizer,
                                batch,
                                SequenceCognitiveLossWeights {
                                    classification: 1.0,
                                    preference: 0.0,
                                    symbolic: 0.0,
                                    contradiction: 0.0,
                                    retrieval: 0.0,
                                },
                            )
                            .map_err(|e| format!("{e:?}"))?;
                    }
                }

                let frozen =
                    BpeCognitiveModel::from_heads(bpe.clone(), model, bpe.fingerprint(), 2)
                        .map_err(|e| format!("{e:?}"))?;
                let bytes = encode_checkpoint(&frozen);
                let hash = checkpoint_hash(&bytes);
                let file = format!("context-{context}-{order}-seed-{seed}.cbpc");
                std::fs::write(out.join(&file), &bytes).map_err(|e| e.to_string())?;
                let restored = load_checkpoint(
                    &std::fs::read(out.join(&file)).map_err(|e| e.to_string())?,
                    hash,
                )
                .map_err(|e| format!("{e:?}"))?;
                if restored != frozen {
                    return Err("checkpoint roundtrip mismatch".into());
                }
                let parameters = restored.heads().parameter_count();
                inventory.push_str(&format!(
                    "{context},{order},{seed},{file},{},{parameters},{},{}\n",
                    hex(&hash),
                    hex(&bpe.fingerprint()),
                    hex(&corpus.hash())
                ));
                for r in rows {
                    let split = match r.split {
                        CorpusSplit::Train => "train",
                        CorpusSplit::Validation => "validation",
                        CorpusSplit::Test => "test",
                    };
                    let p = restored.classify(&r.source).map_err(|e| format!("{e:?}"))?;
                    predictions.push_str(&format!(
                        "{context},{order},{seed},{split},{},{},{},{},{},{parameters}\n",
                        hex(&r.source_hash),
                        r.label,
                        usize::from(p[1] > p[0]),
                        p[1],
                        bpe.required_tokens(&r.source)
                            .map_err(|e| format!("{e:?}"))?
                    ));
                }
                eprintln!("completed context={context} order={order} seed={seed}");
            }
        }
    }
    std::fs::write(out.join("checkpoints.csv"), inventory).map_err(|e| e.to_string())?;
    std::fs::write(out.join("predictions.csv"), predictions).map_err(|e| e.to_string())?;
    Ok(())
}
