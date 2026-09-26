//! Matched diagnostic byte/BPE probe; no runtime activation or expert claim.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{checkpoint_hash, encode_checkpoint, load_checkpoint},
    bpe_cognitive::BpeCognitiveModel,
    bpe_tokenizer::BpeTokenizer,
    ByteTokenizer,
};
use cogno_scirust::*;
use std::io::Write;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: bpe_rust_probe [NEW_OUTPUT_DIRECTORY]".into());
    }
    let output = args.first().map(std::path::PathBuf::from);
    if let Some(dir) = &output {
        std::fs::create_dir(dir).map_err(|e| format!("new output directory: {e}"))?;
    }
    let mut inventory = String::from("seed\tfile\tbytes\tsha256\n");
    let corpus = include_str!("../../../experiments/rust-expert-pilot/corpus.tsv");
    let rows: Vec<_> = corpus
        .lines()
        .map(|line| {
            let f: Vec<_> = line.splitn(4, '\t').collect();
            (
                f[0],
                f[1],
                f[2].parse::<usize>().expect("frozen fixture label"),
                f[3].as_bytes(),
            )
        })
        .collect();
    let train: Vec<_> = rows
        .iter()
        .filter(|r| r.0 == "train")
        .map(|r| r.3)
        .collect();
    let bpe = BpeTokenizer::train(&train, 384, 128).map_err(|e| format!("{e:?}"))?;
    let restored = BpeTokenizer::from_bytes(&bpe.to_bytes()).map_err(|e| format!("{e:?}"))?;
    if restored != bpe {
        return Err("BPE artifact mismatch".into());
    }
    eprintln!(
        "bpe_vocab={} bpe_sha256={:02x?} bpe_artifact_hex={}",
        bpe.vocab_size(),
        bpe.fingerprint(),
        bpe.to_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let byte = ByteTokenizer::try_new(128).map_err(|e| format!("{e:?}"))?;
    println!("arm,seed,stage,split,family,target,prediction,p_compile,tokens,parameters");
    for arm in ["byte", "bpe"] {
        let encode = |payload: &[u8]| -> Result<Vec<u16>, String> {
            if arm == "bpe" {
                let ids = bpe.encode(payload).map_err(|e| format!("{e:?}"))?;
                if bpe.decode(&ids).map_err(|e| format!("{e:?}"))? != payload {
                    return Err("byte reconstruction mismatch".into());
                }
                Ok(ids)
            } else {
                byte.encode(payload).map_err(|e| format!("{e:?}"))
            }
        };
        // Whole-corpus length validation before any optimizer; evaluation rows
        // are only encoded, never used for BPE merges or model gradients.
        let encoded = rows
            .iter()
            .map(|r| encode(r.3))
            .collect::<Result<Vec<_>, _>>()?;
        let a = encode(b"a")?;
        let b = encode(b"b")?;
        let pair = if arm == "bpe" {
            bpe.encode_pair(b"a", b"b").map_err(|e| format!("{e:?}"))?
        } else {
            byte.encode_pair(b"a", b"b").map_err(|e| format!("{e:?}"))?
        };
        for seed in [1, 7, 42] {
            let mut model = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: if arm == "bpe" { bpe.vocab_size() } else { 259 },
                    max_tokens: 128,
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
            let emit = |model: &SequenceCognitiveHeads, stage: &str| -> Result<(), String> {
                for (r, tokens) in rows.iter().zip(&encoded) {
                    let p = model
                        .classification_probabilities(tokens)
                        .map_err(|e| format!("{e:?}"))?;
                    println!(
                        "{arm},{seed},{stage},{},{},{},{},{},{},{}",
                        r.0,
                        r.1,
                        r.2,
                        usize::from(p[1] > p[0]),
                        p[1],
                        tokens.len(),
                        model.parameter_count()
                    );
                }
                Ok(())
            };
            emit(&model, "initial")?;
            let mut optimizer =
                SequenceCognitiveAdamW::try_new(0.003, &model).map_err(|e| format!("{e:?}"))?;
            let candidates: [&[u16]; 2] = [&a, &b];
            for _ in 0..24 {
                for (r, tokens) in rows.iter().zip(&encoded).filter(|(r, _)| r.0 == "train") {
                    let batch = SequenceCognitiveBatch {
                        classification: CognitiveClassification {
                            token_ids: tokens,
                            target_class: r.2,
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
            if arm == "bpe" {
                let frozen =
                    BpeCognitiveModel::from_heads(bpe.clone(), model, bpe.fingerprint(), 2)
                        .map_err(|e| format!("{e:?}"))?;
                let bytes = encode_checkpoint(&frozen);
                let hash = checkpoint_hash(&bytes);
                let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
                let filename = format!("bpe-seed-{seed}.cbpc");
                let persisted = if let Some(dir) = &output {
                    let path = dir.join(&filename);
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map_err(|e| e.to_string())?;
                    file.write_all(&bytes).map_err(|e| e.to_string())?;
                    file.sync_all().map_err(|e| e.to_string())?;
                    std::fs::read(path).map_err(|e| e.to_string())?
                } else {
                    bytes.clone()
                };
                let restored = load_checkpoint(&persisted, hash).map_err(|e| format!("{e:?}"))?;
                if restored != frozen {
                    return Err("trained checkpoint state mismatch".into());
                }
                for r in &rows {
                    let observe = |m: &BpeCognitiveModel| {
                        Ok::<_, cogno_model::bpe_cognitive::BpeCognitiveError>((
                            m.classify(r.3)?,
                            m.preference(r.3)?,
                            m.symbolic(r.3)?,
                            m.contradiction(r.3, b"ab")?,
                            m.retrieve(r.3, &[b"a", b"b"])?,
                        ))
                    };
                    if observe(&frozen).map_err(|e| format!("{e:?}"))?
                        != observe(&restored).map_err(|e| format!("{e:?}"))?
                    {
                        return Err("trained checkpoint signal mismatch".into());
                    }
                }
                emit(restored.heads(), "trained")?;
                inventory.push_str(&format!("{seed}\t{filename}\t{}\t{hex}\n", bytes.len()));
                eprintln!(
                    "checkpoint seed={seed} bytes={} sha256={hex} all_signals_equal=true",
                    bytes.len()
                );
            } else {
                emit(&model, "trained")?;
            }
        }
    }
    if let Some(dir) = output {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("checkpoints.tsv"))
            .map_err(|e| e.to_string())?;
        file.write_all(inventory.as_bytes())
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
    }
    Ok(())
}
