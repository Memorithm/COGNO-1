//! Classification-only V4 diagnostic pilot, not an expert Rust checkpoint.
#![forbid(unsafe_code)]
use cogno_model::{
    encode_sequence_cognitive_artifact, load_sequence_cognitive_artifact, ByteTokenizer,
    SequenceCognitiveExample, SequenceCognitiveModelConfig, SequenceCognitiveTrainer,
};
use cogno_scirust::{SequenceCognitiveHeads, SequenceCognitiveLossWeights};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).ok_or("expected corpus.tsv output-directory")?;
    let output = Path::new(args.get(2).ok_or("expected output-directory")?);
    let content = fs::read_to_string(input).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut families = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    for line in content.lines() {
        let fields: Vec<_> = line.splitn(4, '\t').collect();
        if fields.len() != 4 || !["train", "validation", "test"].contains(&fields[0]) {
            return Err("invalid corpus record".into());
        }
        let label: usize = fields[2].parse().map_err(|_| "invalid label")?;
        if label > 1 || !seen.insert(fields[3]) {
            return Err("invalid label or duplicate source".into());
        }
        if let Some(previous) = families.insert(fields[1], fields[0])
            && previous != fields[0]
        {
            return Err("family leakage between splits".into());
        }
        rows.push((fields[0], fields[1], label, fields[3].as_bytes()));
    }
    for split in ["train", "validation", "test"] {
        for label in [0, 1] {
            if !rows.iter().any(|r| r.0 == split && r.2 == label) {
                return Err("each split must contain both classes".into());
            }
        }
    }
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    // Predeclared configuration: no selection or tuning on validation/test.
    // The other task slots satisfy the joint API but have ZERO loss weights.
    // Their outputs must not be used: shared encoder training changes them too.
    let examples: Vec<_> = rows
        .iter()
        .filter(|r| r.0 == "train")
        .map(|r| SequenceCognitiveExample {
            classification_payload: r.3.to_vec(),
            classification_target: r.2,
            preferred: b"a".to_vec(),
            dispreferred: b"b".to_vec(),
            symbolic_payload: b"a".to_vec(),
            rule_satisfied: vec![true],
            contradiction_left: b"a".to_vec(),
            contradiction_right: b"b".to_vec(),
            contradicts: false,
            retrieval_query: b"a".to_vec(),
            retrieval_candidates: vec![b"a".to_vec(), b"b".to_vec()],
            retrieval_positive_idx: 0,
        })
        .collect();
    println!("seed,stage,split,family,target,prediction,p_compile");
    for seed in [1, 7, 42] {
        let mut config = SequenceCognitiveModelConfig::default();
        config.cognitive.encoder.max_tokens = 128;
        config.cognitive.encoder.embedding_dim = 8;
        config.cognitive.encoder.hidden_dim = 16;
        config.cognitive.encoder.seed = seed;
        config.cognitive.num_classes = 2;
        config.cognitive.num_rules = 1;
        config.epochs = 24;
        config.learning_rate = 0.003;
        config.loss_weights = SequenceCognitiveLossWeights {
            classification: 1.0,
            preference: 0.0,
            symbolic: 0.0,
            contradiction: 0.0,
            retrieval: 0.0,
        };
        let initial =
            SequenceCognitiveHeads::try_new(config.cognitive).map_err(|e| format!("{e:?}"))?;
        let tokenizer = ByteTokenizer::try_new(128).map_err(|e| format!("{e:?}"))?;
        // Preflight all lengths; no truncation and no test-gradient exposure.
        for r in &rows {
            let tokens = tokenizer.encode(r.3).map_err(|e| format!("{e:?}"))?;
            let p = initial
                .classification_probabilities(&tokens)
                .map_err(|e| format!("{e:?}"))?;
            println!(
                "{seed},initial,{},{},{},{},{}",
                r.0,
                r.1,
                r.2,
                usize::from(p[1] > p[0]),
                p[1]
            );
        }
        let trainer = SequenceCognitiveTrainer::try_new(config).map_err(|e| format!("{e:?}"))?;
        let (model, report) = trainer.train(&examples).map_err(|e| format!("{e:?}"))?;
        eprintln!("seed={seed} report={report:?}");
        for r in &rows {
            let p = model
                .classification_probabilities(r.3)
                .map_err(|e| format!("{e:?}"))?;
            println!(
                "{seed},trained,{},{},{},{},{}",
                r.0,
                r.1,
                r.2,
                usize::from(p[1] > p[0]),
                p[1]
            );
        }
        let artifact =
            encode_sequence_cognitive_artifact(model.heads(), model.max_retrieval_candidates())
                .map_err(|e| format!("{e:?}"))?;
        let restored = load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes)
            .map_err(|e| format!("{e:?}"))?;
        if restored.heads() != model.heads() {
            return Err("checkpoint round-trip mismatch".into());
        }
        let hex: String = artifact.bytes.iter().map(|b| format!("{b:02x}")).collect();
        fs::write(output.join(format!("seed-{seed}.cog4.hex")), hex).map_err(|e| e.to_string())?;
        fs::write(output.join(format!("seed-{seed}.manifest.txt")), format!("EXPERIMENTAL; CLASSIFICATION ONLY; NOT ACTIVATED\ncorpus_sha256={:x}\nartifact_sha256={:x}\n{:#?}\n{report:#?}\n", Sha256::digest(content.as_bytes()), Sha256::digest(&artifact.bytes), artifact.manifest)).map_err(|e| e.to_string())?;
    }
    Ok(())
}
