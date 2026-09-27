//! Bounded research runner support. No production model activation.
use cogno_model::{
    bpe_tokenizer::BpeTokenizer,
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
};

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}
pub fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err("file capacity exceeded".into());
    }
    Ok(bytes)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderGraph {
    Dense,
    Gather,
}
impl EncoderGraph {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Gather => "gather",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Protocol {
    pub corpus_sha256: String,
    pub provenance_sha256: String,
    pub seeds: Vec<u64>,
    pub arms: Vec<String>,
    pub epochs: usize,
    pub vocab: usize,
    pub context: usize,
    pub embedding: usize,
    pub hidden: usize,
    pub batch: usize,
    pub learning_rate: f32,
    pub max_updates: usize,
    /// None denotes the unchanged v2 protocol and its dense reference graph.
    pub encoder_graph: Option<EncoderGraph>,
}
impl Protocol {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        if bytes.len() > 4096 || !text.ends_with('\n') {
            return Err("protocol must be bounded newline-terminated UTF-8".into());
        }
        let mut fields = BTreeMap::new();
        for line in text.lines() {
            let (key, value) = line
                .split_once('\t')
                .ok_or("protocol requires key TAB value")?;
            if value.contains(['\t', '\r']) || fields.insert(key, value).is_some() {
                return Err("duplicate/invalid protocol field".into());
            }
        }
        let keys = [
            "version",
            "corpus_sha256",
            "provenance_sha256",
            "seeds",
            "arms",
            "epochs",
            "vocab",
            "context",
            "embedding",
            "hidden",
            "batch",
            "learning_rate",
            "max_updates",
        ];
        let graph = match fields.get("version").copied() {
            Some("rust-train-v2") => None,
            Some("rust-train-v3") => Some(match fields.get("encoder_graph").copied() {
                Some("dense") => EncoderGraph::Dense,
                Some("gather") => EncoderGraph::Gather,
                _ => return Err("v3 requires encoder_graph dense or gather".into()),
            }),
            _ => return Err("unknown protocol version".into()),
        };
        if fields.len() != keys.len() + usize::from(graph.is_some())
            || keys.iter().any(|k| !fields.contains_key(k))
        {
            return Err("unknown/missing protocol fields/version".into());
        }
        for k in ["corpus_sha256", "provenance_sha256"] {
            parse_hash(fields[k]).map_err(|e| format!("{e:?}"))?;
        }
        let number = |key: &str, min, max| -> Result<usize, String> {
            let n = fields[key]
                .parse::<usize>()
                .map_err(|_| format!("invalid {key}"))?;
            if !(min..=max).contains(&n) {
                return Err(format!("out of range {key}"));
            }
            Ok(n)
        };
        let seeds = fields["seeds"]
            .split(',')
            .map(|x| x.parse::<u64>().map_err(|_| "invalid seed".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        let arms: Vec<_> = fields["arms"].split(',').map(str::to_owned).collect();
        if seeds.is_empty()
            || seeds.len() > 8
            || seeds.iter().collect::<BTreeSet<_>>().len() != seeds.len()
            || arms.is_empty()
            || arms.len() > 4
            || arms.iter().collect::<BTreeSet<_>>().len() != arms.len()
            || arms
                .iter()
                .any(|a| !["full", "byte_mix", "half_mix", "cycle_mix"].contains(&a.as_str()))
        {
            return Err("invalid/duplicate seeds or arms".into());
        }
        let learning_rate = fields["learning_rate"]
            .parse::<f32>()
            .map_err(|_| "invalid learning_rate")?;
        if !learning_rate.is_finite()
            || !(0.0..=0.1).contains(&learning_rate)
            || learning_rate == 0.0
        {
            return Err("invalid learning_rate".into());
        }
        Ok(Self {
            corpus_sha256: fields["corpus_sha256"].into(),
            provenance_sha256: fields["provenance_sha256"].into(),
            seeds,
            arms,
            epochs: number("epochs", 1, 100)?,
            vocab: number("vocab", 259, 512)?,
            context: number("context", 3, 512)?,
            embedding: number("embedding", 1, 32)?,
            hidden: number("hidden", 1, 64)?,
            batch: number("batch", 1, 32)?,
            learning_rate,
            max_updates: number("max_updates", 1, 1_000_000)?,
            encoder_graph: graph,
        })
    }
}
pub fn prefix(arm: &str, epoch: usize, merges: usize) -> usize {
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
        _ => unreachable!("validated arm"),
    }
}
pub struct Admission {
    pub protocol: Protocol,
    pub protocol_bytes: Vec<u8>,
    pub corpus: RustCorpus,
    pub tokenizer: BpeTokenizer,
    pub updates: usize,
}
pub fn admit(args: &[String]) -> Result<Admission, String> {
    if args.len() != 4 {
        return Err("expected PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE".into());
    }
    let protocol_bytes = read_bounded(Path::new(&args[0]), 4096)?;
    if digest(&protocol_bytes) != args[1] {
        return Err("protocol digest mismatch".into());
    }
    let protocol = Protocol::parse(&protocol_bytes)?;
    let provenance = read_bounded(Path::new(&args[3]), 8 * 1024 * 1024)?;
    if digest(&provenance) != protocol.provenance_sha256 {
        return Err("provenance digest mismatch".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[2]).map_err(|e| e.to_string())?,
        parse_hash(&protocol.corpus_sha256).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let sources: Vec<_> = corpus
        .records()
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .map(|r| r.source.as_slice())
        .collect();
    let updates = sources.len().div_ceil(protocol.batch)
        * protocol.epochs
        * protocol.seeds.len()
        * protocol.arms.len();
    if updates > protocol.max_updates {
        return Err(format!("update budget exceeded: {updates}"));
    }
    let tokenizer = BpeTokenizer::train(&sources, protocol.vocab, protocol.context)
        .map_err(|e| format!("{e:?}"))?;
    let merges = tokenizer.vocab_size() - 259;
    for source in &sources {
        for arm in &protocol.arms {
            for epoch in 0..protocol.epochs {
                tokenizer
                    .encode_with_merge_prefix(source, prefix(arm, epoch, merges))
                    .map_err(|e| format!("train capacity: {e:?}"))?;
            }
        }
    }
    for row in corpus.records() {
        tokenizer
            .encode(&row.source)
            .map_err(|e| format!("evaluation capacity: {e:?}"))?;
    }
    Ok(Admission {
        protocol,
        protocol_bytes,
        corpus,
        tokenizer,
        updates,
    })
}
pub fn plan(a: &Admission) -> String {
    let mut output = format!("protocol_sha256\t{}\ncorpus_sha256\t{}\nprovenance_sha256\t{}\nrows\t{}\nupdates\t{}\nruns\t{}\nepochs\t{}\nbatch\t{}\nembedding\t{}\nhidden\t{}\nlearning_rate\t{}\ntokenizer_sha256\t{}\n", digest(&a.protocol_bytes), a.protocol.corpus_sha256, a.protocol.provenance_sha256, a.corpus.records().len(), a.updates, a.protocol.seeds.len() * a.protocol.arms.len(), a.protocol.epochs, a.protocol.batch, a.protocol.embedding, a.protocol.hidden, a.protocol.learning_rate, hex(&a.tokenizer.fingerprint()));
    if let Some(graph) = a.protocol.encoder_graph {
        output.push_str(&format!("encoder_graph\t{}\n", graph.name()));
    }
    output
}
#[cfg(test)]
mod tests {
    use super::*;
    fn protocol() -> String {
        format!("version\trust-train-v2\ncorpus_sha256\t{}\nprovenance_sha256\t{}\nseeds\t1,7,42\narms\tfull,cycle_mix\nepochs\t24\nvocab\t384\ncontext\t512\nembedding\t8\nhidden\t16\nbatch\t1\nlearning_rate\t0.003\nmax_updates\t100000\n", "0".repeat(64), "1".repeat(64))
    }
    #[test]
    fn strict_protocol_admission() {
        let p = protocol();
        assert!(Protocol::parse(p.as_bytes()).is_ok());
        for bad in [
            p.replace("0.003", "NaN"),
            p.replace("1,7,42", "1,1"),
            p.replace("full,cycle_mix", "unknown"),
            p.replace("context\t512", "context\t513"),
            format!("{p}epochs\t1\n"),
            p.replace("version\trust-train-v2", "version\tv3"),
        ] {
            assert!(Protocol::parse(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn graph_choice_is_explicit_versioned_and_hash_bound() {
        let v2 = protocol();
        assert_eq!(Protocol::parse(v2.as_bytes()).unwrap().encoder_graph, None);
        let v3 = v2.replace("rust-train-v2", "rust-train-v3");
        for (value, graph) in [
            ("dense", EncoderGraph::Dense),
            ("gather", EncoderGraph::Gather),
        ] {
            let candidate = format!("{v3}encoder_graph\t{value}\n");
            assert_eq!(
                Protocol::parse(candidate.as_bytes()).unwrap().encoder_graph,
                Some(graph)
            );
        }
        for invalid in [
            v3.clone(),
            format!("{v3}encoder_graph\tunknown\n"),
            format!("{v2}encoder_graph\tgather\n"),
            format!("{v3}encoder_graph\tgather\nencoder_graph\tdense\n"),
            format!("{v3}encoder_graph\tgather\nignored\ttrue\n"),
        ] {
            assert!(Protocol::parse(invalid.as_bytes()).is_err());
        }
    }
}

pub fn metrics(
    model: &cogno_scirust::SequenceCognitiveHeads,
    a: &Admission,
    split: CorpusSplit,
) -> Result<(usize, usize, f64), String> {
    let mut count = 0;
    let mut correct = 0;
    let mut nll = 0.0;
    for row in a.corpus.records().iter().filter(|r| r.split == split) {
        let tokens = a
            .tokenizer
            .encode(&row.source)
            .map_err(|e| format!("{e:?}"))?;
        let probs = model
            .classification_probabilities(&tokens)
            .map_err(|e| format!("{e:?}"))?;
        if probs.len() != 2
            || probs
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err("invalid predicted probabilities".into());
        }
        count += 1;
        correct += usize::from(usize::from(probs[1] > probs[0]) == row.label);
        nll -= f64::from(probs[row.label]).clamp(1e-7, 1.0 - 1e-7).ln();
    }
    if count == 0 {
        return Err("empty metric split".into());
    }
    Ok((count, correct, nll / count as f64))
}
pub fn train(a: &Admission, out: &Path, prior: Option<&Path>) -> Result<(), String> {
    use cogno_model::{
        bpe_checkpoint::encode_checkpoint, bpe_cognitive::BpeCognitiveModel,
        training_order::epoch_order,
    };
    use cogno_scirust::{
        CognitiveClassification, SequenceCognitiveAdamW, SequenceCognitiveConfig,
        SequenceCognitiveHeads, SequenceEncoderConfig,
    };
    use std::io::Write;
    if let Some(prior) = prior
        && (read_bounded(&prior.join("protocol.tsv"), 4096)? != a.protocol_bytes
            || read_bounded(&prior.join("plan.tsv"), 4096)? != plan(a).as_bytes())
    {
        return Err("resume protocol mismatch".into());
    }
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("protocol.tsv"), &a.protocol_bytes).map_err(|e| e.to_string())?;
    std::fs::write(out.join("plan.tsv"), plan(a)).map_err(|e| e.to_string())?;
    let p = &a.protocol;
    let rows: Vec<_> = a
        .corpus
        .records()
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .collect();
    let mut completed_updates = 0;
    for arm in &p.arms {
        for &seed in &p.seeds {
            if let Some(prior) = prior {
                let marker = prior.join(format!("{arm}-seed-{seed}.DONE"));
                if marker.try_exists().map_err(|e| e.to_string())? {
                    verify_run(a, prior, arm, seed)?;
                    for name in run_files(arm, seed)
                        .into_iter()
                        .chain([format!("{arm}-seed-{seed}.DONE")])
                    {
                        write_new(
                            &out.join(&name),
                            &read_bounded(&prior.join(&name), 8 * 1024 * 1024)?,
                        )?;
                    }
                    completed_updates += rows.len().div_ceil(p.batch) * p.epochs;
                    eprintln!("reused completed arm={arm} seed={seed}");
                    continue;
                }
            }
            let mut model = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
                encoder: SequenceEncoderConfig {
                    vocab_size: a.tokenizer.vocab_size(),
                    max_tokens: p.context,
                    embedding_dim: p.embedding,
                    hidden_dim: p.hidden,
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
            let mut epoch_log =
                std::fs::File::create(out.join(format!("{arm}-seed-{seed}.epochs.tsv")))
                    .map_err(|e| e.to_string())?;
            writeln!(epoch_log, "epoch\tupdates\ttrain_rows_seen\temitted_tokens\ttrain_count\ttrain_correct\ttrain_nll\tvalidation_count\tvalidation_correct\tvalidation_nll").map_err(|e| e.to_string())?;
            let mut run_updates = 0;
            let mut seen = 0;
            let mut emitted = 0;
            let mut optimizer = SequenceCognitiveAdamW::try_new(p.learning_rate, &model)
                .map_err(|e| format!("{e:?}"))?;
            for epoch in 0..p.epochs {
                let merge_count = prefix(arm, epoch, a.tokenizer.vocab_size() - 259);
                let encoded: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        a.tokenizer
                            .encode_with_merge_prefix(&row.source, merge_count)
                            .map_err(|e| format!("{e:?}"))
                    })
                    .collect::<Result<_, _>>()?;
                let order =
                    epoch_order(rows.len(), seed, epoch as u64).map_err(|e| format!("{e:?}"))?;
                for batch in order.chunks(p.batch) {
                    let examples: Vec<_> = batch
                        .iter()
                        .map(|&i| CognitiveClassification {
                            token_ids: &encoded[i],
                            target_class: rows[i].label,
                        })
                        .collect();
                    match (
                        p.encoder_graph.unwrap_or(EncoderGraph::Dense),
                        examples.len(),
                    ) {
                        (EncoderGraph::Dense, 1) => {
                            model.train_classification_step(&mut optimizer, examples[0])
                        }
                        (EncoderGraph::Dense, _) => {
                            model.train_classification_minibatch_step(&mut optimizer, &examples)
                        }
                        (EncoderGraph::Gather, 1) => {
                            model.train_classification_step_gather(&mut optimizer, examples[0])
                        }
                        (EncoderGraph::Gather, _) => model
                            .train_classification_minibatch_step_gather(&mut optimizer, &examples),
                    }
                    .map_err(|e| format!("{e:?}"))?;
                    completed_updates += 1;
                    run_updates += 1;
                    seen += batch.len();
                    emitted += batch.iter().map(|&i| encoded[i].len()).sum::<usize>();
                }
                let tr = metrics(&model, a, CorpusSplit::Train)?;
                let va = metrics(&model, a, CorpusSplit::Validation)?;
                writeln!(
                    epoch_log,
                    "{}\t{run_updates}\t{seen}\t{emitted}\t{}\t{}\t{}\t{}\t{}\t{}",
                    epoch + 1,
                    tr.0,
                    tr.1,
                    tr.2,
                    va.0,
                    va.1,
                    va.2
                )
                .map_err(|e| e.to_string())?;
                epoch_log.flush().map_err(|e| e.to_string())?;
            }
            let model = BpeCognitiveModel::from_heads(
                a.tokenizer.clone(),
                model,
                a.tokenizer.fingerprint(),
                2,
            )
            .map_err(|e| format!("{e:?}"))?;
            let bytes = encode_checkpoint(&model);
            std::fs::write(out.join(format!("{arm}-seed-{seed}.cbpc")), &bytes)
                .map_err(|e| e.to_string())?;
            let restored = load_run(a, out, arm, seed)?;
            if restored != model {
                return Err("checkpoint round-trip differs".into());
            }
            std::fs::write(
                out.join(format!("{arm}-seed-{seed}.predictions.tsv")),
                predictions(&restored, a, false)?,
            )
            .map_err(|e| e.to_string())?;
            let done = run_manifest(a, out, arm, seed)?;
            write_new(
                &out.join(format!("{arm}-seed-{seed}.DONE")),
                done.as_bytes(),
            )?;
            eprintln!(
                "completed arm={arm} seed={seed} updates={completed_updates} run_sha256={}",
                digest(done.as_bytes())
            );
        }
    }
    if completed_updates != a.updates {
        return Err("update accounting mismatch".into());
    }
    let complete = bundle(a, out)?;
    let hash = digest(complete.as_bytes());
    std::fs::write(out.join("COMPLETE"), complete).map_err(|e| e.to_string())?;
    verify(a, out, &hash)?;
    println!("COMPLETE_SHA256={hash}");
    Ok(())
}

fn graph_admission(a: &Admission, graph: EncoderGraph) -> Result<Admission, String> {
    let original = std::str::from_utf8(&a.protocol_bytes).map_err(|e| e.to_string())?;
    let mut protocol = String::new();
    for line in original.lines() {
        if line.starts_with("encoder_graph\t") {
            continue;
        }
        protocol.push_str(if line.starts_with("version\t") {
            "version\trust-train-v3"
        } else {
            line
        });
        protocol.push('\n');
    }
    protocol.push_str(&format!("encoder_graph\t{}\n", graph.name()));
    Ok(Admission {
        protocol: Protocol::parse(protocol.as_bytes())?,
        protocol_bytes: protocol.into_bytes(),
        corpus: a.corpus.clone(),
        tokenizer: a.tokenizer.clone(),
        updates: a.updates,
    })
}

/// Repeated full training comparisons, with exact artifact parity required.
/// Timings include training, metrics, checkpoint I/O and bundle verification;
/// admitted corpus loading/tokenizer fitting and cross-graph comparisons are excluded.
pub fn compare_graphs(a: &Admission, out: &Path, rounds: usize) -> Result<(), String> {
    use std::io::Write;
    if !(1..=5).contains(&rounds) {
        return Err("comparison rounds must be in 1..=5".into());
    }
    let updates = a
        .updates
        .checked_mul(rounds)
        .and_then(|n| n.checked_mul(2))
        .ok_or("comparison update budget overflow")?;
    if updates > a.protocol.max_updates {
        return Err(format!("comparison update budget exceeded: {updates}"));
    }
    let dense = graph_admission(a, EncoderGraph::Dense)?;
    let gather = graph_admission(a, EncoderGraph::Gather)?;
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    write_new(&out.join("source-protocol.tsv"), &a.protocol_bytes)?;
    let mut samples = std::fs::File::create(out.join("samples.tsv")).map_err(|e| e.to_string())?;
    writeln!(samples, "round\tfirst_graph\tdense_ns\tgather_ns\tmatched_files\tdense_bundle_sha256\tgather_bundle_sha256")
        .map_err(|e| e.to_string())?;
    for round in 0..rounds {
        let paths = [
            out.join(format!("round-{round}-dense")),
            out.join(format!("round-{round}-gather")),
        ];
        let order = if round.is_multiple_of(2) {
            [0, 1]
        } else {
            [1, 0]
        };
        let mut elapsed = [0u128; 2];
        for index in order {
            let start = std::time::Instant::now();
            train(
                if index == 0 { &dense } else { &gather },
                &paths[index],
                None,
            )?;
            elapsed[index] = start.elapsed().as_nanos();
        }
        let matched = compare_training_artifacts(&dense, &paths[0], &paths[1])?;
        let bundles = [
            digest(&read_bounded(&paths[0].join("COMPLETE"), 65536)?),
            digest(&read_bounded(&paths[1].join("COMPLETE"), 65536)?),
        ];
        writeln!(
            samples,
            "{round}\t{}\t{}\t{}\t{matched}\t{}\t{}",
            if order[0] == 0 { "dense" } else { "gather" },
            elapsed[0],
            elapsed[1],
            bundles[0],
            bundles[1]
        )
        .map_err(|e| e.to_string())?;
        samples.flush().map_err(|e| e.to_string())?;
    }
    let complete = format!("rust-graph-comparison-v1\nsource_protocol_sha256\t{}\nrounds\t{rounds}\nupdates\t{updates}\ncheckpoints_per_graph_per_round\t{}\nsamples_sha256\t{}\nmodel_promoted\tfalse\n",
        digest(&a.protocol_bytes), a.protocol.arms.len() * a.protocol.seeds.len(),
        digest(&read_bounded(&out.join("samples.tsv"), 65536)?));
    write_new(&out.join("COMPLETE"), complete.as_bytes())?;
    println!(
        "GRAPH_COMPARISON_COMPLETE_SHA256={}",
        digest(complete.as_bytes())
    );
    Ok(())
}

fn compare_training_artifacts(a: &Admission, left: &Path, right: &Path) -> Result<usize, String> {
    let mut matched = 0;
    for arm in &a.protocol.arms {
        for &seed in &a.protocol.seeds {
            for name in run_files(arm, seed) {
                if read_bounded(&left.join(&name), 8 * 1024 * 1024)?
                    != read_bounded(&right.join(&name), 8 * 1024 * 1024)?
                {
                    return Err(format!("dense/gather training artifact differs: {name}"));
                }
                matched += 1;
            }
        }
    }
    Ok(matched)
}

fn predictions(
    model: &cogno_model::bpe_cognitive::BpeCognitiveModel,
    a: &Admission,
    test_only: bool,
) -> Result<String, String> {
    let mut out =
        String::from("split\tproject\tsource_sha256\ttarget\tprediction\tp0\tp1\ttokens\n");
    for row in a
        .corpus
        .records()
        .iter()
        .filter(|r| (r.split == CorpusSplit::Test) == test_only)
    {
        let probs = model.classify(&row.source).map_err(|e| format!("{e:?}"))?;
        let tokens = model
            .tokenizer()
            .encode(&row.source)
            .map_err(|e| format!("{e:?}"))?;
        let split = match row.split {
            CorpusSplit::Train => "train",
            CorpusSplit::Validation => "validation",
            CorpusSplit::Test => "test",
        };
        out.push_str(&format!(
            "{split}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.project,
            hex(&row.source_hash),
            row.label,
            usize::from(probs[1] > probs[0]),
            probs[0],
            probs[1],
            tokens.len()
        ));
    }
    Ok(out)
}
fn files(a: &Admission) -> Vec<String> {
    let mut names = vec!["protocol.tsv".to_owned(), "plan.tsv".to_owned()];
    for arm in &a.protocol.arms {
        for seed in &a.protocol.seeds {
            for suffix in ["cbpc", "epochs.tsv", "predictions.tsv", "DONE"] {
                names.push(format!("{arm}-seed-{seed}.{suffix}"));
            }
        }
    }
    names
}
fn bundle(a: &Admission, out: &Path) -> Result<String, String> {
    let mut manifest = String::from("rust-train-v2-bundle\nfile\tsha256\n");
    for name in files(a) {
        let bytes = read_bounded(&out.join(&name), 8 * 1024 * 1024)?;
        manifest.push_str(&format!("{name}\t{}\n", digest(&bytes)));
    }
    Ok(manifest)
}
fn load_run(
    a: &Admission,
    out: &Path,
    arm: &str,
    seed: u64,
) -> Result<cogno_model::bpe_cognitive::BpeCognitiveModel, String> {
    use cogno_model::bpe_checkpoint::{checkpoint_hash, load_checkpoint, MAX_BPE_CHECKPOINT_BYTES};
    let bytes = read_bounded(
        &out.join(format!("{arm}-seed-{seed}.cbpc")),
        MAX_BPE_CHECKPOINT_BYTES,
    )?;
    let model = load_checkpoint(&bytes, checkpoint_hash(&bytes)).map_err(|e| format!("{e:?}"))?;
    let cfg = model.heads().config();
    if model.tokenizer() != &a.tokenizer
        || cfg.encoder.seed != seed
        || cfg.encoder.embedding_dim != a.protocol.embedding
        || cfg.encoder.hidden_dim != a.protocol.hidden
        || cfg.encoder.max_tokens != a.protocol.context
        || cfg.num_classes != 2
        || cfg.num_rules != 1
        || cfg.classification_seed != 1
        || cfg.preference_seed != 2
        || cfg.symbolic_seed != 3
        || cfg.contradiction_seed != 4
        || model.candidate_cap() != 2
    {
        return Err("checkpoint configuration differs from protocol".into());
    }
    Ok(model)
}
pub fn verify(a: &Admission, out: &Path, expected: &str) -> Result<(), String> {
    let complete = read_bounded(&out.join("COMPLETE"), 64 * 1024)?;
    if digest(&complete) != expected || complete != bundle(a, out)?.as_bytes() {
        return Err("bundle hash or inventory mismatch".into());
    }
    if read_bounded(&out.join("protocol.tsv"), 4096)? != a.protocol_bytes
        || read_bounded(&out.join("plan.tsv"), 4096)? != plan(a).as_bytes()
    {
        return Err("bundle protocol mismatch".into());
    }
    for arm in &a.protocol.arms {
        for &seed in &a.protocol.seeds {
            verify_run(a, out, arm, seed)?;
            let model = load_run(a, out, arm, seed)?;
            let actual = read_bounded(
                &out.join(format!("{arm}-seed-{seed}.predictions.tsv")),
                8 * 1024 * 1024,
            )?;
            if actual != predictions(&model, a, false)?.as_bytes() {
                return Err("checkpoint predictions differ".into());
            }
        }
    }
    Ok(())
}

fn selection(a: &Admission, out: &Path, expected: &str) -> Result<(String, String), String> {
    verify(a, out, expected)?;
    let mut scores = Vec::new();
    for arm in &a.protocol.arms {
        let mut sum = 0.0;
        let mut correct = 0;
        let mut count = 0;
        for &seed in &a.protocol.seeds {
            let model = load_run(a, out, arm, seed)?;
            let (n, c, nll) = metrics(model.heads(), a, CorpusSplit::Validation)?;
            sum += nll;
            correct += c;
            count += n;
        }
        scores.push((arm, sum / a.protocol.seeds.len() as f64, count, correct));
    }
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i].1 < scores[best].1 {
            best = i;
        }
    }
    let arm = scores[best].0.clone();
    let mut report = format!("rust-train-v2-selection\nprotocol_sha256\t{}\nbundle_sha256\t{expected}\nselected_arm\t{arm}\nrule\tmean_seed_validation_nll_then_protocol_order\narm\tseed_count\tvalidation_observations\tvalidation_correct\tmean_validation_nll\n", digest(&a.protocol_bytes));
    for (name, nll, count, correct) in scores {
        report.push_str(&format!(
            "{name}\t{}\t{count}\t{correct}\t{nll}\n",
            a.protocol.seeds.len()
        ));
    }
    Ok((arm, report))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
pub fn select(a: &Admission, out: &Path, expected: &str, destination: &Path) -> Result<(), String> {
    let (_, report) = selection(a, out, expected)?;
    write_new(destination, report.as_bytes())?;
    println!("SELECTION_SHA256={}", digest(report.as_bytes()));
    Ok(())
}

pub fn test_selected(
    a: &Admission,
    out: &Path,
    expected: &str,
    selection_path: &Path,
    destination: &Path,
) -> Result<(), String> {
    let (selected, report) = selection(a, out, expected)?;
    let frozen = read_bounded(selection_path, 64 * 1024)?;
    if frozen != report.as_bytes() {
        return Err("selection file differs from validation-only decision".into());
    }
    std::fs::create_dir(destination).map_err(|e| e.to_string())?;
    write_new(&destination.join("selection.tsv"), &frozen)?;
    let mut summary =
        String::from("arm\tseed\tselected\tunique_test_sources_per_seed\tcorrect\tmean_nll\n");
    let mut manifest = format!("rust-train-v2-test\nprotocol_sha256\t{}\nbundle_sha256\t{expected}\nselection_sha256\t{}\nfile\tsha256\nselection.tsv\t{}\n", digest(&a.protocol_bytes), digest(&frozen), digest(&frozen));
    // Evaluate only the frozen selected arm; no test-based reselection.
    for arm in [&selected] {
        for &seed in &a.protocol.seeds {
            let model = load_run(a, out, arm, seed)?;
            let (n, correct, nll) = metrics(model.heads(), a, CorpusSplit::Test)?;
            let predictions = predictions(&model, a, true)?;
            let name = format!("{arm}-seed-{seed}.test.tsv");
            write_new(&destination.join(&name), predictions.as_bytes())?;
            manifest.push_str(&format!("{name}\t{}\n", digest(predictions.as_bytes())));
            summary.push_str(&format!(
                "{arm}\t{seed}\t{}\t{n}\t{correct}\t{nll}\n",
                arm == &selected
            ));
        }
    }
    write_new(&destination.join("test-summary.tsv"), summary.as_bytes())?;
    manifest.push_str(&format!(
        "test-summary.tsv\t{}\n",
        digest(summary.as_bytes())
    ));
    write_new(&destination.join("COMPLETE"), manifest.as_bytes())?;
    println!("TEST_COMPLETE_SHA256={}", digest(manifest.as_bytes()));
    Ok(())
}

fn run_files(arm: &str, seed: u64) -> Vec<String> {
    ["cbpc", "epochs.tsv", "predictions.tsv"]
        .iter()
        .map(|suffix| format!("{arm}-seed-{seed}.{suffix}"))
        .collect()
}
fn run_manifest(a: &Admission, out: &Path, arm: &str, seed: u64) -> Result<String, String> {
    let mut manifest = format!(
        "rust-train-v2-run\nprotocol_sha256\t{}\narm\t{arm}\nseed\t{seed}\nfile\tsha256\n",
        digest(&a.protocol_bytes)
    );
    for name in run_files(arm, seed) {
        manifest.push_str(&format!(
            "{name}\t{}\n",
            digest(&read_bounded(&out.join(&name), 8 * 1024 * 1024)?)
        ));
    }
    Ok(manifest)
}
fn verify_run(a: &Admission, out: &Path, arm: &str, seed: u64) -> Result<(), String> {
    if read_bounded(&out.join(format!("{arm}-seed-{seed}.DONE")), 4096)?
        != run_manifest(a, out, arm, seed)?.as_bytes()
    {
        return Err("completed run inventory mismatch".into());
    }
    let model = load_run(a, out, arm, seed)?;
    if read_bounded(
        &out.join(format!("{arm}-seed-{seed}.predictions.tsv")),
        8 * 1024 * 1024,
    )? != predictions(&model, a, false)?.as_bytes()
    {
        return Err("completed run prediction mismatch".into());
    }
    let bytes = read_bounded(
        &out.join(format!("{arm}-seed-{seed}.epochs.tsv")),
        128 * 1024,
    )?;
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let lines: Vec<_> = text.lines().collect();
    if lines.len() != a.protocol.epochs + 1 || !text.ends_with('\n') || lines[0] != "epoch\tupdates\ttrain_rows_seen\temitted_tokens\ttrain_count\ttrain_correct\ttrain_nll\tvalidation_count\tvalidation_correct\tvalidation_nll" { return Err("epoch journal shape differs".into()); }
    let train_rows: Vec<_> = a
        .corpus
        .records()
        .iter()
        .filter(|r| r.split == CorpusSplit::Train)
        .collect();
    let validation_count = a
        .corpus
        .records()
        .iter()
        .filter(|r| r.split == CorpusSplit::Validation)
        .count();
    let mut emitted = 0;
    for (epoch, line) in lines[1..].iter().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 10 {
            return Err("epoch journal columns differ".into());
        }
        let merge_count = prefix(arm, epoch, a.tokenizer.vocab_size() - 259);
        for row in &train_rows {
            emitted += a
                .tokenizer
                .encode_with_merge_prefix(&row.source, merge_count)
                .map_err(|e| format!("{e:?}"))?
                .len();
        }
        let expected = [
            epoch + 1,
            (epoch + 1) * train_rows.len().div_ceil(a.protocol.batch),
            (epoch + 1) * train_rows.len(),
            emitted,
            train_rows.len(),
        ];
        for (field, n) in fields[..5].iter().zip(expected) {
            if field.parse::<usize>().ok() != Some(n) {
                return Err("epoch update/exposure accounting differs".into());
            }
        }
        if fields[7].parse::<usize>().ok() != Some(validation_count) {
            return Err("validation count differs".into());
        }
        for (idx, max) in [(5, train_rows.len()), (8, validation_count)] {
            if fields[idx].parse::<usize>().ok().is_none_or(|n| n > max) {
                return Err("invalid correct count".into());
            }
        }
        for idx in [6, 9] {
            if fields[idx]
                .parse::<f64>()
                .ok()
                .is_none_or(|n| !n.is_finite() || n < 0.0)
            {
                return Err("invalid epoch loss".into());
            }
        }
        if epoch + 1 == a.protocol.epochs {
            let tr = metrics(model.heads(), a, CorpusSplit::Train)?;
            let va = metrics(model.heads(), a, CorpusSplit::Validation)?;
            if fields[5] != tr.1.to_string()
                || fields[6] != tr.2.to_string()
                || fields[8] != va.1.to_string()
                || fields[9] != va.2.to_string()
            {
                return Err("final epoch metrics differ from checkpoint".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod qualification {
    use super::*;
    use std::path::PathBuf;
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            loop {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let path = std::env::temp_dir().join(format!(
                    "cogno-train-v2-{now}-{}",
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("{e}"),
                }
            }
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(root: &Path, batch: usize) -> (Admission, Vec<String>) {
        let mut corpus = String::from("CRUST001\n");
        for (split, labels) in [
            ("train", vec![0, 1, 0]),
            ("validation", vec![0, 1]),
            ("test", vec![0, 1]),
        ] {
            for (i, label) in labels.into_iter().enumerate() {
                let source = format!(
                    "fn {split}_{i}() {{ let x:u8={}; }}",
                    if label == 1 { "1" } else { "\"bad\"" }
                );
                corpus.push_str(&format!(
                    "{split}\t{split}_project\t{label}\t{}\t{}\n",
                    digest(source.as_bytes()),
                    hex(source.as_bytes())
                ));
            }
        }
        let provenance = b"unit test fixture; compiler labels are not a qualification dataset\n";
        let protocol = format!("version\trust-train-v2\ncorpus_sha256\t{}\nprovenance_sha256\t{}\nseeds\t1,7\narms\tfull,cycle_mix\nepochs\t2\nvocab\t259\ncontext\t96\nembedding\t2\nhidden\t3\nbatch\t{batch}\nlearning_rate\t0.003\nmax_updates\t100\n", digest(corpus.as_bytes()), digest(provenance));
        for (name, bytes) in [
            ("corpus", corpus.as_bytes()),
            ("provenance", provenance.as_slice()),
            ("protocol", protocol.as_bytes()),
        ] {
            write_new(&root.join(name), bytes).unwrap();
        }
        let args = vec![
            root.join("protocol").display().to_string(),
            digest(protocol.as_bytes()),
            root.join("corpus").display().to_string(),
            root.join("provenance").display().to_string(),
        ];
        (admit(&args).unwrap(), args)
    }
    #[test]
    fn complete_training_selection_test_and_bundle_tamper_detection() {
        let dir = Scratch::new();
        let (a, _) = fixture(&dir.0, 2);
        let out = dir.0.join("out");
        assert_eq!(a.updates, 16); // 3 rows -> 2 minibatches, 2 epochs, 4 runs.
        train(&a, &out, None).unwrap();
        let hash = digest(&std::fs::read(out.join("COMPLETE")).unwrap());
        verify(&a, &out, &hash).unwrap();
        assert!(train(&a, &out, None).is_err());
        let choice = dir.0.join("selection");
        select(&a, &out, &hash, &choice).unwrap();
        let report = std::fs::read_to_string(&choice).unwrap();
        // No merge rules: the arms are mathematically identical; exact ties use protocol order.
        assert!(report.contains("selected_arm\tfull\n"));
        assert!(report.contains("full\t2\t4\t"));
        let evaluation = dir.0.join("test");
        test_selected(&a, &out, &hash, &choice, &evaluation).unwrap();
        assert_eq!(
            std::fs::read_to_string(evaluation.join("test-summary.tsv"))
                .unwrap()
                .lines()
                .count(),
            3
        );
        for arm in &a.protocol.arms {
            for seed in &a.protocol.seeds {
                let train =
                    std::fs::read_to_string(out.join(format!("{arm}-seed-{seed}.predictions.tsv")))
                        .unwrap();
                assert!(!train.lines().any(|l| l.starts_with("test\t")));
                assert_eq!(train.lines().count(), 6);
                let test_path = evaluation.join(format!("{arm}-seed-{seed}.test.tsv"));
                if arm == "full" {
                    let test = std::fs::read_to_string(test_path).unwrap();
                    assert_eq!(test.lines().count(), 3);
                } else {
                    assert!(!test_path.exists());
                }
            }
        }
        std::fs::write(&choice, "selected_arm\tcycle_mix\n").unwrap();
        let bad_test = dir.0.join("bad-test");
        assert!(test_selected(&a, &out, &hash, &choice, &bad_test).is_err());
        assert!(!bad_test.exists());
        std::fs::write(out.join("full-seed-1.predictions.tsv"), "tampered").unwrap();
        assert!(verify(&a, &out, &hash).is_err());
    }
    #[test]
    fn interrupted_run_restarts_without_losing_completed_seed() {
        let dir = Scratch::new();
        let (a, _) = fixture(&dir.0, 1);
        let original = dir.0.join("original");
        train(&a, &original, None).unwrap();
        let partial = dir.0.join("partial");
        std::fs::create_dir(&partial).unwrap();
        for name in [
            "protocol.tsv".to_owned(),
            "plan.tsv".to_owned(),
            "full-seed-1.DONE".to_owned(),
        ]
        .into_iter()
        .chain(run_files("full", 1))
        {
            std::fs::copy(original.join(&name), partial.join(&name)).unwrap();
        }
        // This is an interrupted inference checkpoint, not restorable optimizer state.
        std::fs::write(partial.join("full-seed-7.cbpc"), "partial").unwrap();
        let resumed = dir.0.join("resumed");
        train(&a, &resumed, Some(&partial)).unwrap();
        for name in files(&a).into_iter().chain(["COMPLETE".to_owned()]) {
            assert_eq!(
                std::fs::read(original.join(&name)).unwrap(),
                std::fs::read(resumed.join(&name)).unwrap(),
                "{name}"
            );
        }
        assert_eq!(
            std::fs::read(partial.join("full-seed-7.cbpc")).unwrap(),
            b"partial"
        );
        std::fs::write(partial.join("full-seed-1.epochs.tsv"), "corrupt").unwrap();
        assert!(train(&a, &dir.0.join("refused"), Some(&partial)).is_err());
    }
    #[test]
    fn admission_rejects_changed_provenance_protocol_and_budget() {
        let dir = Scratch::new();
        let (_, args) = fixture(&dir.0, 1);
        let mut wrong = args.clone();
        wrong[1] = "0".repeat(64);
        assert!(admit(&wrong).is_err());
        let original = std::fs::read_to_string(&args[0]).unwrap();
        let over = original.replace("max_updates\t100", "max_updates\t1");
        std::fs::write(&args[0], &over).unwrap();
        wrong = args.clone();
        wrong[1] = digest(over.as_bytes());
        assert!(admit(&wrong).is_err());
        std::fs::write(&args[0], original).unwrap();
        std::fs::write(&args[3], "changed").unwrap();
        assert!(admit(&args).is_err());
    }
    #[test]
    fn graph_comparison_records_all_rounds_and_checks_artifact_parity() {
        let dir = Scratch::new();
        let (a, _) = fixture(&dir.0, 2);
        let out = dir.0.join("comparison");
        compare_graphs(&a, &out, 2).unwrap();
        let samples = std::fs::read_to_string(out.join("samples.tsv")).unwrap();
        let rows: Vec<_> = samples
            .lines()
            .skip(1)
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .collect();
        assert_eq!(rows.len(), 2);
        for (round, row) in rows.iter().enumerate() {
            assert_eq!(row[0], round.to_string());
            assert_eq!(row[1], if round == 0 { "dense" } else { "gather" });
            assert!(row[2].parse::<u128>().unwrap() > 0);
            assert!(row[3].parse::<u128>().unwrap() > 0);
            assert_eq!(row[4], "12");
            for (graph, column) in [("dense", 5), ("gather", 6)] {
                assert_eq!(
                    row[column],
                    digest(
                        &std::fs::read(out.join(format!("round-{round}-{graph}/COMPLETE")))
                            .unwrap()
                    )
                );
            }
        }
        let complete = std::fs::read_to_string(out.join("COMPLETE")).unwrap();
        assert!(complete.contains("rounds\t2\nupdates\t64\n"));
        assert!(complete.contains(&digest(samples.as_bytes())));
        assert!(compare_graphs(&a, &out, 2).is_err());
        std::fs::write(out.join("round-0-gather/full-seed-1.cbpc"), b"changed").unwrap();
        assert!(compare_training_artifacts(
            &a,
            &out.join("round-0-dense"),
            &out.join("round-0-gather")
        )
        .is_err());
    }

    #[test]
    fn graph_comparison_budget_covers_both_graphs_and_every_round() {
        let dir = Scratch::new();
        let (a, _) = fixture(&dir.0, 1);
        for rounds in [0, 3, 6, usize::MAX] {
            let out = dir.0.join(format!("refused-{rounds}"));
            assert!(compare_graphs(&a, &out, rounds).is_err());
            assert!(!out.exists());
        }
    }

    #[test]
    fn v3_gather_reproduces_dense_training_and_refuses_cross_graph_resume() {
        for batch in [1, 2] {
            let dir = Scratch::new();
            let (dense, mut args) = fixture(&dir.0, batch);
            let dense_out = dir.0.join("dense");
            train(&dense, &dense_out, None).unwrap();
            for graph in ["dense", "gather"] {
                let protocol = format!(
                    "{}encoder_graph\t{graph}\n",
                    std::str::from_utf8(&dense.protocol_bytes)
                        .unwrap()
                        .replace("rust-train-v2", "rust-train-v3")
                );
                std::fs::write(&args[0], &protocol).unwrap();
                // Changing the graph without pinning the new protocol is rejected.
                args[1] = digest(&dense.protocol_bytes);
                assert!(admit(&args).is_err());
                args[1] = digest(protocol.as_bytes());
                let alternative = admit(&args).unwrap();
                assert!(plan(&alternative).contains(&format!("encoder_graph\t{graph}\n")));
                let out = dir.0.join(format!("v3-{graph}"));
                train(&alternative, &out, None).unwrap();
                for arm in &dense.protocol.arms {
                    for &seed in &dense.protocol.seeds {
                        for name in run_files(arm, seed) {
                            assert_eq!(
                                std::fs::read(dense_out.join(&name)).unwrap(),
                                std::fs::read(out.join(&name)).unwrap(),
                                "{name} {graph}"
                            );
                        }
                    }
                }
                let refused = dir.0.join(format!("wrong-resume-{graph}"));
                assert!(train(&alternative, &refused, Some(&dense_out)).is_err());
                assert!(!refused.exists());
                let resumed = dir.0.join(format!("resumed-{graph}"));
                train(&alternative, &resumed, Some(&out)).unwrap();
                assert_eq!(
                    std::fs::read(out.join("COMPLETE")).unwrap(),
                    std::fs::read(resumed.join("COMPLETE")).unwrap()
                );
            }
        }
    }

    #[test]
    fn batch_one_checkpoint_matches_frozen_manual_training() {
        use cogno_scirust::*;
        let dir = Scratch::new();
        let (a, _) = fixture(&dir.0, 1);
        let out = dir.0.join("out");
        train(&a, &out, None).unwrap();
        let p = &a.protocol;
        let mut heads = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: a.tokenizer.vocab_size(),
                max_tokens: p.context,
                embedding_dim: p.embedding,
                hidden_dim: p.hidden,
                seed: 1,
            },
            num_classes: 2,
            num_rules: 1,
            classification_seed: 1,
            preference_seed: 2,
            symbolic_seed: 3,
            contradiction_seed: 4,
        })
        .unwrap();
        let mut optimizer = SequenceCognitiveAdamW::try_new(p.learning_rate, &heads).unwrap();
        let rows: Vec<_> = a
            .corpus
            .records()
            .iter()
            .filter(|r| r.split == CorpusSplit::Train)
            .collect();
        for epoch in 0..p.epochs {
            for i in cogno_model::training_order::epoch_order(rows.len(), 1, epoch as u64).unwrap()
            {
                let tokens = a.tokenizer.encode(&rows[i].source).unwrap();
                heads
                    .train_classification_step(
                        &mut optimizer,
                        CognitiveClassification {
                            token_ids: &tokens,
                            target_class: rows[i].label,
                        },
                    )
                    .unwrap();
            }
        }
        assert_eq!(load_run(&a, &out, "full", 1).unwrap().heads(), &heads);
    }
}
