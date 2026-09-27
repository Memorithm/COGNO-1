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
        if fields.len() != keys.len()
            || keys.iter().any(|k| !fields.contains_key(k))
            || fields["version"] != "rust-train-v2"
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
    format!("protocol_sha256\t{}\ncorpus_sha256\t{}\nprovenance_sha256\t{}\nrows\t{}\nupdates\t{}\nruns\t{}\nepochs\t{}\nbatch\t{}\nembedding\t{}\nhidden\t{}\nlearning_rate\t{}\ntokenizer_sha256\t{}\n", digest(&a.protocol_bytes), a.protocol.corpus_sha256, a.protocol.provenance_sha256, a.corpus.records().len(), a.updates, a.protocol.seeds.len() * a.protocol.arms.len(), a.protocol.epochs, a.protocol.batch, a.protocol.embedding, a.protocol.hidden, a.protocol.learning_rate, hex(&a.tokenizer.fingerprint()))
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
pub fn train(a: &Admission, out: &Path) -> Result<(), String> {
    use cogno_model::{
        bpe_checkpoint::encode_checkpoint, bpe_cognitive::BpeCognitiveModel,
        training_order::epoch_order,
    };
    use cogno_scirust::{
        CognitiveClassification, SequenceCognitiveAdamW, SequenceCognitiveConfig,
        SequenceCognitiveHeads, SequenceEncoderConfig,
    };
    use std::io::Write;
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
                    if examples.len() == 1 {
                        model
                            .train_classification_step(&mut optimizer, examples[0])
                            .map_err(|e| format!("{e:?}"))?;
                    } else {
                        model
                            .train_classification_minibatch_step(&mut optimizer, &examples)
                            .map_err(|e| format!("{e:?}"))?;
                    }
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
            eprintln!("completed arm={arm} seed={seed} updates={completed_updates}");
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
            for suffix in ["cbpc", "epochs.tsv", "predictions.tsv"] {
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
