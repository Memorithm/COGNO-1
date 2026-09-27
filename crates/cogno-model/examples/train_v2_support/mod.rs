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
