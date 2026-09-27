//! Bounded deterministic semantic oracle and opt-in wall-clock benchmark.
#![forbid(unsafe_code)]
use cogno_model::bpe_tokenizer::{
    BpeError, BpeTokenizer, BpeWorkspace, MAX_BPE_BYTES, MAX_BPE_TRAIN_BYTES,
    MAX_BPE_TRAIN_RECORDS, MAX_BPE_VOCAB,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// Deliberately simple independent oracle: recount every adjacency, rebuild and
// validate the complete vocabulary, then replace disjoint occurrences per rank.
fn reference_train(sources: &[&[u8]], target: usize) -> Result<BpeTokenizer, BpeError> {
    let mut model = BpeTokenizer::from_merges(512, &[])?;
    let mut lengths = vec![1usize; 259];
    let mut rules = Vec::new();
    let mut records: Vec<Vec<u16>> = sources
        .iter()
        .map(|s| s.iter().map(|&b| u16::from(b)).collect())
        .collect();
    while model.vocab_size() < target {
        let mut counts = BTreeMap::<(u16, u16), usize>::new();
        for record in &records {
            for pair in record.windows(2) {
                if lengths[pair[0] as usize] + lengths[pair[1] as usize] <= 1024 {
                    *counts.entry((pair[0], pair[1])).or_default() += 1;
                }
            }
        }
        let mut best = None;
        for (pair, count) in counts {
            if count >= 2 && best.is_none_or(|(_, old)| count > old) {
                best = Some((pair, count));
            }
        }
        let Some((pair, _)) = best else {
            break;
        };
        let id = model.vocab_size() as u16;
        lengths.push(lengths[pair.0 as usize] + lengths[pair.1 as usize]);
        rules.push(pair);
        model = BpeTokenizer::from_merges(512, &rules)?;
        for record in &mut records {
            let mut out = Vec::with_capacity(record.len());
            let mut read = 0;
            while read < record.len() {
                if read + 1 < record.len() && (record[read], record[read + 1]) == pair {
                    out.push(id);
                    read += 2;
                } else {
                    out.push(record[read]);
                    read += 1;
                }
            }
            *record = out;
        }
    }
    Ok(model)
}

fn measure_encode(model: &BpeTokenizer, sources: &[&[u8]], heap: bool) -> u128 {
    let mut workspace = BpeWorkspace::new();
    // Warm both instruction paths and allocate reusable heap scratch before timing.
    for &source in sources {
        if heap {
            let _ = model.encode_with_workspace(source, &mut workspace);
        } else {
            let _ = model.encode(source);
        }
    }
    let start = Instant::now();
    for _ in 0..100 {
        for &source in sources {
            if heap {
                let _ = black_box(model.encode_with_workspace(black_box(source), &mut workspace));
            } else {
                let _ = black_box(model.encode(black_box(source)));
            }
        }
    }
    start.elapsed().as_nanos()
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err("usage: bpe_tokenizer_bench VOCAB REPEATS TRAIN_SOURCE... (source <=16 KiB; total <=1 MiB)".into());
    }
    let target: usize = args[1].parse().map_err(|_| "invalid vocabulary")?;
    let repeats: usize = args[2].parse().map_err(|_| "invalid repeats")?;
    if !(259..=MAX_BPE_VOCAB).contains(&target)
        || !(1..=20).contains(&repeats)
        || args.len() - 3 > MAX_BPE_TRAIN_RECORDS
    {
        return Err("invalid bounds".into());
    }
    let mut owned = Vec::new();
    let mut total = 0usize;
    for path in &args[3..] {
        let mut source = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| e.to_string())?
            .take((MAX_BPE_BYTES + 1) as u64)
            .read_to_end(&mut source)
            .map_err(|e| e.to_string())?;
        total += source.len();
        if source.len() > MAX_BPE_BYTES || total > MAX_BPE_TRAIN_BYTES {
            return Err("source capacity".into());
        }
        println!(
            "source_sha256={} bytes={}",
            hex(&Sha256::digest(&source)),
            source.len()
        );
        owned.push(source);
    }
    let sources: Vec<_> = owned.iter().map(Vec::as_slice).collect();
    let model = BpeTokenizer::train(&sources, target, 512).map_err(|e| format!("{e:?}"))?;
    let reference = reference_train(&sources, target).map_err(|e| format!("{e:?}"))?;
    if model.to_bytes() != reference.to_bytes() {
        return Err("training oracle mismatch".into());
    }
    println!(
        "tokenizer_sha256={} vocabulary={} context=512",
        hex(&model.fingerprint()),
        model.vocab_size()
    );
    let prefixes: Vec<_> = sources.iter().map(|s| &s[..s.len().min(128)]).collect();
    let mut workspace = BpeWorkspace::new();
    for (name, cases) in [("whole", &sources), ("prefix128", &prefixes)] {
        let mut accepted = 0;
        for &source in cases {
            let scan = model.encode(source);
            let heap = model
                .encode_with_workspace(source, &mut workspace)
                .map(<[u16]>::to_vec);
            if scan != heap {
                return Err("encoding oracle mismatch".into());
            }
            if scan.is_ok() {
                accepted += 1;
            }
            let chunks = model
                .encode_chunks(source, MAX_BPE_BYTES)
                .map_err(|e| format!("{e:?}"))?;
            let mut restored = Vec::new();
            for chunk in chunks {
                restored.extend(model.decode(&chunk.tokens).map_err(|e| format!("{e:?}"))?);
            }
            if restored != source {
                return Err("lossless chunks mismatch".into());
            }
        }
        println!(
            "workload={name} accepted={accepted} refused={} bytes={}",
            cases.len() - accepted,
            cases.iter().map(|s| s.len()).sum::<usize>()
        );
    }
    println!("sample,operation,implementation,workload,iterations,elapsed_ns");
    for sample in 0..repeats {
        for optimized in if sample % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            let trained = if optimized {
                BpeTokenizer::train(black_box(&sources), target, 512)
            } else {
                reference_train(black_box(&sources), target)
            }
            .map_err(|e| format!("{e:?}"))?;
            let elapsed = start.elapsed().as_nanos();
            if trained.to_bytes() != model.to_bytes() {
                return Err("repeated training mismatch".into());
            }
            println!(
                "{sample},train,{},whole,1,{elapsed}",
                if optimized {
                    "adjacency_delta"
                } else {
                    "full_recount"
                }
            );
        }
        for (name, cases) in [("whole", &sources), ("prefix128", &prefixes)] {
            for heap in if sample % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let elapsed = measure_encode(&model, cases, heap);
                println!(
                    "{sample},encode,{},{name},100,{elapsed}",
                    if heap { "reused_heap" } else { "rank_scan" }
                );
            }
        }
    }
    Ok(())
}
fn main() -> Result<(), String> {
    run(&std::env::args().collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_trainer_preserves_ties_and_overlaps() {
        for sources in [
            vec![b"aaaaaaa".as_slice(), b"abababa"],
            vec![b"fn main() {}", b"let x = 0;"],
        ] {
            assert_eq!(
                reference_train(&sources, 300).unwrap().to_bytes(),
                BpeTokenizer::train(&sources, 300, 512).unwrap().to_bytes()
            );
        }
    }
    #[test]
    fn argument_bounds_fail_before_file_access() {
        for args in [
            vec!["bench"],
            vec!["bench", "513", "3", "missing"],
            vec!["bench", "384", "0", "missing"],
        ] {
            assert!(run(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
    }
}
