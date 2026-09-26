#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus, RustRecord};
#[cfg(test)]
use sha2::{Digest, Sha256};
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn split_name(s: CorpusSplit) -> &'static str {
    match s {
        CorpusSplit::Train => "train",
        CorpusSplit::Validation => "validation",
        CorpusSplit::Test => "test",
    }
}
fn read(path: &str, hash: &str) -> Result<RustCorpus, String> {
    RustCorpus::read(
        std::fs::File::open(path).map_err(|e| e.to_string())?,
        parse_hash(hash).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))
}

use cogno_model::training_order::epoch_order;
fn order(
    rows: &[RustRecord],
    width: usize,
    seed: u64,
    epoch: u64,
) -> Result<Vec<(usize, usize)>, String> {
    if !(1..=16384).contains(&width) {
        return Err("bucket width must be 1..16384 bytes".into());
    }
    let train: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.split == CorpusSplit::Train)
        .collect();
    let permutation = epoch_order(train.len(), seed, epoch).map_err(|e| format!("{e:?}"))?;
    let mut out: Vec<_> = permutation
        .iter()
        .map(|&i| (train[i].0, (train[i].1.source.len() - 1) / width))
        .collect();
    out.sort_by_key(|x| x.1);
    Ok(out)
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: rust_length_order CORPUS SHA BUCKET_BYTES SEED EPOCH".into());
    }
    let c = read(&a[1], &a[2])?;
    let order = order(
        c.records(),
        a[3].parse().map_err(|_| "invalid width")?,
        a[4].parse().map_err(|_| "invalid seed")?,
        a[5].parse().map_err(|_| "invalid epoch")?,
    )?;
    println!("position,corpus_index,source_sha256,split,bytes,bucket");
    for (position, (i, bucket)) in order.iter().enumerate() {
        let r = &c.records()[*i];
        println!(
            "{position},{i},{},{},{},{bucket}",
            hex(&r.source_hash),
            split_name(r.split),
            r.source.len()
        );
    }
    Ok(())
}

#[cfg(test)]
fn fixture(prefix: &str) -> Vec<RustRecord> {
    let mut rows = Vec::new();
    for split in [
        CorpusSplit::Train,
        CorpusSplit::Validation,
        CorpusSplit::Test,
    ] {
        for label in 0..2 {
            let source = format!("{prefix} {} {label}", split_name(split)).into_bytes();
            rows.push(RustRecord {
                split,
                project: format!("{prefix}/{}", split_name(split)),
                label,
                source_hash: Sha256::digest(&source).into(),
                source,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_training_row_once_with_monotonic_buckets() {
        let mut r = [fixture("a"), fixture("bb")].concat();
        r[0].source = vec![b'a'; 2];
        r[1].source = vec![b'a'; 18];
        let got = order(&r, 8, 1, 2).unwrap();
        assert_eq!(got.len(), 4);
        assert!(got.windows(2).all(|p| p[0].1 <= p[1].1));
        assert_eq!(
            got.iter()
                .map(|x| x.0)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1, 6, 7])
        );
        assert_eq!(got, order(&r, 8, 1, 2).unwrap());
    }
    #[test]
    fn boundaries_and_heldout_independence() {
        let mut r = fixture("p");
        r[0].source = vec![b'a'; 8];
        r[1].source = vec![b'a'; 9];
        assert_eq!(order(&r, 8, 0, 0).unwrap(), vec![(0, 0), (1, 1)]);
        let baseline = order(&r, 8, 0, 0).unwrap();
        r[2].source = vec![b'z'; 100];
        assert_eq!(baseline, order(&r, 8, 0, 0).unwrap());
        assert!(order(&r, 0, 0, 0).is_err());
        assert!(order(&r, 16385, 0, 0).is_err());
    }
}
