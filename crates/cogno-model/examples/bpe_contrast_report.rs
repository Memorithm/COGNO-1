//! Explicit contrast-pair diagnostic. A manifest does not establish semantic equivalence.
#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    rust_corpus::{parse_hash, CorpusSplit, RustCorpus},
};
use std::io::Read;

fn observations(args: &[String]) -> Result<Vec<(usize, f64)>, String> {
    let split = match args[5].as_str() {
        "train" => CorpusSplit::Train,
        "validation" => CorpusSplit::Validation,
        "test" => CorpusSplit::Test,
        _ => return Err("unknown split".into()),
    };
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1])
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let model = load_checkpoint(&bytes, parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("{e:?}"))?;
    if model.heads().config().num_classes != 2 {
        return Err("binary classifier required".into());
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[3]).map_err(|e| e.to_string())?,
        parse_hash(&args[4]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    corpus
        .records()
        .iter()
        .filter(|r| r.split == split)
        .map(|r| {
            let p = model.classify(&r.source).map_err(|e| format!("{e:?}"))?;
            if p.len() != 2
                || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || (p[0] + p[1] - 1.0).abs() > 1e-5
            {
                return Err("invalid binary probabilities".into());
            }
            Ok((r.label, f64::from(p[1])))
        })
        .collect()
}

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
type Hash = [u8; 32];
fn pairs(bytes: &[u8]) -> Result<Vec<(Hash, Hash)>, String> {
    if bytes.len() > 65536 {
        return Err("pair manifest exceeds 64 KiB".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 2 {
            return Err("expected pass_sha256 TAB fail_sha256".into());
        }
        let pass = parse_hash(fields[0]).map_err(|e| format!("{e:?}"))?;
        let fail = parse_hash(fields[1]).map_err(|e| format!("{e:?}"))?;
        if !seen.insert(pass) || !seen.insert(fail) {
            return Err("repeated pair member".into());
        }
        result.push((pass, fail));
        if result.len() > 256 {
            return Err("at most 256 pairs".into());
        }
    }
    if result.is_empty() {
        return Err("empty pair manifest".into());
    }
    Ok(result)
}
fn pair_score(pass: (usize, f64), fail: (usize, f64)) -> Result<(bool, bool, f64), String> {
    if pass.0 != 1 || fail.0 != 0 {
        return Err("pair requires compile-pass then compile-fail label".into());
    }
    if [pass.1, fail.1]
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("invalid probability".into());
    }
    Ok((
        pass.1 > 0.5 && fail.1 <= 0.5,
        pass.1 > fail.1,
        pass.1 - fail.1,
    ))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err("usage: bpe_contrast_report CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA train|validation|test PAIRS_TSV PAIRS_SHA".into());
    }
    let mut manifest = Vec::new();
    std::fs::File::open(&a[6])
        .map_err(|e| e.to_string())?
        .take(65537)
        .read_to_end(&mut manifest)
        .map_err(|e| e.to_string())?;
    let expected = parse_hash(&a[7]).map_err(|e| format!("{e:?}"))?;
    let actual: Hash = Sha256::digest(&manifest).into();
    if actual != expected {
        return Err("pair manifest digest mismatch".into());
    }
    let pairs = pairs(&manifest)?;
    let data = observations(&a)?;
    let corpus = RustCorpus::read(
        std::fs::File::open(&a[3]).map_err(|e| e.to_string())?,
        parse_hash(&a[4]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let split = match a[5].as_str() {
        "train" => CorpusSplit::Train,
        "validation" => CorpusSplit::Validation,
        _ => CorpusSplit::Test,
    };
    let lookup: BTreeMap<_, _> = corpus
        .records()
        .iter()
        .filter(|r| r.split == split)
        .map(|r| r.source_hash)
        .zip(data)
        .collect();
    let mut rows = Vec::new();
    let (mut both, mut separated, mut margins) = (0, 0, 0.0);
    for (i, (pass, fail)) in pairs.iter().enumerate() {
        let (b, s, m) = pair_score(
            *lookup
                .get(pass)
                .ok_or("pass source absent from selected split")?,
            *lookup
                .get(fail)
                .ok_or("fail source absent from selected split")?,
        )?;
        both += usize::from(b);
        separated += usize::from(s);
        margins += m;
        rows.push(format!("{i},{},{},{m:.12}", usize::from(b), usize::from(s)));
    }
    println!("checkpoint_sha256,corpus_sha256,pairs_sha256,split,pairs,both_correct,positive_separation,mean_probability_margin");
    println!(
        "{},{},{},{},{},{both},{separated},{:.12}",
        a[2],
        a[4],
        a[7],
        a[5],
        pairs.len(),
        margins / pairs.len() as f64
    );
    println!("pair_index,both_correct,positive_separation,p_pass_minus_p_fail");
    for row in rows {
        println!("{row}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pair_membership_is_unique() {
        let a = "01".repeat(32);
        let b = "02".repeat(32);
        let c = "03".repeat(32);
        assert_eq!(pairs(format!("{a}\t{b}\n").as_bytes()).unwrap().len(), 1);
        assert!(pairs(format!("{a}\t{a}").as_bytes()).is_err());
        assert!(pairs(format!("{a}\t{b}\n{c}\t{a}").as_bytes()).is_err());
    }
    #[test]
    fn bounded_strict_manifest() {
        for data in [
            vec![],
            vec![b'x'; 65537],
            b"invalid\tbad".to_vec(),
            b"\n".to_vec(),
            vec![255],
        ] {
            assert!(pairs(&data).is_err());
        }
        let rows: Vec<_> = (0..514).map(|i| format!("{i:064x}")).collect();
        let text = rows
            .chunks(2)
            .map(|p| format!("{}\t{}", p[0], p[1]))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(pairs(text.as_bytes()).is_err());
    }
    #[test]
    fn separation_is_not_joint_correctness() {
        assert_eq!(pair_score((1, 0.4), (0, 0.2)).unwrap(), (false, true, 0.2));
        assert_eq!(pair_score((1, 0.5), (0, 0.5)).unwrap(), (false, false, 0.0));
        let (b, s, m) = pair_score((1, 0.8), (0, 0.2)).unwrap();
        assert!(b && s);
        assert!((m - 0.6).abs() < 1e-12);
    }
    #[test]
    fn invalid_labels_or_probabilities_refused() {
        for (p, f) in [
            ((0, 0.5), (1, 0.5)),
            ((1, f64::NAN), (0, 0.5)),
            ((1, 0.5), (0, 1.1)),
        ] {
            assert!(pair_score(p, f).is_err());
        }
    }
}
