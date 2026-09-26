#![forbid(unsafe_code)]
use cogno_model::{
    bpe_checkpoint::{load_checkpoint, MAX_BPE_CHECKPOINT_BYTES},
    bpe_cognitive::BpeCognitiveModel,
    rust_corpus::parse_hash,
};
use std::io::Read;
fn model(path: &str, hash: &str) -> Result<BpeCognitiveModel, String> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take((MAX_BPE_CHECKPOINT_BYTES + 1) as u64)
        .read_to_end(&mut b)
        .map_err(|e| e.to_string())?;
    let m = load_checkpoint(&b, parse_hash(hash).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("{e:?}"))?;
    Ok(m)
}

fn compatible(a: &BpeCognitiveModel, b: &BpeCognitiveModel) -> bool {
    let mut x = a.heads().config();
    let mut y = b.heads().config();
    x.encoder.seed = 0;
    y.encoder.seed = 0;
    x.classification_seed = 0;
    y.classification_seed = 0;
    x.preference_seed = 0;
    y.preference_seed = 0;
    x.symbolic_seed = 0;
    y.symbolic_seed = 0;
    x.contradiction_seed = 0;
    y.contradiction_seed = 0;
    x == y
        && a.tokenizer().fingerprint() == b.tokenizer().fingerprint()
        && a.candidate_cap() == b.candidate_cap()
}
fn tensors(m: &BpeCognitiveModel) -> [(&'static str, &[f32]); 11] {
    let h = m.heads();
    [
        ("token_embeddings", h.encoder().token_embeddings()),
        ("position_embeddings", h.encoder().position_embeddings()),
        ("mixing_weights", h.encoder().mixing_weights()),
        ("classification_weights", h.classification_weights()),
        ("classification_bias", h.classification_bias()),
        ("preference_weights", h.preference_weights()),
        ("preference_bias", h.preference_bias()),
        ("symbolic_weights", h.symbolic_weights()),
        ("symbolic_bias", h.symbolic_bias()),
        ("contradiction_weights", h.contradiction_weights()),
        ("contradiction_bias", h.contradiction_bias()),
    ]
}
fn distance(a: &[f32], b: &[f32]) -> Result<(f64, f64, Option<f64>), String> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("invalid tensor pair".into());
    }
    let (mut sq, mut max, mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        let d = x - y;
        sq += d * d;
        max = max.max(d.abs());
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    let cos = if aa == 0.0 || bb == 0.0 {
        None
    } else {
        Some((dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0))
    };
    Ok(((sq / a.len() as f64).sqrt(), max, cos))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: bpe_checkpoint_drift A SHA B SHA".into());
    }
    let x = model(&a[1], &a[2])?;
    let y = model(&a[3], &a[4])?;
    if !compatible(&x, &y) {
        return Err("shape, tokenizer or candidate cap mismatch".into());
    }
    let mut out = Vec::new();
    for ((name, p), (_, q)) in tensors(&x).into_iter().zip(tensors(&y)) {
        let (rms, max, cos) = distance(p, q)?;
        out.push(format!(
            "{name},{},{rms:.12},{max:.12},{}",
            p.len(),
            cos.map_or("NA".into(), |v| format!("{v:.12}"))
        ));
    }
    println!("a_sha256,b_sha256\n{},{}", a[2], a[4]);
    println!("tensor,parameters,rms_difference,max_absolute_difference,cosine");
    for l in out {
        println!("{l}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_geometry() {
        assert_eq!(distance(&[1., 0.], &[0., 1.]).unwrap(), (1., 1., Some(0.)));
        assert_eq!(distance(&[1.], &[-1.]).unwrap(), (2., 2., Some(-1.)));
        assert_eq!(distance(&[0.], &[0.]).unwrap(), (0., 0., None));
    }
    #[test]
    fn identical_nonzero() {
        let (r, m, c) = distance(&[1., 2.], &[1., 2.]).unwrap();
        assert_eq!((r, m), (0., 0.));
        assert!((c.unwrap() - 1.).abs() < 1e-12);
    }
    #[test]
    fn invalid_and_large() {
        assert!(distance(&[], &[]).is_err());
        assert!(distance(&[0.], &[0., 1.]).is_err());
        assert!(distance(&[f32::NAN], &[0.]).is_err());
        assert!(distance(&[f32::MAX], &[-f32::MAX]).unwrap().0.is_finite());
    }
}
