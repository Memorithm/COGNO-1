//! Self-contained research inference checkpoint. Not a production V5 artifact.
//! The caller must obtain the expected digest from a trusted inventory; hashing
//! untrusted bytes and passing that hash is not authentication or promotion.
use crate::bpe_cognitive::BpeCognitiveModel;
use crate::bpe_tokenizer::BpeTokenizer;
use cogno_scirust::{
    SequenceCognitiveConfig, SequenceCognitiveHeads, SequenceEncoder, SequenceEncoderConfig,
    MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_COGNITIVE_PARAMETERS, MAX_SEQUENCE_EMBEDDING_DIM,
    MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_SYMBOLIC_RULES,
};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"CBPC0001";
const HEADER: usize = 96;
/// Upper bound before hashing, parsing or allocating checkpoint tensors.
pub const MAX_BPE_CHECKPOINT_BYTES: usize = HEADER + 1024 + MAX_SEQUENCE_COGNITIVE_PARAMETERS * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpeCheckpointError {
    Size,
    Hash,
    Format,
    Configuration,
    Tokenizer,
    Weights,
}

/// Full checkpoint identity, covering header, tokenizer and all weights.
pub fn checkpoint_hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Persist inference state only. Optimizer moments and training provenance are not included.
pub fn encode_checkpoint(model: &BpeCognitiveModel) -> Vec<u8> {
    let h = model.heads();
    let c = h.config();
    let tokenizer = model.tokenizer().to_bytes();
    let mut out = Vec::with_capacity(HEADER + tokenizer.len() + h.parameter_count() * 4);
    out.extend_from_slice(MAGIC);
    for n in [
        c.encoder.vocab_size,
        c.encoder.max_tokens,
        c.encoder.embedding_dim,
        c.encoder.hidden_dim,
        c.num_classes,
        c.num_rules,
        model.candidate_cap(),
    ] {
        out.extend_from_slice(&(n as u16).to_le_bytes());
    }
    for n in [
        c.encoder.seed,
        c.classification_seed,
        c.preference_seed,
        c.symbolic_seed,
        c.contradiction_seed,
    ] {
        out.extend_from_slice(&n.to_le_bytes());
    }
    out.extend_from_slice(&(tokenizer.len() as u16).to_le_bytes());
    out.extend_from_slice(&model.tokenizer().fingerprint());
    out.extend_from_slice(&tokenizer);
    for tensor in [
        h.encoder().token_embeddings(),
        h.encoder().position_embeddings(),
        h.encoder().mixing_weights(),
        h.classification_weights(),
        h.classification_bias(),
        h.preference_weights(),
        h.preference_bias(),
        h.symbolic_weights(),
        h.symbolic_bias(),
        h.contradiction_weights(),
        h.contradiction_bias(),
    ] {
        for w in tensor {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// Verified descriptive data, with no model activation or training authority.
#[derive(Clone, Debug, PartialEq)]
pub struct BpeCheckpointMetadata {
    pub config: SequenceCognitiveConfig,
    pub candidate_cap: usize,
    pub parameter_count: usize,
    pub checkpoint_bytes: usize,
    pub checkpoint_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
}

/// Validate the complete artifact without allocating numerical tensors.
/// The expected hash must come from the same trusted inventory as for loading.
/// Configuration, tokenizer binding, exact length and every weight are checked.
pub fn inspect_checkpoint(
    bytes: &[u8],
    expected_hash: [u8; 32],
) -> Result<BpeCheckpointMetadata, BpeCheckpointError> {
    let (metadata, _, mut reader) = parse_checkpoint(bytes, expected_hash)?;
    let raw = reader.take(metadata.parameter_count * 4)?;
    if raw
        .chunks_exact(4)
        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
    {
        return Err(BpeCheckpointError::Weights);
    }
    Ok(metadata)
}

fn parse_checkpoint(
    bytes: &[u8],
    expected_hash: [u8; 32],
) -> Result<(BpeCheckpointMetadata, BpeTokenizer, Reader<'_>), BpeCheckpointError> {
    use BpeCheckpointError as E;
    if !(HEADER..=MAX_BPE_CHECKPOINT_BYTES).contains(&bytes.len()) {
        return Err(E::Size);
    }
    if checkpoint_hash(bytes) != expected_hash {
        return Err(E::Hash);
    }
    let mut r = Reader { bytes, pos: 0 };
    if r.take(8)? != MAGIC {
        return Err(E::Format);
    }
    let vocab_size = r.u16()?;
    let max_tokens = r.u16()?;
    let embedding_dim = r.u16()?;
    let hidden_dim = r.u16()?;
    let num_classes = r.u16()?;
    let num_rules = r.u16()?;
    let cap = r.u16()?;
    let seed = r.u64()?;
    let classification_seed = r.u64()?;
    let preference_seed = r.u64()?;
    let symbolic_seed = r.u64()?;
    let contradiction_seed = r.u64()?;
    let tokenizer_len = r.u16()?;
    let mut tokenizer_hash = [0; 32];
    tokenizer_hash.copy_from_slice(r.take(32)?);
    if !(259..=512).contains(&vocab_size)
        || !(3..=512).contains(&max_tokens)
        || !(1..=MAX_SEQUENCE_EMBEDDING_DIM).contains(&embedding_dim)
        || !(1..=MAX_SEQUENCE_HIDDEN_DIM).contains(&hidden_dim)
        || !(2..=MAX_SEQUENCE_CLASSES).contains(&num_classes)
        || !(1..=MAX_SEQUENCE_SYMBOLIC_RULES).contains(&num_rules)
        || !(2..=cogno_scirust::MAX_SEQUENCE_RETRIEVAL_CANDIDATES).contains(&cap)
        || !(12..=1024).contains(&tokenizer_len)
    {
        return Err(E::Configuration);
    }
    let c = SequenceCognitiveConfig {
        encoder: SequenceEncoderConfig {
            vocab_size,
            max_tokens,
            embedding_dim,
            hidden_dim,
            seed,
        },
        num_classes,
        num_rules,
        classification_seed,
        preference_seed,
        symbolic_seed,
        contradiction_seed,
    };
    let count = c.parameter_count().map_err(|_| E::Configuration)?;
    if count > MAX_SEQUENCE_COGNITIVE_PARAMETERS
        || HEADER + tokenizer_len + count * 4 != bytes.len()
    {
        return Err(E::Size);
    }
    let t = BpeTokenizer::from_bytes(r.take(tokenizer_len)?).map_err(|_| E::Tokenizer)?;
    if t.fingerprint() != tokenizer_hash
        || t.vocab_size() != vocab_size
        || t.max_tokens() != max_tokens
    {
        return Err(E::Tokenizer);
    }
    let metadata = BpeCheckpointMetadata {
        config: c,
        candidate_cap: cap,
        parameter_count: count,
        checkpoint_bytes: bytes.len(),
        checkpoint_hash: expected_hash,
        tokenizer_hash,
    };
    Ok((metadata, t, r))
}

/// Reject wrong digest, unknown schema, excess bytes, incompatible vocabulary or nonfinite tensors.
pub fn load_checkpoint(
    bytes: &[u8],
    expected_hash: [u8; 32],
) -> Result<BpeCognitiveModel, BpeCheckpointError> {
    use BpeCheckpointError as E;
    let (metadata, t, mut r) = parse_checkpoint(bytes, expected_hash)?;
    let c = metadata.config;
    let vocab_size = c.encoder.vocab_size;
    let max_tokens = c.encoder.max_tokens;
    let embedding_dim = c.encoder.embedding_dim;
    let hidden_dim = c.encoder.hidden_dim;
    let num_classes = c.num_classes;
    let num_rules = c.num_rules;
    let tokenizer_hash = metadata.tokenizer_hash;
    let cap = metadata.candidate_cap;
    let encoder = SequenceEncoder::from_parts(
        c.encoder,
        r.floats(vocab_size * embedding_dim)?,
        r.floats(max_tokens * embedding_dim)?,
        r.floats(embedding_dim * hidden_dim)?,
    )
    .map_err(|_| E::Weights)?;
    let heads = SequenceCognitiveHeads::from_parts(
        c,
        encoder,
        r.floats(hidden_dim * num_classes)?,
        r.floats(num_classes)?,
        r.floats(hidden_dim)?,
        r.floats(1)?,
        r.floats(hidden_dim * num_rules)?,
        r.floats(num_rules)?,
        r.floats(hidden_dim * 2)?,
        r.floats(2)?,
    )
    .map_err(|_| E::Weights)?;
    if r.pos != bytes.len() {
        return Err(E::Size);
    }
    BpeCognitiveModel::from_heads(t, heads, tokenizer_hash, cap).map_err(|_| E::Configuration)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], BpeCheckpointError> {
        let end = self.pos.checked_add(n).ok_or(BpeCheckpointError::Size)?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(BpeCheckpointError::Size)?;
        self.pos = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<usize, BpeCheckpointError> {
        let mut b = [0; 2];
        b.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(b) as usize)
    }
    fn u64(&mut self) -> Result<u64, BpeCheckpointError> {
        let mut b = [0; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, BpeCheckpointError> {
        let raw = self.take(n.checked_mul(4).ok_or(BpeCheckpointError::Size)?)?;
        raw.chunks_exact(4)
            .map(|b| {
                let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(BpeCheckpointError::Weights)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe_cognitive::tests::fixture;
    #[test]
    fn inspection_matches_model_metadata_without_exposing_weights() {
        let model = fixture();
        let bytes = encode_checkpoint(&model);
        let metadata = inspect_checkpoint(&bytes, checkpoint_hash(&bytes)).unwrap();
        assert_eq!(metadata.config, model.heads().config());
        assert_eq!(metadata.parameter_count, model.heads().parameter_count());
        assert_eq!(metadata.candidate_cap, model.candidate_cap());
        assert_eq!(metadata.tokenizer_hash, model.tokenizer().fingerprint());
        assert_eq!(metadata.checkpoint_bytes, bytes.len());
        assert_eq!(metadata.checkpoint_hash, checkpoint_hash(&bytes));
        assert_eq!(
            inspect_checkpoint(&bytes, [0; 32]),
            Err(BpeCheckpointError::Hash)
        );
    }

    #[test]
    fn inspector_and_loader_agree_on_corruption_and_nonfinite_weights() {
        let bytes = encode_checkpoint(&fixture());
        for end in [0, 7, 95, 96, bytes.len() - 1] {
            let raw = &bytes[..end];
            assert_eq!(
                inspect_checkpoint(raw, checkpoint_hash(raw)).unwrap_err(),
                load_checkpoint(raw, checkpoint_hash(raw)).unwrap_err()
            );
        }
        for offset in [0, 7, 8, 10, 12, 14, 16, 18, 20, 62, 64, 96] {
            let mut raw = bytes.clone();
            raw[offset] ^= 255;
            assert_eq!(
                inspect_checkpoint(&raw, checkpoint_hash(&raw)).unwrap_err(),
                load_checkpoint(&raw, checkpoint_hash(&raw)).unwrap_err()
            );
        }
        let token_bytes = u16::from_le_bytes([bytes[62], bytes[63]]) as usize;
        for offset in [HEADER + token_bytes, bytes.len() - 4] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut raw = bytes.clone();
                raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                assert_eq!(
                    inspect_checkpoint(&raw, checkpoint_hash(&raw)),
                    Err(BpeCheckpointError::Weights)
                );
                assert_eq!(
                    load_checkpoint(&raw, checkpoint_hash(&raw)),
                    Err(BpeCheckpointError::Weights)
                );
            }
        }
        let mut trailing = bytes;
        trailing.extend_from_slice(&[0; 4]);
        assert_eq!(
            inspect_checkpoint(&trailing, checkpoint_hash(&trailing)),
            Err(BpeCheckpointError::Size)
        );
    }

    #[test]
    fn exact_state_and_all_signal_roundtrip() {
        let m = fixture();
        let bytes = encode_checkpoint(&m);
        let restored = load_checkpoint(&bytes, checkpoint_hash(&bytes)).unwrap();
        assert_eq!(m, restored);
        assert_eq!(bytes, encode_checkpoint(&restored));
        assert_eq!(m.classify(b"abab"), restored.classify(b"abab"));
        assert_eq!(m.preference(b"ab"), restored.preference(b"ab"));
        assert_eq!(m.symbolic(b"ab"), restored.symbolic(b"ab"));
        assert_eq!(
            m.contradiction(b"ab", b"ba"),
            restored.contradiction(b"ab", b"ba")
        );
        assert_eq!(
            m.retrieve(b"ab", &[b"a", b"b"]),
            restored.retrieve(b"ab", &[b"a", b"b"])
        );
    }
    #[test]
    fn hostile_checkpoint_rejected_even_with_recomputed_digest() {
        let m = fixture();
        let good = encode_checkpoint(&m);
        assert_eq!(
            load_checkpoint(&good, [0; 32]),
            Err(BpeCheckpointError::Hash)
        );
        for end in [0, 7, 95, 96, good.len() - 1] {
            let b = &good[..end];
            assert!(load_checkpoint(b, checkpoint_hash(b)).is_err());
        }
        for offset in [0, 7, 8, 10, 12, 14, 16, 18, 20, 62, 64, 96] {
            let mut b = good.clone();
            b[offset] ^= 255;
            assert!(
                load_checkpoint(&b, checkpoint_hash(&b)).is_err(),
                "offset {offset}"
            );
        }
        let mut b = good.clone();
        b.extend([0; 4]);
        assert!(load_checkpoint(&b, checkpoint_hash(&b)).is_err());
        let start = HEADER + m.tokenizer().to_bytes().len();
        let mut b = good.clone();
        b[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            load_checkpoint(&b, checkpoint_hash(&b)),
            Err(BpeCheckpointError::Weights)
        );
        let mut b = good.clone();
        b[..8].copy_from_slice(b"CGCOG004");
        assert_eq!(
            load_checkpoint(&b, checkpoint_hash(&b)),
            Err(BpeCheckpointError::Format)
        );
    }
}
