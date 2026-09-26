//! Bounded, byte-preserving BPE research tokenizer (artifact CBPE0001).
//!
//! Implements sequential rank-priority merges, the algorithmic contract also
//! used by SciRust's canonical BPE trainer. This implementation is independent:
//! COGNO framing IDs, hostile-input bounds and binary schema are different.
//! It does not reinterpret or activate any V1–V4 model artifact.

use crate::tokenizer::{BOS_TOKEN, EOS_TOKEN, SEP_TOKEN};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const BASE: usize = 259;
const MAGIC: &[u8; 8] = b"CBPE0001";
/// Vocabulary bound compatible with the current sequence encoder.
pub const MAX_BPE_VOCAB: usize = 512;
/// Maximum input or reconstructed payload bytes per example/pair.
pub const MAX_BPE_BYTES: usize = 16_384;
/// Maximum bytes admitted to the reference in-memory trainer.
pub const MAX_BPE_TRAIN_BYTES: usize = 1_048_576;
/// Maximum records admitted to the reference trainer.
pub const MAX_BPE_TRAIN_RECORDS: usize = 4096;
const MAX_TOKEN_BYTES: usize = 1024;

/// Fail-closed format, capacity and framing errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpeError {
    InvalidConfig,
    Capacity,
    InvalidMerge,
    InvalidArtifact,
    InvalidFraming,
    UnknownToken,
    EmptyTraining,
}

/// Immutable tokenizer: raw bytes 0..255, framing 256..258, merges from 259.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokenizer {
    max_tokens: usize,
    merges: Vec<(u16, u16)>,
    pieces: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    /// Validate all ranks before accepting an externally supplied vocabulary.
    pub fn from_merges(max_tokens: usize, merges: &[(u16, u16)]) -> Result<Self, BpeError> {
        if !(3..=512).contains(&max_tokens) || merges.len() > MAX_BPE_VOCAB - BASE {
            return Err(BpeError::InvalidConfig);
        }
        let mut pieces: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
        pieces.extend([vec![], vec![], vec![]]);
        let mut seen = BTreeSet::new();
        for &(a, b) in merges {
            let a = usize::from(a);
            let b = usize::from(b);
            if a >= pieces.len()
                || b >= pieces.len()
                || (256..BASE).contains(&a)
                || (256..BASE).contains(&b)
            {
                return Err(BpeError::InvalidMerge);
            }
            if pieces[a].len() + pieces[b].len() > MAX_TOKEN_BYTES {
                return Err(BpeError::Capacity);
            }
            let mut piece = pieces[a].clone();
            piece.extend_from_slice(&pieces[b]);
            if !seen.insert(piece.clone()) {
                return Err(BpeError::InvalidMerge);
            }
            pieces.push(piece);
        }
        Ok(Self {
            max_tokens,
            merges: merges.to_vec(),
            pieces,
        })
    }

    /// Learn only from a caller-provided TRAIN partition, never infer a split.
    /// Frequency ties use the smallest pair of IDs; merges never cross records.
    pub fn train(train: &[&[u8]], vocab: usize, max_tokens: usize) -> Result<Self, BpeError> {
        if !(BASE..=MAX_BPE_VOCAB).contains(&vocab) {
            return Err(BpeError::InvalidConfig);
        }
        if train.len() > MAX_BPE_TRAIN_RECORDS {
            return Err(BpeError::Capacity);
        }
        let mut total = 0usize;
        for text in train {
            if text.len() > MAX_BPE_BYTES {
                return Err(BpeError::Capacity);
            }
            total = total.checked_add(text.len()).ok_or(BpeError::Capacity)?;
            if total > MAX_BPE_TRAIN_BYTES {
                return Err(BpeError::Capacity);
            }
        }
        if total == 0 {
            return Err(BpeError::EmptyTraining);
        }
        let mut model = Self::from_merges(max_tokens, &[])?;
        let mut records: Vec<Vec<u16>> = train
            .iter()
            .map(|s| s.iter().map(|&b| u16::from(b)).collect())
            .collect();
        while model.vocab_size() < vocab {
            let mut counts = BTreeMap::<(u16, u16), usize>::new();
            for record in &records {
                for pair in record.windows(2) {
                    let key = (pair[0], pair[1]);
                    if model.pieces[usize::from(key.0)].len()
                        + model.pieces[usize::from(key.1)].len()
                        <= MAX_TOKEN_BYTES
                    {
                        *counts.entry(key).or_default() += 1;
                    }
                }
            }
            let mut best = None;
            for (pair, frequency) in counts {
                if frequency >= 2 && best.is_none_or(|(_, n)| frequency > n) {
                    best = Some((pair, frequency));
                }
            }
            let Some((pair, _)) = best else {
                break;
            };
            let id = model.vocab_size() as u16;
            let mut merges = model.merges.clone();
            merges.push(pair);
            model = Self::from_merges(max_tokens, &merges)?;
            for record in &mut records {
                merge(record, pair, id);
            }
        }
        Ok(model)
    }

    /// Actual vocabulary size (training can stop before the requested target).
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    /// Maximum framed token count accepted by this artifact.
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn raw(&self, bytes: &[u8]) -> Result<Vec<u16>, BpeError> {
        if bytes.len() > MAX_BPE_BYTES {
            return Err(BpeError::Capacity);
        }
        let mut ids: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();
        for (rank, &pair) in self.merges.iter().enumerate() {
            merge(&mut ids, pair, (BASE + rank) as u16);
        }
        Ok(ids)
    }

    /// Encode one payload without truncation, including BOS/EOS.
    pub fn encode(&self, bytes: &[u8]) -> Result<Vec<u16>, BpeError> {
        let ids = self.raw(bytes)?;
        if ids.len() + 2 > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        let mut out = vec![BOS_TOKEN];
        out.extend(ids);
        out.push(EOS_TOKEN);
        Ok(out)
    }

    /// Encode independently on each side; no merge can cross SEP.
    pub fn encode_pair(&self, left: &[u8], right: &[u8]) -> Result<Vec<u16>, BpeError> {
        if left
            .len()
            .checked_add(right.len())
            .ok_or(BpeError::Capacity)?
            > MAX_BPE_BYTES
        {
            return Err(BpeError::Capacity);
        }
        let left = self.raw(left)?;
        let right = self.raw(right)?;
        if left.len() + right.len() + 3 > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        let mut out = vec![BOS_TOKEN];
        out.extend(left);
        out.push(SEP_TOKEN);
        out.extend(right);
        out.push(EOS_TOKEN);
        Ok(out)
    }

    /// Decode a single framed payload exactly, including non-UTF-8 bytes.
    pub fn decode(&self, ids: &[u16]) -> Result<Vec<u8>, BpeError> {
        if ids.len() < 2 || ids.first() != Some(&BOS_TOKEN) || ids.last() != Some(&EOS_TOKEN) {
            return Err(BpeError::InvalidFraming);
        }
        if ids.len() > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        let mut size = 0usize;
        for &id in &ids[1..ids.len() - 1] {
            if (BOS_TOKEN..=SEP_TOKEN).contains(&id) {
                return Err(BpeError::InvalidFraming);
            }
            let piece = self
                .pieces
                .get(usize::from(id))
                .ok_or(BpeError::UnknownToken)?;
            size += piece.len();
            if size > MAX_BPE_BYTES {
                return Err(BpeError::Capacity);
            }
        }
        let mut out = Vec::with_capacity(size);
        for &id in &ids[1..ids.len() - 1] {
            out.extend_from_slice(&self.pieces[usize::from(id)]);
        }
        Ok(out)
    }

    /// Canonical little-endian schema: magic, max_tokens, merge count, pairs.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(self.max_tokens as u16).to_le_bytes());
        out.extend_from_slice(&(self.merges.len() as u16).to_le_bytes());
        for &(a, b) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Reject oversized, unknown-version, truncated and trailing-byte artifacts.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BpeError> {
        if bytes.len() < 12 || bytes.len() > 12 + 4 * (MAX_BPE_VOCAB - BASE) || &bytes[..8] != MAGIC
        {
            return Err(BpeError::InvalidArtifact);
        }
        let max = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        let count = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
        if bytes.len() != 12 + count * 4 {
            return Err(BpeError::InvalidArtifact);
        }
        let merges: Vec<_> = bytes[12..]
            .chunks_exact(4)
            .map(|p| {
                (
                    u16::from_le_bytes([p[0], p[1]]),
                    u16::from_le_bytes([p[2], p[3]]),
                )
            })
            .collect();
        Self::from_merges(max, &merges)
    }

    /// Identity binds the ordered vocabulary, framing version and context cap.
    pub fn fingerprint(&self) -> [u8; 32] {
        Sha256::digest(self.to_bytes()).into()
    }
}

fn merge(ids: &mut Vec<u16>, pair: (u16, u16), output: u16) {
    let mut read = 0;
    let mut write = 0;
    while read < ids.len() {
        if read + 1 < ids.len() && (ids[read], ids[read + 1]) == pair {
            ids[write] = output;
            read += 2;
        } else {
            ids[write] = ids[read];
            read += 1;
        }
        write += 1;
    }
    ids.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_all_bytes_and_unseen_identifiers() {
        let t = BpeTokenizer::train(&[b"fn main() {} fn main() {}"], 300, 512).unwrap();
        for bytes in [
            (0..=255).collect::<Vec<u8>>(),
            b"\xff\0\n\t fn unseen_xyz() {}".to_vec(),
            vec![],
        ] {
            assert_eq!(t.decode(&t.encode(&bytes).unwrap()).unwrap(), bytes);
        }
    }
    #[test]
    fn deterministic_roundtrip_and_compression() {
        let data: [&[u8]; 2] = [b"abcabc", b"abcabc"];
        let a = BpeTokenizer::train(&data, 300, 512).unwrap();
        let b = BpeTokenizer::train(&[data[1], data[0]], 300, 512).unwrap();
        assert_eq!(a, b);
        assert!(a.encode(data[0]).unwrap().len() < data[0].len() + 2);
        assert_eq!(BpeTokenizer::from_bytes(&a.to_bytes()).unwrap(), a);
        assert_ne!(
            a.fingerprint(),
            BpeTokenizer::from_merges(128, &a.merges)
                .unwrap()
                .fingerprint()
        );
    }
    #[test]
    fn ranks_overlaps_and_pair_boundary() {
        let a = BpeTokenizer::from_merges(512, &[(97, 97), (259, 97)]).unwrap();
        assert_eq!(a.encode(b"aaa").unwrap(), vec![BOS_TOKEN, 260, EOS_TOKEN]);
        assert_eq!(
            a.encode_pair(b"a", b"a").unwrap(),
            vec![BOS_TOKEN, 97, SEP_TOKEN, 97, EOS_TOKEN]
        );
        let a = BpeTokenizer::from_merges(512, &[(98, 99), (97, 98)]).unwrap();
        assert_eq!(
            a.encode(b"abc").unwrap(),
            vec![BOS_TOKEN, 97, 259, EOS_TOKEN]
        );
    }
    #[test]
    fn hostile_artifacts_and_ids_are_rejected() {
        for rules in [
            vec![(259, 0)],
            vec![(BOS_TOKEN, 97)],
            vec![(97, 98), (97, 98)],
        ] {
            assert!(BpeTokenizer::from_merges(512, &rules).is_err());
        }
        let t = BpeTokenizer::from_merges(512, &[]).unwrap();
        let good = t.to_bytes();
        for end in 0..good.len() {
            assert!(BpeTokenizer::from_bytes(&good[..end]).is_err());
        }
        let mut extra = good.clone();
        extra.push(0);
        assert!(BpeTokenizer::from_bytes(&extra).is_err());
        assert!(t.decode(&[BOS_TOKEN, 999, EOS_TOKEN]).is_err());
        assert!(t.decode(&[BOS_TOKEN, SEP_TOKEN, EOS_TOKEN]).is_err());
        assert!(t.decode(&[97, 98]).is_err());
    }
    #[test]
    fn bounded_before_training_and_after_compression() {
        assert!(BpeTokenizer::train(&[], 300, 512).is_err());
        assert!(BpeTokenizer::train(&[b"a"], 513, 512).is_err());
        let t = BpeTokenizer::from_merges(3, &[]).unwrap();
        assert!(t.encode(b"aa").is_err());
        assert!(t.encode(&vec![0; MAX_BPE_BYTES + 1]).is_err());
        assert!(BpeTokenizer::train(&[&vec![0; MAX_BPE_BYTES + 1]], 300, 512).is_err());
        let t = BpeTokenizer::from_merges(3, &[(97, 97)]).unwrap();
        assert_eq!(t.decode(&t.encode(b"aa").unwrap()).unwrap(), b"aa");
    }

    #[test]
    fn trainer_and_expansion_limits_are_enforced() {
        assert_eq!(
            BpeTokenizer::train(&vec![&b"a"[..]; MAX_BPE_TRAIN_RECORDS + 1], 259, 512),
            Err(BpeError::Capacity)
        );
        let record = vec![0; MAX_BPE_BYTES];
        assert_eq!(
            BpeTokenizer::train(&vec![record.as_slice(); 65], 259, 512),
            Err(BpeError::Capacity)
        );
        let mut rules = vec![(97, 97)];
        for id in 259..268 {
            rules.push((id, id));
        }
        let t = BpeTokenizer::from_merges(512, &rules).unwrap();
        let mut ids = vec![BOS_TOKEN];
        ids.extend([268; 17]);
        ids.push(EOS_TOKEN);
        assert_eq!(t.decode(&ids), Err(BpeError::Capacity));
        rules.push((268, 268));
        assert_eq!(
            BpeTokenizer::from_merges(512, &rules),
            Err(BpeError::Capacity)
        );
        let mut artifact = t.to_bytes();
        artifact[7] = b'2';
        assert_eq!(
            BpeTokenizer::from_bytes(&artifact),
            Err(BpeError::InvalidArtifact)
        );
    }

    #[test]
    fn varied_binary_inputs_roundtrip_after_artifact_reload() {
        let t =
            BpeTokenizer::train(&[b"abababab", b"abcabcabc", b"\0\xff\0\xff"], 300, 512).unwrap();
        let t = BpeTokenizer::from_bytes(&t.to_bytes()).unwrap();
        let mut state = 42u32;
        for len in 0..=510 {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8
                })
                .collect();
            assert_eq!(t.decode(&t.encode(&bytes).unwrap()).unwrap(), bytes);
        }
    }
}
