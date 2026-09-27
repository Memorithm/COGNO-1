//! Bounded, byte-preserving BPE research tokenizer (artifact CBPE0001).
//!
//! Implements sequential rank-priority merges, the algorithmic contract also
//! used by SciRust's canonical BPE trainer. This implementation is independent:
//! COGNO framing IDs, hostile-input bounds and binary schema are different.
//! It does not reinterpret or activate any V1–V4 model artifact.

use crate::tokenizer::{BOS_TOKEN, EOS_TOKEN, SEP_TOKEN};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

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
    ranks: BTreeMap<(u16, u16), u16>,
}

/// Exact half-open byte range in the original source for one framed token.
/// BOS and EOS have zero-width ranges. Offsets are not Unicode character
/// positions: an arbitrary byte tokenizer may split a UTF-8 code point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokenSpan {
    pub token: u16,
    pub start: usize,
    pub end: usize,
}

/// Caller-owned scratch space for rank-heap encoding. Reusing this value avoids
/// rebuilding its allocations. It stores no model parameters and may be reused
/// with a different tokenizer. Input remains bounded by `MAX_BPE_BYTES`.
#[derive(Debug, Default)]
pub struct BpeWorkspace {
    nodes: Vec<BpeNode>,
    candidates: BinaryHeap<Reverse<(u16, usize, usize)>>,
    output: Vec<u16>,
}

#[derive(Debug)]
struct BpeNode {
    id: u16,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

impl BpeWorkspace {
    /// Empty scratch buffers; storage grows only for admitted bounded inputs.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BpeTokenizer {
    /// Validate all ranks before accepting an externally supplied vocabulary.
    pub fn from_merges(max_tokens: usize, merges: &[(u16, u16)]) -> Result<Self, BpeError> {
        if !(3..=512).contains(&max_tokens) || merges.len() > MAX_BPE_VOCAB - BASE {
            return Err(BpeError::InvalidConfig);
        }
        let mut pieces: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
        pieces.extend([vec![], vec![], vec![]]);
        let mut model = Self {
            max_tokens,
            merges: Vec::with_capacity(merges.len()),
            pieces,
            ranks: BTreeMap::new(),
        };
        for &pair in merges {
            model.append_merge(pair)?;
        }
        Ok(model)
    }

    // Validate just the next rank. The already accepted prefix is immutable.
    fn append_merge(&mut self, pair: (u16, u16)) -> Result<(), BpeError> {
        let (a, b) = (usize::from(pair.0), usize::from(pair.1));
        if a >= self.pieces.len()
            || b >= self.pieces.len()
            || (256..BASE).contains(&a)
            || (256..BASE).contains(&b)
        {
            return Err(BpeError::InvalidMerge);
        }
        if self.pieces[a].len() + self.pieces[b].len() > MAX_TOKEN_BYTES {
            return Err(BpeError::Capacity);
        }
        let mut piece = self.pieces[a].clone();
        piece.extend_from_slice(&self.pieces[b]);
        if self.pieces[BASE..].contains(&piece) {
            return Err(BpeError::InvalidMerge);
        }
        self.pieces.push(piece);
        self.ranks.insert(pair, self.merges.len() as u16);
        self.merges.push(pair);
        Ok(())
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
        let mut counts = BTreeMap::<(u16, u16), usize>::new();
        for record in &records {
            for pair in record.windows(2) {
                *counts.entry((pair[0], pair[1])).or_default() += 1;
            }
        }
        while model.vocab_size() < vocab {
            let mut best = None;
            for (&pair, &frequency) in &counts {
                if frequency >= 2 && best.is_none_or(|(_, n)| frequency > n) {
                    best = Some((pair, frequency));
                }
            }
            let Some((pair, _)) = best else {
                break;
            };
            let id = model.vocab_size() as u16;
            model.append_merge(pair)?;
            for record in &mut records {
                merge_counted(record, pair, id, &mut counts, &model.pieces);
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

    /// Measure framed length even when it exceeds this artifact's context cap.
    /// The raw input byte bound still applies. This never authorizes inference.
    pub fn required_tokens(&self, bytes: &[u8]) -> Result<usize, BpeError> {
        Ok(self.raw(bytes)?.len() + 2)
    }

    /// Measure an independently encoded pair, including BOS/SEP/EOS.
    pub fn required_pair_tokens(&self, left: &[u8], right: &[u8]) -> Result<usize, BpeError> {
        if left
            .len()
            .checked_add(right.len())
            .ok_or(BpeError::Capacity)?
            > MAX_BPE_BYTES
        {
            return Err(BpeError::Capacity);
        }
        Ok(self.raw(left)?.len() + self.raw(right)?.len() + 3)
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

    /// Encode with lossless source locations for Rust diagnostics and attribution.
    /// Uses the same capacity checks and IDs as `encode`, including framing.
    pub fn encode_with_offsets(&self, bytes: &[u8]) -> Result<Vec<BpeTokenSpan>, BpeError> {
        let ids = self.encode(bytes)?;
        let mut cursor = 0;
        Ok(ids
            .into_iter()
            .map(|token| {
                let start = cursor;
                cursor += self.pieces[usize::from(token)].len();
                BpeTokenSpan {
                    token,
                    start,
                    end: cursor,
                }
            })
            .collect())
    }

    /// Exact rank-priority encoding using a reusable heap and linked positions.
    /// This opt-in path has the same framing, IDs and refusal rules as `encode`.
    /// An error leaves no usable output in the workspace. Stale heap entries are
    /// discarded by validating live adjacency before each merge.
    pub fn encode_with_workspace<'a>(
        &self,
        bytes: &[u8],
        workspace: &'a mut BpeWorkspace,
    ) -> Result<&'a [u16], BpeError> {
        workspace.output.clear();
        if bytes.len() > MAX_BPE_BYTES {
            return Err(BpeError::Capacity);
        }
        workspace.nodes.clear();
        workspace.candidates.clear();
        workspace
            .nodes
            .extend(bytes.iter().enumerate().map(|(i, &b)| BpeNode {
                id: u16::from(b),
                prev: i.checked_sub(1),
                next: (i + 1 < bytes.len()).then_some(i + 1),
                alive: true,
            }));
        for left in 0..bytes.len().saturating_sub(1) {
            self.queue_candidate(workspace, left);
        }
        let mut count = bytes.len();
        while let Some(Reverse((rank, left, right))) = workspace.candidates.pop() {
            if !workspace.nodes[left].alive
                || !workspace.nodes[right].alive
                || workspace.nodes[left].next != Some(right)
                || (workspace.nodes[left].id, workspace.nodes[right].id)
                    != self.merges[usize::from(rank)]
            {
                continue;
            }
            workspace.nodes[left].id = (BASE + usize::from(rank)) as u16;
            workspace.nodes[left].next = workspace.nodes[right].next;
            workspace.nodes[right].alive = false;
            if let Some(next) = workspace.nodes[left].next {
                workspace.nodes[next].prev = Some(left);
            }
            count -= 1;
            if let Some(prev) = workspace.nodes[left].prev {
                self.queue_candidate(workspace, prev);
            }
            self.queue_candidate(workspace, left);
        }
        if count + 2 > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        workspace.output.push(BOS_TOKEN);
        workspace
            .output
            .extend(workspace.nodes.iter().filter(|n| n.alive).map(|n| n.id));
        workspace.output.push(EOS_TOKEN);
        Ok(&workspace.output)
    }

    fn queue_candidate(&self, workspace: &mut BpeWorkspace, left: usize) {
        if let Some(right) = workspace.nodes[left].next {
            let pair = (workspace.nodes[left].id, workspace.nodes[right].id);
            if let Some(&rank) = self.ranks.get(&pair) {
                workspace.candidates.push(Reverse((rank, left, right)));
            }
        }
    }

    /// Encode bounded batches while retaining one outcome per source in order.
    /// Aggregate bounds are checked before any encoding. A source exceeding its
    /// byte/context cap is a per-record refusal, never silently skipped or cut.
    /// The outer limit is 4096 records and 1 MiB total source bytes; workspace
    /// allocation is reused across records, and returned token vectors are owned.
    pub fn encode_batch(
        &self,
        sources: &[&[u8]],
        workspace: &mut BpeWorkspace,
    ) -> Result<Vec<Result<Vec<u16>, BpeError>>, BpeError> {
        workspace.output.clear();
        if sources.len() > MAX_BPE_TRAIN_RECORDS {
            return Err(BpeError::Capacity);
        }
        let total = sources
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.len()))
            .ok_or(BpeError::Capacity)?;
        if total > MAX_BPE_TRAIN_BYTES {
            return Err(BpeError::Capacity);
        }
        Ok(sources
            .iter()
            .map(|source| {
                self.encode_with_workspace(source, workspace)
                    .map(<[u16]>::to_vec)
            })
            .collect())
    }

    /// Training augmentation only: apply exactly the first `merge_count` ranks.
    /// Zero exposes byte fallback IDs; the full prefix is identical to `encode`.
    /// This does not mutate the tokenizer or change its inference/artifact contract.
    /// Invalid prefixes and sequences exceeding the usual context cap are refused.
    pub fn encode_with_merge_prefix(
        &self,
        bytes: &[u8],
        merge_count: usize,
    ) -> Result<Vec<u16>, BpeError> {
        if merge_count > self.merges.len() {
            return Err(BpeError::InvalidConfig);
        }
        if bytes.len() > MAX_BPE_BYTES {
            return Err(BpeError::Capacity);
        }
        let mut ids: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();
        for (rank, &pair) in self.merges[..merge_count].iter().enumerate() {
            merge(&mut ids, pair, (BASE + rank) as u16);
        }
        if ids.len() + 2 > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        ids.insert(0, BOS_TOKEN);
        ids.push(EOS_TOKEN);
        Ok(ids)
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
        let payload = &ids[1..ids.len() - 1];
        let size = self.decoded_size(payload)?;
        Ok(self.decode_payload(payload, size))
    }

    /// Decode exactly one BOS/left/SEP/right/EOS frame, bounding both sides together.
    pub fn decode_pair(&self, ids: &[u16]) -> Result<(Vec<u8>, Vec<u8>), BpeError> {
        if ids.len() < 3 || ids.first() != Some(&BOS_TOKEN) || ids.last() != Some(&EOS_TOKEN) {
            return Err(BpeError::InvalidFraming);
        }
        if ids.len() > self.max_tokens {
            return Err(BpeError::Capacity);
        }
        let payload = &ids[1..ids.len() - 1];
        let separator = payload
            .iter()
            .position(|&id| id == SEP_TOKEN)
            .ok_or(BpeError::InvalidFraming)?;
        let left = &payload[..separator];
        let right = &payload[separator + 1..];
        let left_size = self.decoded_size(left)?;
        let right_size = self.decoded_size(right)?;
        if left_size + right_size > MAX_BPE_BYTES {
            return Err(BpeError::Capacity);
        }
        Ok((
            self.decode_payload(left, left_size),
            self.decode_payload(right, right_size),
        ))
    }

    fn decoded_size(&self, ids: &[u16]) -> Result<usize, BpeError> {
        let mut size = 0usize;
        for &id in ids {
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
        Ok(size)
    }

    fn decode_payload(&self, ids: &[u16], size: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(size);
        for &id in ids {
            out.extend_from_slice(&self.pieces[usize::from(id)]);
        }
        out
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

// Update only adjacency edges changed by a left-to-right non-overlapping merge.
// In particular, repeated symbols count overlapping candidates during selection,
// while application still merges disjoint occurrences, matching the reference.
fn merge_counted(
    ids: &mut Vec<u16>,
    pair: (u16, u16),
    output: u16,
    counts: &mut BTreeMap<(u16, u16), usize>,
    pieces: &[Vec<u8>],
) {
    let mut read = 0;
    let mut write = 0usize;
    while read < ids.len() {
        if read + 1 < ids.len() && (ids[read], ids[read + 1]) == pair {
            let prev = write.checked_sub(1).map(|p| ids[p]);
            let next = ids.get(read + 2).copied();
            for edge in [
                prev.map(|v| (v, pair.0)),
                Some(pair),
                next.map(|v| (pair.1, v)),
            ]
            .into_iter()
            .flatten()
            {
                if pieces[usize::from(edge.0)].len() + pieces[usize::from(edge.1)].len()
                    <= MAX_TOKEN_BYTES
                {
                    let n = counts.get_mut(&edge).expect("counted live adjacency");
                    *n -= 1;
                    if *n == 0 {
                        counts.remove(&edge);
                    }
                }
            }
            for edge in [prev.map(|v| (v, output)), next.map(|v| (output, v))]
                .into_iter()
                .flatten()
            {
                if pieces[usize::from(edge.0)].len() + pieces[usize::from(edge.1)].len()
                    <= MAX_TOKEN_BYTES
                {
                    *counts.entry(edge).or_default() += 1;
                }
            }
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

fn merge(ids: &mut Vec<u16>, pair: (u16, u16), output: u16) {
    // Leave unchanged prefixes untouched and return immediately for absent pairs.
    let Some(first) = ids.windows(2).position(|w| (w[0], w[1]) == pair) else {
        return;
    };
    let mut read = first;
    let mut write = first;
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
    fn reference_train(train: &[&[u8]], vocab: usize) -> BpeTokenizer {
        let mut model = BpeTokenizer::from_merges(512, &[]).unwrap();
        let mut records: Vec<Vec<u16>> = train
            .iter()
            .map(|s| s.iter().map(|&b| u16::from(b)).collect())
            .collect();
        while model.vocab_size() < vocab {
            let mut counts = BTreeMap::<(u16, u16), usize>::new();
            for r in &records {
                for p in r.windows(2) {
                    if model.pieces[p[0] as usize].len() + model.pieces[p[1] as usize].len()
                        <= MAX_TOKEN_BYTES
                    {
                        *counts.entry((p[0], p[1])).or_default() += 1;
                    }
                }
            }
            let mut best = None;
            for (p, n) in counts {
                if n >= 2 && best.is_none_or(|(_, old)| n > old) {
                    best = Some((p, n));
                }
            }
            let Some((p, _)) = best else {
                break;
            };
            let id = model.vocab_size() as u16;
            let mut rules = model.merges.clone();
            rules.push(p);
            model = BpeTokenizer::from_merges(512, &rules).unwrap();
            for r in &mut records {
                reference_merge(r, p, id);
            }
        }
        model
    }

    #[test]
    fn batch_preserves_refusals_order_and_aggregate_bounds() {
        let t = BpeTokenizer::from_merges(4, &[(97, 98)]).unwrap();
        let mut workspace = BpeWorkspace::new();
        let oversized = vec![0; MAX_BPE_BYTES + 1];
        let sources = [b"ab".as_slice(), b"abc", b"abcd", oversized.as_slice(), b""];
        assert_eq!(
            t.encode_batch(&sources, &mut workspace).unwrap(),
            sources.iter().map(|s| t.encode(s)).collect::<Vec<_>>()
        );
        assert!(t.encode_batch(&[], &mut workspace).unwrap().is_empty());
        assert_eq!(
            t.encode_batch(&vec![b"".as_slice(); MAX_BPE_TRAIN_RECORDS + 1], &mut workspace),
            Err(BpeError::Capacity)
        );
        let full = vec![0; MAX_BPE_TRAIN_BYTES];
        assert_eq!(
            t.encode_batch(&[&full, b"a"], &mut workspace),
            Err(BpeError::Capacity)
        );
        assert!(workspace.output.is_empty());
        assert_eq!(
            t.encode_batch(&[b"ab"], &mut workspace).unwrap(),
            [t.encode(b"ab")]
        );
    }

    #[test]
    fn byte_offsets_cover_exact_rust_source_and_binary_payloads() {
        let source =
            r####"fn café<'a>(s: &'a str) -> &'a str { r###"日本語\n"###; s }"####.as_bytes();
        let t = BpeTokenizer::train(&[source, source], 300, 512).unwrap();
        for bytes in [source, b"\xff\0", b""] {
            let spans = t.encode_with_offsets(bytes).unwrap();
            assert_eq!(
                spans.iter().map(|s| s.token).collect::<Vec<_>>(),
                t.encode(bytes).unwrap()
            );
            let mut cursor = 0;
            for span in &spans {
                assert_eq!(span.start, cursor);
                assert_eq!(&bytes[span.start..span.end], t.pieces[span.token as usize]);
                cursor = span.end;
            }
            assert_eq!(cursor, bytes.len());
            assert_eq!(spans.first().unwrap().start, 0);
            assert_eq!(spans.last().unwrap().end, bytes.len());
        }
        let tiny = BpeTokenizer::from_merges(3, &[]).unwrap();
        assert_eq!(tiny.encode_with_offsets(b"ab"), Err(BpeError::Capacity));
    }

    #[test]
    fn heap_encoding_matches_rank_scans_and_workspace_reuse() {
        let mut workspace = BpeWorkspace::new();
        for rules in [
            vec![],
            vec![(0, 0), (259, 0), (1, 2)],
            vec![(1, 2), (0, 1), (0, 259)],
        ] {
            let t = BpeTokenizer::from_merges(512, &rules).unwrap();
            for n in 0..=8u32 {
                for mut x in 0..3usize.pow(n) {
                    let bytes: Vec<u8> = (0..n)
                        .map(|_| {
                            let b = (x % 3) as u8;
                            x /= 3;
                            b
                        })
                        .collect();
                    assert_eq!(
                        t.encode_with_workspace(&bytes, &mut workspace).unwrap(),
                        t.encode(&bytes).unwrap()
                    );
                }
            }
        }
        let t = BpeTokenizer::train(&[b"fn main() { let x = vec![1,2]; }"], 300, 512).unwrap();
        for n in [0, 1, 511, 512, MAX_BPE_BYTES, MAX_BPE_BYTES + 1] {
            let bytes = vec![b'x'; n];
            assert_eq!(
                t.encode_with_workspace(&bytes, &mut workspace)
                    .map(<[u16]>::to_vec),
                t.encode(&bytes)
            );
        }
        assert!(workspace.output.is_empty());
        assert_eq!(
            t.encode_with_workspace(b"", &mut workspace).unwrap(),
            [BOS_TOKEN, EOS_TOKEN]
        );
    }

    #[test]
    fn maintained_training_counts_match_full_recount() {
        for n in 1..=7u32 {
            for mut x in 0..3usize.pow(n) {
                let text: Vec<u8> = (0..n)
                    .map(|_| {
                        let b = (x % 3) as u8;
                        x /= 3;
                        b
                    })
                    .collect();
                let records = [text.as_slice(), text.as_slice(), b"aabaaabaa".as_slice()];
                let actual = BpeTokenizer::train(&records, 275, 512).unwrap();
                assert_eq!(actual, reference_train(&records, 275));
            }
        }
        let repeated = vec![b'a'; 16384];
        assert_eq!(
            BpeTokenizer::train(&[&repeated], 512, 512).unwrap(),
            reference_train(&[&repeated], 512)
        );
    }

    #[test]
    fn incremental_vocabulary_matches_rebuilt_prefixes_and_rejects_atomically() {
        let mut t = BpeTokenizer::from_merges(512, &[]).unwrap();
        let rules = [(97, 98), (259, 259), (260, 99)];
        for (i, &rule) in rules.iter().enumerate() {
            t.append_merge(rule).unwrap();
            assert_eq!(t, BpeTokenizer::from_merges(512, &rules[..=i]).unwrap());
        }
        let saved = t.clone();
        for invalid in [(97, 98), (999, 0), (BOS_TOKEN, 0)] {
            assert_eq!(t.append_merge(invalid), Err(BpeError::InvalidMerge));
            assert_eq!(t, saved);
        }
    }

    #[test]
    fn training_prefix_preserves_full_inference_and_exposes_fallback() {
        let tokenizer = BpeTokenizer::from_merges(512, &[(97, 98), (259, 259)]).unwrap();
        let before = tokenizer.to_bytes();
        for bytes in [b"abab".as_slice(), b"", b"\xff\0ab", b"aaaaaaaa"] {
            assert_eq!(
                tokenizer.encode_with_merge_prefix(bytes, 2),
                tokenizer.encode(bytes)
            );
            for count in 0..=2 {
                let encoded = tokenizer.encode_with_merge_prefix(bytes, count).unwrap();
                assert_eq!(tokenizer.decode(&encoded).unwrap(), bytes);
            }
        }
        assert_eq!(
            tokenizer.encode_with_merge_prefix(b"abab", 0).unwrap(),
            vec![256, 97, 98, 97, 98, 257]
        );
        assert_eq!(
            tokenizer.encode_with_merge_prefix(b"abab", 1).unwrap(),
            vec![256, 259, 259, 257]
        );
        assert_eq!(tokenizer.to_bytes(), before);
    }

    #[test]
    fn training_prefix_rejects_capacity_without_truncation() {
        let tokenizer = BpeTokenizer::from_merges(3, &[(97, 98)]).unwrap();
        assert!(tokenizer.encode(b"ab").is_ok());
        assert_eq!(
            tokenizer.encode_with_merge_prefix(b"ab", 0),
            Err(BpeError::Capacity)
        );
        assert_eq!(
            tokenizer.encode_with_merge_prefix(b"", 2),
            Err(BpeError::InvalidConfig)
        );
        assert_eq!(
            tokenizer.encode_with_merge_prefix(&vec![97; MAX_BPE_BYTES + 1], 1),
            Err(BpeError::Capacity)
        );
    }

    fn reference_merge(ids: &mut Vec<u16>, pair: (u16, u16), output: u16) {
        let mut out = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                out.push(output);
                i += 2;
            } else {
                out.push(ids[i]);
                i += 1;
            }
        }
        *ids = out;
    }

    #[test]
    fn optimized_merge_matches_reference_for_overlaps_and_absent_pairs() {
        // Exhaust all ternary sequences through length eight and every pair.
        for len in 0..=8 {
            for mut value in 0..3usize.pow(len) {
                let ids: Vec<_> = (0..len)
                    .map(|_| {
                        let id = (value % 3) as u16;
                        value /= 3;
                        id
                    })
                    .collect();
                for a in 0..=3 {
                    for b in 0..=3 {
                        let mut expected = ids.clone();
                        let mut actual = ids.clone();
                        reference_merge(&mut expected, (a, b), 4);
                        merge(&mut actual, (a, b), 4);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn optimized_encoding_keeps_rank_ids_and_artifact_identity() {
        let t = BpeTokenizer::train(&[b"fn main() { let a = 1; }", b"abababab"], 300, 512).unwrap();
        let bytes = t.to_bytes();
        for input in [
            b"fn main() { let a = 1; }".as_slice(),
            b"abababaa",
            b"",
            b"unseen\xff",
        ] {
            let mut expected: Vec<_> = input.iter().map(|&b| u16::from(b)).collect();
            for (rank, &pair) in t.merges.iter().enumerate() {
                reference_merge(&mut expected, pair, (BASE + rank) as u16);
            }
            let actual = t.encode(input).unwrap();
            assert_eq!(&actual[1..actual.len() - 1], expected);
        }
        assert_eq!(t.to_bytes(), bytes);
        assert_eq!(
            BpeTokenizer::from_bytes(&bytes).unwrap().fingerprint(),
            t.fingerprint()
        );
    }

    #[test]
    fn preflight_counts_do_not_relax_context_admission() {
        let t = BpeTokenizer::from_merges(3, &[(97, 97)]).unwrap();
        let identity = t.fingerprint();
        assert_eq!(
            t.required_tokens(b"aa").unwrap(),
            t.encode(b"aa").unwrap().len()
        );
        assert_eq!(t.required_tokens(b"aaaa").unwrap(), 4);
        assert_eq!(t.encode(b"aaaa"), Err(BpeError::Capacity));
        assert_eq!(t.required_pair_tokens(b"a", b"a").unwrap(), 5);
        assert_eq!(t.encode_pair(b"a", b"a"), Err(BpeError::Capacity));
        assert!(t.required_tokens(&vec![0; MAX_BPE_BYTES + 1]).is_err());
        assert!(t
            .required_pair_tokens(&vec![0; MAX_BPE_BYTES], b"a")
            .is_err());
        assert_eq!(t.fingerprint(), identity);
    }
    #[test]
    fn pair_roundtrip_and_hostile_framing() {
        let t = BpeTokenizer::train(&[b"abababab"], 280, 512).unwrap();
        for (left, right) in [(b"".as_slice(), b"".as_slice()), (b"ab\xff", b"\0ab")] {
            assert_eq!(
                t.decode_pair(&t.encode_pair(left, right).unwrap()).unwrap(),
                (left.to_vec(), right.to_vec())
            );
        }
        for ids in [
            vec![BOS_TOKEN, EOS_TOKEN],
            vec![BOS_TOKEN, 97, EOS_TOKEN],
            vec![BOS_TOKEN, SEP_TOKEN, SEP_TOKEN, EOS_TOKEN],
            vec![BOS_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN],
            vec![BOS_TOKEN, SEP_TOKEN, 999, EOS_TOKEN],
        ] {
            assert!(t.decode_pair(&ids).is_err());
        }
        let mut rules = vec![(97, 97)];
        for id in 259..268 {
            rules.push((id, id));
        }
        let t = BpeTokenizer::from_merges(512, &rules).unwrap();
        let mut ids = vec![BOS_TOKEN];
        ids.extend([268; 8]);
        ids.push(SEP_TOKEN);
        ids.extend([268; 9]);
        ids.push(EOS_TOKEN);
        assert_eq!(t.decode_pair(&ids), Err(BpeError::Capacity));
    }
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
