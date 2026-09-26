//! Experimental frozen BPE cognitive model. No runtime activation authority.
use crate::bpe_tokenizer::{BpeError, BpeTokenizer};
use cogno_scirust::{SciRustError, SequenceCognitiveHeads, MAX_SEQUENCE_RETRIEVAL_CANDIDATES};

/// Binding, input admission or numerical failure.
#[derive(Clone, Debug, PartialEq)]
pub enum BpeCognitiveError {
    Tokenizer(BpeError),
    Numerical(SciRustError),
    IncompatibleTokenizer,
    InvalidCandidateCap,
}
impl From<BpeError> for BpeCognitiveError {
    fn from(value: BpeError) -> Self {
        Self::Tokenizer(value)
    }
}
impl From<SciRustError> for BpeCognitiveError {
    fn from(value: SciRustError) -> Self {
        Self::Numerical(value)
    }
}

/// Maximum examples in one synchronous classification request.
pub const MAX_BPE_CLASSIFICATION_BATCH: usize = 64;
/// Aggregate raw-byte bound, in addition to each tokenizer input bound.
pub const MAX_BPE_CLASSIFICATION_BATCH_BYTES: usize = 262_144;

/// A batch never returns partial predictions. An example error identifies its row.
#[derive(Clone, Debug, PartialEq)]
pub enum BpeBatchError {
    Capacity,
    Example {
        index: usize,
        source: BpeCognitiveError,
    },
}

/// Three single-payload research signals evaluated with one tokenization.
/// Numerical encoders remain independent; this is not a fused-kernel API.
#[derive(Clone, Debug, PartialEq)]
pub struct BpePayloadSignals {
    pub classification: Vec<f32>,
    pub preference: f32,
    pub symbolic: Vec<f32>,
}

/// Owns an immutable tokenizer and its matching numerical heads.
/// Outputs are research signals, not policy, expertise or permission to act.
#[derive(Clone, Debug, PartialEq)]
pub struct BpeCognitiveModel {
    tokenizer: BpeTokenizer,
    heads: SequenceCognitiveHeads,
    candidate_cap: usize,
}
impl BpeCognitiveModel {
    /// The caller supplies the tokenizer fingerprint recorded during training.
    /// This checks consistency, not the truth of training provenance.
    pub fn from_heads(
        tokenizer: BpeTokenizer,
        heads: SequenceCognitiveHeads,
        training_tokenizer_hash: [u8; 32],
        candidate_cap: usize,
    ) -> Result<Self, BpeCognitiveError> {
        if tokenizer.fingerprint() != training_tokenizer_hash
            || tokenizer.vocab_size() != heads.config().encoder.vocab_size
            || tokenizer.max_tokens() != heads.config().encoder.max_tokens
        {
            return Err(BpeCognitiveError::IncompatibleTokenizer);
        }
        if !(2..=MAX_SEQUENCE_RETRIEVAL_CANDIDATES).contains(&candidate_cap) {
            return Err(BpeCognitiveError::InvalidCandidateCap);
        }
        Ok(Self {
            tokenizer,
            heads,
            candidate_cap,
        })
    }
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }
    pub fn heads(&self) -> &SequenceCognitiveHeads {
        &self.heads
    }
    pub fn candidate_cap(&self) -> usize {
        self.candidate_cap
    }
    pub fn classify(&self, bytes: &[u8]) -> Result<Vec<f32>, BpeCognitiveError> {
        Ok(self
            .heads
            .classification_probabilities(&self.tokenizer.encode(bytes)?)?)
    }
    /// Classify a bounded batch in input order, with exact scalar semantics.
    /// Every input is admitted before any inference starts. No partial output is
    /// exposed if an input or numerical computation fails. An empty batch is valid.
    pub fn classify_batch(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<f32>>, BpeBatchError> {
        if inputs.len() > MAX_BPE_CLASSIFICATION_BATCH {
            return Err(BpeBatchError::Capacity);
        }
        let total = inputs
            .iter()
            .try_fold(0usize, |total, input| total.checked_add(input.len()))
            .ok_or(BpeBatchError::Capacity)?;
        if total > MAX_BPE_CLASSIFICATION_BATCH_BYTES {
            return Err(BpeBatchError::Capacity);
        }
        let encoded: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                self.tokenizer
                    .encode(bytes)
                    .map_err(|source| BpeBatchError::Example {
                        index,
                        source: source.into(),
                    })
            })
            .collect::<Result<_, _>>()?;
        encoded
            .iter()
            .enumerate()
            .map(|(index, tokens)| {
                self.heads
                    .classification_probabilities(tokens)
                    .map_err(|source| BpeBatchError::Example {
                        index,
                        source: source.into(),
                    })
            })
            .collect()
    }

    /// Admit/tokenize once for all single-payload heads. A failure returns no
    /// partial result, and neither model weights nor tokenizer state changes.
    pub fn payload_signals(&self, bytes: &[u8]) -> Result<BpePayloadSignals, BpeCognitiveError> {
        let tokens = self.tokenizer.encode(bytes)?;
        Ok(BpePayloadSignals {
            classification: self.heads.classification_probabilities(&tokens)?,
            preference: self.heads.preference_score(&tokens)?,
            symbolic: self.heads.symbolic_satisfactions(&tokens)?,
        })
    }

    pub fn preference(&self, bytes: &[u8]) -> Result<f32, BpeCognitiveError> {
        Ok(self
            .heads
            .preference_score(&self.tokenizer.encode(bytes)?)?)
    }
    pub fn symbolic(&self, bytes: &[u8]) -> Result<Vec<f32>, BpeCognitiveError> {
        Ok(self
            .heads
            .symbolic_satisfactions(&self.tokenizer.encode(bytes)?)?)
    }
    pub fn contradiction(&self, left: &[u8], right: &[u8]) -> Result<[f32; 2], BpeCognitiveError> {
        Ok(self
            .heads
            .contradiction_probabilities(&self.tokenizer.encode_pair(left, right)?)?)
    }
    pub fn retrieve(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
    ) -> Result<Vec<f32>, BpeCognitiveError> {
        if !(2..=self.candidate_cap).contains(&candidates.len()) {
            return Err(BpeCognitiveError::InvalidCandidateCap);
        }
        let query = self.tokenizer.encode(query)?;
        let encoded: Vec<_> = candidates
            .iter()
            .map(|c| self.tokenizer.encode(c))
            .collect::<Result<_, _>>()?;
        let refs: Vec<_> = encoded.iter().map(Vec::as_slice).collect();
        Ok(self.heads.retrieval_similarities(&query, &refs)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cogno_scirust::{SequenceCognitiveConfig, SequenceEncoderConfig};
    pub(crate) fn fixture() -> BpeCognitiveModel {
        let t = BpeTokenizer::from_merges(32, &[(97, 98)]).unwrap();
        let h = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: t.vocab_size(),
                max_tokens: 32,
                embedding_dim: 4,
                hidden_dim: 8,
                seed: 42,
            },
            num_classes: 2,
            num_rules: 2,
            classification_seed: 1,
            preference_seed: 2,
            symbolic_seed: 3,
            contradiction_seed: 4,
        })
        .unwrap();
        let hash = t.fingerprint();
        BpeCognitiveModel::from_heads(t, h, hash, 4).unwrap()
    }
    #[test]
    fn combined_payload_signals_match_scalar_and_reject_overflow() {
        let model = fixture();
        let before = model.clone();
        for payload in [b"ab".as_slice(), b"", b"\xff\0", b"abab"] {
            let signals = model.payload_signals(payload).unwrap();
            assert_eq!(signals.classification, model.classify(payload).unwrap());
            assert_eq!(signals.preference, model.preference(payload).unwrap());
            assert_eq!(signals.symbolic, model.symbolic(payload).unwrap());
        }
        assert_eq!(
            model.payload_signals(&[0; 31]),
            Err(BpeCognitiveError::Tokenizer(BpeError::Capacity))
        );
        assert_eq!(model, before);
    }

    #[test]
    fn batch_matches_scalar_in_order_and_keeps_model_immutable() {
        let model = fixture();
        let before = model.clone();
        let inputs: [&[u8]; 5] = [b"ab", b"", b"abab", b"\xff\0", b"ab"];
        let expected: Vec<_> = inputs.iter().map(|s| model.classify(s).unwrap()).collect();
        assert_eq!(model.classify_batch(&inputs).unwrap(), expected);
        assert_eq!(model.classify_batch(&[]).unwrap(), Vec::<Vec<f32>>::new());
        assert_eq!(
            model
                .classify_batch(&[b"a".as_slice(); MAX_BPE_CLASSIFICATION_BATCH])
                .unwrap()
                .len(),
            MAX_BPE_CLASSIFICATION_BATCH
        );
        assert_eq!(model, before);
    }

    #[test]
    fn batch_refuses_limits_and_reports_late_invalid_row() {
        let model = fixture();
        let before = model.clone();
        assert_eq!(
            model.classify_batch(&[b"a".as_slice(); MAX_BPE_CLASSIFICATION_BATCH + 1]),
            Err(BpeBatchError::Capacity)
        );
        let big = vec![0; 16_384];
        assert_eq!(
            model.classify_batch(&[big.as_slice(); 17]),
            Err(BpeBatchError::Capacity)
        );
        assert_eq!(
            model.classify_batch(&[b"ab", &[0; 33]]),
            Err(BpeBatchError::Example {
                index: 1,
                source: BpeCognitiveError::Tokenizer(BpeError::Capacity),
            })
        );
        assert_eq!(model, before);
        assert_eq!(
            model.classify_batch(&[b"ab"]).unwrap(),
            vec![model.classify(b"ab").unwrap()]
        );
    }

    #[test]
    fn binding_rejects_same_size_different_vocabulary_and_context() {
        let m = fixture();
        let other = BpeTokenizer::from_merges(32, &[(98, 97)]).unwrap();
        assert!(BpeCognitiveModel::from_heads(
            other,
            m.heads.clone(),
            m.tokenizer.fingerprint(),
            4
        )
        .is_err());
        let other = BpeTokenizer::from_merges(31, &[(97, 98)]).unwrap();
        assert!(BpeCognitiveModel::from_heads(
            other.clone(),
            m.heads.clone(),
            other.fingerprint(),
            4
        )
        .is_err());
        let other = BpeTokenizer::from_merges(32, &[]).unwrap();
        assert!(BpeCognitiveModel::from_heads(
            other.clone(),
            m.heads.clone(),
            other.fingerprint(),
            4
        )
        .is_err());
        assert!(BpeCognitiveModel::from_heads(
            m.tokenizer.clone(),
            m.heads.clone(),
            m.tokenizer.fingerprint(),
            1
        )
        .is_err());
    }
    #[test]
    fn all_signals_match_numerical_heads_and_inputs_are_bounded() {
        let m = fixture();
        let ids = m.tokenizer.encode(b"abab").unwrap();
        assert_eq!(
            m.classify(b"abab").unwrap(),
            m.heads.classification_probabilities(&ids).unwrap()
        );
        assert_eq!(
            m.preference(b"abab").unwrap(),
            m.heads.preference_score(&ids).unwrap()
        );
        assert_eq!(
            m.symbolic(b"abab").unwrap(),
            m.heads.symbolic_satisfactions(&ids).unwrap()
        );
        let pair = m.tokenizer.encode_pair(b"ab", b"ba").unwrap();
        assert_eq!(
            m.contradiction(b"ab", b"ba").unwrap(),
            m.heads.contradiction_probabilities(&pair).unwrap()
        );
        let a = m.tokenizer.encode(b"ab").unwrap();
        let b = m.tokenizer.encode(b"ba").unwrap();
        assert_eq!(
            m.retrieve(b"abab", &[b"ab", b"ba"]).unwrap(),
            m.heads.retrieval_similarities(&ids, &[&a, &b]).unwrap()
        );
        assert!(m.classify(&[0; 33]).is_err());
        assert!(m.retrieve(b"ab", &[]).is_err());
        assert!(m.retrieve(b"ab", &[b"a".as_slice(); 5]).is_err());
    }
}
