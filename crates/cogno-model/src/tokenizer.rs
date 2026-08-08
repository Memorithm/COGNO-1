//! Deterministic, bounded tokenizer for the future sequence model.
//!
//! The tokenizer is deliberately byte based: every input byte maps to exactly
//! one token, so arbitrary hostile payloads have no UTF-8 dependency and no
//! out-of-vocabulary path. Framing tokens make single and paired inputs
//! unambiguous. This module does not change the historical v1/v2 artifact
//! tokenizer contract; a future sequence-model artifact must bind this
//! descriptor explicitly.

use sha2::{Digest, Sha256};

/// Raw byte tokens occupy ids 0..=255.
pub const BYTE_TOKEN_COUNT: u16 = 256;
/// Beginning-of-sequence marker.
pub const BOS_TOKEN: u16 = 256;
/// End-of-sequence marker.
pub const EOS_TOKEN: u16 = 257;
/// Separator used for paired inputs.
pub const SEP_TOKEN: u16 = 258;
/// Exact vocabulary size of the deterministic byte tokenizer.
pub const BYTE_TOKENIZER_VOCAB_SIZE: usize = 259;
/// Maximum sequence length accepted by this tokenizer contract.
pub const MAX_BYTE_TOKENIZER_TOKENS: usize = 512;
/// Canonical descriptor hashed into future sequence-model manifests.
pub const BYTE_TOKENIZER_DESCRIPTOR: &[u8] =
    b"cogno-byte-tokenizer-v2;raw=0..255;bos=256;eos=257;sep=258;max=512";

/// Fail-closed tokenizer errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteTokenizerError {
    InvalidMaximum,
    TokenCapacityExceeded { requested: usize, maximum: usize },
    LengthOverflow,
    AllocationFailed,
}

/// Stateless byte tokenizer with an explicit per-instance token cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteTokenizer {
    max_tokens: usize,
}

impl ByteTokenizer {
    /// Construct a tokenizer whose output can contain at least `BOS, EOS` and
    /// never exceeds the global sequence-token bound.
    pub fn try_new(max_tokens: usize) -> Result<Self, ByteTokenizerError> {
        if !(2..=MAX_BYTE_TOKENIZER_TOKENS).contains(&max_tokens) {
            return Err(ByteTokenizerError::InvalidMaximum);
        }
        Ok(Self { max_tokens })
    }

    #[must_use]
    pub const fn max_tokens(self) -> usize {
        self.max_tokens
    }

    /// Encode one arbitrary byte payload as `[BOS] payload [EOS]`.
    pub fn encode(&self, payload: &[u8]) -> Result<Vec<u16>, ByteTokenizerError> {
        let requested = payload
            .len()
            .checked_add(2)
            .ok_or(ByteTokenizerError::LengthOverflow)?;
        self.ensure_capacity(requested)?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(requested)
            .map_err(|_| ByteTokenizerError::AllocationFailed)?;
        tokens.push(BOS_TOKEN);
        tokens.extend(payload.iter().map(|&byte| u16::from(byte)));
        tokens.push(EOS_TOKEN);
        Ok(tokens)
    }

    /// Encode two arbitrary byte payloads as
    /// `[BOS] left [SEP] right [EOS]`.
    pub fn encode_pair(&self, left: &[u8], right: &[u8]) -> Result<Vec<u16>, ByteTokenizerError> {
        let requested = left
            .len()
            .checked_add(right.len())
            .and_then(|value| value.checked_add(3))
            .ok_or(ByteTokenizerError::LengthOverflow)?;
        self.ensure_capacity(requested)?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(requested)
            .map_err(|_| ByteTokenizerError::AllocationFailed)?;
        tokens.push(BOS_TOKEN);
        tokens.extend(left.iter().map(|&byte| u16::from(byte)));
        tokens.push(SEP_TOKEN);
        tokens.extend(right.iter().map(|&byte| u16::from(byte)));
        tokens.push(EOS_TOKEN);
        Ok(tokens)
    }

    fn ensure_capacity(&self, requested: usize) -> Result<(), ByteTokenizerError> {
        if requested > self.max_tokens {
            return Err(ByteTokenizerError::TokenCapacityExceeded {
                requested,
                maximum: self.max_tokens,
            });
        }
        Ok(())
    }
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self {
            max_tokens: MAX_BYTE_TOKENIZER_TOKENS,
        }
    }
}

/// SHA-256 fingerprint of the exact sequence-tokenizer contract.
#[must_use]
pub fn byte_tokenizer_hash() -> [u8; 32] {
    Sha256::digest(BYTE_TOKENIZER_DESCRIPTOR).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_maps_to_its_exact_token_without_oov() {
        let tokenizer = ByteTokenizer::default();
        let payload: Vec<u8> = (0u8..=u8::MAX).collect();
        let tokens = tokenizer.encode(&payload).expect("bounded encoding");
        assert_eq!(tokens.len(), 258);
        assert_eq!(tokens[0], BOS_TOKEN);
        assert_eq!(tokens[257], EOS_TOKEN);
        for (byte, &token) in payload.iter().zip(&tokens[1..257]) {
            assert_eq!(token, u16::from(*byte));
        }
    }

    #[test]
    fn paired_framing_is_exact_and_order_preserving() {
        let tokenizer = ByteTokenizer::try_new(16).expect("tokenizer");
        assert_eq!(
            tokenizer.encode_pair(b"ab", b"CD").expect("pair"),
            vec![BOS_TOKEN, 97, 98, SEP_TOKEN, 67, 68, EOS_TOKEN]
        );
        assert_ne!(
            tokenizer.encode_pair(b"ab", b"CD").expect("left-right"),
            tokenizer.encode_pair(b"CD", b"ab").expect("right-left")
        );
    }

    #[test]
    fn capacity_includes_framing_before_allocation() {
        let tokenizer = ByteTokenizer::try_new(5).expect("tokenizer");
        assert_eq!(tokenizer.encode(b"abc").expect("exact fit").len(), 5);
        assert_eq!(
            tokenizer.encode(b"abcd"),
            Err(ByteTokenizerError::TokenCapacityExceeded {
                requested: 6,
                maximum: 5,
            })
        );
        assert_eq!(
            tokenizer.encode_pair(b"a", b"bc"),
            Err(ByteTokenizerError::TokenCapacityExceeded {
                requested: 6,
                maximum: 5,
            })
        );
    }

    #[test]
    fn invalid_limits_fail_closed() {
        assert_eq!(
            ByteTokenizer::try_new(1),
            Err(ByteTokenizerError::InvalidMaximum)
        );
        assert_eq!(
            ByteTokenizer::try_new(MAX_BYTE_TOKENIZER_TOKENS + 1),
            Err(ByteTokenizerError::InvalidMaximum)
        );
    }

    #[test]
    fn descriptor_hash_is_stable_and_domain_specific() {
        assert_eq!(byte_tokenizer_hash(), byte_tokenizer_hash());
        assert_ne!(
            byte_tokenizer_hash(),
            crate::artifact::neural_tokenizer_hash()
        );
    }
}
