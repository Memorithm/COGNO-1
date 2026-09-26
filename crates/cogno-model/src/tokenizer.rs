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

/// Strict decoding errors; malformed streams never silently discard markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteDecodeError {
    Capacity,
    Framing,
    NonByteToken,
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
        let mut tokens = Vec::new();
        self.encode_into(payload, &mut tokens)?;
        Ok(tokens)
    }

    /// Replace a caller-owned buffer with one framed payload, reusing its capacity.
    /// On any error the previous buffer contents remain unchanged.
    pub fn encode_into(
        &self,
        payload: &[u8],
        tokens: &mut Vec<u16>,
    ) -> Result<(), ByteTokenizerError> {
        let requested = payload
            .len()
            .checked_add(2)
            .ok_or(ByteTokenizerError::LengthOverflow)?;
        self.prepare_output(tokens, requested)?;
        tokens.push(BOS_TOKEN);
        tokens.extend(payload.iter().map(|&byte| u16::from(byte)));
        tokens.push(EOS_TOKEN);
        Ok(())
    }

    /// Encode two arbitrary byte payloads as `[BOS] left [SEP] right [EOS]`.
    pub fn encode_pair(&self, left: &[u8], right: &[u8]) -> Result<Vec<u16>, ByteTokenizerError> {
        let mut tokens = Vec::new();
        self.encode_pair_into(left, right, &mut tokens)?;
        Ok(tokens)
    }

    /// Replace a caller-owned buffer with a framed pair, reusing its capacity.
    /// On any error the previous buffer contents remain unchanged.
    pub fn encode_pair_into(
        &self,
        left: &[u8],
        right: &[u8],
        tokens: &mut Vec<u16>,
    ) -> Result<(), ByteTokenizerError> {
        let requested = left
            .len()
            .checked_add(right.len())
            .and_then(|value| value.checked_add(3))
            .ok_or(ByteTokenizerError::LengthOverflow)?;
        self.prepare_output(tokens, requested)?;
        tokens.push(BOS_TOKEN);
        tokens.extend(left.iter().map(|&byte| u16::from(byte)));
        tokens.push(SEP_TOKEN);
        tokens.extend(right.iter().map(|&byte| u16::from(byte)));
        tokens.push(EOS_TOKEN);
        Ok(())
    }

    /// Decode a canonical single stream without interpreting arbitrary bytes as UTF-8.
    pub fn decode(&self, tokens: &[u16]) -> Result<Vec<u8>, ByteDecodeError> {
        let payload = self.decode_frame(tokens)?;
        Self::decode_bytes(payload)
    }

    /// Decode exactly one SEP-delimited pair. Repeated separators are refused.
    pub fn decode_pair(&self, tokens: &[u16]) -> Result<(Vec<u8>, Vec<u8>), ByteDecodeError> {
        let payload = self.decode_frame(tokens)?;
        let separator = payload
            .iter()
            .position(|&id| id == SEP_TOKEN)
            .ok_or(ByteDecodeError::Framing)?;
        let left = Self::decode_bytes(&payload[..separator])?;
        let right = Self::decode_bytes(&payload[separator + 1..])?;
        Ok((left, right))
    }

    fn decode_frame<'a>(&self, tokens: &'a [u16]) -> Result<&'a [u16], ByteDecodeError> {
        if tokens.len() > self.max_tokens {
            return Err(ByteDecodeError::Capacity);
        }
        if tokens.len() < 2
            || tokens.first() != Some(&BOS_TOKEN)
            || tokens.last() != Some(&EOS_TOKEN)
        {
            return Err(ByteDecodeError::Framing);
        }
        Ok(&tokens[1..tokens.len() - 1])
    }

    fn decode_bytes(tokens: &[u16]) -> Result<Vec<u8>, ByteDecodeError> {
        tokens
            .iter()
            .map(|&id| u8::try_from(id).map_err(|_| ByteDecodeError::NonByteToken))
            .collect()
    }

    fn prepare_output(
        &self,
        tokens: &mut Vec<u16>,
        requested: usize,
    ) -> Result<(), ByteTokenizerError> {
        self.ensure_capacity(requested)?;
        // Reserve before clearing so an allocation refusal preserves the old output.
        tokens
            .try_reserve_exact(requested.saturating_sub(tokens.len()))
            .map_err(|_| ByteTokenizerError::AllocationFailed)?;
        tokens.clear();
        Ok(())
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
    fn strict_decoding_roundtrips_all_bytes_and_pairs() {
        let tokenizer = ByteTokenizer::default();
        let bytes: Vec<u8> = (0..=255).collect();
        for length in 0..=bytes.len() {
            let payload = &bytes[..length];
            assert_eq!(
                tokenizer
                    .decode(&tokenizer.encode(payload).unwrap())
                    .unwrap(),
                payload
            );
            let pair = tokenizer.encode_pair(payload, b"\xff\0").unwrap();
            assert_eq!(
                tokenizer.decode_pair(&pair).unwrap(),
                (payload.to_vec(), b"\xff\0".to_vec())
            );
        }
    }

    #[test]
    fn strict_decoding_rejects_noncanonical_frames_and_capacity() {
        let tokenizer = ByteTokenizer::try_new(8).unwrap();
        for bad in [
            vec![],
            vec![256],
            vec![0, 257],
            vec![256, 0],
            vec![256, 256, 257],
            vec![256, 259, 257],
            vec![256, 258, 257],
        ] {
            assert!(tokenizer.decode(&bad).is_err());
        }
        assert!(tokenizer.decode_pair(&[256, 257]).is_err());
        assert!(tokenizer.decode_pair(&[256, 258, 258, 257]).is_err());
        assert_eq!(tokenizer.decode(&[256; 9]), Err(ByteDecodeError::Capacity));
    }

    #[test]
    fn reusable_buffers_match_all_byte_values_and_keep_allocation() {
        let t = ByteTokenizer::default();
        let mut buffer = Vec::with_capacity(MAX_BYTE_TOKENIZER_TOKENS);
        let allocation = buffer.as_ptr();
        let payload: Vec<u8> = (0..=255).collect();
        for len in 0..=payload.len() {
            t.encode_into(&payload[..len], &mut buffer).unwrap();
            assert_eq!(buffer, t.encode(&payload[..len]).unwrap());
            assert_eq!(buffer.as_ptr(), allocation);
            t.encode_pair_into(&payload[..len], b"\xff\0", &mut buffer)
                .unwrap();
            assert_eq!(buffer, t.encode_pair(&payload[..len], b"\xff\0").unwrap());
            assert_eq!(buffer.as_ptr(), allocation);
        }
    }

    #[test]
    fn reusable_buffer_preserves_previous_result_on_refusal() {
        let t = ByteTokenizer::try_new(5).unwrap();
        let mut buffer = vec![999; 20];
        let before = buffer.clone();
        assert_eq!(
            t.encode_into(b"abcd", &mut buffer),
            Err(ByteTokenizerError::TokenCapacityExceeded {
                requested: 6,
                maximum: 5
            })
        );
        assert_eq!(buffer, before);
        assert_eq!(
            t.encode_pair_into(b"ab", b"c", &mut buffer),
            Err(ByteTokenizerError::TokenCapacityExceeded {
                requested: 6,
                maximum: 5
            })
        );
        assert_eq!(buffer, before);
        t.encode_into(b"abc", &mut buffer).unwrap();
        assert_eq!(buffer, vec![BOS_TOKEN, 97, 98, 99, EOS_TOKEN]);
        t.encode_pair_into(b"", b"", &mut buffer).unwrap();
        assert_eq!(buffer, vec![BOS_TOKEN, SEP_TOKEN, EOS_TOKEN]);
    }

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
