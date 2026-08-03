//! Pre-allocated request scratch buffers (COGNO-1 V2 §14).
//!
//! Hot paths receive their workspaces pre-allocated; functions do not retain
//! these references beyond the request. In critical paths, `Vec::new`,
//! `String::new`, `Box::new`, `format!`, `to_string`, `to_owned`, and
//! `collect::<Vec<_>>` / `collect::<String>` are forbidden (§14). This module
//! only defines the views; the runtime owns and pools the underlying storage.

use crate::ids::{CandidateScore, RuleRef, TokenId, ValidationResult};

/// Borrowed, pre-allocated workspace handed to a request. Every field is a
/// `&mut` slice over already-allocated storage; no allocation happens here.
#[derive(Debug)]
pub struct RequestScratch<'a> {
    pub token_ids: &'a mut [TokenId],
    pub logits: &'a mut [f32],
    pub validator_results: &'a mut [ValidationResult],
    pub rule_refs: &'a mut [RuleRef],
    pub candidate_scores: &'a mut [CandidateScore],
    pub output_bytes: &'a mut [u8],
}

impl RequestScratch<'_> {
    /// Length of the token-id slot. The caller checks slot capacities before
    /// writing; this struct performs no allocation and no panics.
    #[must_use]
    pub fn token_slot(&self) -> usize {
        self.token_ids.len()
    }

    /// Length of the output-byte slot.
    #[must_use]
    pub fn output_slot(&self) -> usize {
        self.output_bytes.len()
    }
}
