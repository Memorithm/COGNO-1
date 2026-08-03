//! KV cache controller (COGNO-1 V2 §17).
//!
//! The KV cache has a fixed or explicitly reserved capacity; it never grows
//! without control. Context truncation is never silent: every truncation
//! produces a [`ContextReport`] the runtime records in the audit log.

use cogno_core::{ContextReport, KvCachePolicy, MemoryError};

/// Errors raised by the KV controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvError {
    ContextTooLarge,
    Overflow,
}

/// Deterministic KV cache admission. Holds the reserved capacity and policy
/// and decides which tokens are admitted, evicted or rejected.
#[derive(Debug)]
pub struct KvController {
    pub capacity_tokens: usize,
    pub policy: KvCachePolicy,
}

impl KvController {
    /// Construct with a hard capacity and a chosen eviction policy.
    #[must_use]
    pub fn new(capacity_tokens: usize, policy: KvCachePolicy) -> Self {
        Self {
            capacity_tokens,
            policy,
        }
    }

    /// Admit `requested_tokens` against the capacity under the policy.
    /// Returns a [`ContextReport`] describing how many tokens are kept and
    /// how many are dropped; never silent when `dropped_tokens > 0`.
    pub fn admit(&self, requested_tokens: usize) -> Result<ContextReport, KvError> {
        match self.policy {
            KvCachePolicy::RejectOnOverflow => {
                if requested_tokens > self.capacity_tokens {
                    Err(KvError::ContextTooLarge)
                } else {
                    Ok(ContextReport {
                        requested_tokens,
                        admitted_tokens: requested_tokens,
                        dropped_tokens: 0,
                        policy: self.policy,
                    })
                }
            }
            KvCachePolicy::SlidingWindow { window_tokens } => {
                // Admit at most (window, capacity) of the requested tokens;
                // the rest are the sliding-window evictions.
                let cap = window_tokens.min(self.capacity_tokens);
                let admitted = cap.min(requested_tokens);
                let dropped = requested_tokens
                    .checked_sub(admitted)
                    .ok_or(KvError::Overflow)?;
                Ok(ContextReport {
                    requested_tokens,
                    admitted_tokens: admitted,
                    dropped_tokens: dropped,
                    policy: self.policy,
                })
            }
            KvCachePolicy::PrefixPinnedSlidingWindow {
                prefix_tokens,
                window_tokens,
            } => {
                let p = prefix_tokens.min(self.capacity_tokens);
                let w = window_tokens.min(self.capacity_tokens.saturating_sub(p));
                let admit_budget = p.checked_add(w).ok_or(KvError::Overflow)?;
                let admitted = admit_budget.min(requested_tokens);
                let dropped = requested_tokens
                    .checked_sub(admitted)
                    .ok_or(KvError::Overflow)?;
                Ok(ContextReport {
                    requested_tokens,
                    admitted_tokens: admitted,
                    dropped_tokens: dropped,
                    policy: self.policy,
                })
            }
        }
    }

    /// Convert the controller's capacity to bytes under checked arithmetic,
    /// using the §11 KV formula via the core.
    pub fn capacity_bytes(
        &self,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        element_bytes: usize,
        batch_size: usize,
    ) -> Result<usize, MemoryError> {
        cogno_core::checked_kv_cache_bytes(
            layers,
            self.capacity_tokens,
            kv_heads,
            head_dim,
            element_bytes,
            batch_size,
        )
    }
}
