//! Bounded, pre-allocated, fallible KV cache (COGNO-1 §SciRust #8, §17).
//!
//! Fixed capacity, pre-allocated at construction (warm-up, §13). Pushing past
//! capacity fails with [`KvPushError`] according to the chosen
//! [`cogno_core::KvCachePolicy`]; no growth, no silent drop on
//! `RejectOnOverflow`. Truncation is never silent: the controller surfaces a
//! [`cogno_core::ContextReport`].

use cogno_core::ContextReport;
pub use cogno_core::KvCachePolicy;

/// Errors raised by a push into a bounded KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPushError {
    /// `RejectOnOverflow` policy: the slot was full.
    Full,
    /// Sliding window evicted the oldest entry (not silent; the caller
    /// inspects `KvPushOutcome::Evicted`).
    Evicted,
    /// Capacity was 0 at construction (misconfiguration).
    ZeroCapacity,
}

/// Outcome of a push.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPushOutcome {
    Admitted,
    Evicted { dropped_token_id: u64 },
    Refused,
}

/// Bounded KV cache. Stores token ids (u64) and a parallel value dimension
/// (`head_dim` floats per token, per head, per layer); the layout is
/// pre-allocated at construction.
#[derive(Debug)]
pub struct BoundedKvCache {
    pub capacity_tokens: usize,
    pub policy: KvCachePolicy,
    pub token_ids: Vec<u64>,
    pub values: Vec<f32>,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub len: usize,
}

impl BoundedKvCache {
    /// Construct with full pre-allocation (warm-up, §13). `capacity_tokens`
    /// must be non-zero.
    pub fn try_new(
        capacity_tokens: usize,
        head_dim: usize,
        num_heads: usize,
        num_layers: usize,
        policy: KvCachePolicy,
    ) -> Result<Self, KvPushError> {
        if capacity_tokens == 0 || head_dim == 0 || num_heads == 0 || num_layers == 0 {
            return Err(KvPushError::ZeroCapacity);
        }
        let total = capacity_tokens
            .checked_mul(head_dim)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(num_layers))
            .ok_or(KvPushError::ZeroCapacity)?;
        if total == 0 {
            return Err(KvPushError::ZeroCapacity);
        }
        Ok(Self {
            capacity_tokens,
            policy,
            token_ids: vec![0; capacity_tokens],
            values: vec![0.0; total],
            head_dim,
            num_heads,
            num_layers,
            len: 0,
        })
    }

    /// Push a token id. Behavior follows `KvCachePolicy`.
    pub fn push(&mut self, token_id: u64) -> KvPushOutcome {
        match self.policy {
            KvCachePolicy::RejectOnOverflow => {
                if self.len >= self.capacity_tokens {
                    return KvPushOutcome::Refused;
                }
                self.token_ids[self.len] = token_id;
                self.len += 1;
                KvPushOutcome::Admitted
            }
            KvCachePolicy::SlidingWindow { window_tokens } => {
                let win = window_tokens.min(self.capacity_tokens);
                if self.len < win {
                    self.token_ids[self.len] = token_id;
                    self.len += 1;
                    return KvPushOutcome::Admitted;
                }
                // Evict the oldest: shift left by 1 and append.
                let dropped = self.token_ids[0];
                self.token_ids.copy_within(1..win, 0);
                self.token_ids[win - 1] = token_id;
                self.len = win;
                KvPushOutcome::Evicted {
                    dropped_token_id: dropped,
                }
            }
            KvCachePolicy::PrefixPinnedSlidingWindow {
                prefix_tokens,
                window_tokens,
            } => {
                let prefix = prefix_tokens.min(self.capacity_tokens);
                let win = window_tokens.min(self.capacity_tokens.saturating_sub(prefix));
                let cap = prefix + win;
                if self.len < cap {
                    self.token_ids[self.len] = token_id;
                    self.len += 1;
                    return KvPushOutcome::Admitted;
                }
                // Evict the oldest *after* the pinned prefix.
                let dropped = self.token_ids[prefix];
                self.token_ids.copy_within(prefix + 1..cap, prefix);
                self.token_ids[cap - 1] = token_id;
                self.len = cap;
                KvPushOutcome::Evicted {
                    dropped_token_id: dropped,
                }
            }
        }
    }

    /// Report a context admission decision (never silent, §17).
    pub fn report(&self, requested_tokens: usize) -> ContextReport {
        let admitted = self.len.min(requested_tokens);
        ContextReport {
            requested_tokens,
            admitted_tokens: admitted,
            dropped_tokens: requested_tokens.saturating_sub(admitted),
            policy: self.policy,
        }
    }
}
