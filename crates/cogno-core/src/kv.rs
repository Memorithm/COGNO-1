//! KV cache policy (COGNO-1 V2 §17).
//!
//! The KV cache has a fixed or explicitly reserved capacity; it never grows
//! without control across tokens. The runtime chooses a policy explicitly.
//! Context truncation is never silent; a `ContextReport` is produced whenever
//! truncation happens.

/// Eviction policy for the KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCachePolicy {
    /// Refuse to admit tokens beyond the reserved capacity.
    RejectOnOverflow,
    /// Keep only the most recent `window_tokens`.
    SlidingWindow { window_tokens: usize },
    /// Pin the first `prefix_tokens` and keep a sliding window of
    /// `window_tokens` after them.
    PrefixPinnedSlidingWindow {
        prefix_tokens: usize,
        window_tokens: usize,
    },
}

/// Report on a context admission decision. `dropped_tokens > 0` means the
/// context was truncated, which is never silent (§17).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextReport {
    pub requested_tokens: usize,
    pub admitted_tokens: usize,
    pub dropped_tokens: usize,
    pub policy: KvCachePolicy,
}

impl ContextReport {
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.dropped_tokens > 0
    }
}
