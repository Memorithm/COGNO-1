//! Backpressure policy (COGNO-1 V2 §16).
//!
//! Every queue carries a capacity, a full-queue behavior, a maximum deadline,
//! and a saturation metric. For interactive requests the default is
//! `RejectNewest`; an existing request is never dropped silently.

/// Behavior when a bounded queue is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueFullPolicy {
    /// Reject the incoming request. Default for interactive requests.
    RejectNewest,
    /// Drop the oldest queued entry. Must not be silent: the drop is logged.
    RejectOldest,
    /// Block the producer until space frees or the deadline elapses.
    BlockWithDeadline,
}
