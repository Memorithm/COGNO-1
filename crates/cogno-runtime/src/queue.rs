//! Bounded queue with explicit backpressure (COGNO-1 V2 §16).
//!
//! Every queue carries a capacity, a full-queue behavior, a maximum deadline
//! and a saturation metric. Existing entries are never dropped silently: under
//! `RejectNewest`, an incoming entry is refused while the queue is preserved.
//!
//! The MVP exposes synchronous try-push/try-pop; an async variant would layer
//! `BlockWithDeadline` on top using the same bounds.

use cogno_core::QueueFullPolicy;

/// Errors raised by a bounded queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueError {
    Full,
    DeadlineExceeded,
}

/// Saturation metric for monitoring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueStats {
    pub capacity: usize,
    pub len: usize,
    pub rejections: u64,
}

/// Bounded FIFO queue. The MVP exposes synchronous try-push/try-pop; an async
/// variant would layer `BlockWithDeadline` on top using the same bounds.
#[derive(Debug)]
pub struct BoundedQueue<T> {
    inner: std::collections::VecDeque<T>,
    capacity: usize,
    pub policy: QueueFullPolicy,
    pub stats: QueueStats,
}

impl<T> BoundedQueue<T> {
    /// Construct with an explicit capacity and full-queue policy. Capacity 0
    /// is rejected: a zero-capacity queue carries no information and signals
    /// misconfiguration.
    pub fn try_new(capacity: usize, policy: QueueFullPolicy) -> Result<Self, QueueError> {
        if capacity == 0 {
            return Err(QueueError::Full);
        }
        Ok(Self {
            inner: std::collections::VecDeque::with_capacity(capacity),
            capacity,
            policy,
            stats: QueueStats {
                capacity,
                len: 0,
                rejections: 0,
            },
        })
    }

    /// Try to enqueue. Behavior under saturation follows the policy:
    ///  - `RejectNewest` (default for interactive): refuse the new entry,
    ///    keep the queue intact, `rejections += 1`.
    ///  - `RejectOldest`: drop the oldest, push the new. The drop is **not**
    ///    silent — the caller is informed via [`QueueStats::rejections`].
    ///  - `BlockWithDeadline`: synchronous variant refuses with
    ///    `DeadlineExceeded` since this runtime does not block.
    pub fn try_push(&mut self, value: T) -> Result<(), QueueError> {
        if self.inner.len() < self.capacity {
            self.inner.push_back(value);
            self.stats.len = self.inner.len();
            return Ok(());
        }
        match self.policy {
            QueueFullPolicy::RejectNewest | QueueFullPolicy::BlockWithDeadline => {
                self.stats.rejections = self.stats.rejections.saturating_add(1);
                Err(match self.policy {
                    QueueFullPolicy::BlockWithDeadline => QueueError::DeadlineExceeded,
                    _ => QueueError::Full,
                })
            }
            QueueFullPolicy::RejectOldest => {
                self.inner.pop_front();
                self.inner.push_back(value);
                self.stats.rejections = self.stats.rejections.saturating_add(1);
                self.stats.len = self.inner.len();
                Ok(())
            }
        }
    }

    /// Dequeue the oldest entry.
    pub fn try_pop(&mut self) -> Option<T> {
        let v = self.inner.pop_front();
        self.stats.len = self.inner.len();
        v
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.inner.len() == self.capacity
    }
}
