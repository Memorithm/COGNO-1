//! Bounded collection (COGNO-1 V2 §15).
//!
//! Any collection influenced by external input must carry a maximum capacity.
//! `BoundedVec` refuses overruns with `CapacityError` rather than growing
//! silently. Hot-path buffers use `RequestScratch` (§14); `BoundedVec` is for
//! bounded container state whose growth happens outside the stable-regime hot
//! path (initialization / warm-up / offline tooling / report builders).

/// A `Vec` capped at `max_len`. Pushing or extending past the cap fails with
/// `CapacityError`; the inner state is left intact on failure (no partial
/// extension).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedVec<T> {
    inner: Vec<T>,
    max_len: usize,
}

/// Overflow of a bounded container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityError {
    Exceeded { requested: usize, maximum: usize },
}

impl<T> BoundedVec<T> {
    /// Construct an empty `BoundedVec` capped at `max_len`. `max_len == 0`
    /// is rejected: a zero-capacity bounded container carries no information
    /// and almost always signals a misconfiguration.
    pub fn try_new(max_len: usize) -> Result<Self, CapacityError> {
        if max_len == 0 {
            return Err(CapacityError::Exceeded {
                requested: 0,
                maximum: 0,
            });
        }
        Ok(Self {
            inner: Vec::new(),
            max_len,
        })
    }

    /// Try to append one element. Fails closed if at capacity; inner untouched.
    pub fn try_push(&mut self, value: T) -> Result<(), CapacityError> {
        if self.inner.len() >= self.max_len {
            return Err(CapacityError::Exceeded {
                requested: self.inner.len().saturating_add(1),
                maximum: self.max_len,
            });
        }
        self.inner.push(value);
        Ok(())
    }

    /// Try to extend from a slice. Fails closed if the new length exceeds the
    /// cap; inner is untouched (`.extend_from_slice` is only called after the
    /// bound check, so no partial extension occurs).
    pub fn try_extend_from_slice(&mut self, values: &[T]) -> Result<(), CapacityError>
    where
        T: Clone,
    {
        let new_len =
            self.inner
                .len()
                .checked_add(values.len())
                .ok_or(CapacityError::Exceeded {
                    requested: usize::MAX,
                    maximum: self.max_len,
                })?;
        if new_len > self.max_len {
            return Err(CapacityError::Exceeded {
                requested: new_len,
                maximum: self.max_len,
            });
        }
        self.inner.extend_from_slice(values);
        Ok(())
    }

    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// View the contents as a borrowed slice — no ownership transfer, no copy.
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
