//! Admission control before allocation (COGNO-1 V2 §12).
//!
//! Before tokenizing or allocating a cache, the runtime computes a
//! [`RequestEstimate`] via checked arithmetic and admits the request only if
//! it fits the remaining budget. A non-admissible request is rejected before
//! its content is fully loaded whenever the protocol allows.

use cogno_core::{MemoryBudget, MemoryError, RequestEstimate};

/// Errors raised by the admission gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    Memory(MemoryError),
}

impl From<MemoryError> for AdmissionError {
    fn from(e: MemoryError) -> Self {
        AdmissionError::Memory(e)
    }
}

/// Admission controller. Pure: holds a budget reference and answers fits/no.
#[derive(Debug)]
pub struct Admission<'b> {
    pub budget: &'b MemoryBudget,
    /// Currently reserved bytes (e.g., model weights + tokenizer + warm-up
    /// caches). Conservative low-water mark; the runtime never over-allocates
    /// past `hard_limit_bytes`.
    pub reserved: usize,
}

impl<'b> Admission<'b> {
    /// Construct with the byte quantity already committed (initialization +
    /// warm-up). The budget must already be validated.
    pub fn new(budget: &'b MemoryBudget, reserved: usize) -> Self {
        Self { budget, reserved }
    }

    fn available(&self) -> usize {
        self.budget.hard_limit_bytes.saturating_sub(self.reserved)
    }

    /// Compute the (checked) estimate for a request and enforce admission
    /// against `hard_limit - reserved`. Returns the estimate on success.
    pub fn admit(
        &self,
        input_bytes: usize,
        estimated_tokens: usize,
        kv_cache_bytes: usize,
        workspace_bytes: usize,
        output_reserve_bytes: usize,
    ) -> Result<RequestEstimate, AdmissionError> {
        // Reject oversized inputs/outputs before computing totals.
        if input_bytes > self.budget.max_input_bytes {
            return Err(AdmissionError::Memory(MemoryError::CapacityExceeded {
                requested: input_bytes,
                maximum: self.budget.max_input_bytes,
            }));
        }
        if output_reserve_bytes > self.budget.max_output_bytes {
            return Err(AdmissionError::Memory(MemoryError::CapacityExceeded {
                requested: output_reserve_bytes,
                maximum: self.budget.max_output_bytes,
            }));
        }
        if estimated_tokens > self.budget.max_context_tokens {
            return Err(AdmissionError::Memory(MemoryError::ContextTooLarge));
        }
        if kv_cache_bytes > self.budget.kv_cache_limit_bytes {
            return Err(AdmissionError::Memory(MemoryError::CapacityExceeded {
                requested: kv_cache_bytes,
                maximum: self.budget.kv_cache_limit_bytes,
            }));
        }
        let est = RequestEstimate::try_new(
            input_bytes,
            estimated_tokens,
            kv_cache_bytes,
            workspace_bytes,
            output_reserve_bytes,
        )?;
        est.fits(self.available())?;
        Ok(est)
    }
}
