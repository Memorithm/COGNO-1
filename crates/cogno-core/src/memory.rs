//! Memory budget with checked arithmetic (COGNO-1 V2 §11–§13).
//!
//! No `let bytes = layers * tokens * heads * head_dim * element_size;`. All
//! size math uses `checked_add` / `checked_mul` / `checked_sub` and surfaces
//! `MemoryError::ArithmeticOverflow` on overflow.

/// Errors raised by memory operations. Matches the §13 enum exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryError {
    ArithmeticOverflow,
    BudgetExceeded { requested: usize, available: usize },
    CapacityExceeded { requested: usize, maximum: usize },
    AllocationFailed,
    QueueFull,
    ContextTooLarge,
    OutputTooLarge,
}

/// Errors raised when validating a `MemoryBudget`. Separate from `MemoryError`
/// to keep the §13 enum faithful; budget construction has its own failure
/// modes (sub-budget sum, mandatory-zero limits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    SubBudgetSumExceedsHardLimit { sum: usize, hard_limit: usize },
    ArithmeticOverflow,
    MandatoryLimitZero { field: &'static str },
}

/// Explicit memory budget (§11). Fields are `pub` per the spec, but a valid
/// budget must be obtained via `try_new` (or built then `validate`d); runtime
/// consumers of a budget treat an unvalidated one as non-admissible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub hard_limit_bytes: usize,

    pub model_weights_limit_bytes: usize,
    pub tokenizer_limit_bytes: usize,
    pub kv_cache_limit_bytes: usize,
    pub tensor_workspace_limit_bytes: usize,
    pub request_scratch_limit_bytes: usize,
    pub profile_cache_limit_bytes: usize,
    pub event_buffer_limit_bytes: usize,

    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub max_batch_size: usize,
    pub max_candidates: usize,
    pub max_concurrent_requests: usize,
    pub max_queue_depth: usize,
}

impl MemoryBudget {
    /// Validate the §11 invariants with checked arithmetic:
    ///  - the 7 sub-budgets sum to at most `hard_limit_bytes`;
    ///  - no addition overflows while accumulating the sum;
    ///  - every mandatory limit is non-zero.
    ///
    /// Sub-budgets (weights, tokenizer, KV cache, workspace, scratch, profile
    /// cache, event buffer) may be `0`, meaning "not used". The `hard_limit`
    /// and every `max_*` field are mandatory.
    pub fn validate(&self) -> Result<(), BudgetError> {
        macro_rules! require_nonzero {
            ($field:ident) => {
                if self.$field == 0 {
                    return Err(BudgetError::MandatoryLimitZero {
                        field: stringify!($field),
                    });
                }
            };
        }

        require_nonzero!(hard_limit_bytes);
        require_nonzero!(max_input_bytes);
        require_nonzero!(max_output_bytes);
        require_nonzero!(max_context_tokens);
        require_nonzero!(max_output_tokens);
        require_nonzero!(max_batch_size);
        require_nonzero!(max_candidates);
        require_nonzero!(max_concurrent_requests);
        require_nonzero!(max_queue_depth);

        let sub = [
            self.model_weights_limit_bytes,
            self.tokenizer_limit_bytes,
            self.kv_cache_limit_bytes,
            self.tensor_workspace_limit_bytes,
            self.request_scratch_limit_bytes,
            self.profile_cache_limit_bytes,
            self.event_buffer_limit_bytes,
        ];

        let mut sum: usize = 0;
        for &v in sub.iter() {
            sum = sum.checked_add(v).ok_or(BudgetError::ArithmeticOverflow)?;
        }

        if sum > self.hard_limit_bytes {
            return Err(BudgetError::SubBudgetSumExceedsHardLimit {
                sum,
                hard_limit: self.hard_limit_bytes,
            });
        }
        Ok(())
    }

    /// Construct and validate a budget in one call. The explicit argument list
    /// is intentional: every limit must be supplied, never defaulted, never
    /// silently grown beyond the configured limit (§13).
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        hard_limit_bytes: usize,
        model_weights_limit_bytes: usize,
        tokenizer_limit_bytes: usize,
        kv_cache_limit_bytes: usize,
        tensor_workspace_limit_bytes: usize,
        request_scratch_limit_bytes: usize,
        profile_cache_limit_bytes: usize,
        event_buffer_limit_bytes: usize,
        max_input_bytes: usize,
        max_output_bytes: usize,
        max_context_tokens: usize,
        max_output_tokens: usize,
        max_batch_size: usize,
        max_candidates: usize,
        max_concurrent_requests: usize,
        max_queue_depth: usize,
    ) -> Result<Self, BudgetError> {
        let b = Self {
            hard_limit_bytes,
            model_weights_limit_bytes,
            tokenizer_limit_bytes,
            kv_cache_limit_bytes,
            tensor_workspace_limit_bytes,
            request_scratch_limit_bytes,
            profile_cache_limit_bytes,
            event_buffer_limit_bytes,
            max_input_bytes,
            max_output_bytes,
            max_context_tokens,
            max_output_tokens,
            max_batch_size,
            max_candidates,
            max_concurrent_requests,
            max_queue_depth,
        };
        b.validate()?;
        Ok(b)
    }
}

/// Estimate of a request's memory footprint, computed before any allocation
/// (§12). `total_bytes` excludes `estimated_tokens` (a count, not bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestEstimate {
    pub input_bytes: usize,
    pub estimated_tokens: usize,
    pub kv_cache_bytes: usize,
    pub workspace_bytes: usize,
    pub output_reserve_bytes: usize,
    pub total_bytes: usize,
}

impl RequestEstimate {
    /// Construct an estimate with checked arithmetic. Returns
    /// `MemoryError::ArithmeticOverflow` if the component sum wraps.
    pub fn try_new(
        input_bytes: usize,
        estimated_tokens: usize,
        kv_cache_bytes: usize,
        workspace_bytes: usize,
        output_reserve_bytes: usize,
    ) -> Result<Self, MemoryError> {
        let total_bytes = input_bytes
            .checked_add(kv_cache_bytes)
            .ok_or(MemoryError::ArithmeticOverflow)?
            .checked_add(workspace_bytes)
            .ok_or(MemoryError::ArithmeticOverflow)?
            .checked_add(output_reserve_bytes)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        Ok(Self {
            input_bytes,
            estimated_tokens,
            kv_cache_bytes,
            workspace_bytes,
            output_reserve_bytes,
            total_bytes,
        })
    }

    /// Admission check against an available byte budget.
    pub fn fits(&self, available: usize) -> Result<(), MemoryError> {
        if self.total_bytes > available {
            return Err(MemoryError::BudgetExceeded {
                requested: self.total_bytes,
                available,
            });
        }
        Ok(())
    }
}

/// Checked KV cache size for a standard autoregressive Transformer.
///
/// `KV bytes = 2 × layers × tokens × kv_heads × head_dim × element_bytes × batch`
///
/// The factor 2 covers the two caches, keys and values. The backend MUST verify
/// its own real layout before relying on this estimate (§11).
pub fn checked_kv_cache_bytes(
    layers: usize,
    tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    element_bytes: usize,
    batch_size: usize,
) -> Result<usize, MemoryError> {
    const KV_FACTOR: usize = 2;
    let mut bytes = KV_FACTOR;
    bytes = bytes
        .checked_mul(layers)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    bytes = bytes
        .checked_mul(tokens)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    bytes = bytes
        .checked_mul(kv_heads)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    bytes = bytes
        .checked_mul(head_dim)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    bytes = bytes
        .checked_mul(element_bytes)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    bytes = bytes
        .checked_mul(batch_size)
        .ok_or(MemoryError::ArithmeticOverflow)?;
    Ok(bytes)
}
