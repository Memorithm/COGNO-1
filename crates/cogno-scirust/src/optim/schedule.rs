//! Explicit, opt-in learning-rate schedule; does not advance optimizer state.
use crate::{SciRustError, SciRustResult};

/// Linear warmup followed by cosine decay to a strictly positive floor.
/// Query by the number of *successful* optimizer updates already completed.
/// Failed updates must reuse their rate. Checkpoint callers persist these four
/// constructor arguments alongside the model and optimizer step counter.
#[derive(Clone, Debug)]
pub struct WarmupCosine {
    peak: f32,
    floor: f32,
    warmup_steps: u64,
    total_steps: u64,
}
impl WarmupCosine {
    pub fn try_new(
        peak: f32,
        floor: f32,
        warmup_steps: u64,
        total_steps: u64,
    ) -> SciRustResult<Self> {
        if !peak.is_finite() || !floor.is_finite() || floor <= 0.0 || peak < floor {
            return Err(SciRustError::NonFinite);
        }
        if total_steps < 2 || warmup_steps >= total_steps {
            return Err(SciRustError::Empty);
        }
        Ok(Self {
            peak,
            floor,
            warmup_steps,
            total_steps,
        })
    }
    /// Step zero uses the first warmup rate (or the peak without warmup).
    /// The final budgeted step and all subsequent queries return the floor.
    pub fn learning_rate(&self, completed_steps: u64) -> f32 {
        let floor = f64::from(self.floor);
        let delta = f64::from(self.peak) - floor;
        if completed_steps >= self.total_steps - 1 {
            return self.floor;
        }
        if completed_steps < self.warmup_steps {
            return (floor + delta * (completed_steps + 1) as f64 / self.warmup_steps as f64)
                as f32;
        }
        let progress = if self.warmup_steps == 0 {
            completed_steps as f64 / (self.total_steps - 1) as f64
        } else {
            (completed_steps - self.warmup_steps + 1) as f64
                / (self.total_steps - self.warmup_steps) as f64
        };
        (floor + delta * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
    }
}
