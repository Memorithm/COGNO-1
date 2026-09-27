//! Weighted microbatch accumulation before a single optimizer step.
use super::validate_workspace_size;
use crate::{SciRustError, SciRustResult};

/// Bounded f64 accumulator. Pass each microbatch's mean gradient and its sample
/// count as the weight, so an incomplete final batch is not over-represented.
/// No parameter/optimizer updates occur until the caller explicitly steps.
#[derive(Clone, Debug)]
pub struct GradientAccumulator {
    sum: Vec<f64>,
    total_weight: f64,
    batches: u64,
}
impl GradientAccumulator {
    pub fn try_new(elements: usize) -> SciRustResult<Self> {
        validate_workspace_size(elements)?;
        Ok(Self {
            sum: vec![0.0; elements],
            total_weight: 0.0,
            batches: 0,
        })
    }
    /// Atomically admit a microbatch. Invalid late elements leave all sums intact.
    pub fn add(&mut self, gradient: &[f32], weight: f64) -> SciRustResult<()> {
        if gradient.len() != self.sum.len() {
            return Err(SciRustError::Shape {
                lhs: vec![gradient.len()],
                rhs: vec![self.sum.len()],
            });
        }
        let total = self.total_weight + weight;
        if !weight.is_finite() || weight <= 0.0 || !total.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        let batches = self.batches.checked_add(1).ok_or(SciRustError::Overflow)?;
        for (&sum, &g) in self.sum.iter().zip(gradient) {
            if !g.is_finite() || !(sum + f64::from(g) * weight).is_finite() {
                return Err(SciRustError::NonFinite);
            }
        }
        for (sum, &g) in self.sum.iter_mut().zip(gradient) {
            *sum += f64::from(g) * weight;
        }
        self.total_weight = total;
        self.batches = batches;
        Ok(())
    }
    /// Write the sample-weighted gradient mean without clearing the accumulator.
    pub fn mean_into(&self, output: &mut [f32]) -> SciRustResult<()> {
        if output.len() != self.sum.len() {
            return Err(SciRustError::Shape {
                lhs: vec![output.len()],
                rhs: vec![self.sum.len()],
            });
        }
        if self.batches == 0 {
            return Err(SciRustError::Empty);
        }
        for &sum in &self.sum {
            if !((sum / self.total_weight) as f32).is_finite() {
                return Err(SciRustError::NonFinite);
            }
        }
        for (out, &sum) in output.iter_mut().zip(&self.sum) {
            *out = (sum / self.total_weight) as f32;
        }
        Ok(())
    }
    pub fn batches(&self) -> u64 {
        self.batches
    }
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }
    /// Start the next optimizer batch while retaining allocated storage.
    pub fn clear(&mut self) {
        self.sum.fill(0.0);
        self.total_weight = 0.0;
        self.batches = 0;
    }
}
