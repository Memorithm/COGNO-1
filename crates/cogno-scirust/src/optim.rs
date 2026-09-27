//! AdamW / AMSGrad optimizers with checked arithmetic (COGNO-1 §SciRust #7).
//!
//! Invalid steps fail before changing parameters or optimizer state; counters
//! use checked arithmetic rather than wrapping.
//! The optimizer is pluggable so future variants (Lion, signSGD) layer in
//! without touching the autograd.

use crate::error::{SciRustError, SciRustResult};

/// Optimizer trait. `step` consumes the gradient for one parameter and updates
/// the parameter in place.
pub trait Optimizer {
    fn step(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()>;
}

/// Per-parameter optimizer state (m, v, optional v_hat for AMSGrad, step).
#[derive(Clone, Debug)]
pub struct ParamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub v_hat: Vec<f32>,
    pub step: u64,
}

impl ParamState {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            v_hat: vec![0.0; n],
            step: 0,
        }
    }
}

/// AdamW (decoupled weight decay).
#[derive(Clone, Debug)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub state: ParamState,
}

impl AdamW {
    pub fn try_new(lr: f32, n: usize) -> SciRustResult<Self> {
        let opt = Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            state: ParamState::new(n),
        };
        validate_hyperparams(opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay)?;
        Ok(opt)
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        validate_state(&self.state, param, grad)?;
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        // Stage the entire update: even arithmetic overflow at the last element
        // must leave both moments and parameters available for a safe retry.
        let mut next = self.clone();
        let mut values = param.to_vec();
        next.step_candidate(&mut values, grad)?;
        param.copy_from_slice(&values);
        *self = next;
        Ok(())
    }
}

impl AdamW {
    fn step_candidate(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        if param.len() != grad.len() {
            return Err(SciRustError::Shape {
                lhs: vec![param.len()],
                rhs: vec![grad.len()],
            });
        }
        // Re-validate on every step: the hyperparameter fields are plain data
        // and a host could mutate them after construction; an out-of-domain
        // value (e.g. beta1 == 1.0) would divide by zero and silently produce
        // NaN parameters. Fail closed before any state changes (S10).
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        self.state.step = self
            .state
            .step
            .checked_add(1)
            .ok_or(SciRustError::Overflow)?;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let lr = self.lr;
        let wd = self.weight_decay;
        let t = self.state.step as f32;
        let bias1 = 1.0 - b1.powf(t);
        let bias2 = 1.0 - b2.powf(t);
        for i in 0..param.len() {
            let g = grad[i];
            if !g.is_finite() {
                return Err(SciRustError::NonFinite);
            }
            self.state.m[i] = b1 * self.state.m[i] + (1.0 - b1) * g;
            self.state.v[i] = b2 * self.state.v[i] + (1.0 - b2) * g * g;
            crate::error::ensure_finite(self.state.m[i])?;
            crate::error::ensure_finite(self.state.v[i])?;
            let m_hat = self.state.m[i] / bias1;
            let v_hat = self.state.v[i] / bias2;
            // Decoupled weight decay: param -= lr * (wd * param + m_hat / (sqrt(v_hat) + eps))
            let denom = v_hat.sqrt() + eps;
            let update = m_hat / denom;
            let candidate = param[i] - lr * (wd * param[i] + update);
            // A non-finite result must surface, never silently poison the
            // parameter for every subsequent step.
            if !candidate.is_finite() {
                return Err(SciRustError::NonFinite);
            }
            param[i] = candidate;
        }
        Ok(())
    }
}

/// Upper bound for opt-in reusable optimizer workspaces.
pub const MAX_OPTIMIZER_WORKSPACE_ELEMENTS: usize = 1_048_576;

/// Reusable transactional storage for [`AdamW::step_with_workspace`].
/// One workspace belongs to one parameter shape and can be reused across steps.
#[derive(Clone, Debug)]
pub struct AdamWWorkspace {
    candidate: AdamW,
    values: Vec<f32>,
}

impl AdamWWorkspace {
    pub fn try_new(elements: usize) -> SciRustResult<Self> {
        validate_workspace_size(elements)?;
        Ok(Self {
            candidate: AdamW::try_new(1.0, elements)?,
            values: vec![0.0; elements],
        })
    }
}

fn validate_workspace_size(elements: usize) -> SciRustResult<()> {
    if elements == 0 {
        return Err(SciRustError::Empty);
    }
    if elements > MAX_OPTIMIZER_WORKSPACE_ELEMENTS {
        return Err(SciRustError::CapacityExceeded {
            requested: elements,
            maximum: MAX_OPTIMIZER_WORKSPACE_ELEMENTS,
        });
    }
    Ok(())
}

impl AdamW {
    /// Same arithmetic and atomic failure semantics as `step`, with reusable
    /// candidate buffers. A successful call makes no new heap allocations.
    /// Scratch contents after a failed call are unspecified; retry is safe.
    pub fn step_with_workspace(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        workspace: &mut AdamWWorkspace,
    ) -> SciRustResult<()> {
        validate_state(&self.state, param, grad)?;
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        if workspace.values.len() != param.len() {
            return Err(SciRustError::Shape {
                lhs: vec![workspace.values.len()],
                rhs: vec![param.len()],
            });
        }
        workspace.candidate.lr = self.lr;
        workspace.candidate.beta1 = self.beta1;
        workspace.candidate.beta2 = self.beta2;
        workspace.candidate.eps = self.eps;
        workspace.candidate.weight_decay = self.weight_decay;
        workspace.candidate.state.m.copy_from_slice(&self.state.m);
        workspace.candidate.state.v.copy_from_slice(&self.state.v);
        workspace
            .candidate
            .state
            .v_hat
            .copy_from_slice(&self.state.v_hat);
        workspace.candidate.state.step = self.state.step;
        workspace.values.copy_from_slice(param);
        workspace
            .candidate
            .step_candidate(&mut workspace.values, grad)?;
        param.copy_from_slice(&workspace.values);
        std::mem::swap(self, &mut workspace.candidate);
        Ok(())
    }
}

/// AMSGrad (uses the max of past v to avoid negative learning rates on sparse
/// gradients).
#[derive(Clone, Debug)]
pub struct AmsGrad {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub state: ParamState,
}

impl AmsGrad {
    pub fn try_new(lr: f32, n: usize) -> SciRustResult<Self> {
        let opt = Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            state: ParamState::new(n),
        };
        validate_hyperparams(opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay)?;
        Ok(opt)
    }
}

impl Optimizer for AmsGrad {
    fn step(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        validate_state(&self.state, param, grad)?;
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        // Stage the entire update: even arithmetic overflow at the last element
        // must leave both moments and parameters available for a safe retry.
        let mut next = self.clone();
        let mut values = param.to_vec();
        next.step_candidate(&mut values, grad)?;
        param.copy_from_slice(&values);
        *self = next;
        Ok(())
    }
}

impl AmsGrad {
    fn step_candidate(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        if param.len() != grad.len() {
            return Err(SciRustError::Shape {
                lhs: vec![param.len()],
                rhs: vec![grad.len()],
            });
        }
        // Same re-validation as AdamW: hyperparameters are mutable plain
        // data; out-of-domain values must fail closed before any update.
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        self.state.step = self
            .state
            .step
            .checked_add(1)
            .ok_or(SciRustError::Overflow)?;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let lr = self.lr;
        let wd = self.weight_decay;
        for i in 0..param.len() {
            let g = grad[i];
            if !g.is_finite() {
                return Err(SciRustError::NonFinite);
            }
            self.state.m[i] = b1 * self.state.m[i] + (1.0 - b1) * g;
            self.state.v[i] = b2 * self.state.v[i] + (1.0 - b2) * g * g;
            crate::error::ensure_finite(self.state.m[i])?;
            crate::error::ensure_finite(self.state.v[i])?;
            self.state.v_hat[i] = self.state.v_hat[i].max(self.state.v[i]);
            let denom = self.state.v_hat[i].sqrt() + eps;
            let update = self.state.m[i] / denom;
            let candidate = param[i] - lr * (wd * param[i] + update);
            if !candidate.is_finite() {
                return Err(SciRustError::NonFinite);
            }
            param[i] = candidate;
        }
        Ok(())
    }
}

/// Reusable transactional storage for [`AmsGrad::step_with_workspace`].
/// One workspace belongs to one parameter shape and can be reused across steps.
#[derive(Clone, Debug)]
pub struct AmsGradWorkspace {
    candidate: AmsGrad,
    values: Vec<f32>,
}

impl AmsGradWorkspace {
    pub fn try_new(elements: usize) -> SciRustResult<Self> {
        validate_workspace_size(elements)?;
        Ok(Self {
            candidate: AmsGrad::try_new(1.0, elements)?,
            values: vec![0.0; elements],
        })
    }
}

impl AmsGrad {
    /// Same arithmetic and atomic failure semantics as `step`, with reusable
    /// candidate buffers. A successful call makes no new heap allocations.
    /// Scratch contents after a failed call are unspecified; retry is safe.
    pub fn step_with_workspace(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        workspace: &mut AmsGradWorkspace,
    ) -> SciRustResult<()> {
        validate_state(&self.state, param, grad)?;
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        if workspace.values.len() != param.len() {
            return Err(SciRustError::Shape {
                lhs: vec![workspace.values.len()],
                rhs: vec![param.len()],
            });
        }
        workspace.candidate.lr = self.lr;
        workspace.candidate.beta1 = self.beta1;
        workspace.candidate.beta2 = self.beta2;
        workspace.candidate.eps = self.eps;
        workspace.candidate.weight_decay = self.weight_decay;
        workspace.candidate.state.m.copy_from_slice(&self.state.m);
        workspace.candidate.state.v.copy_from_slice(&self.state.v);
        workspace
            .candidate
            .state
            .v_hat
            .copy_from_slice(&self.state.v_hat);
        workspace.candidate.state.step = self.state.step;
        workspace.values.copy_from_slice(param);
        workspace
            .candidate
            .step_candidate(&mut workspace.values, grad)?;
        param.copy_from_slice(&workspace.values);
        std::mem::swap(self, &mut workspace.candidate);
        Ok(())
    }
}

/// Validate every hyperparameter against its domain. `beta` must lie in
/// `[0, 1)` (a value of `1` zeroes the bias correction denominator), `eps`
/// must be strictly positive and finite, the learning rate strictly positive,
/// and weight decay non-negative.
fn validate_hyperparams(lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) -> SciRustResult<()> {
    let ok = lr > 0.0
        && lr.is_finite()
        && (0.0..1.0).contains(&beta1)
        && beta1.is_finite()
        && (0.0..1.0).contains(&beta2)
        && beta2.is_finite()
        && eps > 0.0
        && eps.is_finite()
        && wd >= 0.0
        && wd.is_finite();
    if !ok {
        return Err(SciRustError::NonFinite);
    }
    Ok(())
}

fn validate_state(state: &ParamState, param: &[f32], grad: &[f32]) -> SciRustResult<()> {
    for length in [grad.len(), state.m.len(), state.v.len(), state.v_hat.len()] {
        if length != param.len() {
            return Err(SciRustError::Shape {
                lhs: vec![length],
                rhs: vec![param.len()],
            });
        }
    }
    for &value in param
        .iter()
        .chain(grad)
        .chain(&state.m)
        .chain(&state.v)
        .chain(&state.v_hat)
    {
        crate::error::ensure_finite(value)?;
    }
    if state.v.iter().chain(&state.v_hat).any(|&value| value < 0.0) {
        return Err(SciRustError::NonFinite);
    }
    Ok(())
}

/// Maximum number of gradient elements inspected by one global clipping call.
pub const MAX_CLIPPED_GRADIENT_ELEMENTS: usize = 1_048_576;

/// Clip all supplied tensors by a single global L2 norm, returning the norm
/// before clipping. No optimizer state or parameters are touched.
///
/// Uses f64 accumulation and scaling to avoid overflow on finite f32 gradients.
/// Validates the entire bounded input before mutation; invalid inputs leave every
/// tensor unchanged. Empty tensors are allowed, but the total must be nonzero.
pub fn clip_global_gradient_norm(
    gradients: &mut [&mut [f32]],
    max_norm: f64,
) -> SciRustResult<f64> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(SciRustError::NonFinite);
    }
    let mut count = 0usize;
    let mut squared = 0.0f64;
    for tensor in gradients.iter() {
        count = count
            .checked_add(tensor.len())
            .ok_or(SciRustError::Overflow)?;
        if count > MAX_CLIPPED_GRADIENT_ELEMENTS {
            return Err(SciRustError::CapacityExceeded {
                requested: count,
                maximum: MAX_CLIPPED_GRADIENT_ELEMENTS,
            });
        }
        for &value in tensor.iter() {
            if !value.is_finite() {
                return Err(SciRustError::NonFinite);
            }
            squared += f64::from(value) * f64::from(value);
        }
    }
    if count == 0 {
        return Err(SciRustError::Empty);
    }
    let norm = squared.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for tensor in gradients.iter_mut() {
            for value in tensor.iter_mut() {
                *value = (f64::from(*value) * scale) as f32;
            }
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod clipping_tests {
    use super::*;

    #[test]
    fn clips_across_tensor_boundaries_with_one_scale() {
        let mut left = [3.0];
        let mut right = [4.0, 0.0];
        assert_eq!(
            clip_global_gradient_norm(&mut [&mut left, &mut right], 2.5).unwrap(),
            5.0
        );
        assert_eq!(left, [1.5]);
        assert_eq!(right, [2.0, 0.0]);
        assert_eq!(
            clip_global_gradient_norm(&mut [&mut left, &mut right], 3.0).unwrap(),
            2.5
        );
        assert_eq!(left, [1.5]);
    }

    #[test]
    fn rejects_late_corruption_without_partial_mutation() {
        let mut left = [3.0, 4.0];
        let mut right = [f32::INFINITY];
        assert_eq!(
            clip_global_gradient_norm(&mut [&mut left, &mut right], 1.0),
            Err(SciRustError::NonFinite)
        );
        assert_eq!(left, [3.0, 4.0]);
        assert_eq!(right, [f32::INFINITY]);
        assert!(clip_global_gradient_norm(&mut [&mut left], 0.0).is_err());
        assert_eq!(left, [3.0, 4.0]);
        assert_eq!(
            clip_global_gradient_norm(&mut [], 1.0),
            Err(SciRustError::Empty)
        );
        let mut large = vec![0.0; MAX_CLIPPED_GRADIENT_ELEMENTS + 1];
        assert!(matches!(
            clip_global_gradient_norm(&mut [&mut large], 1.0),
            Err(SciRustError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn extreme_finite_and_zero_gradients_are_supported() {
        let mut extreme = [f32::MAX, -f32::MAX];
        let norm = clip_global_gradient_norm(&mut [&mut extreme], 1.0).unwrap();
        assert!(norm > f64::from(f32::MAX));
        assert!((f64::from(extreme[0]).hypot(f64::from(extreme[1])) - 1.0).abs() < 1e-7);
        let mut zero = [0.0, -0.0];
        assert_eq!(
            clip_global_gradient_norm(&mut [&mut zero], 1.0).unwrap(),
            0.0
        );
        assert_eq!(zero[1].to_bits(), (-0.0f32).to_bits());
    }
}

mod accumulate;
mod checkpoint;
pub use accumulate::GradientAccumulator;
mod schedule;
pub use schedule::WarmupCosine;
