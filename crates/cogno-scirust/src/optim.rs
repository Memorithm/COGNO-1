//! AdamW / AMSGrad optimizers with checked arithmetic (COGNO-1 §SciRust #7).
//!
//! No `panic!`; the step count and weight updates saturate rather than wrap.
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
        validate_hyperparams(lr)?;
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            state: ParamState::new(n),
        })
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        if param.len() != grad.len() {
            return Err(SciRustError::Shape {
                lhs: vec![param.len()],
                rhs: vec![grad.len()],
            });
        }
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
            let m_hat = self.state.m[i] / bias1;
            let v_hat = self.state.v[i] / bias2;
            // Decoupled weight decay: param -= lr * (wd * param + m_hat / (sqrt(v_hat) + eps))
            let denom = v_hat.sqrt() + eps;
            let update = m_hat / denom;
            param[i] -= lr * (wd * param[i] + update);
        }
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
        validate_hyperparams(lr)?;
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            state: ParamState::new(n),
        })
    }
}

impl Optimizer for AmsGrad {
    fn step(&mut self, param: &mut [f32], grad: &[f32]) -> SciRustResult<()> {
        if param.len() != grad.len() {
            return Err(SciRustError::Shape {
                lhs: vec![param.len()],
                rhs: vec![grad.len()],
            });
        }
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
            self.state.v_hat[i] = self.state.v_hat[i].max(self.state.v[i]);
            let denom = self.state.v_hat[i].sqrt() + eps;
            let update = self.state.m[i] / denom;
            param[i] -= lr * (wd * param[i] + update);
        }
        Ok(())
    }
}

fn validate_hyperparams(lr: f32) -> SciRustResult<()> {
    if lr <= 0.0 || !lr.is_finite() {
        return Err(SciRustError::NonFinite);
    }
    Ok(())
}
