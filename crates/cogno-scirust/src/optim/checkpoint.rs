//! Bounded, versioned AdamW checkpoint with parameters and optimizer state.
//! This wire format is not authenticated: callers must separately pin its hash
//! and bind parameter ordering/model metadata before accepting external bytes.
use super::{validate_hyperparams, validate_state, validate_workspace_size, AdamW, ParamState};
use crate::{SciRustError, SciRustResult};

impl AdamW {
    /// Encode exact f32 bits, counters, and parameters for deterministic resume.
    pub fn checkpoint(&self, parameters: &[f32]) -> SciRustResult<Vec<u8>> {
        validate_workspace_size(parameters.len())?;
        validate_state(&self.state, parameters, parameters)?;
        validate_hyperparams(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay)?;
        let mut out = Vec::with_capacity(40 + 16 * parameters.len());
        out.extend_from_slice(b"CADAM001");
        out.extend_from_slice(&(parameters.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.state.step.to_le_bytes());
        for x in [self.lr, self.beta1, self.beta2, self.eps, self.weight_decay] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for values in [parameters, &self.state.m, &self.state.v, &self.state.v_hat] {
            for x in values {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Decode only after validating the header, exact length, and resource cap.
    /// Invalid moments, parameters, or hyperparameters are rejected.
    pub fn from_checkpoint(bytes: &[u8]) -> SciRustResult<(Self, Vec<f32>)> {
        if bytes.len() < 40 || &bytes[..8] != b"CADAM001" {
            return Err(SciRustError::Empty);
        }
        let u32_at =
            |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        let n = u32_at(8) as usize;
        validate_workspace_size(n)?;
        let expected = n
            .checked_mul(16)
            .and_then(|x| x.checked_add(40))
            .ok_or(SciRustError::Overflow)?;
        if bytes.len() != expected {
            return Err(SciRustError::Shape {
                lhs: vec![bytes.len()],
                rhs: vec![expected],
            });
        }
        let step = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);
        let float_at = |i| f32::from_bits(u32_at(i));
        let vector = |offset: usize| (0..n).map(|i| float_at(offset + i * 4)).collect::<Vec<_>>();
        let opt = Self {
            lr: float_at(20),
            beta1: float_at(24),
            beta2: float_at(28),
            eps: float_at(32),
            weight_decay: float_at(36),
            state: ParamState {
                step,
                m: vector(40 + 4 * n),
                v: vector(40 + 8 * n),
                v_hat: vector(40 + 12 * n),
            },
        };
        let parameters = vector(40);
        validate_state(&opt.state, &parameters, &parameters)?;
        validate_hyperparams(opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay)?;
        Ok((opt, parameters))
    }
}
