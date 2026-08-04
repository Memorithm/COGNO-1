//! Confidence calibration head (COGNO-1 §SciRust #5).
//!
//! Maps a raw score `z` to a calibrated confidence in basis points `0..=10_000`
//! via a temperature `T`: `confidence_bps = round(10_000 * sigmoid(z / T))`.
//! `T` is learnable (the optimizer adjusts it). The output is clamped to the
//! `CognoProposalView::confidence_bps` range so it always validates against
//! the strict §2 schema (any value `> 10_000` is rejected upstream).

use crate::error::{SciRustError, SciRustResult};
use crate::tensor::Tensor;

/// Calibration head. Owns a temperature parameter.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub temperature: f32,
    pub max_bps: u16,
}

impl Calibration {
    /// Construct with a temperature. `temperature` must be positive finite.
    pub fn try_new(temperature: f32) -> SciRustResult<Self> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        Ok(Self {
            temperature,
            max_bps: 10_000,
        })
    }

    /// Sigmoid(z/T), numerically stable.
    fn sigmoid(&self, z: f32) -> f32 {
        let x = z / self.temperature;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// Calibrate a raw score `z` to basis points, clamped to `0..=max_bps`.
    #[must_use]
    pub fn calibrate_bps(&self, z: f32) -> u16 {
        let s = self.sigmoid(z);
        let bps = (s * f32::from(self.max_bps)).round();
        let clamped = bps.clamp(0.0, f32::from(self.max_bps));
        clamped as u16
    }

    /// Batch calibrate. Returns a `CalibratedConfidence` per input.
    pub fn calibrate_batch(&self, z: &Tensor) -> SciRustResult<CalibratedConfidence> {
        if z.shape.rank() != 1 {
            return Err(SciRustError::Shape {
                lhs: z.shape.as_slice().to_vec(),
                rhs: vec![1],
            });
        }
        let bps: Vec<u16> = z.data.iter().map(|&v| self.calibrate_bps(v)).collect();
        Ok(CalibratedConfidence {
            bps,
            temperature: self.temperature,
        })
    }

    /// Set the temperature (used by the optimizer to step the parameter).
    pub fn set_temperature(&mut self, t: f32) -> SciRustResult<()> {
        if t <= 0.0 || !t.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        self.temperature = t;
        Ok(())
    }
}

/// Result of a batch calibration. `temperature` is `f32` so we only derive
/// `PartialEq` (not `Eq`: floats are not `Eq`).
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedConfidence {
    pub bps: Vec<u16>,
    pub temperature: f32,
}

impl CalibratedConfidence {
    #[must_use]
    pub fn all_within_range(&self) -> bool {
        self.bps.iter().all(|&b| b <= 10_000)
    }
}
