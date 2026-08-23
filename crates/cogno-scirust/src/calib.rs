//! Confidence calibration head (COGNO-1 §SciRust #5).
//!
//! Maps a raw score `z` to a calibrated confidence in basis points `0..=10_000`
//! via a temperature `T`: `confidence_bps = round(10_000 * sigmoid(z / T))`.
//! The temperature can be fitted post-hoc on a bounded held-out binary batch by
//! minimizing negative log-likelihood. Fitting optimizes `log(T)` so positivity
//! is structural and retains the best observed temperature, preventing the
//! calibration pass from committing a worse held-out NLL.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::tensor::Tensor;
use crate::{AdamW, Optimizer};

/// Maximum held-out calibration observations consumed by one fit.
pub const MAX_CALIBRATION_EXAMPLES: usize = 4_096;
/// Maximum deterministic optimizer epochs in one calibration fit.
pub const MAX_CALIBRATION_EPOCHS: u32 = 4_096;

/// Bounded post-hoc temperature fitting configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationFitConfig {
    pub epochs: u32,
    pub learning_rate: f32,
    pub min_temperature: f32,
    pub max_temperature: f32,
}

impl Default for CalibrationFitConfig {
    fn default() -> Self {
        Self {
            epochs: 128,
            learning_rate: 0.05,
            min_temperature: 0.05,
            max_temperature: 20.0,
        }
    }
}

impl CalibrationFitConfig {
    fn validate(self) -> SciRustResult<()> {
        if self.epochs == 0 || self.epochs > MAX_CALIBRATION_EPOCHS {
            return Err(SciRustError::CapacityExceeded {
                requested: self.epochs as usize,
                maximum: MAX_CALIBRATION_EPOCHS as usize,
            });
        }
        if self.learning_rate <= 0.0
            || self.learning_rate > 1.0
            || !self.learning_rate.is_finite()
            || self.min_temperature <= 0.0
            || !self.min_temperature.is_finite()
            || !self.max_temperature.is_finite()
            || self.max_temperature <= self.min_temperature
        {
            return Err(SciRustError::NonFinite);
        }
        Ok(())
    }
}

/// Deterministic summary of one held-out temperature fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationFitReport {
    pub examples: usize,
    pub epochs: u32,
    pub initial_nll: f32,
    pub final_nll: f32,
    pub temperature: f32,
}

/// Calibration head. Owns a positive temperature parameter.
#[derive(Clone, Debug, PartialEq)]
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
        stable_sigmoid(z / self.temperature)
    }

    /// Calibrate a raw score `z` to basis points, clamped to `0..=max_bps`.
    ///
    /// A non-finite `z` maps to `0` bps — the **most conservative** confidence
    /// — rather than silently round-tripping NaN into a plausible-looking
    /// value. Batch callers that must surface the corruption use
    /// [`Self::try_calibrate_batch`].
    #[must_use]
    pub fn calibrate_bps(&self, z: f32) -> u16 {
        if !z.is_finite() {
            return 0;
        }
        let s = self.sigmoid(z);
        let bps = (s * f32::from(self.max_bps)).round();
        let clamped = bps.clamp(0.0, f32::from(self.max_bps));
        clamped as u16
    }

    /// Batch calibrate. Returns an error instead of a confidence when any
    /// input score is non-finite: a corrupted score must never be published
    /// as a valid-looking confidence (surfaced, not propagated).
    pub fn try_calibrate_batch(&self, z: &Tensor) -> SciRustResult<CalibratedConfidence> {
        if z.data.iter().any(|v| !v.is_finite()) {
            return Err(SciRustError::NonFinite);
        }
        self.calibrate_batch(z)
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

    /// Mean binary negative log-likelihood at the current temperature.
    ///
    /// `targets[i] == true` denotes the positive class for `logits[i]`.
    pub fn binary_nll(&self, logits: &[f32], targets: &[bool]) -> SciRustResult<f32> {
        validate_binary_batch(logits, targets)?;
        binary_nll_at_temperature(logits, targets, self.temperature)
    }

    /// Fit the temperature on an explicit held-out binary calibration set.
    ///
    /// Only the scalar temperature is updated; model weights/logits remain
    /// frozen. Optimization occurs in `log(T)` space with AdamW weight decay
    /// disabled. The best observed held-out NLL is retained, so this method
    /// never commits an optimizer iterate that is worse than its bounded start.
    pub fn fit_binary(
        &mut self,
        logits: &[f32],
        targets: &[bool],
        config: CalibrationFitConfig,
    ) -> SciRustResult<CalibrationFitReport> {
        config.validate()?;
        validate_binary_batch(logits, targets)?;

        let bounded_start = self
            .temperature
            .clamp(config.min_temperature, config.max_temperature);
        ensure_finite(bounded_start)?;
        let initial_nll = binary_nll_at_temperature(logits, targets, bounded_start)?;
        let mut best_temperature = bounded_start;
        let mut best_nll = initial_nll;

        let min_log_temperature = config.min_temperature.ln();
        let max_log_temperature = config.max_temperature.ln();
        ensure_finite(min_log_temperature)?;
        ensure_finite(max_log_temperature)?;
        let mut log_temperature = vec![bounded_start.ln()];
        let mut optimizer = AdamW::try_new(config.learning_rate, 1)?;
        optimizer.weight_decay = 0.0;
        let divisor = logits.len() as f32;
        ensure_finite(divisor)?;

        for _ in 0..config.epochs {
            let temperature = log_temperature[0].exp();
            ensure_finite(temperature)?;
            let mut gradient = 0.0f32;
            for (&logit, &target) in logits.iter().zip(targets) {
                let scaled = logit / temperature;
                ensure_finite(scaled)?;
                let probability = stable_sigmoid(scaled);
                let target = if target { 1.0 } else { 0.0 };
                // d NLL / d log(T) = -(sigmoid(z/T) - y) * z / T.
                gradient += -(probability - target) * logit / temperature;
                ensure_finite(gradient)?;
            }
            gradient /= divisor;
            ensure_finite(gradient)?;
            optimizer.step(&mut log_temperature, &[gradient])?;
            log_temperature[0] = log_temperature[0].clamp(min_log_temperature, max_log_temperature);

            let candidate_temperature = log_temperature[0].exp();
            ensure_finite(candidate_temperature)?;
            let candidate_nll = binary_nll_at_temperature(logits, targets, candidate_temperature)?;
            if candidate_nll < best_nll {
                best_nll = candidate_nll;
                best_temperature = candidate_temperature;
            }
        }

        self.set_temperature(best_temperature)?;
        Ok(CalibrationFitReport {
            examples: logits.len(),
            epochs: config.epochs,
            initial_nll,
            final_nll: best_nll,
            temperature: best_temperature,
        })
    }

    /// Set the temperature after validating positivity/finiteness.
    pub fn set_temperature(&mut self, temperature: f32) -> SciRustResult<()> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        self.temperature = temperature;
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

fn validate_binary_batch(logits: &[f32], targets: &[bool]) -> SciRustResult<()> {
    if logits.is_empty() {
        return Err(SciRustError::Empty);
    }
    if logits.len() != targets.len() {
        return Err(SciRustError::Shape {
            lhs: vec![logits.len()],
            rhs: vec![targets.len()],
        });
    }
    if logits.len() > MAX_CALIBRATION_EXAMPLES {
        return Err(SciRustError::CapacityExceeded {
            requested: logits.len(),
            maximum: MAX_CALIBRATION_EXAMPLES,
        });
    }
    for &logit in logits {
        ensure_finite(logit)?;
    }
    Ok(())
}

fn binary_nll_at_temperature(
    logits: &[f32],
    targets: &[bool],
    temperature: f32,
) -> SciRustResult<f32> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(SciRustError::NonFinite);
    }
    let mut loss = 0.0f32;
    for (&logit, &target) in logits.iter().zip(targets) {
        let scaled = logit / temperature;
        ensure_finite(scaled)?;
        let softplus = if scaled >= 0.0 {
            scaled + (-scaled).exp().ln_1p()
        } else {
            scaled.exp().ln_1p()
        };
        let target = if target { 1.0 } else { 0.0 };
        loss += softplus - target * scaled;
        ensure_finite(loss)?;
    }
    let divisor = logits.len() as f32;
    ensure_finite(divisor)?;
    let mean = loss / divisor;
    ensure_finite(mean)?;
    Ok(mean)
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit_config() -> CalibrationFitConfig {
        CalibrationFitConfig {
            epochs: 96,
            learning_rate: 0.05,
            min_temperature: 0.1,
            max_temperature: 20.0,
        }
    }

    #[test]
    fn deterministic_temperature_fit_reduces_held_out_nll() {
        let logits = [4.0, 4.0, -4.0, -4.0];
        let targets = [true, false, false, true];
        let mut left = Calibration::try_new(1.0).expect("left");
        let mut right = Calibration::try_new(1.0).expect("right");
        let left_report = left
            .fit_binary(&logits, &targets, fit_config())
            .expect("left fit");
        let right_report = right
            .fit_binary(&logits, &targets, fit_config())
            .expect("right fit");

        assert_eq!(left, right);
        assert_eq!(left_report, right_report);
        assert!(left_report.final_nll < left_report.initial_nll);
        assert!(left_report.temperature > 1.0);
        assert!((0.1..=20.0).contains(&left_report.temperature));
        assert_eq!(
            left_report.final_nll,
            left.binary_nll(&logits, &targets).expect("nll")
        );
    }

    #[test]
    fn fit_never_commits_worse_nll_than_bounded_start() {
        let logits = [2.0, -2.0, 1.0, -1.0];
        let targets = [true, false, true, false];
        let mut calibration = Calibration::try_new(1.0).expect("calibration");
        let report = calibration
            .fit_binary(&logits, &targets, fit_config())
            .expect("fit");
        assert!(report.final_nll <= report.initial_nll);
        assert!(calibration.temperature.is_finite());
        assert!(calibration.temperature > 0.0);
    }

    #[test]
    fn hostile_calibration_batches_fail_closed() {
        let mut calibration = Calibration::try_new(1.0).expect("calibration");
        assert_eq!(
            calibration.fit_binary(&[], &[], fit_config()),
            Err(SciRustError::Empty)
        );
        assert!(matches!(
            calibration.fit_binary(&[1.0, 2.0], &[true], fit_config()),
            Err(SciRustError::Shape { .. })
        ));
        assert_eq!(
            calibration.fit_binary(&[f32::NAN], &[true], fit_config()),
            Err(SciRustError::NonFinite)
        );
    }

    #[test]
    fn hostile_fit_configuration_fails_closed() {
        let mut calibration = Calibration::try_new(1.0).expect("calibration");
        let mut config = fit_config();
        config.epochs = 0;
        assert!(matches!(
            calibration.fit_binary(&[1.0], &[true], config),
            Err(SciRustError::CapacityExceeded { .. })
        ));
        let mut config = fit_config();
        config.min_temperature = 2.0;
        config.max_temperature = 1.0;
        assert_eq!(
            calibration.fit_binary(&[1.0], &[true], config),
            Err(SciRustError::NonFinite)
        );
    }
}
