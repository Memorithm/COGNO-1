//! Longitudinal scientific-taste tracking across verified generations.
//!
//! History is bounded and deterministic. This module never mutates a live
//! runtime profile; it only reports whether a newer generation is stable,
//! improving, stale, or contradictory relative to prior verified observations.

use std::collections::BTreeMap;

/// Maximum number of retained observations per preference.
pub const MAX_HISTORY_PER_PREFERENCE: usize = 256;
/// Confidence change that is considered material drift.
pub const MATERIAL_DRIFT_BPS: u16 = 1_000;
/// Number of generations without observation after which a preference is stale.
pub const STALE_GENERATION_GAP: u64 = 32;

/// One verified longitudinal observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteHistoryObservation {
    pub generation: u64,
    pub preference_id: u64,
    pub confidence_bps: u16,
    pub active: bool,
    pub contradicted: bool,
}

/// Stability classification for one preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteDriftState {
    New,
    Stable,
    Strengthening,
    Weakening,
    Contradicted,
    Stale,
}

/// Derived longitudinal status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteLongitudinalStatus {
    pub preference_id: u64,
    pub first_generation: u64,
    pub last_generation: u64,
    pub observations: u16,
    pub latest_confidence_bps: u16,
    pub drift: TasteDriftState,
    pub eligible_for_supersession: bool,
}

/// Invalid longitudinal input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteLongitudinalError {
    ZeroGeneration,
    ConfidenceOutOfRange(u16),
    NonMonotonicObservation(u64),
    HistoryFull(u64),
}

/// Bounded deterministic longitudinal index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TasteLongitudinalTracker {
    observations: BTreeMap<u64, Vec<TasteHistoryObservation>>,
}

impl TasteLongitudinalTracker {
    /// Append a verified observation. Generations must increase per preference.
    pub fn append(
        &mut self,
        observation: TasteHistoryObservation,
    ) -> Result<(), TasteLongitudinalError> {
        if observation.generation == 0 {
            return Err(TasteLongitudinalError::ZeroGeneration);
        }
        if observation.confidence_bps > 10_000 {
            return Err(TasteLongitudinalError::ConfidenceOutOfRange(
                observation.confidence_bps,
            ));
        }
        let history = self
            .observations
            .entry(observation.preference_id)
            .or_default();
        if history.len() >= MAX_HISTORY_PER_PREFERENCE {
            return Err(TasteLongitudinalError::HistoryFull(
                observation.preference_id,
            ));
        }
        if history
            .last()
            .is_some_and(|previous| observation.generation <= previous.generation)
        {
            return Err(TasteLongitudinalError::NonMonotonicObservation(
                observation.preference_id,
            ));
        }
        history.push(observation);
        Ok(())
    }

    /// Derive status as of `current_generation` without mutating history.
    #[must_use]
    pub fn status(
        &self,
        preference_id: u64,
        current_generation: u64,
    ) -> Option<TasteLongitudinalStatus> {
        let history = self.observations.get(&preference_id)?;
        let first = *history.first()?;
        let latest = *history.last()?;
        let previous = history.iter().rev().nth(1).copied();

        let drift = if latest.contradicted {
            TasteDriftState::Contradicted
        } else if current_generation.saturating_sub(latest.generation) >= STALE_GENERATION_GAP {
            TasteDriftState::Stale
        } else if let Some(previous) = previous {
            classify_delta(previous.confidence_bps, latest.confidence_bps)
        } else {
            TasteDriftState::New
        };

        Some(TasteLongitudinalStatus {
            preference_id,
            first_generation: first.generation,
            last_generation: latest.generation,
            observations: u16::try_from(history.len()).unwrap_or(u16::MAX),
            latest_confidence_bps: latest.confidence_bps,
            drift,
            eligible_for_supersession: matches!(
                drift,
                TasteDriftState::Contradicted | TasteDriftState::Stale | TasteDriftState::Weakening
            ),
        })
    }

    /// Number of tracked preferences.
    #[must_use]
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

fn classify_delta(previous: u16, latest: u16) -> TasteDriftState {
    if latest >= previous.saturating_add(MATERIAL_DRIFT_BPS) {
        TasteDriftState::Strengthening
    } else if previous >= latest.saturating_add(MATERIAL_DRIFT_BPS) {
        TasteDriftState::Weakening
    } else {
        TasteDriftState::Stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weakening_is_reported_without_mutating_profile() {
        let mut tracker = TasteLongitudinalTracker::default();
        tracker
            .append(TasteHistoryObservation {
                generation: 1,
                preference_id: 7,
                confidence_bps: 9_000,
                active: true,
                contradicted: false,
            })
            .expect("first");
        tracker
            .append(TasteHistoryObservation {
                generation: 2,
                preference_id: 7,
                confidence_bps: 7_000,
                active: true,
                contradicted: false,
            })
            .expect("second");
        let status = tracker.status(7, 2).expect("status");
        assert_eq!(status.drift, TasteDriftState::Weakening);
        assert!(status.eligible_for_supersession);
    }

    #[test]
    fn stale_history_is_detected_by_generation_gap() {
        let mut tracker = TasteLongitudinalTracker::default();
        tracker
            .append(TasteHistoryObservation {
                generation: 1,
                preference_id: 7,
                confidence_bps: 8_000,
                active: true,
                contradicted: false,
            })
            .expect("append");
        assert_eq!(
            tracker
                .status(7, 1 + STALE_GENERATION_GAP)
                .map(|status| status.drift),
            Some(TasteDriftState::Stale)
        );
    }

    #[test]
    fn non_monotonic_history_fails_closed() {
        let mut tracker = TasteLongitudinalTracker::default();
        let observation = TasteHistoryObservation {
            generation: 2,
            preference_id: 7,
            confidence_bps: 8_000,
            active: true,
            contradicted: false,
        };
        tracker.append(observation).expect("first");
        assert_eq!(
            tracker.append(observation),
            Err(TasteLongitudinalError::NonMonotonicObservation(7))
        );
    }
}
