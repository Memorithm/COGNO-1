//! Scalar reward engine (COGNO-1 V2 §8, Phase 4 prerequisite).
//!
//! The reward engine is a pure, deterministic integer scalar. It runs **after**
//! the lexicographic gates (structural, hard, capability, privacy), so it can
//! never compensate a hard violation (S4). Float reward machinery
//! (log-probabilities) belongs to Phase 4 and lives behind the model/backend
//! boundary; the core exposes only the integer scoring surface so the
//! tie-break is platform-independent.

use crate::ids::CandidateScore;
use crate::reward::Reward;
use crate::MemoryError;

/// Integer scoring parameters. All fields are `i64`/`u16` so addition cannot
/// drift on platform float semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewardParams {
    pub confidence_weight: i64,
    pub evidence_weight: i64,
    pub safety_penalty: i64,
}

impl RewardParams {
    pub const DEFAULT: Self = Self {
        confidence_weight: 1,
        evidence_weight: 4,
        // Safety penalty is a *negative* contribution; it can only lower the
        // score. The lexicographic gate already rejected hard violations, so
        // this penalty is for soft safety smells, not for compensable ones.
        safety_penalty: -1024,
    };
}

/// Scalar reward engine. Deterministic, allocation-free, no float.
#[derive(Clone, Debug, Default)]
pub struct RewardEngine;

impl RewardEngine {
    /// Score a candidate from its proposal view and a reward signal. Returns
    /// `MemoryError::ArithmeticOverflow` if the integer sum wraps.
    pub fn score(
        &self,
        confidence_bps: u16,
        evidence_count: usize,
        signal: Reward,
        params: RewardParams,
    ) -> Result<CandidateScore, MemoryError> {
        let conf = i64::from(confidence_bps);
        let ev = match i64::try_from(evidence_count) {
            Ok(v) => v,
            Err(_) => return Err(MemoryError::ArithmeticOverflow),
        };

        let term_a = conf
            .checked_mul(params.confidence_weight)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        let term_b = ev
            .checked_mul(params.evidence_weight)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        let term_c = params.safety_penalty;

        let total = term_a
            .checked_add(term_b)
            .ok_or(MemoryError::ArithmeticOverflow)?
            .checked_add(term_c)
            .ok_or(MemoryError::ArithmeticOverflow)?
            .checked_add(signal.0)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        Ok(CandidateScore::new(total))
    }
}
