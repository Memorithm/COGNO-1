//! Deterministic integer reward (COGNO-1 V2 §8 score component).
//!
//! Deliberately an integer so the lexicographic ordering and tie-break are
//! fully deterministic across platforms. Float reward machinery
//! (log-probabilities) belongs to Phase 4 and lives behind the model/backend
//! boundary, never in `cogno-core`. A reward can never cancel a hard
//! constraint (S4): the lexicographic `decide` checks hard constraints before
//! the score is ever considered.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reward(pub i64);

impl Reward {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }
}
