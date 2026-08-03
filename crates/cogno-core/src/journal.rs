//! Append-only event journal (COGNO-1 V2 §18, S6).
//!
//! The journal is the source of truth; the active profile is a derived view.
//! This module is a pure-Rust, in-memory representation used by tests and by
//! the runtime's journal layer. On disk, the runtime serializes events to
//! `events.log` (append-only), `snapshot.bin` (compacted state),
//! `manifest.json` (versions + fingerprints) and `profile.md` (non-canonical
//! human view). The core itself does no filesystem I/O (crate invariant), so
//! this module exposes the data model and deterministic append/compact logic.

use crate::evidence::{EvidenceId, EvidenceOrigin};
use crate::origin::InputOrigin;

/// Opaque event identifier. Monotonic per journal; duplicates are detected by
/// `fingerprint` (a model can't fabricate authority by replaying an id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u64);

/// Fingerprint over the event's payload and provenance. The runtime computes
/// it (SHA-256 in §21); the core only carries it. Two events with equal
/// `fingerprint` are treated as duplicates (§9 — duplicates counted once).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub [u8; 32]);

impl Fingerprint {
    pub const ZERO: Self = Self([0u8; 32]);

    #[must_use]
    pub const fn is_zero(&self) -> bool {
        let z = &self.0;
        z[0] == 0
            && z[1] == 0
            && z[2] == 0
            && z[3] == 0
            && z[4] == 0
            && z[5] == 0
            && z[6] == 0
            && z[7] == 0
            && z[8] == 0
            && z[9] == 0
            && z[10] == 0
            && z[11] == 0
            && z[12] == 0
            && z[13] == 0
            && z[14] == 0
            && z[15] == 0
            && z[16] == 0
            && z[17] == 0
            && z[18] == 0
            && z[19] == 0
            && z[20] == 0
            && z[21] == 0
            && z[22] == 0
            && z[23] == 0
            && z[24] == 0
            && z[25] == 0
            && z[26] == 0
            && z[27] == 0
            && z[28] == 0
            && z[29] == 0
            && z[30] == 0
            && z[31] == 0
    }
}

/// A journal event. Every field that influences persisted state carries
/// provenance (S6): `origin` (typed) and `evidence_origin` (authority).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalEvent {
    pub id: EventId,
    pub fingerprint: Fingerprint,
    pub origin: InputOrigin,
    pub evidence_origin: EvidenceOrigin,
    pub evidence_id: EvidenceId,
    /// Category bucket the runtime uses to derive profile entries. Opaque to
    /// the core; kept here so replay is self-contained.
    pub category_tag: u16,
    /// Bounded payload. The runtime enforces `max_payload_bytes` (§18) before
    /// appending; the core trusts the bound because the runtime is the
    /// authority.
    pub payload: Vec<u8>,
}

/// Append-only journal. Deduplicates by `fingerprint` (§9). Keeps the order in
/// which unique events were appended; compaction is a separate, deterministic
/// pass (`compact`).
#[derive(Debug, Default)]
pub struct Journal {
    events: Vec<JournalEvent>,
    seen: std::collections::HashSet<Fingerprint>,
    next_id: u64,
}

/// Journal append outcome. `Duplicate` means the fingerprint was already
/// present; the event is **not** appended again (counted once, §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendOutcome {
    Appended { id: EventId },
    Duplicate { id: EventId },
}

impl Journal {
    /// Append an event. Returns `Duplicate` (with the original id) if the
    /// fingerprint was already present; the journal state is unchanged in that
    /// case (no partial modification, §13 allocation principle extended to
    /// state).
    pub fn append(&mut self, mut event: JournalEvent) -> AppendOutcome {
        if let Some(existing) = self
            .events
            .iter()
            .find(|e| e.fingerprint == event.fingerprint)
        {
            return AppendOutcome::Duplicate { id: existing.id };
        }
        event.id = EventId(self.next_id);
        self.next_id = self.next_id.checked_add(1).expect("event id overflow");
        self.seen.insert(event.fingerprint);
        self.events.push(event);
        AppendOutcome::Appended {
            id: EventId(self.next_id - 1),
        }
    }

    /// Iterate events in append order. Used by the profile derivation and the
    /// replay tests.
    pub fn iter(&self) -> impl Iterator<Item = &JournalEvent> {
        self.events.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Reset to empty. Used by tests and by snapshot rebuild; the on-disk
    /// `events.log` is the permanent record and is never reset by the core.
    pub fn clear(&mut self) {
        self.events.clear();
        self.seen.clear();
        self.next_id = 0;
    }
}
