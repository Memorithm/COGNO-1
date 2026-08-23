//! Derived profile view (COGNO-1 V2 §18).
//!
//! The active profile is a **view derived** from the append-only journal. The
//! canonical `profile.md` is a non-canonical human view; the runtime serializes
//! derived state to `snapshot.bin`. This module computes the deterministic
//! derivation so that the replay tests can assert: same journal ⇒ same
//! profile.
//!
//! Rules lives one of `RuleState`s (§9). A rule moves `Candidate -> Active`
//! only by a deterministic policy (`min_evidence` distinct non-model proofs).
//! Model inference alone never creates a hard rule (§9). Contradictory rules
//! become `Conflicted`, never silently dropped.

use crate::evidence::{EvidenceId, EvidenceOrigin, RuleState};
use crate::journal::Journal;
use crate::semantic::SemanticMemoryBudget;
use std::collections::{BTreeMap, HashSet};

/// Category bucket carried by a `JournalEvent`. Reused here as the rule key.
type CategoryTag = u16;

/// A rule derived from the journal. The runtime holds the payload; the core
/// tracks state and evidence count so the lifecycle is deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedRule {
    pub state: RuleState,
    pub hard: bool,
    pub distinct_evidence: u32,
    pub model_only: bool,
    pub conflicts: Vec<CategoryTag>,
}

impl DerivedRule {
    /// A rule becomes `Active` only with at least `min_evidence` distinct
    /// non-model proofs (§9). A model-only rule stays `Quarantined`.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, RuleState::Active)
    }

    #[must_use]
    pub const fn is_hard(&self) -> bool {
        self.hard
    }
}

/// Deterministic derivation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivationPolicy {
    /// Minimum distinct non-model evidence to promote `Candidate -> Active`.
    pub min_evidence: u32,
    /// Whether a rule with any model evidence but no user/formal evidence is
    /// kept (in quarantine). Always true per §9.
    pub quarantine_model_only: bool,
}

impl DerivationPolicy {
    pub const DEFAULT: Self = Self {
        min_evidence: 2,
        quarantine_model_only: true,
    };
}

/// Derived profile, computed from a `Journal` under a `DerivationPolicy`.
#[derive(Debug, Default)]
pub struct Profile {
    pub rules: BTreeMap<CategoryTag, DerivedRule>,
}

impl Profile {
    /// Derive a profile from a journal. Pure and deterministic: same journal
    /// and policy ⇒ same profile (replay invariant S6/S7).
    ///
    /// Contradiction handling (§9, T06): a `UserRejection` **and** an
    /// approval/statement for the same category mark it `Conflicted`,
    /// regardless of the order the events were appended in. Both pieces of
    /// evidence are kept in the journal (the journal is append-only); the
    /// derived view surfaces the conflict so the host can arbitrate.
    ///
    /// `distinct_evidence` counts **distinct evidence ids** (§9): replaying
    /// the same id with different fingerprints cannot inflate authority.
    pub fn derive(journal: &Journal, policy: DerivationPolicy) -> Self {
        let mut p = Self::default();
        let budget = SemanticMemoryBudget::MVP_SAFE;
        // Per-category derivation state, folded in append order but reduced
        // order-independently at the end.
        let mut approved_seen: BTreeMap<CategoryTag, bool> = BTreeMap::new();
        let mut rejected_seen: BTreeMap<CategoryTag, bool> = BTreeMap::new();
        let mut seen_ids_per_rule: BTreeMap<CategoryTag, HashSet<EvidenceId>> = BTreeMap::new();

        for ev in journal.iter() {
            if !budget.allows_payload(ev.payload.len()) {
                continue;
            }
            let entry = p.rules.entry(ev.category_tag).or_insert(DerivedRule {
                state: RuleState::Quarantined,
                hard: false,
                distinct_evidence: 0,
                model_only: true,
                conflicts: Vec::new(),
            });
            let tag = ev.category_tag;

            let is_rejection = matches!(ev.evidence_origin, EvidenceOrigin::UserRejection);
            if is_rejection {
                rejected_seen.insert(tag, true);
            }

            let authoritative = !matches!(
                ev.evidence_origin,
                EvidenceOrigin::ModelInference
                    | EvidenceOrigin::ImportedProfile
                    | EvidenceOrigin::ToolObservation
            );
            if authoritative {
                if !is_rejection {
                    approved_seen.insert(tag, true);
                }
                // Count distinct evidence ids only; duplicates of an already
                // seen id never inflate `distinct_evidence`.
                let ids = seen_ids_per_rule.entry(tag).or_default();
                if ids.insert(ev.evidence_id) {
                    entry.distinct_evidence = entry.distinct_evidence.saturating_add(1);
                }
                entry.model_only = false;
                if ev.evidence_origin.can_create_hard_rule() {
                    entry.hard = true;
                }
            }
        }

        for (&tag, entry) in p.rules.iter_mut() {
            let conflicted = approved_seen.contains_key(&tag) && rejected_seen.contains_key(&tag);
            if conflicted {
                // A conflicted rule is never `Active`: the host must arbitrate
                // before it can be re-promoted. The evidence is preserved.
                entry.conflicts.push(tag);
                entry.state = RuleState::Conflicted;
            } else if entry.model_only && policy.quarantine_model_only {
                entry.state = RuleState::Quarantined;
            } else if entry.distinct_evidence >= policy.min_evidence {
                entry.state = RuleState::Active;
            } else {
                entry.state = RuleState::Candidate;
            }
        }
        p
    }

    #[must_use]
    pub fn rule(&self, tag: CategoryTag) -> Option<&DerivedRule> {
        self.rules.get(&tag)
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.rules.values().filter(|r| r.is_active()).count()
    }

    #[must_use]
    pub fn hard_count(&self) -> usize {
        self.rules.values().filter(|r| r.is_hard()).count()
    }
}
