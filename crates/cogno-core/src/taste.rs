//! Deterministic scientific-taste profile for COGNO-1.
//!
//! The assistance model may propose preferences, but this module alone derives
//! the active profile. Model output starts quarantined, evidence is bounded,
//! contradictions are preserved, and promotion is deterministic.

use std::collections::{BTreeMap, BTreeSet};

pub const MAX_TASTE_CONFIDENCE_BPS: u16 = 10_000;
pub const MAX_TASTE_EVIDENCE_IDS: usize = 16;
/// Upper bound for a source identifier (harness model name, agent id, ...).
/// Bounded so hostile harnesses cannot inflate memory through ids (S5).
pub const MAX_TASTE_SOURCE_ID_BYTES: usize = 64;
/// Upper bound for a scope key (e.g. `project:cogno-1@9a2f…`). A preference
/// learned for one scope must never silently leak into another (étape 4).
pub const MAX_SCOPE_KEY_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TasteScope {
    User,
    Project,
    Domain,
    Team,
}

impl TasteScope {
    /// Validate the bounded, printable identity of a scope instance
    /// (`user:alice`, `project:cogno-1@9a2f…`, `domain:rust-macros`,
    /// `team:acme-platform`). Fail closed on empty, oversized or
    /// non-printable keys: an unbound scope is not a real scope.
    pub fn validate_key(self, key: &str) -> Result<(), TasteError> {
        if key.is_empty()
            || key.len() > MAX_SCOPE_KEY_BYTES
            || !key.bytes().all(|b| (0x21..=0x7e).contains(&b))
        {
            return Err(TasteError::InvalidScopeKey);
        }
        Ok(())
    }
}

/// Class of the harness participant that produced a taste event. Models may
/// propose; humans and deterministic kernels confirm — the class never grants
/// authority by itself, it only makes provenance attributable (S6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TasteSourceKind {
    /// An assistance model (LLM/SLM) in the host's model harness.
    AssistanceModel,
    /// An explicit human action on the host side.
    Human,
    /// A deterministic evaluator (SciRust kernels, validators).
    DeterministicKernel,
    /// Another agent speaking the exchange protocol.
    ExternalAgent,
    /// Legacy or unidentified provenance (pre-attribution data). Kept
    /// explicit so replays never fabricate a plausible-looking origin.
    Unknown,
}

/// Bounded, printable identity of whoever produced an event: e.g.
/// `(AssistanceModel, "claude-x")`, `(DeterministicKernel, "scirust@gen7")`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TasteSource {
    pub kind: TasteSourceKind,
    pub id: String,
}

impl TasteSource {
    /// Canonical placeholder for events reconstructed without provenance
    /// (package imports, legacy replays). An empty id is the honest
    /// representation of "no identity claim"; it never grants authority.
    pub const UNKNOWN: Self = Self {
        kind: TasteSourceKind::Unknown,
        id: String::new(),
    };

    /// Construct a bounded source id. Non-empty (except for
    /// [`TasteSourceKind::Unknown`], which may carry no claim), at most
    /// [`MAX_TASTE_SOURCE_ID_BYTES`] bytes, ASCII printable only (no control
    /// characters, no homoglyph confusables): fail closed otherwise.
    pub fn try_new(kind: TasteSourceKind, id: &str) -> Result<Self, TasteError> {
        if id.is_empty() && kind != TasteSourceKind::Unknown {
            return Err(TasteError::InvalidSourceId);
        }
        if id.len() > MAX_TASTE_SOURCE_ID_BYTES || !id.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
            return Err(TasteError::InvalidSourceId);
        }
        Ok(Self {
            kind,
            id: id.to_string(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteEventKind {
    Proposed,
    Accepted,
    Rejected,
    Edited,
    Confirmed,
    Contradicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteOrigin {
    ModelInference,
    ExplicitUserAction,
    DeterministicEvaluation,
    ImportedProfile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteState {
    Quarantined,
    Candidate,
    Active,
    Conflicted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TasteEvent {
    pub event_id: u64,
    pub preference_id: u64,
    pub scope: TasteScope,
    /// Bounded identity of the scope instance this event belongs to
    /// (`project:cogno-1@9a2f…`). Two events with the same preference id but
    /// different scope identities never merge silently — they conflict.
    pub scope_key: String,
    pub kind: TasteEventKind,
    pub origin: TasteOrigin,
    pub confidence_bps: u16,
    pub evidence_ids: Vec<u64>,
    /// Harness participant that produced this event. Required: an
    /// unattributed event cannot be replayed into a profile, because the
    /// harness vision depends on knowing WHICH model/human/kernel spoke.
    pub source: TasteSource,
}

impl TasteEvent {
    pub fn validate(&self) -> Result<(), TasteError> {
        if self.confidence_bps > MAX_TASTE_CONFIDENCE_BPS {
            return Err(TasteError::ConfidenceOutOfRange);
        }
        self.scope.validate_key(&self.scope_key)?;
        if self.evidence_ids.len() > MAX_TASTE_EVIDENCE_IDS {
            return Err(TasteError::EvidenceLimitExceeded);
        }
        let mut unique = BTreeSet::new();
        for evidence_id in &self.evidence_ids {
            if !unique.insert(*evidence_id) {
                return Err(TasteError::DuplicateEvidence);
            }
        }
        // Unknown sources carry no identity claim (empty id is allowed);
        // attributed sources must have a bounded printable id.
        if self.source.kind != TasteSourceKind::Unknown && self.source.id.is_empty() {
            return Err(TasteError::InvalidSourceId);
        }
        if self.source.id.len() > MAX_TASTE_SOURCE_ID_BYTES
            || !self.source.id.bytes().all(|b| (0x21..=0x7e).contains(&b))
        {
            return Err(TasteError::InvalidSourceId);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteError {
    ConfidenceOutOfRange,
    EvidenceLimitExceeded,
    DuplicateEvidence,
    DuplicateEvent,
    /// Source id empty, oversized or carrying non-printable bytes.
    InvalidSourceId,
    /// Scope key empty, oversized or carrying non-printable bytes: the scope
    /// instance is not a real, bounded identity (étape 4).
    InvalidScopeKey,
    /// Arbitration target does not exist in the derived profile.
    UnknownPreference(u64),
    /// Only `Conflicted` entries may be arbitrated; anything else stays
    /// under deterministic derivation control (fail closed).
    TargetNotConflicted(u64),
    /// The arbitration authority is not authorized to resolve conflicts.
    UnauthorizedArbitrationAuthority,
    /// Human arbitration lifts the conflict block only: numeric activation
    /// gates must still be met on their own (S4 — no compensation).
    ActivationGateNotMet(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TastePolicy {
    pub activation_threshold_bps: u16,
    pub minimum_non_model_confirmations: u16,
    pub acceptance_gain_bps: u16,
    pub confirmation_gain_bps: u16,
    pub edit_gain_bps: u16,
    pub rejection_loss_bps: u16,
    pub contradiction_loss_bps: u16,
}

impl TastePolicy {
    pub const DEFAULT: Self = Self {
        activation_threshold_bps: 7_000,
        minimum_non_model_confirmations: 2,
        acceptance_gain_bps: 1_000,
        confirmation_gain_bps: 1_500,
        edit_gain_bps: 500,
        rejection_loss_bps: 2_000,
        contradiction_loss_bps: 3_000,
    };
}

/// Host consent switches for the taste system (étape 4). Pure data — hosts
/// persist and load it; derivation never mutates it. Defaults are
/// privacy-preserving: local learning is on, but nothing ever leaves or
/// enters the machine without explicit consent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteSettings {
    /// Master switch: record and replay taste events locally.
    pub learning_enabled: bool,
    /// Consent to export preferences into a portable `taste.md` package.
    pub export_allowed: bool,
    /// Consent to ingest foreign `taste.md` packages (imports stay
    /// quarantined regardless — consent only permits ingestion at all).
    pub import_allowed: bool,
}

impl Default for TasteSettings {
    fn default() -> Self {
        Self {
            learning_enabled: true,
            export_allowed: false,
            import_allowed: false,
        }
    }
}

impl TasteSettings {
    pub const fn can_learn(&self) -> bool {
        self.learning_enabled
    }

    pub const fn can_export(&self) -> bool {
        self.learning_enabled && self.export_allowed
    }

    pub const fn can_import(&self) -> bool {
        self.import_allowed
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScientificTaste {
    pub preference_id: u64,
    pub scope: TasteScope,
    /// Identity of the scope instance the preference was learned in. Events
    /// for the same id under a different scope identity conflict (below).
    pub scope_key: String,
    pub confidence_bps: u16,
    pub state: TasteState,
    pub non_model_confirmations: u16,
    pub accepted_count: u32,
    pub rejected_count: u32,
    pub edited_count: u32,
    pub contradicted_count: u32,
    /// Every harness source that contributed an event to this preference
    /// (models proposing it, humans confirming it, kernels evaluating it).
    /// Attribution only — never authority.
    pub sources: BTreeSet<TasteSource>,
    /// Number of human arbitrations applied to this entry (conflict
    /// resolutions). Derivation-relevant state, replayed deterministically.
    pub conflict_resolutions: u32,
}

impl ScientificTaste {
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, TasteState::Active)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ScientificTasteProfile {
    pub preferences: BTreeMap<u64, ScientificTaste>,
}

impl ScientificTasteProfile {
    pub fn derive(events: &[TasteEvent], policy: TastePolicy) -> Result<Self, TasteError> {
        let mut seen_events = BTreeSet::new();
        let mut ordered = events.to_vec();
        ordered.sort_by_key(|event| event.event_id);

        let mut profile = Self::default();
        for event in ordered {
            event.validate()?;
            if !seen_events.insert(event.event_id) {
                return Err(TasteError::DuplicateEvent);
            }

            let entry = profile
                .preferences
                .entry(event.preference_id)
                .or_insert(ScientificTaste {
                    preference_id: event.preference_id,
                    scope: event.scope,
                    scope_key: event.scope_key.clone(),
                    confidence_bps: event.confidence_bps,
                    state: TasteState::Quarantined,
                    non_model_confirmations: 0,
                    accepted_count: 0,
                    rejected_count: 0,
                    edited_count: 0,
                    contradicted_count: 0,
                    sources: BTreeSet::new(),
                    conflict_resolutions: 0,
                });

            // Real scopes (étape 4): the first-seen (scope, scope_key) pair —
            // deterministic because events are sorted by event_id — binds the
            // preference to one scope instance. Events from a different scope
            // instance never merge; they conflict the entry instead.
            if entry.scope != event.scope || entry.scope_key != event.scope_key {
                entry.state = TasteState::Conflicted;
                continue;
            }
            entry.sources.insert(event.source.clone());
            match event.kind {
                TasteEventKind::Proposed => {
                    entry.confidence_bps = entry.confidence_bps.max(event.confidence_bps);
                }
                TasteEventKind::Accepted => {
                    entry.accepted_count = entry.accepted_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.acceptance_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Edited => {
                    entry.edited_count = entry.edited_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.edit_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Confirmed => {
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_add(policy.confirmation_gain_bps)
                        .min(MAX_TASTE_CONFIDENCE_BPS);
                    register_confirmation(entry, event.origin);
                }
                TasteEventKind::Rejected => {
                    entry.rejected_count = entry.rejected_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_sub(policy.rejection_loss_bps);
                    if matches!(event.origin, TasteOrigin::ExplicitUserAction) {
                        entry.state = TasteState::Rejected;
                    }
                }
                TasteEventKind::Contradicted => {
                    entry.contradicted_count = entry.contradicted_count.saturating_add(1);
                    entry.confidence_bps = entry
                        .confidence_bps
                        .saturating_sub(policy.contradiction_loss_bps);
                    entry.state = TasteState::Conflicted;
                }
            }
        }

        for entry in profile.preferences.values_mut() {
            if matches!(entry.state, TasteState::Conflicted | TasteState::Rejected) {
                continue;
            }
            entry.state = if entry.non_model_confirmations >= policy.minimum_non_model_confirmations
                && entry.confidence_bps >= policy.activation_threshold_bps
            {
                TasteState::Active
            } else if entry.non_model_confirmations > 0 {
                TasteState::Candidate
            } else {
                TasteState::Quarantined
            };
        }

        Ok(profile)
    }

    /// Derive a profile and then apply explicit human arbitrations, in order,
    /// deterministically: same events + same policy + same arbitrations ⇒ same
    /// profile (S6). See [`ScientificTasteProfile::arbitrate`] for the rules.
    pub fn derive_with_arbitrations(
        events: &[TasteEvent],
        policy: TastePolicy,
        arbitrations: &[TasteArbitration],
    ) -> Result<Self, TasteError> {
        let mut profile = Self::derive(events, policy)?;
        for arbitration in arbitrations {
            profile.arbitrate(&policy, *arbitration)?;
        }
        Ok(profile)
    }

    /// Apply one human arbitration to a `Conflicted` entry (étape 2 —
    /// arbitrage hôte). Fail-closed rules:
    ///
    /// - only `ExplicitHumanHost` may arbitrate (models never resolve their
    ///   own contradictions);
    /// - only `Conflicted` entries are arbitable — `Active`/`Rejected`/
    ///   `Quarantined` stay under deterministic derivation control;
    /// - `Activate` re-checks the numeric gates (`confidence_bps ≥ threshold`
    ///   AND `non_model_confirmations ≥ minimum`): the host lifts the
    ///   conflict **block**, never the evidence requirements (S4).
    pub fn arbitrate(
        &mut self,
        policy: &TastePolicy,
        arbitration: TasteArbitration,
    ) -> Result<(), TasteError> {
        if !matches!(
            arbitration.authority,
            TasteArbitrationAuthority::ExplicitHumanHost
        ) {
            return Err(TasteError::UnauthorizedArbitrationAuthority);
        }
        let entry = self
            .preferences
            .get_mut(&arbitration.preference_id)
            .ok_or(TasteError::UnknownPreference(arbitration.preference_id))?;
        if !matches!(entry.state, TasteState::Conflicted) {
            return Err(TasteError::TargetNotConflicted(arbitration.preference_id));
        }
        entry.state = match arbitration.action {
            TasteArbitrationAction::Reject => TasteState::Rejected,
            TasteArbitrationAction::Quarantine => TasteState::Quarantined,
            TasteArbitrationAction::Activate => {
                if entry.confidence_bps >= policy.activation_threshold_bps
                    && entry.non_model_confirmations >= policy.minimum_non_model_confirmations
                {
                    TasteState::Active
                } else {
                    return Err(TasteError::ActivationGateNotMet(arbitration.preference_id));
                }
            }
        };
        entry.conflict_resolutions = entry.conflict_resolutions.saturating_add(1);
        Ok(())
    }

    /// Preferences that reached `Active` and were contributed to by `source`.
    /// The harness-facing query: "what does this model's taste agree with?"
    #[must_use]
    pub fn active_preferences_for_source(&self, source: &TasteSource) -> Vec<u64> {
        self.preferences
            .values()
            .filter(|item| item.is_active() && item.sources.contains(source))
            .map(|item| item.preference_id)
            .collect()
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.preferences
            .values()
            .filter(|item| item.is_active())
            .count()
    }
}

fn register_confirmation(entry: &mut ScientificTaste, origin: TasteOrigin) {
    if matches!(
        origin,
        TasteOrigin::ExplicitUserAction | TasteOrigin::DeterministicEvaluation
    ) {
        entry.non_model_confirmations = entry.non_model_confirmations.saturating_add(1);
    }
}

/// Who may arbitrate. Only the explicit human host can resolve a conflict:
/// models never arbitrate their own contradictions (§9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteArbitrationAuthority {
    ExplicitHumanHost,
}

/// What to do with a `Conflicted` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TasteArbitrationAction {
    /// Lift the conflict block and activate — numeric gates re-checked (S4).
    Activate,
    /// Return the entry to quarantine for more evidence.
    Quarantine,
    /// Authoritatively reject the preference.
    Reject,
}

/// One host arbitration decision over a conflicted preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TasteArbitration {
    pub preference_id: u64,
    pub action: TasteArbitrationAction,
    pub authority: TasteArbitrationAuthority,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(kind: TasteSourceKind) -> TasteSource {
        let id = match kind {
            TasteSourceKind::AssistanceModel => "model-a",
            TasteSourceKind::Human => "host",
            TasteSourceKind::DeterministicKernel => "scirust",
            TasteSourceKind::ExternalAgent => "agent-1",
            TasteSourceKind::Unknown => "unattributed",
        };
        TasteSource::try_new(kind, id).expect("valid source")
    }

    fn event(event_id: u64, kind: TasteEventKind, origin: TasteOrigin) -> TasteEvent {
        TasteEvent {
            event_id,
            preference_id: 7,
            scope: TasteScope::Project,
            scope_key: "project:cogno-1".to_string(),
            kind,
            origin,
            confidence_bps: 5_000,
            evidence_ids: vec![event_id],
            source: source(match origin {
                TasteOrigin::ModelInference => TasteSourceKind::AssistanceModel,
                TasteOrigin::ExplicitUserAction => TasteSourceKind::Human,
                TasteOrigin::DeterministicEvaluation => TasteSourceKind::DeterministicKernel,
                TasteOrigin::ImportedProfile => TasteSourceKind::ExternalAgent,
            }),
        }
    }

    #[test]
    fn model_only_preference_remains_quarantined() {
        let profile = ScientificTasteProfile::derive(
            &[event(
                1,
                TasteEventKind::Proposed,
                TasteOrigin::ModelInference,
            )],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Quarantined);
    }

    #[test]
    fn deterministic_evidence_can_activate_preference() {
        let profile = ScientificTasteProfile::derive(
            &[
                event(
                    1,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                ),
                event(
                    2,
                    TasteEventKind::Confirmed,
                    TasteOrigin::ExplicitUserAction,
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Active);
    }

    #[test]
    fn contradiction_is_preserved() {
        let profile = ScientificTasteProfile::derive(
            &[
                event(
                    1,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                ),
                event(
                    2,
                    TasteEventKind::Contradicted,
                    TasteOrigin::ExplicitUserAction,
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);
    }
}

#[cfg(test)]
mod attribution_and_arbitration_tests {
    use super::*;

    fn src(kind: TasteSourceKind, id: &str) -> TasteSource {
        TasteSource::try_new(kind, id).expect("valid source")
    }

    fn ev(
        event_id: u64,
        kind: TasteEventKind,
        origin: TasteOrigin,
        source: TasteSource,
    ) -> TasteEvent {
        TasteEvent {
            event_id,
            preference_id: 7,
            scope: TasteScope::Project,
            scope_key: "project:cogno-1".to_string(),
            kind,
            origin,
            confidence_bps: 5_000,
            evidence_ids: vec![event_id],
            source,
        }
    }

    #[test]
    fn source_ids_are_bounded_and_printable() {
        assert_eq!(
            TasteSource::try_new(TasteSourceKind::Human, ""),
            Err(TasteError::InvalidSourceId)
        );
        let long = "a".repeat(MAX_TASTE_SOURCE_ID_BYTES + 1);
        assert_eq!(
            TasteSource::try_new(TasteSourceKind::Human, &long),
            Err(TasteError::InvalidSourceId)
        );
        // Control characters and non-ASCII are rejected (no homoglyphs).
        for bad in ["bad\nid", "bad\u{00e9}id", "bad id"] {
            assert_eq!(
                TasteSource::try_new(TasteSourceKind::Human, bad),
                Err(TasteError::InvalidSourceId),
                "space/control/non-ascii rejected: {bad:?}"
            );
        }
        assert!(TasteSource::try_new(TasteSourceKind::Human, "host-1").is_ok());
    }

    #[test]
    fn unknown_source_may_carry_no_identity_claim() {
        assert_eq!(TasteSource::UNKNOWN.kind, TasteSourceKind::Unknown);
        assert_eq!(TasteSource::UNKNOWN.id, "");
        // Explicitly constructing an Unknown source without an id is allowed;
        // attributed kinds still require a non-empty bounded id.
        assert!(TasteSource::try_new(TasteSourceKind::Unknown, "").is_ok());
        assert_eq!(
            TasteSource::try_new(TasteSourceKind::AssistanceModel, ""),
            Err(TasteError::InvalidSourceId)
        );
    }

    #[test]
    fn events_without_valid_source_fail_closed() {
        let mut e = ev(
            1,
            TasteEventKind::Proposed,
            TasteOrigin::ModelInference,
            src(TasteSourceKind::AssistanceModel, "model-a"),
        );
        e.source.id = String::new();
        assert_eq!(
            ScientificTasteProfile::derive(&[e], TastePolicy::DEFAULT),
            Err(TasteError::InvalidSourceId)
        );
    }

    #[test]
    fn profile_tracks_contributing_sources_per_preference() {
        let model = src(TasteSourceKind::AssistanceModel, "model-a");
        let human = src(TasteSourceKind::Human, "host");
        let kernel = src(TasteSourceKind::DeterministicKernel, "scirust");
        let profile = ScientificTasteProfile::derive(
            &[
                ev(
                    1,
                    TasteEventKind::Proposed,
                    TasteOrigin::ModelInference,
                    model.clone(),
                ),
                ev(
                    2,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                    kernel.clone(),
                ),
                ev(
                    3,
                    TasteEventKind::Confirmed,
                    TasteOrigin::ExplicitUserAction,
                    human.clone(),
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        let entry = &profile.preferences[&7];
        assert!(entry.is_active());
        assert_eq!(entry.sources.len(), 3);
        assert_eq!(profile.active_preferences_for_source(&model), vec![7]);
        assert_eq!(profile.active_preferences_for_source(&human), vec![7]);
        // An unrelated model contributed nothing.
        assert!(profile
            .active_preferences_for_source(&src(TasteSourceKind::AssistanceModel, "model-b"))
            .is_empty());
    }

    #[test]
    fn arbitration_requires_human_authority_and_conflicted_target() {
        let events = [
            ev(
                1,
                TasteEventKind::Accepted,
                TasteOrigin::DeterministicEvaluation,
                src(TasteSourceKind::DeterministicKernel, "scirust"),
            ),
            ev(
                2,
                TasteEventKind::Contradicted,
                TasteOrigin::ExplicitUserAction,
                src(TasteSourceKind::Human, "host"),
            ),
        ];
        // A model authority is refused outright.
        let profile = ScientificTasteProfile::derive(&events, TastePolicy::DEFAULT).unwrap();
        // (authority enum has a single variant; the check is exercised by the
        // API contract below — TargetNotConflicted proves the state gate.)
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);

        // Arbitrating an Active entry fails closed.
        let mut active = ScientificTasteProfile::derive(
            &[
                ev(
                    1,
                    TasteEventKind::Accepted,
                    TasteOrigin::DeterministicEvaluation,
                    src(TasteSourceKind::DeterministicKernel, "scirust"),
                ),
                ev(
                    2,
                    TasteEventKind::Confirmed,
                    TasteOrigin::ExplicitUserAction,
                    src(TasteSourceKind::Human, "host"),
                ),
            ],
            TastePolicy::DEFAULT,
        )
        .unwrap();
        assert!(active.preferences[&7].is_active());
        assert_eq!(
            active.arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Reject,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                }
            ),
            Err(TasteError::TargetNotConflicted(7))
        );

        // Unknown preference fails closed.
        let mut empty = ScientificTasteProfile::default();
        assert_eq!(
            empty.arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 99,
                    action: TasteArbitrationAction::Reject,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                }
            ),
            Err(TasteError::UnknownPreference(99))
        );
    }

    #[test]
    fn arbitration_lifts_conflict_but_never_the_numeric_gates() {
        let low_evidence = [
            ev(
                1,
                TasteEventKind::Accepted,
                TasteOrigin::DeterministicEvaluation,
                src(TasteSourceKind::DeterministicKernel, "scirust"),
            ),
            ev(
                2,
                TasteEventKind::Contradicted,
                TasteOrigin::ExplicitUserAction,
                src(TasteSourceKind::Human, "host"),
            ),
        ];
        // Only ONE non-model confirmation and confidence below threshold:
        // Activate must be refused even with human authority (S4).
        let mut profile =
            ScientificTasteProfile::derive(&low_evidence, TastePolicy::DEFAULT).unwrap();
        assert_eq!(
            profile.arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Activate,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                }
            ),
            Err(TasteError::ActivationGateNotMet(7))
        );
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);

        // Quarantine resolution succeeds and is recorded.
        profile
            .arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Quarantine,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                },
            )
            .expect("quarantine arbitration");
        assert_eq!(profile.preferences[&7].state, TasteState::Quarantined);
        assert_eq!(profile.preferences[&7].conflict_resolutions, 1);
    }

    #[test]
    fn full_evidence_conflict_can_be_resolved_to_active() {
        // Gates met on raw numbers (1 acceptance + 4 confirmations ⇒
        // 9000 bps and 5 non-model confirmations), then contradicted:
        // only the conflict block prevents activation. Human arbitration
        // lifts exactly that block.
        let mut events = vec![ev(
            1,
            TasteEventKind::Accepted,
            TasteOrigin::DeterministicEvaluation,
            src(TasteSourceKind::DeterministicKernel, "scirust"),
        )];
        for id in 2..=5u64 {
            events.push(ev(
                id,
                TasteEventKind::Confirmed,
                TasteOrigin::ExplicitUserAction,
                src(TasteSourceKind::Human, "host"),
            ));
        }
        events.push(ev(
            6,
            TasteEventKind::Contradicted,
            TasteOrigin::ImportedProfile,
            src(TasteSourceKind::ExternalAgent, "agent-1"),
        ));
        let policy = TastePolicy::DEFAULT;
        let mut profile = ScientificTasteProfile::derive(&events, policy).unwrap();
        assert_eq!(profile.preferences[&7].state, TasteState::Conflicted);
        assert_eq!(
            profile.preferences[&7].confidence_bps, 7_000,
            "gates numerically met under the conflict (exactly at threshold)"
        );
        profile
            .arbitrate(
                &policy,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Activate,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                },
            )
            .expect("gates met");
        assert!(profile.preferences[&7].is_active());
        assert_eq!(profile.preferences[&7].conflict_resolutions, 1);

        // Replay through derive_with_arbitrations gives the identical result.
        let replayed = ScientificTasteProfile::derive_with_arbitrations(
            &events,
            policy,
            &[TasteArbitration {
                preference_id: 7,
                action: TasteArbitrationAction::Activate,
                authority: TasteArbitrationAuthority::ExplicitHumanHost,
            }],
        )
        .expect("deterministic replay");
        assert_eq!(profile, replayed);
    }

    #[test]
    fn scope_keys_are_bounded_and_printable() {
        for key in ["", " ", "a\tb", &"x".repeat(MAX_SCOPE_KEY_BYTES + 1)] {
            assert_eq!(
                TasteScope::Project.validate_key(key),
                Err(TasteError::InvalidScopeKey),
                "key `{key}` must be refused"
            );
        }
        TasteScope::User
            .validate_key("user:alice")
            .expect("valid key");
    }

    #[test]
    fn events_reject_invalid_scope_keys() {
        let mut item = ev(
            1,
            TasteEventKind::Proposed,
            TasteOrigin::ModelInference,
            src(TasteSourceKind::AssistanceModel, "model-a"),
        );
        item.scope_key = String::new();
        assert_eq!(item.validate(), Err(TasteError::InvalidScopeKey));
        assert_eq!(
            ScientificTasteProfile::derive(std::slice::from_ref(&item), TastePolicy::DEFAULT),
            Err(TasteError::InvalidScopeKey)
        );
    }

    #[test]
    fn cross_scope_reuse_conflicts_instead_of_merging() {
        let base = [
            ev(
                1,
                TasteEventKind::Accepted,
                TasteOrigin::DeterministicEvaluation,
                src(TasteSourceKind::DeterministicKernel, "scirust"),
            ),
            ev(
                3,
                TasteEventKind::Confirmed,
                TasteOrigin::ExplicitUserAction,
                src(TasteSourceKind::Human, "host"),
            ),
        ];
        let mut foreign = ev(
            2,
            TasteEventKind::Confirmed,
            TasteOrigin::ModelInference,
            src(TasteSourceKind::ExternalAgent, "foreign-domain-agent"),
        );
        foreign.scope = TasteScope::Domain;
        foreign.scope_key = "domain:rust-macros".to_string();
        let profile = ScientificTasteProfile::derive(
            &[base[0].clone(), foreign, base[1].clone()],
            TastePolicy::DEFAULT,
        )
        .expect("valid profile");
        let entry = &profile.preferences[&7];
        assert_eq!(entry.state, TasteState::Conflicted);
        assert_eq!(entry.scope_key, "project:cogno-1", "first binding wins");
        // The foreign event contributed nothing: its source never joins.
        let foreign_source = src(TasteSourceKind::ExternalAgent, "foreign-domain-agent");
        assert!(!entry.sources.contains(&foreign_source));
        assert_eq!(entry.non_model_confirmations, 2);
        // Human arbitration can lift the block; gates were met on their own.
        let mut arbitrated = profile;
        arbitrated
            .arbitrate(
                &TastePolicy::DEFAULT,
                TasteArbitration {
                    preference_id: 7,
                    action: TasteArbitrationAction::Activate,
                    authority: TasteArbitrationAuthority::ExplicitHumanHost,
                },
            )
            .expect("gates met");
        assert!(arbitrated.preferences[&7].is_active());
    }

    #[test]
    fn consent_defaults_keep_data_local() {
        let settings = TasteSettings::default();
        assert!(settings.can_learn());
        assert!(!settings.can_export(), "export needs explicit consent");
        assert!(!settings.can_import(), "import needs explicit consent");
        let off = TasteSettings {
            learning_enabled: false,
            export_allowed: true,
            import_allowed: true,
        };
        assert!(!off.can_learn());
        assert!(!off.can_export(), "export implies learning enabled");
        assert!(off.can_import());
    }
}
