//! Lifetimes and retention (COGNO-1 V2 §19).

/// Lifetime class of a piece of memory/retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    EphemeralRequest,
    Session,
    Project,
    UserGlobal,
    SecurityAudit,
}

/// Retention policy. The MVP-safe defaults keep identifiers, categories,
/// fingerprints, statistics, extracted rules, minimal provenance and validator
/// results — not raw prompts, model outputs, or diffs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionPolicy {
    pub max_session_events: usize,
    pub max_project_events: usize,
    pub max_global_rules: usize,
    pub retain_raw_prompts: bool,
    pub retain_model_outputs: bool,
    pub retain_diffs: bool,
}

impl RetentionPolicy {
    /// MVP-safe defaults recommended in §19. Retains no raw content.
    pub const MVP_SAFE: Self = Self {
        max_session_events: 1024,
        max_project_events: 4096,
        max_global_rules: 512,
        retain_raw_prompts: false,
        retain_model_outputs: false,
        retain_diffs: false,
    };

    #[must_use]
    pub const fn retains_any_raw(self) -> bool {
        self.retain_raw_prompts || self.retain_model_outputs || self.retain_diffs
    }
}
