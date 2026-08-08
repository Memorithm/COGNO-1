# Neural model generation implementation map

This map records the code paths introduced by the controlled model-persistence phase.

- `crates/cogno-runtime/src/model_generation.rs` — immutable, SHA-256-linked model generation manifests and replay chain.
- `crates/cogno-runtime/src/model_persistence.rs` — host-attested transactional commit, strict `MODEL_CURRENT`, canonical `ModelManifest` persistence, bounded artifact reads, and restart replay.
- `crates/cogno-runtime/tests/model_generation_persistence.rs` — public deterministic replay and missing-predecessor adversarial coverage.
- `docs/MODEL_GENERATION_PERSISTENCE.md` — authority, disk layout, commit ordering, replay checks, and deliberately deferred concurrency/crash-recovery hardening.

The model-generation namespace is independent from Scientific Taste generations. No API in this phase hot-swaps a live model or grants model-origin authority.
