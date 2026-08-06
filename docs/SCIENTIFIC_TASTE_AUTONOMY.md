# Autonomous Scientific Taste Phase

This phase extends the deterministic scientific-taste substrate beyond verified profile persistence into a bounded autonomous evaluation loop.

## Scope

The implementation adds five layers:

1. **Autonomous orchestration** — coordinates bounded cross-component observations, promotion evidence, deterministic benchmark evaluation, and longitudinal tracking without mutating a live runtime profile.
2. **Deterministic benchmark** — measures oracle agreement, abstention, aggregate regret, improvements, regressions, and unchanged cases using integer-only metrics.
3. **Longitudinal tracking** — records bounded per-preference history across generations and classifies stable, strengthening, weakening, contradicted, and stale states.
4. **SCIAGENT / SciRust exchange boundary** — model observations remain quarantined; only deterministic SciRust evaluation or explicit user action can become persistent validation evidence.
5. **System adversarial coverage** — exercises corruption, digest mismatch, forked generations, duplicate generation commits, unknown rollback, conflicting evidence, and model self-promotion attempts.

## Authority boundary

The autonomous coordinator never writes or replaces a live verified profile. Generation persistence remains transactional and restart-time profile installation remains sealed. The model cannot validate its own observations, grant itself authority, or bypass deterministic evaluation.

## Determinism

All benchmark metrics are integer-only. Candidate and validation processing is bounded. Stable identifiers are sorted before derivation where input ordering could otherwise affect output. Longitudinal observations are strictly monotonic per preference.
