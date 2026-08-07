# Autonomous Scientific Taste Implementation Map

- `scientific_exchange.rs`: trust-boundary classification for SciAgent/SciRust observations.
- `taste_orchestrator.rs`: deterministic promotion-state derivation.
- `taste_benchmark.rs`: integer-only quality and regret metrics.
- `taste_longitudinal.rs`: bounded multi-generation drift history.
- `taste_autonomy.rs`: high-level coordination of exchange, orchestration, benchmark, and history.
- `taste_campaign.rs`: bounded transactional coordination of multiple autonomous generations with campaign-level rollback.
- `tests/autonomous_taste_cycle.rs`: multi-generation end-to-end autonomous pass.
- `tests/scientific_taste_campaign.rs`: public campaign determinism and model non-authority coverage.
- `tests/scientific_taste_system_adversarial.rs`: system-level fail-closed coverage.
