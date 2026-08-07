# Scientific Taste Campaign Phase

This phase extends the merged autonomous scientific-taste substrate with a bounded multi-generation campaign runner.

## Scope

- evaluate up to 64 autonomous generations in one deterministic campaign;
- require strictly increasing generation identifiers;
- preserve SciAgent model observations as non-authoritative quarantine input;
- accumulate longitudinal status only through existing autonomous evaluation rules;
- commit longitudinal tracker changes only if every generation succeeds;
- return a deterministic campaign report with per-generation autonomous reports and final longitudinal statuses.

## Fail-closed transaction boundary

Campaign evaluation runs against a cloned longitudinal tracker. Any invalid generation, exchange record, benchmark input, promotion input, or longitudinal update aborts the campaign and leaves the caller's tracker unchanged.

## Authority boundary

The campaign runner is report-only. It does not write, replace, install, or hot-swap a verified runtime taste profile. Runtime profile installation remains governed by the existing transactional generation persistence, restart-time verification, and sealed installation boundary.

## Acceptance criteria

- `cargo fmt --all -- --check`
- `RUSTFLAGS="-D warnings" cargo test --locked --workspace --all-targets`
- `cargo clippy --locked --workspace --all-targets -- -D warnings`
- public API integration coverage demonstrates deterministic replay and model non-authority
- unit coverage demonstrates campaign-level rollback on failure and rejection of non-monotonic generations
