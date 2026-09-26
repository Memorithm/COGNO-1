# Native cross-corpus identity audit

```sh
cargo +1.97.1 run -p cogno-model --example rust_corpus_overlap -- full /tmp/cogno-rust-admitted/corpus.crust 766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee test-only /tmp/cogno-external-admitted/corpus.crust c2285c3e478509399da4fe130f8d8c060d5dfba283d4d4eafa510857e540086d
cargo +1.97.1 test -p cogno-model --example rust_corpus_overlap
```

Each side independently specifies `full` or `test-only` admission and an expected
SHA-256. The bounded CRUST001 readers validate internal source hashes, class
presence and within-corpus project separation before cross-corpus comparisons.

Sorted CSV reports identical source hashes and shared project identifiers,
including splits, record counts and conflicting labels for identical sources.
Project rows use `label_conflict=false`: projects legitimately contain both
labels, and only identical-source rows test label agreement. Summary counts go
to stderr. Any overlap produces nonzero exit status after the full CSV; a clean
comparison produces only a CSV header and exits zero. Invalid input also fails
nonzero, with a validation error rather than an overlap report.

Measured 2026-09-26: the existing 24-record diagnostic versus the existing
8-record external panel has **zero exact source or project-identifier overlaps**.
Comparing the diagnostic to itself intentionally fails, with 24 source and
12 project matches. This validates detection, not semantic independence:
renamed projects, copied fragments, transformed examples and common upstream
ancestry require separate provenance/near-duplicate review. Both panels have
already been observed and do not become fresh holdouts through this check.

SciRust pipelines can call this gate for CRUST001 datasets before a training
campaign without a service dependency or a model-training step.
