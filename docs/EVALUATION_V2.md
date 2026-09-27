# Bound multi-seed evaluation

These native Rust examples consume an existing compiler-acceptance experiment. They do not generate Rust code, authenticate an upstream project, or promote a model. Test predictions are already observed; exploratory reports do not create a fresh holdout.

Every command begins with six positional arguments: `PREDICTIONS.csv SHA256 CORPUS.crust SHA256 GROUPS.tsv SHA256`. Digests must come from a separately reviewed inventory. Group files have exactly `project<TAB>group<TAB>split` as header, one line per corpus project, exact split agreement, and no group crossing splits. Domain names supplied as groups remain domains, not independent upstream projects.

`cargo run --release -p cogno-model --example v2_evidence_audit -- ...` validates every prediction against the hash-pinned corpus. One arm, 1–64 seeds, every admitted source exactly once per seed, binary probabilities, consistent labels/argmax, and bounded LF text are required. The only argmax tolerance is 1e-7 around a rounded f32 tie. Reports include input SHA bindings. Rows repeated across seeds are never counted as independent source samples.
