# Explicit project-group partitioning

```sh
cargo run -p cogno-model --example rust_project_split -- \
  CORPUS CORPUS_SHA GROUPS_TSV GROUPS_SHA NEW_DIRECTORY
```

This Rust-only utility remaps **an already admitted full CRUST001 corpus** using a curator-provided assignment. It is not an importer for raw unpartitioned sources. Both input SHA-256 values must come from an independently trusted inventory. No new crate or runtime service is required.

The mapping is UTF-8 TSV, with this exact header and a final LF:

```text
project	group	split
upstream/project-a	upstream-family-a	train
upstream/project-b	upstream-family-a	train
upstream/project-c	upstream-family-c	validation
upstream/project-d	upstream-family-d	test
```

The displayed column separators are literal tabs. Every corpus project must have exactly one row, with no unknown projects. A declared group may contain several projects, but all must share one target split. Project/group IDs must be 1–128 bytes and contain only ASCII letters, digits, `_`, `.`, `/`, or `-`. Splits are exactly `train`, `validation`, or `test`. CR, blank rows, extra columns and duplicate assignments are rejected. Limits are 4096 project rows and 1 MiB of inventory bytes; corpus bounds remain those of `RustCorpus`.

Only the split field changes. Source bytes, source SHA-256, project identity, binary label, row order and number of records are preserved. The result passes full corpus admission again, including exact-source uniqueness and both labels in each partition. Mapping-row order does not affect the corpus or report. There is no random assignment, class-balancing heuristic, source renaming or truncation.

The fresh output directory contains:

- `corpus.crust`: the re-admitted corpus.
- `groups.tsv`: the exact input assignment, preserving its independent hash.
- `report.tsv`: project-sorted original/target partitions and per-label row counts.
- `COMPLETE`: written last, binding input corpus, inventory, output corpus and report digests.

An existing destination is refused. Each file uses exclusive creation and `sync_all`. A write failure leaves an incomplete directory for diagnosis; this is not an atomic directory transaction. Consumers must check the complete manifest and artifact hashes. Use a trusted destination parent directory.

## Interpretation

Groups are assertions supplied by the curator, not proof that projects are independent. Related forks, generated variants and shared upstream sources should receive a common group. Synthetic family aliases remain synthetic: regrouping does not turn them into independent real projects. Source and license provenance must remain linked through the preserved source identities in the original inventory.

**Moving previously observed validation/test records cannot create a new unseen holdout.** This tool explicitly allows partition redesign and reports it; use the result for a declared new experiment, retaining the old inventory and protocol. Freeze group decisions before fitting or evaluating a new model. Compiler labels are retained without recompilation, so this utility makes no new compiler or model-quality claim.

The output uses the existing native corpus contract and can be consumed by COGNO-1 and exported for SciRust or RemoteOps workflows without inventing another dataset format.
