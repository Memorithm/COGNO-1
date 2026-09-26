# Bounded Rust evaluation pipeline

`scripts/admit_rust_corpus.py INPUT.jsonl NEW_DIRECTORY --license MIT` accepts
declared, already reviewed classification records. Each JSON object requires
source, its SHA-256, binary integer label, split (train/validation/test), project,
immutable revision (40/64 hex), license, classification (Public/Internal), compiler
and provenance. Approval is an explicit caller-supplied license allowlist, not
automated legal review. Metadata is checked for consistency, not independently
authenticated. Compiler labels must have been verified upstream; this tool does
not compile or execute submitted code and does not prove label correctness.

Admission rejects duplicate JSON keys, exact source duplicates, project identities
shared across splits and partitions lacking either label. Source bytes are never
normalized. Near duplicates, forks, aliases and shared upstream code still require
curation: project identity is supplied by the caller. Limits: 4 MiB JSONL,
4,096 rows, 16 KiB/source, 1 MiB total source. This is a bounded research pipeline,
not massive corpus ingestion. Failures never overwrite an existing directory.

Output includes retained provenance, an integrity manifest and `corpus.crust`:
ASCII `CRUST001` header followed by tab-separated split, project, label, source
SHA-256 and lowercase hex-encoded UTF-8 source. Hex preserves tabs and newlines
without introducing another runtime dependency. The manifest's corpus digest
must travel through a trusted inventory; self-reported hashes are not signatures.

The existing 24-snippet pilot is reused diagnostic data. Grouping its synthetic
families is not evidence of generalization to independent real-world projects.
No training or quality gain is established by passing corpus admission.

`cogno_model::rust_corpus::RustCorpus::read` accepts a bounded stream and an
independent expected corpus digest. It rechecks wire grammar, per-source hashes,
UTF-8, sizes, duplicates, project separation and both labels in every split.
It does not read or authenticate the provenance sidecar. No new Cargo dependency
is needed. This reader is a consistency boundary, not runtime activation authority.

## Coverage before training

`rust_corpus_coverage CORPUS EXPECTED_SHA256` reads the admitted corpus, learns
BPE merges from train only (target vocabulary 384, context 128), and reports
source bytes, accepted token totals and rejected example counts for each split.
Byte and BPE totals include framing; totals exclude rejected inputs, so compare
compression only when coverage is equal. No truncation or model training occurs.

`python3 scripts/prepare_rust_diagnostic.py NEW_DIRECTORY` adapts the frozen
24-snippet diagnostic with its original compiler declarations and source revision.
Synthetic family IDs are explicitly prefixed `synthetic/`; they are not independent
real-world projects. The source file hash is pinned and changed data is rejected.
This is an integration fixture, not an enlarged evaluation dataset.

## Fresh-process checkpoint evaluation

`bpe_corpus_eval CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA train|validation|test`
loads identified inference state and a validated corpus, then emits classification
CSV only for the selected split. It contains no optimizer, source compiler or
code executor. Binary class count, corpus/checkpoint identities and capacity
errors are checked; all selected predictions are computed successfully before
any CSV is emitted. There is no skipping or silent truncation.

For the reused integration fixture:

```sh
cargo +1.97.1 build --release --locked -p cogno-model --examples
python3 scripts/verify_rust_pipeline.py --examples-dir target/release/examples --checkpoints /tmp/new-bpe-checkpoints --output /tmp/new-rust-evaluation
```

The verifier requires the checkpoint inventory frozen in Git, prepares the pinned
diagnostic corpus and checks coverage against committed measurements. It launches
nine separate evaluators (three seeds × three partitions), checks all 72 outputs
against frozen references and tests refusal of wrong model/corpus hashes. New
output directories are mandatory. Results are also checked in isolated offline CI.
This validates persistence and evaluation plumbing, not independent Rust quality.
