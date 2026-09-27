# External Rust source acquisition v2

This pipeline acquires real source files from three pinned upstream repositories,
retains their MIT license texts, removes diagnostic comments within a conservative
lexical subset, and compiles **both original and derived sources**. It produces
native CRUST001 and source provenance without any network requirement during
qualification. Acquisition tooling uses the Python standard library; the Rust
compiler and existing native corpus consumer remain the execution boundary.

## Fixed observed panel

| Upstream repository | Immutable revision | Split | Accepted sources |
|---|---|---|---:|
| rust-lang/rust | a22b02eaecd6ac937d752139c79d0159c725932d | train | 7 |
| rust-lang/rustlings | a650509c789da1656f813392b16aa1fa043b7f3e | validation | 7 |
| dtolnay/trybuild | 4b511198467970a3ec448df3e3837f53e0677940 | test | 4 |

Upstream URLs, source and license paths, byte lengths, SHA-256 values, intended
labels and observed error codes are in `experiments/external-corpus-v2/inventory.json`.
Bytes are retained under SHA filenames in `cache/`; original licenses are included.
The revision is an upstream commit, not an invented project or synthetic alias.
All files and revisions from a repository share the same project identity.
Related forks must be declared in the same lineage group in `splits.json`.
These group declarations do not prove semantic or historical independence.

This is an **18-source acquisition/qualification smoke panel**, not a benchmark
of model expertise. All cases were inspected and compiled during curation. No
checkpoint was trained or selected. No accuracy claim or blinded holdout claim
is made. Rustlings exercises and solutions remain together in validation.
Trybuild's `run-fail.rs` has compile label 1: its runtime panic is never executed,
and compile success is not runtime correctness. Classes are not balanced within
each split. Large-project compilation and dependencies are outside this scope.

Ten rejected candidates are recorded with immutable paths and reasons: nine
trybuild files duplicate the same `fn main() {}` and one Rustlings solution falls
outside the conservative namespace audit. They are not counted as accepted data.
The accepted panel has no exact or whitespace-normalized match to **328 known
records** (24 pilot, 288 curriculum, eight previous upstream originals and eight
derived versions). `known-registry.json` pins those repository artifacts.
Whitespace normalization also affects strings: it is a conservative exclusion,
not semantic deduplication. Near-duplicates, shared idioms, upstream lineage and
other unlisted corpora are not certified independent.

## Reproduce offline

From the repository root:

```sh
RUSTC_PINNED="$(rustup which --toolchain 1.97.1 rustc)"
python3 scripts/external_corpus_qualify.py /tmp/cogno-external-v2-new \
  --rustc "$RUSTC_PINNED"
PYTHONPATH=scripts python3 -m unittest discover -s scripts \
  -p 'test_external_corpus_*.py' -v
```

The destination must not exist. Qualification reproduces all **36 original and
derived metadata-only compilations** and compares compiler evidence, corpus and
provenance bytes with the frozen artifacts. It validates the known-corpus hashes
and copies the exact known inventory used into its result for durable inspection.
Admitted corpus SHA-256:
`09263fd03150916481db10a7c636ee9ecdd5e13b9c10637efb716433d0e1113d`.

## Acquire another reviewed inventory

```sh
python3 scripts/external_corpus_fetch.py inventory.json INVENTORY_SHA cache
python3 scripts/external_corpus_fetch.py inventory.json INVENTORY_SHA cache --offline
python3 scripts/external_corpus_pipeline.py \
  inventory.json INVENTORY_SHA splits.json SPLITS_SHA cache NEW_OUTPUT \
  --rustc "$RUSTC_PINNED" --known known.jsonl KNOWN_SHA
```

The download phase permits only canonical immutable GitHub raw URLs and refuses
redirects. Expected digests must come from a reviewed independent inventory;
checksums alone do not authenticate authorship or establish license ownership.
The pipeline reads only the verified local cache. Additional `--known PATH SHA`
arguments extend contamination checking; at least one known inventory is required.

Bounds: at most 256 sources, 16 KiB per source, 64 KiB per license, 1 MiB inventory;
known sources at most 8192 records/4 MiB decoded source bytes. Corpus admission adds
its own unchanged limits. IDs and paths are constrained. Dataset comments with
harness dependencies/configuration, nonlocal modules, includes/environment macros,
unstable features, raw strings, block comments and unsupported quoted literals
are refused for manual review. This conservative audit is **not a Rust parser**.
Raw source hashes and comment-removal line numbers are retained; raw and derived
compiler outcomes must agree with exact reviewed diagnostic categories.

Compilation requires the release and full commit of rustc 1.97.1, edition 2021,
`--emit=metadata --error-format=json`, no Cargo/build scripts and no source execution.
On POSIX it applies CPU/address-space/file-size limits and a 20-second timeout.
Crashes, timeouts, missing compiler, missing dependencies, uncoded errors and
unexpected diagnostics stop admission. These limits are not a security sandbox
for arbitrary untrusted Rust: compile only reviewed fixtures in an isolated worker
when sourcing untrusted code. No network or Thor reservation is implied.

Output includes original and derived sources, licenses, inventory and split
snapshots, compiler observations, known-source snapshots, admitted corpus and
provenance. `COMPLETE.json` is written last. A disk failure can leave incomplete
output; this is not an atomic filesystem transaction. Retain all provenance and
license artifacts when passing the corpus to COGNO-1, SciRust or RemoteOps.
