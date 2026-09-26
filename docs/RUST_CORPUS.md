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
