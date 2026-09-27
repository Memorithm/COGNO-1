# Exact BPE acceleration and source locations

`CBPE0001` still serializes the same max-context field and ordered merge pairs.
Byte IDs, framing, tie breaks, left-to-right overlap handling, pair boundaries,
and refusal rules have not changed. These changes improve tokenizer operations;
they do not demonstrate Rust reasoning or code generation quality.

## Training

The trainer appends only the new validated vocabulary piece. It counts corpus
adjacencies once, then subtracts and adds the edges affected by each disjoint
merge. Pair selection remains maximum frequency with the smallest pair of IDs
breaking ties. No pair crosses a source boundary. The 1 MiB, 4096-record,
16 KiB-per-record and 512-vocabulary bounds remain in force.

For admitted corpora, prefer `BpeTokenizer::train_from_view(&view, vocab, context)`
where `view = corpus.training_view()?`. A standalone artifact can be produced by:

```sh
cargo run --release -p cogno-model --example bpe_train_corpus -- \
  CORPUS CORPUS_SHA256 384 512 NEW_ARTIFACT
```

The destination must not exist. The CLI reports admitted training count and
artifact SHA256. Changing held-out sources cannot affect training through the
view. This does not certify project provenance or compiler labels by itself.

## Inference and long source processing

- `encode` retains the rank-scan path. `encode_with_workspace` offers an exact
  rank-heap alternative with caller-owned buffers. Heap candidates are checked
  against live adjacency before merging. Left positions resolve equal-rank
  overlaps. At most `n - 1 + 2(n - 1)` candidates are inserted for `n` input bytes.
- `encode_batch` reuses this workspace, retains source order and reports every
  individual refusal. Aggregate bounds are checked before encoding any record.
- `encode_with_offsets` returns half-open byte offsets, including zero-width
  BOS/EOS ranges. Offsets are not character columns or syntax-node boundaries.
- `encode_chunks` divides the full final token stream into bounded contexts with
  exact source byte ranges and explicit BOS/EOS framing. `max_chunks` bounds
  output allocation and excess refuses the whole source. It neither silently
  truncates nor changes single-context `encode` admission. Recombining decoded
  chunks reconstructs every byte, including invalid UTF-8. A chunk need not be a
  valid Rust fragment. Preserve original project/source/split identity when using
  chunks for training: correlated chunks are not new independent observations.

## Measured scope

Raw results: [`timings.txt`](../experiments/tokenizer-engine-v2/timings.txt).
The release benchmark uses three actual repository Rust files, 28,041 bytes,
from source commit `c55761a0ec257672b0d029cc3eea37c4ea761024`:
`crates/cogno-model/src/tokenizer.rs`, `crates/cogno-model/src/rust_corpus.rs`, and
`crates/cogno-core/src/lib.rs`. Input SHA256 values appear in the raw output.
The benchmark fits all three sources solely to measure tokenizer work. They are
not a held-out evaluation corpus, and no task accuracy is estimated.

Environment: x86_64 Linux 6.18.44, glibc 2.39, 9 CPU affinity slots, Rust 1.97.1
(`8bab26f4f68e0e26f0bb7960be334d5b520ea452`), Cargo release defaults. Five samples
alternate implementation order; encoding warms each path then times 100 passes.
Other jobs shared the machine. These are wall-clock observations, not a stable
latency guarantee or a Thor measurement.

| Operation | Reference median | New path median | Observed ratio |
|---|---:|---:|---:|
| Train 384-token vocabulary | 258.992 ms, full recount | 18.294 ms, adjacency deltas | 14.16× faster |
| Encode three 128-byte prefixes, 100 passes | 4.427 ms, rank scan | 2.867 ms, reusable heap | 1.54× faster |
| Encode three whole files, 100 passes | 352.908 ms, rank scan | 530.783 ms, reusable heap | heap 1.50× slower |

The reference trainer is an independent full-recount implementation that rebuilds
and validates the vocabulary at each rank, not a separately compiled historical
binary. Both produce byte-identical artifacts. Whole files **all exceed the
512-token context**: those measurements include refusal after full tokenization,
not successful long-context model inference. All three prefixes are admitted.
Because heap performance depends on workload, it remains opt-in; the default
encoder has not been switched. No end-to-end neural throughput claim follows.

The benchmark compares every input's scan/heap outcome, validates lossless
chunk reconstruction, and verifies identical artifacts on every repeated fit:

```sh
cargo run --release -p cogno-model --example bpe_tokenizer_bench -- \
  384 5 crates/cogno-model/src/tokenizer.rs \
  crates/cogno-model/src/rust_corpus.rs crates/cogno-core/src/lib.rs
```

To reproduce the recorded input identities after files evolve, export each path
from the pinned source commit using `git show COMMIT:PATH` and pass those three
exported files. Repeats are restricted to 1–20; all supplied files are training
input, bounded before fitting. Tokenizer fingerprint for the recorded inputs:
`092d8817b519a57499b679932b0d22ad6dde4c710a9ca5b237dfc5d30f744bed`.

A separate CLI smoke run on the frozen domain corpus
`fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f`
used exactly 192 training rows and reproduced its published tokenizer fingerprint
`e9d665b509f568308584d853850dabfd19bf27c9b7e1e1166dce785317add7a6`.
No validation/test rows were fitted, and no model was retrained or promoted by
this tokenizer benchmark.
