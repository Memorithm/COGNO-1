# Bounded Rust BPE: experimental foundation

The `cogno-model::bpe_tokenizer` module implements an independent bounded version
of the sequential rank-priority BPE algorithm, informed by the existing SciRust
canonical trainer contract. No SciRust source is copied or dynamically linked.
This is not binary-compatible with SciAgent's JSON artifact: COGNO keeps raw
bytes 0..255, BOS=256, EOS=257, SEP=258 and merge IDs from 259.

Sources inspected: Memorithm/scirust commit
`8479ab7a7ed20db2b8fae8a779a31c4c94f4bb0e`,
`scirust-sciagent/src/canonical_bpe_train.rs` and
`scirust-sciagent/src/elastic_text_tokenizer.rs`.

## Contract

- No new dependencies. SHA-256 uses the already present model dependency.
- Arbitrary bytes round-trip exactly: no Unicode normalization, whitespace
  stripping or lossy UTF-8 conversion. Invalid tokens/framing are errors.
- Pair members are encoded independently; merges cannot consume SEP.
- Vocabulary <=512, context <=512 tokens, input/reconstructed bytes <=16,384,
  expansion per token <=1,024 bytes. Capacity errors never truncate input.
- The reference trainer admits <=4,096 records and <=1 MiB of total raw input.
  It is an in-memory correctness baseline, not a massive-training engine.
- Training learns one global pair per iteration, minimum frequency two, ties
  resolved by pair ID. Record boundaries remain separate. The caller supplies
  only its training partition; this module cannot infer source-project splits.
- CBPE0001 stores context and ordered merge pairs in little endian. Forward or
  special-token references, duplicate byte expansions, oversized, truncated,
  unknown-version and trailing-byte artifacts are rejected. The fingerprint
  covers exact artifact bytes; it is not a signature.

## Compatibility and activation

V1–V4 artifact loaders, byte-tokenizer hashes, inference authority and activation
gates remain unchanged. CBPE0001 is a **tokenizer artifact**, not a complete model
checkpoint. `bpe_rust_probe` exercises the real shared cognitive numerical heads
using BPE token IDs and a matching embedding vocabulary. It does not add BPE to
the production `SequenceCognitiveModel` or runtime activation surfaces.

The next integration must bind this fingerprint and vocabulary to the weights
in a distinct versioned model format, including pair framing and all five heads.
Silently changing token IDs under an existing V4 checkpoint is prohibited.

## Diagnostic experiment

```sh
cargo +1.97.1 run --release --locked -p cogno-model --example bpe_rust_probe
```

The frozen 24-snippet Rust fixture is diagnostic data already examined in the
earlier pilot, **not a new untouched holdout**. Train BPE on its 16 training
snippets only, target vocabulary 384, context 128. Byte and BPE arms use seeds
1/7/42, encoder widths 8/16, 24 epochs and LR 0.003. Only classification loss
is enabled; other head slots are placeholders and unqualified. The larger BPE
embedding table changes parameter count and initialization offsets; equal
epochs/widths are NOT equal parameter count or compute. No speed or expert-Rust
quality claim follows from shorter sequences.

Retain all seed/split predictions, actual vocabulary/parameter counts and
tokenizer bytes. Full-language corpus evaluation, fast encoding kernels,
scalable tokenizer training and an untouched project-separated quality panel
remain required before promotion.
