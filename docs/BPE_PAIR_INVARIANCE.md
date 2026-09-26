# bpe_pair_invariance

Run `cargo run --release -p cogno-model --example bpe_pair_invariance -- CHECKPOINT SHA CORPUS SHA SPLIT PAIRS_TSV PAIRS_SHA`.

The hash-bound LF-terminated TSV contains `original_source_sha256<TAB>variant_source_sha256`, with no header, at most 2,048 pairs. Both sources must exist in the selected admitted split and have equal compiler labels. Reports probability shifts and class flips for explicitly curated transformations, including whitespace/comment variants if supplied. **Equal compiler labels do not establish semantic equivalence.** This tool neither strips comments nor assumes arbitrary text editing preserves Rust behavior. Reversed duplicates, self-pairs, missing sources and cross-split pairs are rejected.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
