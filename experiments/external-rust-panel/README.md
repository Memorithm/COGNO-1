# External Rust compiler panel — 2026-09-26

Eight small standalone borrow-checker tests selected from **rust-lang/rust**,
revision `a22b02eaecd6ac937d752139c79d0159c725932d`, before evaluating the frozen
COGNO checkpoints. This is a small, manually selected single-project panel,
not a representative or multi-project benchmark. No example is used for training.

Source: https://github.com/rust-lang/rust/tree/a22b02eaecd6ac937d752139c79d0159c725932d/tests/ui/borrowck

`upstream/` retains original files. `source/` removes line comments (including
expected-error and run-pass annotations), trailing whitespace and blank lines.
The eight files were read before this transformation; none has a string literal
containing `//`. This is not a general Rust comment-stripping algorithm. Executable
code is otherwise retained, including indentation and string literals. Both
versions must be compiled to establish that the transformation preserves labels.
MIT permission notice is retained in `LICENSE-MIT`; copyright remains with
The Rust Project Contributors. `panel.json` pins both byte identities.

Expected labels: three compile-success and five compile-failure examples. These
are hypotheses from upstream annotations until verified using the pinned local
compiler. Compile outcome is not runtime correctness. Code is never executed.
Use edition 2021 and rustc 1.97.1; do not use upstream harness flags implicitly.
The majority-class baseline is 5/8, or 62.5%.

Compiler validation completed with `rustc 1.97.1 (8bab26f4f 2026-07-14)`:
all eight original/derived pairs had identical outcomes and error codes, matching
the three success/five failure expectations. `compiler-results.json` records the
observations. Test-only CRUST001 digest:
`c2285c3e478509399da4fe130f8d8c060d5dfba283d4d4eafa510857e540086d`.

Reproduce with `python3 scripts/prepare_external_rust.py NEW_DIRECTORY`.
The script refuses source bytes or inventory differing from the reviewed pins;
it only emits compiler metadata, with a 20-second timeout per invocation. This
is not a general-purpose sandbox for arbitrary Rust submissions. Output must be
a new directory; failures before admission produce no accepted corpus.

All three COGNO checkpoint identities were frozen in
`experiments/bpe-rust-pilot/checkpoints.tsv` before this selection. The tokenizer
and weights must not be fitted or retuned on this panel. Once its results have
been examined it is a regression panel, not an untouched future holdout.
