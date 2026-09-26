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

## First frozen-checkpoint result

All three seeds (1/7/42) accepted seven examples and refused one at context 128:
`borrowck-binding-mutbl`. Every accepted input was predicted compile-failure.
Each checkpoint therefore got 5/7 accepted cases right (71.43%), with 5 true
negatives, 2 false negatives and 0 true positives. This **only matches the constant
majority-class baseline on the accepted subset**. On all eight inputs, there are
five correct answers, two incorrect answers and one refusal; coverage is 87.5%.
Do not report 71.43% as evidence of expert Rust ability or improvement.

`seed-*.csv` retains every probability, prediction and refusal; `results.json`
includes checkpoint identities and the confusion matrices. There was no training,
tokenizer fitting or checkpoint selection using these eight cases. No latency
measurement or statistical generalization claim is made.

Reproduce with `scripts/evaluate_external_rust.py --executable PATH_TO_bpe_external_eval
--checkpoints CHECKPOINT_DIRECTORY --output NEW_DIRECTORY`. It validates the frozen
checkpoint inventory, rechecks compiler labels, admits only test rows, then runs
three separate inference processes. No refused input is truncated or omitted.

## Frozen regression gate

`scripts/verify_external_rust.py` reruns the compiler checks and evaluation,
compares all 24 seed/case rows to these frozen files (absolute probability
tolerance 1e-6), compares the full compiler report and aggregate metrics, and
requires wrong model/corpus hashes to fail before producing any predictions.
It is invoked by the source-distribution verification inside network-isolated
CI, using the pinned compiler binary and newly regenerated frozen checkpoints.
No upstream repository is contacted in that verification. The retained MIT
license travels with the source bundle.
The verifier also rejects exact original/derived source overlap with any of the
24 pinned earlier diagnostic records. This does not detect semantic near duplicates.

This panel is now observed regression data. Subsequent model selection must use
training/validation sources distinct from it, followed by a new unopened panel
for a generalization claim. First priorities exposed by this result are the
constant negative prediction and one context refusal, not financial deployment.
