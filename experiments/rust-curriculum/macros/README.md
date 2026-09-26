# Original synthetic Rust curriculum: macros

This slice contains **24 original Rust snippets in 12 semantic contrast pairs**,
with 12 compiler-success and 12 compiler-failure cases. It is preparation data
for a future classification experiment; no checkpoint was trained or selected
using it in this change. No model-quality improvement is claimed.

Each pair changes a domain-specific semantic condition rather
than simply renaming identifiers. Both members of a pair stay in the same split:
16 train, 4 validation, 4 test. `project` identifies a **synthetic family**, never
an independent upstream project. Families within the same Rust concept can still
be semantically close; this grouping does not establish independent project
holdout performance. All snippets and intended outcomes were examined during
authoring. The four test rows are synthetic diagnostics, not an unopened external
benchmark or evidence of expert Rust skill. Compile success does not imply runtime
correctness, safety, algorithmic quality, or financial suitability.

## Provenance and compiler evidence

Authoring origin: original COGNO-1 fixtures, 2026-09-26, with no upstream extraction.
The original fixture sources are covered by the local `LICENSE-MIT` notice.
`fixtures.json` pins every source SHA256, expected binary outcome, expected
compiler diagnostic category, family, and split. The record `revision` is the
SHA256 of that inventory, **not an invented Git revision**. The verifier embeds
that inventory hash and checks the source bytes before compiling a private copy.

Every source was compiled with **rustc 1.97.1 (8bab26f4f 2026-07-14)**,
full compiler commit `8bab26f4f68e0e26f0bb7960be334d5b520ea452`, edition 2021,
`--emit=metadata --error-format=json`. No source program was executed.
`compiler-results.json` preserves the observed labels, exit codes, and diagnostic
categories. `corpus.jsonl` contains the observed compiler labels, admitted using
`scripts/admit_rust_corpus.py`. Failed launches, timeouts, crashes, unrecognized
errors, and outcomes differing from the reviewed expectations stop preparation;
they cannot silently become negative training labels. `LIFETIME` denotes rustc's
uncoded error with exact text `lifetime may not live long enough` where present.

File names, expected outcomes, diagnostic messages, family descriptions, and
provenance **must not be included in model inputs**. The input is `source` only;
`label` is the binary compile target. This maintains the existing corpus reader
contract. No source is truncated to satisfy a model context limit.

## Reproduce

From the repository root, with the pinned toolchain installed:

```sh
python3 experiments/rust-curriculum/macros/verify.py /tmp/cogno-macros-new
python3 -m unittest discover -s scripts -p 'test_curriculum_macros.py' -v
```

The output directory must not exist. It receives `corpus.crust`, `corpus.jsonl`,
`provenance.jsonl`, the admitted manifest, and a fresh compiler report. An optional
`--rustc /absolute/path/to/rustc` avoids invoking rustup; both release and full
compiler commit must match. Reproduction requires no network or third-party
Python package. The verifier has a 20-second timeout per compiler invocation;
it is a checker for these reviewed bounded fixtures, not a sandbox for arbitrary
untrusted Rust code. The domain CI recompiles all 24 and compares their evidence
byte-for-byte to the retained artifacts.

## Families

| Synthetic family | Partition | Semantic contrast |
| --- | --- | --- |
| expression-result | train | expression macro expansion obeys expected type |
| hygiene-local | train | macro receives call-site local explicitly |
| repeated-element | train | repetition still requires homogeneous array element types |
| generated-return | train | generated function body must match declared output |
| generated-field | train | macro-generated field accesses must exist |
| mutable-expansion | train | macro assignment respects call-site mutability |
| macro-move | train | duplicated macro operand can move a value twice |
| macro-bound | train | generated generic implementation requires its trait bound |
| nested-expansion | validation | nested macros preserve expression type requirements |
| macro-method | validation | method invocation in expansion must resolve |
| generated-pattern | test | generated match must be exhaustive |
| macro-const | test | macro expansion in const must be const-evaluable |

The existing pilot and upstream regression-panel artifacts remain unchanged.
Any next training run must freeze its protocol before evaluation and preserve
all seeds, failures, context refusals, and a constant-class baseline. A future
external quality claim requires a new independent evaluation panel.
