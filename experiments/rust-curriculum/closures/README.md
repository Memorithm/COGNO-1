# Original synthetic Rust curriculum: closures

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
python3 experiments/rust-curriculum/closures/verify.py /tmp/cogno-closures-new
python3 -m unittest discover -s scripts -p 'test_curriculum_closures.py' -v
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
| fn-mut-capture | train | mutating captures require FnMut |
| fn-once-consume | train | consuming captures requires FnOnce |
| binding-mutability | train | calling a mutable capture requires mutable closure binding |
| capture-to-pointer | train | capturing closure cannot coerce to function pointer |
| return-type | train | closure return must meet expected function result |
| argument-lock | train | first call fixes inferred closure argument type |
| call-once | train | a consumed closure cannot be called twice |
| move-owner | train | move closure owns its non-Copy capture |
| boxed-lifetime | validation | returned closure must own local captured value |
| send-capture | validation | Send closures cannot capture Rc |
| argument-arity | test | closure calls require the declared argument count |
| mutable-reference | test | closure bound determines mutability of reference parameter |

The existing pilot and upstream regression-panel artifacts remain unchanged.
Any next training run must freeze its protocol before evaluation and preserve
all seeds, failures, context refusals, and a constant-class baseline. A future
external quality claim requires a new independent evaluation panel.
