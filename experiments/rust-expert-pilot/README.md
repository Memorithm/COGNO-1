# Rust specialization: diagnostic pilot, not expert qualification

Priority as of 2026-09-26: specialize COGNO-1 on expert Rust tasks first; financial training is deferred. COGNO V4 produces typed scores, not source-code completions. This pilot evaluates only its compile/reject classifier. It is not a coding LLM benchmark.

## What actually ran

24 original synthetic snippets, validated with **rustc 1.97.1, edition 2024** using metadata-only compilation, without executing the programs. Compiler rejection does not prove runtime unsafety; successful compilation does not prove correctness or performance. `corpus.jsonl` records source, SHA-256, Internal classification, provenance and compiler diagnostics. There are no imported repositories, copied documentation passages or financial data.

Eight training families (16 snippets): moves, overlapping mutable borrows, lifetimes, Send, Sync, Sized, associated types and const generics. Two validation families (four snippets): dyn compatibility and Pin/Unpin. Two test families (four snippets): generic associated types and higher-ranked trait bounds. Each complete family and its valid/invalid pair stay in one split. This is intentionally tiny and hand-designed; family separation is not proof of independence from all conceptual overlap.

All three seeds (1, 7, 42) were declared before evaluation. Each uses a newly initialized V4 model, 3,326 parameters, byte vocabulary 259, maximum 128 tokens, embedding width 8, hidden width 16, 24 epochs, learning rate 0.003. This is training from initialization, not fine-tuning a proven expert checkpoint. No truncation; overlength inputs fail. Only classification has nonzero loss weight. Auxiliary task slots contain placeholders required by the joint API; their heads are **untrained and unusable**, and their losses have no task-quality meaning. The shared encoder changes, so this is not a drop-in replacement for a previously trained multitask model.

| Seed | Initial train | Trained train | Initial validation | Trained validation | Initial test | Trained test |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/16 | 13/16 | 2/4 | 2/4 | 2/4 | 2/4 |
| 7 | 8/16 | 16/16 | 2/4 | 4/4 | 2/4 | 1/4 |
| 42 | 8/16 | 12/16 | 2/4 | 3/4 | 2/4 | 2/4 |

A constant-class baseline scores 50% in each balanced split. **No held-out improvement is demonstrated.** Training accuracy is not an expert qualification. There was no selection, early stopping or hyperparameter tuning on validation/test. These test results are now diagnostic knowledge; future design iterations need an additional untouched evaluation set.

The three checkpoints are stored as hexadecimal COG4 bytes with readable manifests and hashes. The trainer verifies exact parameter equality after encode/load. They are experimental, classification-only, not activated in any runtime. Predictions include every sample before and after training; no successful seed is selectively reported. The current 128-token pilot and architecture are not adequate evidence for whole-crate or long-range Rust reasoning.

## Reproduce

Run at repository root (Python standard library only):

```sh
python experiments/rust-expert-pilot/prepare.py
cargo +1.97.1 run --release --locked -p cogno-model --example rust_expert_pilot -- \
  experiments/rust-expert-pilot/corpus.tsv experiments/rust-expert-pilot/checkpoints \
  > experiments/rust-expert-pilot/predictions.csv
```

The preparation step checks expected compiler outcomes and refuses compiler crashes, timeouts, unexpected labels and duplicate code. The example rejects duplicate sources, cross-split family leakage, missing classes and overlength inputs. This is a controlled synthetic experiment, not a general importer for untrusted training corpora.

## Next expert-Rust curriculum

Expand with independently authored project-level task families: lifetime and variance diagnostics, GAT/HRTB, Send/Sync and async cancellation, Pin projection, trait coherence and dyn compatibility, safe FFI boundaries, error handling and performance regressions. For each task retain provenance, licensing, compiler/toolchain, expected diagnostics, executable behavioral tests where applicable, and corrected alternatives. Split by source project and transformation family before generating variants. Keep compiler diagnostics and answer keys out of model inputs.

Evaluate separately: balanced classification accuracy and calibration; ranking of behaviorally verified repairs; retrieval over relevant and hard-negative Rust code; contradiction detection on explicit semantic claims. Require gains over constant, lexical and untrained baselines on enough untouched tasks, across seeds, before claiming useful specialization. No target percentage is established by this pilot. Any tokenizer/context/architecture change must be versioned and compared with the current encoder on the same training budget. Financial training remains out of this phase.

Reference semantics (consulted 2026-09-26; the pinned compiler is the label oracle):
- https://doc.rust-lang.org/reference/trait-bounds.html
- https://doc.rust-lang.org/std/pin/index.html
