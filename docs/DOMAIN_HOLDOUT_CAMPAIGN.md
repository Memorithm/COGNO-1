# Fixed domain holdout campaign — 2026-09-27

Purpose: diagnose generalization to entire synthetic Rust domains with three
times the previous training rows, without claiming independent upstream projects.
The 288 sources already exist in this repository. They are known synthetic data,
not a newly blinded benchmark. Source strings, compiler labels and provenance
remain unchanged. Every case is recertified with pinned rustc 1.97.1 before admission.

Splits fixed before new model runs:

| Split | Domains | Rows |
|---|---|---:|
| Train | ownership, borrowing, traits, lifetimes, pattern_matching, closures, iterators, error_handling | 192 |
| Validation | smart_pointers, async_futures | 48 |
| Test | macros, concurrency | 48 |

All synthetic families of a domain share its split through the explicit
hash-bound `rust_project_split` inventory. The old within-domain split assignments
remain recorded in its report. This is domain separation, not proof that semantic
near-duplicates cannot cross domains. Do not pool these results with the old
64/16/16 experiment: both training and evaluation distributions change.

Frozen admitted corpus SHA256:
`fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f`.
Frozen groups SHA256:
`ea343f65801e0da63b6de455bf9731fca8586c3d1c39b63ce1aab7675fd8c606`.

Use the existing four prefix schedules, seeds 1/7/42, target BPE vocabulary 384,
context 512, embedding 8, hidden 16, 24 epochs and AdamW 0.003. Fit tokenization
and weights on train only; use full BPE for evaluation. No early stopping,
hyperparameter search or best-seed selection. Preflight rejects any capacity
failure; never truncate. Select the arm by mean validation NLL across all three
seeds (clip probabilities to [1e-7,1-1e-7], ties use existing arm order). Test
observations have no influence on selection. Retain all arms and seeds.

Report per-split/per-seed accuracy and NLL, mean arm metrics, uniform-probability
NLL ln(2), per-domain test accuracy, tokenizer/checkpoint/corpus hashes, and
coverage. The same 48 test sources across three seeds are not 144 independent
examples. No runtime promotion, expert Rust or financial capability claim.

Preparation: `scripts/prepare_domain_holdout.py` compiles the 12 domains and
invokes the bounded Rust grouping tool. Run `bpe_curriculum_probe CORPUS SHA OUT
--domain-holdout-v1`. Original invocations keep the frozen 64/16/16 campaign.
For selection use `select(rows, validation_count=48)` from
`scripts/select_curriculum_probe.py`; its original default stays 16.

This bounded CPU diagnostic precedes external corpus admission and any massive
Thor training. A successful compiler/toolchain preflight is not a reservation
of Thor resources or evidence that an unattended training job has started.
