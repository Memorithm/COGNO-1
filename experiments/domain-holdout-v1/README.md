# Whole-domain Rust diagnostic — 2026-09-27

Protocol PR #191 was published before these 12 candidate runs. See the
[fixed protocol](../../docs/DOMAIN_HOLDOUT_CAMPAIGN.md). All 288 known synthetic
sources were recertified using rustc 1.97.1. Eight whole domains supply 192 train
rows, two other domains supply 48 validation rows, and macros plus concurrency
supply 48 test rows. No test domain contributes tokenizer fitting or gradients.
Grouping identities and previous split assignments remain in groups.tsv and
group-report.tsv; admission.json binds the original and regrouped corpora.

| Arm | Mean validation NLL | Mean validation accuracy | Mean test accuracy | Mean test NLL |
|---|---:|---:|---:|---:|
| full | 1.155352722 | 61.11% | 55.56% | 1.216771427 |
| byte_mix | 0.681855283 | 59.72% | 54.86% | 0.734183599 |
| half_mix | 0.700644437 | 59.03% | 59.03% | 0.696846705 |
| cycle_mix | 0.674545317 | 59.03% | 60.42% | 0.678491229 |

The fixed validation-NLL rule selects **cycle_mix**, retaining all seeds. Its
test counts are 29/48, 28/48 and 30/48 for seeds 1/7/42, versus 28/48, 26/48,
26/48 for full BPE. The descriptive mean difference is 4.86 percentage points.
Validation accuracy is lower than full BPE despite better NLL. Both selected
validation and test NLL are modestly below ln(2)=0.693147181 for constant 0.5
probabilities. Balanced constant-class test accuracy is 50%.

The same 48 sources repeat across seeds; they are **not 144 independent cases**.
Contrastive pairs and synthetic families are correlated. Two held-out domains
are not independent upstream repositories. Sources were previously published,
and the schedule was informed by earlier experiments. Do not claim statistical
significance, expert Rust capability, or improvement over the previous campaign's
58.33%: its test set and training set differ. No model is promoted.

This task predicts compiler acceptance of short Rust snippets. It does not test
program execution, code generation, patch synthesis, whole-crate correctness or
financial reasoning. Full BPE reaches 96.53% mean training accuracy while its
test NLL is poor; cycle_mix training accuracy is 69.97%. This diagnoses a serious
gap between fitting the training examples and useful generalization.

All 3,456 predictions, 12 checkpoint hashes, 36 exposure records and per-domain
metrics are retained, including weaker arms. Raw checkpoint binaries can be
reconstructed; they are not committed. `summarize_domain_holdout.py` verifies
complete source/arm/seed coverage, probabilities and checkpoint byte identities.
It rejects class/probability contradictions outside a 1e-7 band around f32 ties.

`verify_domain_holdout.py` recompiles all 288 cases, retrains all 12 models,
checks every checkpoint hash and prediction, and recomputes metrics. The
dedicated CI runs it. This x86 CPU reference does not imply bit-identical results
on every architecture; Thor qualification is recorded separately. Timing and
GPU acceleration were not measured in this campaign.

## Thor reproduction

RemoteOps run 36295174922 completed on Thor aarch64 using pinned COGNO commit
`df657c04ad254fbd92aaca8e4fc52fa2488afbfe` and rustc 1.97.1. All 12 checkpoint
inventories and hashes match this x86 reference. The entire predictions.csv
matches SHA256 `051f5e2936629a81c92c89e3cf6161b7240d4e28dbcbc2c9e193362a15c356d3`.
The arm metrics and validation-only selection also match. `thor-summary.json`
preserves the reported identities and run provenance. This demonstrates one
fixed experiment's CPU reproduction across x86 and ARM64, not universal bitwise
portability, GPU use, massive training, or a throughput improvement. Thor retains
the checkpoints, full predictions, source snapshot and compiler evidence in its
exclusive run directory; no production model was installed.
