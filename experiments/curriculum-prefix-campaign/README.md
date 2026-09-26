# Curriculum prefix exposure — 2026-09-26

Protocol, runner and validation-only selector published before candidate training
in PR #168. See [fixed protocol](../../docs/CURRICULUM_PREFIX_CAMPAIGN.md).
Local CPU, Rust1.97.1. Four original compiler-verified synthetic domains,
64training/16validation/16test rows; twelve candidate checkpoints retained by
SHA256, all1,152predictions retained, no binaries committed or runtime promotion.

| Arm | Validation mean NLL | Mean validation accuracy | Test correct across seeds1/7/42 | Mean test accuracy |
|---|---:|---:|---|---:|
| full | 1.054817456 | 52.08% | 8/16,9/16,7/16 | 50.00% |
| byte_mix | 0.708567719 | 43.75% | 10/16,5/16,8/16 | 47.92% |
| half_mix | 0.721085077 | 52.08% | 9/16,8/16,9/16 | 54.17% |
| cycle_mix | 0.695401419 | 41.67% | 11/16,8/16,9/16 | 58.33% |

The fixed selector chose **cycle_mix**, using validation NLL across all three
seeds. The selection metric was NLL, not accuracy: validation accuracy actually
decreases from25/48 to20/48 seed-example predictions. Its test mean NLL is0.691481936 versus0.965233340 for full BPE training.
However, constant probabilities0.5 give NLL ln(2)=0.693147181: selected validation
NLL is slightly worse than that uninformed baseline. Improved loss versus an
overconfident baseline is not demonstrated Rust expertise.

Test accuracy improves descriptively from24/48 to28/48 seed-example predictions.
Those are the same16sources repeated with three seeds, **not48independent
examples**. The fixtures are synthetic, tiny, previously published and correlated
within families. No statistical/generalization or expert capability claim.
No seed is discarded; all other arms, including byte_mix degradation, remain.

Active training token IDs rise from161(full) to170(cycle_mix). Full-inference
test tokens whose IDs never appeared in training fall from10/893 to8/893;
validation remains18/703. This small coverage change does not establish the
cause of the score difference. Prefix schedules also change sequence lengths
and compute, despite matched architecture, initializations and optimizer steps.
The eight newly added curriculum domains are excluded from this experiment.

Reproduction rebuilds every checkpoint and compares identities, predictions,
selection and exposure counts using `scripts/verify_curriculum_probe.py`.
