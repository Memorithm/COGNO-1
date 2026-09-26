# Fixed curriculum prefix campaign — 2026-09-26

Protocol fixed before the first candidate training. This is a small synthetic
classification study, not expert Rust code generation or a financial model.

Use only the four existing domains ownership, borrowing, traits, lifetimes.
`scripts/prepare_rust_curriculum.py` reruns their pinned rustc1.97.1 metadata
certification and combines them with full corpus admission. Expected corpus
SHA256: `9d7ed2ef18a30f760696dff926c1c2a2684446d12905c9a205018b6a95f50376`.
There are64training/16validation/16test examples; pairs stay within a synthetic
family partition. These are not independent upstream projects. The eight new
domains in this50-PR series are excluded from this training/selection campaign.

BPE target vocabulary384, context512, embedding8, hidden16, two classes,
one symbolic rule. Encoder seeds1/7/42; head seeds1/2/3/4. Classification-only
AdamW learning rate0.003,24epochs, each training row once per epoch using the
same seeded epoch permutation for every arm. All train representations must
fit context before any optimizer runs; never truncate or skip sources.

Four fixed arms, with M learned merges and zero-based epoch:

- `full`: always M.
- `byte_mix`: even epochs M, odd epochs0.
- `half_mix`: even epochs M, odd epochs floor(M/2).
- `cycle_mix`: cycle0, floor(M/2), M.

Vocabulary fitting uses training sources only. Every evaluation uses the same
full M merges. Thus model parameters, initialization, steps and final inference
are matched, while sequence lengths and training compute differ. There is no
matched-compute or speed claim. This is deterministic prefix augmentation,
not random per-merge BPE dropout.

Retain all12checkpoints and every trained prediction. Select one arm across
all3seeds by mean validation NLL, clipping reported probabilities to[1e-7,1-1e-7].
Exact ties prefer the arm order above. Never select a favorable seed or use
any test prediction for selection. Test performance is descriptive on the
previously published synthetic fixtures, not a fresh benchmark. No automatic
runtime promotion. Record training token-ID exposure and test token novelty;
reduced novelty alone does not demonstrate a quality improvement.

Commands after a release build:

```sh
python3 scripts/prepare_rust_curriculum.py /tmp/new-curriculum
./target/release/examples/bpe_curriculum_probe /tmp/new-curriculum/corpus.crust 9d7ed2ef18a30f760696dff926c1c2a2684446d12905c9a205018b6a95f50376 /tmp/new-prefix-candidates
python3 scripts/select_curriculum_probe.py /tmp/new-prefix-candidates/predictions.csv
```
