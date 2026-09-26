# CPU inference: direct sequence embeddings

The read-only `SequenceEncoder::forward` now looks up token and position embeddings directly, then streams projection, ReLU and pooling. It no longer builds an autograd tape or dense selectors. Training still uses the unchanged differentiable tape. Public APIs, artifact layout, weights and admission limits are unchanged. All parameter arrays are checked for finiteness before inference. Encoder scratch consists of embedding_dim + hidden_dim f32 values, excluding parameters, inputs and allocator overhead.

## Measurement (2026-09-26)

Synthetic V4 classification probabilities, vocab=259, max_tokens=128, embedding_dim=32, hidden_dim=64. Seeds 1/7/42 each receive eight identical joint AdamW updates on synthetic data. Each length receives five warmups and 100 timed calls per seed. Output logging and training are outside the timed interval. See `experiments/inference-cpu/summary.json` for hardware, hashes and per-seed medians. Three sequential processes run baseline, candidate, baseline again; no CPU affinity or isolated hardware guarantee. Baseline is commit `14fffd91d4b97438b165ac6ac6d6638728585e39`; both binaries use rustc 1.98.1 (48a229cea 2026-09-01) and the same repository release profile. The probe was reformatted between builds without semantic changes.

| Tokens | Baseline median (µs) | Candidate (µs) | Baseline repeat (µs) | Conservative ratio |
|---:|---:|---:|---:|---:|
| 16 | 173.49 | 28.33 | 173.36 | 6.12× |
| 64 | 676.08 | 97.59 | 690.41 | 6.93× |
| 128 | 1374.32 | 192.54 | 1348.74 | 7.00× |

All 900 candidate output probability triples match the baseline exactly in round-trip f32 CSV representation. Independent tests compare the unchanged tape across 54 shape/seed/length/repetition configurations, invalid inputs, overflowing arithmetic and signed-zero/extreme finite weights. Existing joint training tests also pass.

These are local CPU observations, not universal latency guarantees, task-quality improvements, financial evidence, or exchange execution measurements. The five V4 heads use this encoder; only classification is timed here. The legacy SequenceClassifier logits path is separate and not accelerated by this change.

Reproduce with `cargo +stable run --release --locked -p cogno-scirust --example inference_probe`. For the baseline, use the same example on the baseline commit. Do not compare different compiler versions or profiles.
