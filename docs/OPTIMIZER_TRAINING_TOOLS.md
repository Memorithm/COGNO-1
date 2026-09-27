# Optional numerical training tools

These safe-Rust APIs are available under `cogno_scirust::optim` and
`cogno_scirust::losses`. Existing training defaults are unchanged. They do not
make the current classifier a Rust code generator.

* `AdamWWorkspace` / `AmsGradWorkspace` retain candidate buffers, preserving the
  original arithmetic and all-or-nothing parameter/state updates. Allocate once
  for each parameter shape, then call `step_with_workspace`. Workspaces are
  scratch, not checkpoints. Error recovery overwrites their candidate contents.
* `AdamW::checkpoint` and `from_checkpoint` encode parameters, hyperparameters,
  moments and step in `CADAM001`. The bounded format preserves f32 bits. It does
  **not** authenticate bytes or identify parameter ordering. Store a trusted
  content hash and model/tokenizer/schedule metadata separately before accepting
  an externally supplied checkpoint. AMSGrad checkpoints are not supported.
* `GradientAccumulator` sums weighted microbatch gradients in f64. For a mean
  gradient, pass the microbatch's sample count; call `mean_into`, clip if desired,
  and perform one optimizer update. Explicitly clear for the next update.
* `SoftTargetCrossEntropy` keeps gradients connected to logits. Smoothing uses
  `(1 - alpha) * one_hot + alpha / classes`; alpha zero is ordinary cross entropy.
  Target probabilities require unit mass within 1e-6. No implicit normalization.
* `weighted_mean_loss` normalizes finite nonnegative sample weights in f64 before
  building tape operations. Zero weights mask examples; an all-zero batch fails.
* `WarmupCosine` is explicitly opt-in. Query with the number of successful
  optimizer updates already completed. Retrying a failed update uses the same
  rate. Persist constructor arguments with checkpoint metadata. Its positive
  floor is compatible with the existing strictly positive AdamW learning rate.

## Reproduction

```
cargo test -p cogno-scirust --offline
cargo run -p cogno-scirust --release --offline --example optimizer_training_probe
```

The probe trains a two-parameter classifier on four synthetic numeric examples,
using smoothing, weighted microbatches, scheduling and a midpoint resume. It
checks loss decreases, then compares legacy and reusable AdamW over seven
alternating pairs of 256 updates on 16,384 parameters. Every pair must produce
identical parameter/optimizer bytes. It prints raw nanosecond observations and
medians, with no timing pass threshold. The timings exclude initial optimizer
allocation and final checkpoint encoding. They include candidate copying and
validation; there is no claim that those costs disappear.

This is an optimizer microbenchmark, not model end-to-end training, compiler
acceptance evaluation, GPU measurement or evidence of Rust expertise. Shared
host load, compiler and CPU affect timings; measure on the deployment host.

One local x86_64 release run with Rust 1.97.1 produced a synthetic loss decrease
from 0.693147182 to 0.146234949 over 120 updates, including checkpoint resume.
All seven paired microbenchmarks produced identical checkpoint bytes. Median
legacy time was 80,813,341 ns and reusable time was 80,220,969 ns (ratio 1.0074).
This near-parity observation does not establish a meaningful throughput gain.
Raw paired times (legacy ns, reusable ns), in execution order:

```
190802477 80220969
79592883 86941674
80813341 78240969
79627120 79768890
81007479 78913786
79818905 80303186
80876186 80395337
```
