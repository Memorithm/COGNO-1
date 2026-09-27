# Opt-in sequence training paths

The original dense-selector graph and default classifier methods remain unchanged.
The new paths reuse the existing tensors, optimizer and safe Rust autograd. No
new external dependency, model promotion or architecture flag in legacy artifacts
is introduced.

## APIs and intended use

- `SequenceEncoder::append_to_tape_gather` and
  `SequenceClassifier::{loss_and_gradients_gather, train_step_gather}` replace
  one-hot selector matmuls with differentiable row gathers. Repeated token ids
  accumulate into the same embedding row. Values, gradients and twelve successive
  AdamW updates are checked against the dense path by the runnable probe.
- `sequence::SequenceWorkspace` and `forward_with_workspace` reuse two bounded
  scratch vectors. They cache no model parameters; same-width models can safely
  reuse scratch. The returned slice is valid until the next mutable scratch use.
- `token_features` returns local token-plus-position ReLU features.
  `forward_weighted`, `append_to_tape_weighted` and the classifier's
  `logits_weighted`, `loss_and_gradients_weighted`, `train_step_weighted` use
  nonnegative normalized pooling weights. A zero mask preserves original
  positions and removes that row's contribution to the pooled representation.
  These are external masks, not learned attention or attribution scores.
- `forward_contextual`, `append_to_tape_contextual` and classifier
  `logits_contextual`, `loss_and_gradients_contextual`, `train_step_contextual`
  add `strength * previous_token_embedding` before position addition and the
  existing projection. Strength lies in `[0,1]`; the first token repeats itself
  at the left boundary. This provides adjacent-token interaction before ReLU,
  without attention, additional trainable tensors or a new optimizer.

Mask and contextual strength must be recorded alongside an experiment and used
consistently at inference. The existing artifact layout does not encode these
controls. Loading contextual-trained parameters and invoking the default method
would evaluate a different function. These APIs do not establish code generation
or Rust competence.

## Reproduction

```sh
cargo test --locked -p cogno-scirust --all-targets
cargo run --release --locked -p cogno-scirust --example sequence_paths_probe -- 10
```

The example accepts 1–1000 iterations per round, warms each operation three times,
and reports the median of seven rounds. It checks exact dense/gather gradient
and multi-update parity before measuring. Timings use a single process; operation
order is fixed and no cross-host confidence intervals are claimed. The contextual
one-example optimization smoke is not a held-out evaluation.

Recorded local run: x86_64, rustc 1.97.1 (8bab26f4f 2026-07-14), release profile.
Dimensions: vocabulary 384, capacity 512, input length 192, embedding width 8,
hidden width 16, two classes. These measurements are shape-specific CPU timings,
not a production latency promise.

| Measurement | Result |
|---|---:|
| Dense loss + backward, median ns | 20,399,911 |
| Gather loss + backward, median ns | 151,758 |
| Contextual loss + backward, median ns | 170,473 |
| Direct encoder inference, median ns | 31,017 |
| Workspace encoder inference, median ns | 32,588 |
| Dense largest tensor, elements | 98,304 |
| Gather largest tensor, elements | 4,096 |
| Contextual one-example NLL, before | 0.66544294 |
| Contextual one-example NLL, after 24 updates | 0.38394734 |

The largest-tensor figure is an admission bound, not total resident memory. The
gather gain primarily removes dense selector work; it is not evidence of better
accuracy. Workspace reuse did **not** improve latency in this run, despite avoiding
fresh scratch-vector allocation. No held-out result or model promotion follows
from this microbenchmark.
