# Explicit classification-only cognitive training

`SequenceCognitiveHeads::classification_loss_and_gradients` computes the same
classification NLL and connected gradients as the joint objective with weights
`[1, 0, 0, 0, 0]`, using one encoder graph and no dummy observations.
`train_classification_step` updates only the encoder and classification head.
Inactive heads and their optimizer moments/counters remain unchanged.

This is deliberately opt-in. A zero-weight joint step continues to evaluate all
objectives and update all head optimizers, including their weight decay. Thus
full-model equality with that existing path is not promised. Active gradients
and parameter updates are tested for exact equality for 3 seeds and 8 steps.
A unit test also preserves nonzero inactive moments and counters across a step.
Invalid input leaves both model and optimizer unchanged.

A local synthetic CPU diagnostic on 2026-09-26 used Rust 1.97.1, release mode,
x86_64 Intel Xeon Platinum 8573C, one model with vocab 32, context/sequence 16,
embedding 8, hidden 16, 2 classes and 2 retrieval candidates. For 1,000 calls,
joint gradients took 245,118 microseconds and classification-only gradients
30,031 microseconds (8.162x for that run). This excludes optimizer updates and
is neither a Thor measurement nor a model-quality result. Shared-host timing
varies; no timing threshold runs in CI.

Reproduce the diagnostic with:

```sh
cargo +1.97.1 test --release -p cogno-scirust --test classification_only_training classification_only_cpu_probe -- --ignored --nocapture
```

Existing training entry points and published checkpoint goldens are unchanged.
