# Frozen-encoder classification probe

A diagnostic head-only API complements the existing end-to-end classifier:

- `SequenceClassifier::loss_and_head_gradients` detaches the bounded encoder
  feature vector and differentiates only the dense classification head.
- `train_head_step` uses `sequence_classifier::FrozenClassifierHeadAdamW`, which
  has two tensor optimizers and no encoder optimizer state.
- Rejected updates leave the head and optimizer unchanged; successful updates
  cannot decay or otherwise change encoder tensors.

This provides a controlled representation probe: fit a head while holding the
encoder fixed, then evaluate on an independently admitted validation/test split.
It can help separate head fitting from representation adaptation, but no Rust
expertise or generalization improvement is demonstrated by adding the API.
Existing training and artifact formats remain unchanged. There are no new
external dependencies.

Tests check exact head-gradient/loss agreement with end-to-end differentiation
across seeds, sequence lengths and classes. A repeated synthetic-label fit checks
loss reduction, deterministic updates and bit-for-bit encoder preservation.
Invalid targets, token IDs, empty sequences and context overflow must preserve
both the model and optimizer.
