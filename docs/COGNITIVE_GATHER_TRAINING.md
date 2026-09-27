# Gather training for the cognitive heads

The `SequenceCognitiveHeads` classification-only path can now use the existing
bounded row-gather encoder via `classification_loss_and_gradients_gather` and
`train_classification_step_gather`. It removes dense selector matrices of size
tokens × vocabulary and tokens × maximum context. Gradients, parameters and
optimizer state retain their existing dense layout.

This closes an integration gap: the previous gather implementation was available
on `SequenceClassifier`, while the configurable Rust trainer uses the cognitive
heads. The new API is opt-in; the existing training entry points remain the dense
reference. Checkpoint formats and deterministic runtime authority are unchanged.

Public integration tests compare complete gradients, losses, model parameters
and optimizer states after nine updates for each of three seeds and three token
layouts, including repeated token IDs and maximum context. Invalid token IDs,
empty/oversized sequences and invalid classes must leave state unchanged.
No throughput improvement or model-quality gain is inferred from these tests.
