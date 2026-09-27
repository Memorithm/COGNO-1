# Bounded Rust classifier training v2

`cargo run --release --locked -p cogno-model --example rust_train_v2 -- plan PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE`

The UTF-8 protocol is newline-terminated, one key TAB value per line. Exactly these keys are required:

```text
version	rust-train-v2
corpus_sha256	<64 lowercase hex>
provenance_sha256	<64 lowercase hex>
seeds	1,7,42
arms	full,cycle_mix
epochs	24
vocab	384
context	512
embedding	8
hidden	16
batch	1
learning_rate	0.003
max_updates	100000
```

The protocol's supplied SHA-256 pins its exact bytes. The corpus is admitted using CRUST001 source hashes and project disjointness rules. The provenance file is bound by exact-byte SHA-256; this does **not** certify licenses, labels, project independence or compiler outcomes. Those must be established before preregistration. Project IDs are declarations, not proof of independent upstream projects.

Only train rows fit BPE. Every configured representation of every training row and the full-BPE evaluation representation of every row must fit; no truncation. The plan rejects a run exceeding `max_updates`, counting the last partial minibatch. Maximum bounds: 100 epochs, 8 distinct seeds, 4 distinct arms, batch 32, embedding 32, hidden 64, vocabulary/context 512. This is a bounded CPU research workflow, not a distributed training system.

Arms are `full`, `byte_mix`, `half_mix`, `cycle_mix`, with the frozen curriculum probe's merge-prefix schedules. Order in the protocol is significant for deterministic selection ties. No model is promoted or enabled by this command. The current target is binary compiler acceptance, not code generation or Rust expertise.

## Train every preregistered run

`rust_train_v2 train PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE NEW_OUTPUT_DIR`

Creates a new directory (existing directories are refused), stores the exact protocol and admitted plan, and trains every arm × seed for the full epoch budget. Train indices use the frozen `epoch_order` permutation. Consecutive chunks form minibatches with a final partial batch; there is no oversampling, class balancing or dropped row. Minibatch gradients are averaged by actual batch size. Batch one calls the frozen single-row training step, retaining arithmetic and head seeds 1/2/3/4. No validation or test examples contribute gradients. Each run writes a loadable CBPC checkpoint containing its fitted BPE tokenizer.
