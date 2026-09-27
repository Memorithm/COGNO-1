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

## Epoch accounting

Every run emits an `.epochs.tsv` journal after each complete epoch, including actual updates, rows seen, emitted tokens including framing, and train/validation counts, correct predictions and mean NLL. Both metric splits use full BPE independent of training arm. Probability clipping is fixed at `[1e-7, 1-1e-7]`; tied class probabilities choose class 0. Test predictions are not evaluated during training. Validation is observed only for reporting: no early stopping, budget changes or best-epoch selection. A journal without a completed checkpoint is a partial run, not admissible evidence.

## Completed bundle admission

Each checkpoint is reloaded and compared exactly with its in-memory model before exporting source-bound train/validation predictions. `COMPLETE` is written last and binds the exact protocol, plan, all epoch journals, checkpoints and prediction files. Training prints `COMPLETE_SHA256`; retain that value in an independent trusted record. A hash supplied from the same untrusted directory is not authentication.

`rust_train_v2 verify PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE OUTPUT_DIR COMPLETE_SHA`

Verification rejects incomplete inventories, changed bytes or protocol/configuration mismatches and independently recomputes every stored prediction from its checkpoint. A valid bundle certifies reproducible artifact contents, not that a claimed training history or compiler label is truthful. No test outcomes enter the completed training bundle.

## Validation-only arm selection

`rust_train_v2 select PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE OUTPUT_DIR COMPLETE_SHA NEW_SELECTION_FILE`

The selector first verifies the complete Cartesian product of preregistered arms and seeds, then recomputes final-checkpoint validation NLL. It selects the lowest mean across **all** seeds, with exact ties resolved by protocol arm order. The report retains every arm, seed count and repeated observation count. It does not choose the best seed or epoch. Repeated source observations across seeds are not independent test cases. Selection is recorded in a new file with both protocol and bundle identities; an existing selection file is refused.

## Test after frozen selection

`rust_train_v2 test PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE OUTPUT_DIR COMPLETE_SHA SELECTION_FILE NEW_TEST_DIRECTORY`

This command requires the exact validation-only decision before evaluating test outcomes. It exports every seed of the selected arm **and all preregistered controls**, never changing selection. A separate completed test bundle binds the training bundle, selection and every prediction. Its per-seed source count must not be multiplied into a claim of independent samples. Test sources were admitted and checked for capacity before training, but their outcomes were not computed by the training or selection commands. This separation is procedural: filesystem permissions do not prevent an operator from reading the source corpus or repeatedly running new protocols. Preregister protocols before inspecting outcomes.
