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

This command requires the exact validation-only decision before evaluating test outcomes. It exports every seed of the selected arm only, never changing selection. Test predictions for unselected arms are not computed or exported. A separate completed test bundle binds the training bundle, selection and every prediction. Its per-seed source count must not be multiplied into a claim of independent samples. Test sources were admitted and checked for capacity before training, but their outcomes were not computed by the training or selection commands. This separation is procedural: filesystem permissions do not prevent an operator from reading the source corpus or repeatedly running new protocols. Preregister protocols before inspecting outcomes.

## Interrupted experiments

`rust_train_v2 resume PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE PRIOR_OUTPUT NEW_OUTPUT`

Each completed run writes a `.DONE` manifest after checkpoint round-trip and prediction export. Resume validates protocol identity, file hashes, model configuration, recomputed predictions, every epoch's row/update/token accounting and final-checkpoint metrics. It copies only complete valid runs into a **new** output directory. Partial runs restart deterministically from epoch zero: inference checkpoints do not contain AdamW moments and are never presented as optimizer resumes. A malformed `.DONE` fails closed instead of silently retraining. The prior output is preserved.

Resume is for an experiment directory under the operator's control. Its local per-run hashes detect accidental corruption but are not authenticated provenance; completed bundle verification still requires an independently retained `COMPLETE_SHA`. Neither run markers nor final journals prove intermediate optimization steps against a malicious writer. Retain emitted `run_sha256` values externally when stronger audit binding is required.

## Qualification gates

`cargo test --locked -p cogno-model --example rust_train_v2`

The bounded regression fixtures exercise actual optimization, partial minibatch accounting, all-seed selection and exact ties, source-bound test export, changed provenance/budget refusal, corrupted artifacts and selection refusal, complete-run reuse plus deterministic restart, and byte-identical model parameters versus the frozen batch-one step. Test fixture labels are not an expert Rust benchmark. No speed or quality improvement is inferred from successful qualification; those require a separately preregistered corpus and measured results.

## Opt-in v3 encoder graph

The same commands also accept `version` set to `rust-train-v3`, with exactly one
additional required field: `encoder_graph` set to `dense` or `gather`. V2 retains
its exact schema and dense behavior; adding a graph field to v2 is rejected.
There is no automatic graph selection or environment-variable override.

Gather uses the cognitive classification APIs for both single observations and
mean minibatches. It removes dense selector matrices while retaining dense
gradients and AdamW states. It does not change the model architecture, parameter
count, tokenizer or CBPC checkpoint format. The v3 plan records the graph, and
the exact protocol hash binds it to every completed experiment. Resume across
protocols/graphs is rejected before creating a destination.

Tests reproduce checkpoint bytes, epoch journals and predictions for v2 dense,
v3 dense and v3 gather, with two arms, two seeds, and batches one and two. They
also verify complete-run replay and changed-protocol refusal. These bounded
fixtures establish arithmetic equivalence; end-to-end timing requires a separate
recorded experiment, and model quality is unchanged in these comparisons.

## Paired full-training timing

`rust_train_v2 compare-graphs PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE NEW_OUTPUT ROUNDS`

This explicit experiment compares v3 dense and gather variants of an admitted
protocol, preserving every other field. It accepts 1–5 rounds, alternates which
graph runs first, and retains both complete training directories for every round.
The total update budget includes **both graphs × every round × every arm × every
seed**. Invalid rounds or an excessive total budget fail before creating output.

Every pair must reproduce identical checkpoint bytes, epoch journals and
train/validation predictions before a timing row is accepted. The sample table
records every elapsed nanosecond count, order, matched-file count and both bundle
identities. The final marker binds the input protocol and sample table and is
written after all rounds pass. Its hash is printed for independent retention.
No fastest-round selection, speedup assertion or test-based model selection occurs.

Timing includes training, metrics, checkpoint I/O and bundle verification. Shared
corpus admission/BPE fitting and the cross-graph byte comparison are excluded.
Results depend on the declared shape, data, build profile and host; retain those
alongside raw samples. This is a CPU training comparison, not inference latency,
GPU scaling, or evidence of improved prediction quality.
