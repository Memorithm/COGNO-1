# Order/context ablation protocol

Fixed before running candidates: contexts 128 and 256, each with preserved row
order or deterministic epoch shuffling; seeds 1/7/42; BPE target vocabulary 384;
embedding/hidden widths 8/16; 24 epochs; AdamW learning rate 0.003; only the
classification objective weighted 1, all other objective weights 0.

`training_order::epoch_order` is a versioned-by-source bounded permutation utility:
Fisher-Yates, SplitMix64 with documented constants, epoch-specific state and bounded
rejection sampling. It sees only the training row count. Each training row occurs
once per epoch. Evaluation rows are never included in optimizer batches or BPE fitting.

Choose one **configuration** across all three seeds by mean validation negative
log-likelihood (probabilities clipped to [1e-7,1-1e-7] for reporting only). Exact
ties use context then preserved-before-shuffled order. Do not select a favorable
seed. Retain every prediction, checkpoint identity and score. Do not use original
test rows or the already observed external panel for this choice.

This is exploratory work on the previously examined 24-row diagnostic corpus.
Its validation split is not fresh or large enough to establish generalization.
The external panel is already observed regression data; it is not a new holdout.
Increasing context changes position-table parameter count and initialization
offsets; this is not a matched-parameter/compute comparison or a speed benchmark.
No existing baseline artifacts are rewritten. No runtime activation is granted.
