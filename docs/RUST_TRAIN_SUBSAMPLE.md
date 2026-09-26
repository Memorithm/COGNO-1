# Balanced train subsampling
`cargo run -p cogno-model --example rust_train_subsample -- CORPUS SHA PER_CLASS SEED NEW_OUTPUT`

Ranks train records separately per binary class using SHA-256(seed little-endian u64 || source SHA-256), selecting exactly PER_CLASS from each. Original row order and every held-out record are preserved. Selection is invariant to input ordering and never depends on held-out content. This balances individual rows, not families or projects; use only for declared ablations. Fails on insufficient classes, invalid bounds, admission failure or existing output. Prints the output SHA-256. Input identity must be independently trusted. No improvement claim is implied by balanced sampling.
