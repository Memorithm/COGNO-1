# bpe_paired_bootstrap

Run `cargo run --release -p cogno-model --example bpe_paired_bootstrap -- A SHA B SHA CORPUS SHA SPLIT SEED`.

Reports B-minus-A accuracy and a paired row bootstrap percentile interval using exactly 2,000 resamples, deterministic SplitMix64 and bounded rejection sampling. Nearest-rank 2.5% and 97.5% quantiles use sorted indices 49 and 1949. Maximum work is 8,192,000 draws before rare rejection retries. This resamples rows, **not projects**: correlated families invalidate an independent-sample interpretation. The interval is an exploratory diagnostic and does not correct repeated selection.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
