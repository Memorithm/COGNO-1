# bpe_paired_disagreement

Run `cargo run --release -p cogno-model --example bpe_paired_disagreement -- A SHA B SHA CORPUS SHA SPLIT`.

Evaluates both frozen models against exactly the same source records, retaining the four paired correctness cells. Reports the two-sided exact binomial McNemar diagnostic `min(1, 2 P[Binomial(b+c, 0.5) <= min(b,c)])`, computed in log space. No independence assumption is justified for related code examples: the p-value is descriptive, not evidence of generalization, especially after repeated model selection. Models may have different tokenizers; both must accept every selected source.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
