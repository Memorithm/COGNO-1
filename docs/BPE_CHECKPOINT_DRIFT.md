# bpe_checkpoint_drift

Run `cargo run --release -p cogno-model --example bpe_checkpoint_drift -- A SHA B SHA`.

Verifies complete hashes and checkpoint contents, then compares all eleven tensors by RMS difference, maximum absolute difference and cosine. Shapes, tokenizer fingerprint and candidate cap must match; initialization seed metadata may differ. Zero-vector cosine is `NA`. Arithmetic is f64 even for extreme finite f32 weights. No tensor alignment/permutation correction is attempted: distances are coordinate-wise and do not establish functional or quality differences. This tool has no corpus argument.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
