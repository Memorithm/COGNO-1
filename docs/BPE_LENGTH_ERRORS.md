# bpe_length_errors

Run `cargo run --release -p cogno-model --example bpe_length_errors -- CHECKPOINT SHA CORPUS SHA SPLIT`.

Fixed token-length strata: 0–32, 33–64, 65–128, 129–256, 257–512, 513+. Counts required BPE tokens including framing **before** the context limit, then separately reports accepted, correct, and capacity-refused rows. No silent truncation; empty accepted bins report `NA`. Token counts depend on this checkpoint's tokenizer; cross-model strata need not represent the same sources.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
