# bpe_probability_saturation

Run `cargo run --release -p cogno-model --example bpe_probability_saturation -- CHECKPOINT SHA CORPUS SHA SPLIT`.

Reports fixed extreme-probability counts (`p <= 1e-6`, `p >= 1 - 1e-6`), incorrectly confident predictions, probability range, and mean binary entropy in nats. Endpoint entropy terms are zero without clipping. This diagnoses output saturation only; no claim about hidden-feature saturation or learned competence follows. Thresholds are fixed descriptive bins and are never fitted on evaluation rows.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
