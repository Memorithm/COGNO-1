# bpe_project_confusion

Run `cargo run --release -p cogno-model --example bpe_project_confusion -- CHECKPOINT SHA CORPUS SHA SPLIT`.

Produces deterministic TSV confusion counts per admitted project/family, with absent-class recall reported as `NA`. Classification ties choose class zero. Projects are not averaged as though they were equally sized samples; raw support is retained. This exposes aggregate scores that conceal failure on a particular family.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
