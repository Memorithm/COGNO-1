# bpe_inference_timing

Run `cargo run --release -p cogno-model --example bpe_inference_timing -- CHECKPOINT SHA CORPUS SHA SPLIT ITERATIONS`.

Single calling thread, 1–100 iterations and at most 32,768 timed calls. One untimed pass warms the model and provides a bit-exact probability reference checked after every timed call. Times include tokenization, classification, allocation and probability validation; checkpoint/corpus loading and parity comparison are excluded. Nearest-rank p50/p95/p99 are over all calls, so source mix affects results. No thread affinity, OS isolation or CPU-frequency guarantee is implied. Record machine/compiler and use release builds before comparing runs. Timing does not establish throughput under concurrency.

Research diagnostic only. No threshold fitting, checkpoint mutation, promotion, or claim of Rust expertise. Hashes must come from an independent trusted inventory. `external` accepts only a test-only CRUST001 corpus; other splits require the full admitted corpus. Capacity failures abort before any report is printed unless explicitly counted. All paths are explicit caller inputs, with bounded reads; no input code is executed.
