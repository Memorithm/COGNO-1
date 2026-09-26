# Rust corpus pipeline diagnostic — 2026-09-26

Measured with Rust 1.97.1 release and the already examined 24-snippet synthetic
pilot. `coverage.csv` is actual output, not a forecast. Admitted CRUST001 SHA-256:
`766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee`.
No examples were rejected at context 128. BPE learns only the 16 train rows and
reproduces the prior tokenizer identity `ebcf64dd3088a68809d95f17c2839fec0430b381a80bc7ac428b3cb90f9c99cf`.

This validates JSONL admission → lossless wire → Rust parsing → coverage. It
does not validate compiler labels independently, acquire new projects, measure
inference speed or establish expert Rust performance. Byte totals include two
framing tokens per accepted record. Synthetic families are not real projects.
