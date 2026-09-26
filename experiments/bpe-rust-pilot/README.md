# Bounded BPE diagnostic — 2026-09-26

Source baseline: e208cd4f7817836e76984aa3aa70edd783b6b7f9 plus the implementation
in this change. Rust 1.97.1, release profile; no remote training or GPU used.
Reproduce with `cargo +1.97.1 run --release --locked -p cogno-model --example bpe_rust_probe`.
`predictions.csv` retains every initial/trained prediction; `tokenizer.txt`
retains the actual vocabulary, SHA-256 and complete canonical artifact in hex.
The tokenizer is learned exclusively on 16 training snippets. The 24-snippet
fixture was already inspected in earlier experiments: this is not a fresh holdout.

## Observations

Token totals include framing and count each snippet once, not once per seed.

| Partition | Byte tokens | BPE tokens | Reduction |
|---|---:|---:|---:|
| Train (16) | 791 | 207 | 73.83% |
| Validation (4) | 247 | 155 | 37.25% |
| Test (4) | 217 | 151 | 30.41% |

Accuracy after 24 epochs, seeds always ordered 1 / 7 / 42:

| Arm | Train | Validation | Test |
|---|---|---|---|
| Byte | 13/16, 16/16, 12/16 | 2/4, 4/4, 3/4 | 2/4, 1/4, 2/4 |
| BPE | 16/16, 16/16, 14/16 | 2/4, 2/4, 2/4 | 2/4, 2/4, 2/4 |

Byte vocabulary 259 / 3,326 parameters; BPE vocabulary 384 / 4,326 parameters.
The byte results reproduce the previous pilot. The BPE test accuracy is only
50% on each seed and validation worsens on two seeds. There is no demonstrated
expert Rust capability or reliable quality improvement. Widths, seeds, epochs
and learning rate match; parameter count, initialization offsets and compute do
not. No latency measurement was made, and compression is not a speed benchmark.

This is classification-only training through the cognitive numerical heads,
not code generation and not qualification of all five objectives. Production
V1–V4 loaders and activation remain unchanged; CBPE0001 is tokenizer-only.

## Checkpoint integration batch

The subsequent batch adds paired decoding, an immutable research model binding,
CBPC0001 inference checkpoints and post-training reload checks. On 2026-09-26,
Rust 1.97.1 release, all three trained checkpoints were 17,912 bytes each.
`checkpoints.tsv` records the locally measured identities; binaries are regenerated
by the probe, not checked into Git. Every prediction remained identical to the
original CSV. Successful outputs for all five signals and every weight were
exactly preserved across reload; this adds no evidence of model quality.

```sh
cargo +1.97.1 run --release --locked -p cogno-model --example bpe_rust_probe -- /tmp/new-bpe-checkpoints > /tmp/bpe-predictions.csv
python3 scripts/check_bpe_evidence.py /tmp/bpe-predictions.csv --checkpoints /tmp/new-bpe-checkpoints
```

The evidence checker requires the complete frozen row set, unchanged categorical
outputs/token/parameter counts and absolute probability drift <=1e-6. This is a
regression check on reused diagnostic data, not a general quality benchmark.
It checks inventory sizes, digests and filenames; semantic binary validation is
performed by the Rust loader during the probe. A self-supplied inventory is not
authentication. CPU/compiler changes may require investigating numerical drift;
do not replace the reference merely to turn a failed check green.

The isolated offline workflow now runs this verification too. No datasets are
downloaded at inference time. Production activation, independent project-separated
Rust evaluation, scalable corpus/tokenizer training and Thor execution remain
outside this completed integration batch. The bounded byte-preserving format and
inventory verification can be reused by other ecosystem research pilots without
adding a running SciRust or RemoteOps service to model inference.
