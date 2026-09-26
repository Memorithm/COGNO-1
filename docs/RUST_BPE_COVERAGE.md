# Train-only BPE context coverage

```sh
cargo +1.97.1 run -p cogno-model --example rust_bpe_coverage -- /tmp/cogno-rust-admitted/corpus.crust 766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee 384 128
cargo +1.97.1 test -p cogno-model --example rust_bpe_coverage
```

Arguments are the full admitted corpus, independently expected SHA-256, requested
vocabulary, and context. Optionally append a test-only corpus and its expected
SHA-256. Both inputs use the bounded CRUST001 readers with explicit partition
requirements. Only the full corpus's `train` rows train the tokenizer; optional
external rows cannot influence its vocabulary. The tokenizer is trained anew,
so this tool describes that configuration, not an arbitrary saved checkpoint.

CSV contains training and reporting corpus identities, tokenizer fingerprint,
actual vocabulary, context, source identity, split, source bytes, required framed
tokens, admission status and label. Required counts include BOS/EOS and remain
available for context-refused records. No source is truncated. All reports are
validated before CSV output begins; invalid arguments/artifacts fail nonzero.

Measured on 2026-09-26, target vocabulary 384 and context 128:

| Split | Accepted / total | Maximum required tokens |
|---|---:|---:|
| train | 16/16 | 20 |
| validation | 4/4 | 57 |
| test | 4/4 | 48 |

This diagnostic has no context refusals at 128. Increasing context alone cannot
recover a refused sample in this particular corpus. Acceptance is not predictive
accuracy, and this already-observed small corpus is not a new holdout.

The command can be used in SciRust dataset pipelines emitting CRUST001 without
adding an external service or training on evaluation records.
