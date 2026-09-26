# Native train-only BPE vocabulary utilization

```sh
cargo +1.97.1 run -p cogno-model --example rust_bpe_utilization -- /tmp/cogno-rust-admitted/corpus.crust 766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee 384 128 > utilization.csv
cargo +1.97.1 test -p cogno-model --example rust_bpe_utilization
```

The bounded SHA-verified full CRUST001 corpus supplies only training records to
BPE learning. CSV lists every byte/merge token's emitted occurrences in each
split, including unused tokens, alongside corpus/tokenizer identities and
context. BOS/EOS/SEP are excluded. Stderr reports accepted/refused records,
byte and merge occurrences, active merge types, and occurrences of tokens not
emitted by any context-admitted training record. No model is trained or changed.

**A token unseen in admitted training is not an unknown token.** Byte fallback
makes arbitrary admitted bytes representable. A byte can occur inside a training
merge without its standalone token ever being emitted. Intermediate merges may
also be replaced by later merges, so not all learned merges appear in final
sequences. The reported counts describe final token IDs, not source characters
or how many merge operations occurred.

Context-refused records contribute no token counts and are explicitly counted.
If any training record is refused, “unseen” is relative to admitted training
records only; the tokenizer nevertheless learned from every train record. No
truncation or hidden context widening occurs. For complete-corpus length/refusal
analysis use a context-coverage report before interpreting utilization.

Measured on 2026-09-26, vocabulary 384 (125 merges), context 128:

| Split | Accepted/refused | Byte occurrences | Merge occurrences | Active merge types | Unseen-in-train occurrences |
|---|---:|---:|---:|---:|---:|
| train | 16/0 | 24 | 151 | 68 | 0 |
| validation | 4/0 | 86 | 61 | 19 | 105 |
| test | 4/0 | 106 | 37 | 16 | 105 |

Thus 105 of 147 validation payload tokens and 105 of 143 test payload tokens
use IDs absent from final training sequences in this tiny diagnostic. This is a
concrete coverage discrepancy to investigate when expanding the corpus; it does
not establish the cause of classifier failure, or demonstrate improved Rust
capability. These are already-observed diagnostic partitions.

The deterministic CSV can be reused by SciRust corpus tooling without an
external service dependency. Token IDs are meaningful only with their specific
tokenizer fingerprint.
