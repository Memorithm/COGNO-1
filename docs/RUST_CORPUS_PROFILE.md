# Native admitted-corpus composition report

`rust_corpus_profile` reads a CRUST001 corpus through the bounded native reader,
checking an independently supplied SHA-256 before reporting. `full` requires all
three splits with both labels; `test-only` explicitly requires only test records.
It does not verify the truth of compiler labels or source licensing.

```sh
cargo +1.97.1 run -p cogno-model --example rust_corpus_profile -- full /tmp/cogno-rust-admitted/corpus.crust 766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee
cargo +1.97.1 test -p cogno-model --example rust_corpus_profile
```

The deterministic CSV includes corpus identity, each split's two class counts,
unique project identifiers, total source bytes, and byte-length quantiles. Median
and p95 use nearest rank (`ceil(p*n)-1`), so an even-size median is the lower
middle value. Lengths count bytes, not characters or tokenizer tokens. Project
counts refer to admitted identifiers, not independently verified repositories.

Measured on the existing diagnostic corpus on 2026-09-26:

| Split | Rows | Labels 0/1 | Projects | Bytes | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 16 | 8/8 | 8 | 759 | 28 | 51 | 60 | 60 |
| validation | 4 | 2/2 | 2 | 239 | 39 | 42 | 91 | 91 |
| test | 4 | 2/2 | 2 | 209 | 44 | 47 | 66 | 66 |

These describe a tiny already-observed diagnostic set, not Rust expertise or
representative real-world performance. This command is reusable by SciRust data
preparation pipelines accepting CRUST001, with no new runtime dependency.
