# Per-source sparse token activation matrix
`cargo run -p cogno-model --example rust_source_tokens -- CORPUS SHA VOCAB CONTEXT`

Learns BPE only from training sources in the hash-checked corpus, then emits one CSV row per source/token pair with occurrence count and whether the token is unseen in context-admitted training rows. Explicit refusal rows retain source identity; refused sources contribute no partial activations. Framing tokens are excluded. Corpus and tokenizer SHA values go to stderr.

This complements the aggregate utilization report by identifying which sources activate each token and where held-out novelty occurs. It is not a learned neural activation or an explanation of predictions. Training sources beyond context still participate in BPE vocabulary learning, but not in the admitted-training activation reference; the column name makes this distinction explicit.
