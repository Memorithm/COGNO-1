# Cross-split byte similarity audit
`cargo run -p cogno-model --example rust_corpus_similarity -- CORPUS SHA MIN_JACCARD_PPM`

Hash-checks the complete corpus and emits cross-split pairs whose set Jaccard similarity on whitespace-removed byte trigrams meets the inclusive integer threshold (0 to 1,000,000 ppm). Output includes exact intersection/union counts. At most 512 rows and 2048 normalized bytes per source; oversized input is rejected, never truncated. Empty/short normalized sources use one whole-source gram.

This is a bounded diagnostic, not Rust parsing or semantic duplicate detection. Whitespace inside strings is removed too: inspect flagged original sources before any decision. No source, label or split is changed and no automatic decontamination is claimed. Especially useful when assembling compiler curricula for COGNO and SciRust.
