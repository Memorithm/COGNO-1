# Checked corpus union
`cargo run -p cogno-model --example rust_corpus_union -- LEFT SHA RIGHT SHA NEW_OUTPUT`

Reads two independently hash-checked full CRUST001 corpora. Concatenation must pass the complete admission contract again: total bounds, exact-source duplicates, project split leakage and all six split/label combinations. No automatic deduplication or split reassignment occurs. Output uses exclusive creation and prints its SHA-256. In a failed disk write remove the incomplete output before retrying. The supplied input digests require an independent trusted inventory; hashing does not authenticate provenance. This is reusable for SciRust corpus preparation without runtime dependencies.
