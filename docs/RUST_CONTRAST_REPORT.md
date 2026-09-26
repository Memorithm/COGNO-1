# Explicit Rust contrast-pair report

`cargo run --release -p cogno-model --example bpe_contrast_report -- CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA test PAIRS_TSV PAIRS_SHA`

The full admitted CRUST001 corpus, frozen binary BPE checkpoint and pair manifest are independently SHA256 checked. A manifest has no header: each line is the lowercase source SHA256 of a compile-pass member, a TAB, then the source SHA256 of its compile-fail partner. It is bounded to 64 KiB / 256 pairs; each source may occur once only. All members must be present in the requested split (train, validation or test), with labels 1 and 0 respectively. Select and freeze the manifest before inspecting model probabilities. No automatic pair mining or pair selection is performed. Membership alone does not establish that two programs differ by one semantic intervention: human review remains necessary.

For every specified pair the CLI reports joint correctness (`p_pass > .5` and `p_fail <= .5`), positive separation (`p_pass > p_fail`) and the signed probability margin. The summary preserves counts and mean margin across all specified pairs. Positive separation can occur when both predictions are fail: it must not be represented as joint correctness. Ties predict fail and are not positive separation.

Every selected-split record is inferred before output; a capacity error or invalid probability aborts the complete report, never silently changing a denominator. No training, Rust compilation or execution occurs. Original baseline artifacts remain unchanged. This diagnostic does not establish expert Rust ability or generalization. Unit tests cover strict bounds, duplicate membership, labels, nonfinite probabilities, ties, and the distinction between ranking and joint correctness.
