# Length-bucket training order export
`cargo run -p cogno-model --example rust_length_order -- CORPUS SHA BUCKET_BYTES SEED EPOCH`

Exports every training row exactly once. Start with the existing deterministic bounded epoch permutation, then stable-sort by ascending byte-length bucket `(source_bytes - 1) / BUCKET_BYTES`; within-bucket permutation is preserved. Width is 1..16384 bytes. Held-out rows never enter the order. CSV records original zero-based corpus index and source SHA so an external trainer can verify mapping.

Byte length is not token count, execution cost or Rust difficulty. This changes sample ordering and may change training outcomes; it is an explicit experimental curriculum, never an implicit replacement of the baseline. No speedup or quality improvement is claimed. SciRust trainers may consume the same auditable index order.
