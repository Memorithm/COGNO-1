# Explicit contrast-pair manifest admission
`cargo run -p cogno-model --example rust_pair_manifest -- CORPUS SHA MANIFEST MANIFEST_SHA`

Manifest has no header: `pair_id<TAB>family<TAB>fail_source_sha256<TAB>pass_source_sha256`, terminated by LF. Both files require independent expected SHA-256 values. Every corpus source must occur exactly once, each pair must share project and split and have labels 0 then 1, and each declared family must remain within one split. IDs are unique, ASCII path-like names; manifest is bounded to 256 KiB. Validated output adds the split.

Families are declared by the curator, never inferred from filenames or source syntax. Structural acceptance does not prove a pair differs in exactly one semantic feature; it makes subsequent contrast evaluation auditable. No compiler invocation or label inference occurs.
