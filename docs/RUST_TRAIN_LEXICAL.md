# Training-only byte lexical diagnostics
`cargo run -p cogno-model --example rust_train_lexical -- CORPUS SHA`

Emits raw byte/line counts, ASCII alphanumeric-or-underscore run count and distinct count, non-ASCII byte count, and raw brace peak/unbalanced counts for training rows only. Labels and source digests allow auditing whether superficial cues correlate with the target. A final LF terminates an existing line; it does not add an empty line.

These are byte heuristics, not a Rust lexer/parser, compiler complexity estimate or competence score. Numbers count as word runs; comments and string literals count; braces inside strings affect raw depth. Unicode is reported as bytes, not characters. It does not inspect held-out content to select features, rank examples or modify training automatically.
