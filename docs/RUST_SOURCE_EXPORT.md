# Source export for compiler reproduction
`cargo run -p cogno-model --example rust_source_export -- CORPUS SHA NEW_DIRECTORY`

After full hash-checked corpus admission, exclusively creates the output directory and exports original source bytes to lowercase `<source_sha256>.rs` filenames. A SHA-sorted manifest records source identity, split, project, label and byte count. Each file is flushed with `sync_all`; `COMPLETE`, written last, binds the original corpus digest and manifest digest. No compiler or source is executed.

An existing destination is refused, including symlinks. On failure the incomplete fresh directory remains for diagnosis, without a successful completion marker. The directory is not an atomic filesystem transaction and `COMPLETE` is a completion convention, not authentication: recheck digests before consuming files. Use a trusted parent directory. This enables independent compiler reproduction in SciRust or RemoteOps without Python or a network service.
