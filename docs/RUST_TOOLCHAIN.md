# Rust toolchain policy

COGNO-1 tracks the latest stable Rust release on `main`.

Current baseline:

- Rust: **1.97.1**
- Cargo: toolchain bundled with Rust 1.97.1
- Language edition: **2024**
- Rustfmt style edition: **2021** during the compiler migration
- Workspace `rust-version`: **1.97.1**

The toolchain is pinned in `rust-toolchain.toml`. GitHub Actions installs the same exact release for formatting, Clippy, tests, documentation, frozen release builds, and dependency-policy checks. CI also verifies that the pinned channel, workspace `rust-version`, and language edition cannot silently drift apart.

The language edition and rustfmt style edition are deliberately decoupled for this migration. Rust 2024 semantics are enabled immediately, while `rustfmt.toml` retains style edition 2021 so the compiler/toolchain upgrade does not create an unrelated repository-wide formatting rewrite. A later dedicated change may move formatting to style edition 2024 and apply the resulting mechanical diff.

When a newer stable Rust release is adopted, update `rust-toolchain.toml`, `[workspace.package].rust-version`, every explicit CI toolchain pin, and this document in the same pull request. The migration is accepted only after all locked/frozen gates pass with `RUSTFLAGS=-D warnings`.

COGNO-1 does not claim compatibility with older Rust compilers once `main` advances its baseline. Older release branches may retain their historical compiler requirement.
