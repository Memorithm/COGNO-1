# Rust toolchain policy

COGNO-1 tracks the latest stable Rust release on `main`.

Current baseline:

- Rust: **1.97.1**
- Cargo: toolchain bundled with Rust 1.97.1
- Edition: **2024**
- Workspace `rust-version`: **1.97.1**

The toolchain is pinned in `rust-toolchain.toml`. GitHub Actions installs the same exact release for formatting, Clippy, tests, documentation, frozen release builds, and dependency-policy checks. CI also verifies that the pinned channel, workspace `rust-version`, and edition cannot silently drift apart.

When a newer stable Rust release is adopted, update `rust-toolchain.toml`, `[workspace.package].rust-version`, every explicit CI toolchain pin, and this document in the same pull request. The migration is accepted only after all locked/frozen gates pass with `RUSTFLAGS=-D warnings`.

COGNO-1 does not claim compatibility with older Rust compilers once `main` advances its baseline. Older release branches may retain their historical compiler requirement.
