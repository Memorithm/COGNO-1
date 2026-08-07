# Rust 1.97.1 migration

COGNO-1 moved its development baseline from Rust 1.75 / edition 2021 to Rust 1.97.1 / edition 2024.

The migration is intentionally explicit and reproducible:

- `rust-toolchain.toml` pins Rust 1.97.1;
- `[workspace.package]` requires `rust-version = "1.97.1"`;
- the workspace uses edition 2024;
- every GitHub Actions Rust job installs Rust 1.97.1 explicitly;
- CI verifies the toolchain pin, workspace compiler requirement, and edition before accepting dependency-policy checks;
- existing `Cargo.lock`, `--locked`, and `--frozen` supply-chain gates are preserved.

Rust 1.97.1 is the current stable point release as of 2026-08-08. Future stable upgrades should update all toolchain declarations atomically in one reviewed change.
