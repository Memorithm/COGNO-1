# ADR-0001: COGNO-1 Rust toolchain island

- **Status:** accepted
- **Date:** 2026-09-27
- **Issue:** COGNO-1 #91

## Context

Memorithm repositories intentionally have different Rust compatibility targets.
COGNO-1 currently declares workspace edition 2024 and rust-version = "1.97.1",
pins the same compiler in rust-toolchain.toml, and verifies the exact version in
the dependency-policy CI job. The Rust50 qualification scripts compile the
external panel with rustc 1.97.1, and the RemoteOps Thor qualification pins the
same toolchain. Saved compiler outcomes and reproducibility evidence therefore
include this toolchain identity.

The organisation's other MSRV declarations must not be raised implicitly to
accommodate COGNO-1. Conversely, lowering only COGNO-1 would require rebuilding
the pinned evidence and checking every workspace target, offline bundle and
RemoteOps qualification path.

## Decision

COGNO-1 remains an explicit Rust 1.97.1 compatibility island until a separate,
reviewed MSRV-reduction campaign proves another toolchain across the complete
workspace and evidence pipeline.

The boundary is enforced by:

- Cargo.toml rust-version = "1.97.1";
- rust-toolchain.toml channel = "1.97.1";
- CI checks that compare the compiler, manifest and toolchain file exactly;
- qualification scripts that reject a compiler with another version;
- RemoteOps jobs that invoke rustc +1.97.1 and cargo +1.97.1.

This ADR makes no claim that 1.97.1 is the smallest compiler version capable of
building the sources. It records the reproducibility and evidence boundary.

## Consequences

COGNO contributors need Rust 1.97.1 for the supported COGNO workflow. A caller
using an older organisation-wide toolchain must consume a released COGNO protocol
or artifact rather than silently compiling the workspace with a different
compiler. COGNO changes must not modify SciRust, CCOS, FLAT-ATTENTION, TDI or
other repositories' MSRV declarations as a side effect.

The current external-corpus compiler observations remain tied to rustc 1.97.1.
They are metadata-only qualification evidence and must be regenerated if the
toolchain changes.

## Required process for a future reduction

A future MSRV proposal must use a separate branch and:

1. select the candidate stable toolchain and record its exact version;
2. run a version matrix over every workspace crate, all targets, documentation,
   offline distribution and qualification scripts;
3. regenerate the external compiler observations and compare all hashes;
4. rerun the RemoteOps CPU qualification at the candidate revision;
5. update Cargo.toml, rust-toolchain.toml, CI, scripts, documentation and
   evidence together;
6. merge only when the complete evidence verifier passes.

Until those steps succeed, changing the pin is a reproducibility regression.
