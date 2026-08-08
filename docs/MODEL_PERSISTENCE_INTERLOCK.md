# Native model-persistence interlock

Reviewed neural model generations are serialized at the public mutation boundary by a root-scoped OS file lock on `.MODEL-COMMIT.lock`.

## Invariants

- The raw persistence module is crate-private; external callers cannot bypass the interlock by importing its commit function directly.
- `commit_reviewed_model_generation` validates the persistence root and lock object, opens the persistent lock file read/write, calls `File::try_lock`, then re-validates the reserved filesystem namespace while holding the exclusive mutation lock.
- Contention never waits. `TryLockError::WouldBlock` is surfaced fail-closed as `ModelPersistenceError::Io` with `ErrorKind::WouldBlock`.
- The lock handle remains alive for the complete transaction. The OS releases ownership when the handle/process closes, so a process crash does not leave a logically owned stale lock behind.
- The lock file may remain on disk permanently; its pathname is only a rendezvous object and its bytes carry no authority.
- Loading and replaying persisted model generations remains read-only and does not require the exclusive mutation lock, but it performs the same reserved-object preflight before reading selected state.

This interlock does not grant model authority, activate Meta, install weights into a live runtime, enable tools, or mutate policy.

## Filesystem object-type hardening

The public commit and replay paths inspect reserved objects with `symlink_metadata` before ordinary file reads or opens may follow pathnames.

The following invariants are enforced fail-closed:

- the persistence root itself must be a real directory, not a symbolic link or another object type;
- `.MODEL-COMMIT.lock`, `MODEL_CURRENT`, and `.MODEL_CURRENT.tmp`, when present, must be real regular files;
- every `model-generation-N` entry must use a canonical decimal generation in `1..=4096` and must be a real directory;
- each finalized generation must contain real regular `model.bin`, `model.manifest`, and `generation.manifest` files;
- every `.model-stage-N` entry must use a canonical bounded generation and must be a real directory;
- symbolic links and unexpected object types in any of those reserved positions are rejected before model artifact loading or persistence mutation proceeds.

Unknown unrelated root entries are ignored; malformed names that enter the reserved model-generation or staging namespaces are rejected.

## Crash recovery after generation publication

A process may crash after `model-generation-N/` has been atomically renamed into place but before `MODEL_CURRENT` advances. A retry under the native interlock may select that already-published generation only when all of the following are true:

1. `MODEL_CURRENT` does not already select generation `N`;
2. generation 1 has no prior selected history, or generation `N > 1` extends exactly the generation currently selected by `MODEL_CURRENT`;
3. the reviewed candidate still passes the hostile neural artifact loader;
4. the persisted `generation.manifest` is byte-for-byte identical to the manifest reconstructed from the reviewed candidate, its held-out metrics, and the verified predecessor digest;
5. the persisted `model.manifest` is byte-for-byte identical to the canonical external model manifest reconstructed from the reviewed candidate;
6. the persisted `model.bin` is byte-for-byte identical to the reviewed candidate artifact;
7. only after all identity checks succeed may a residual `.MODEL_CURRENT.tmp` be replaced and `MODEL_CURRENT` advance atomically.

Recovery reads are bounded by the exact reviewed artifact length or fixed manifest width. A foreign, truncated, oversized, tampered, forked, or metric-mismatched generation is never selected.

## Remaining pathname race boundary

This phase rejects pre-existing symlinks and unexpected filesystem object types using safe Rust `std` APIs. It does **not** claim complete resistance to a hostile process that can rewrite path components concurrently after preflight. Portable safe `std` does not currently expose a directory-handle-relative `openat2`-style primitive with no-follow semantics for the complete tree.

COGNO-owned writers are serialized by the native OS interlock. A deployment that grants an untrusted external principal concurrent write access to the model persistence root therefore remains outside the supported trust boundary. Any future stronger pathname-race hardening must preserve `#![forbid(unsafe_code)]` and the existing artifact, generation-chain, authority, and crash-recovery checks.
