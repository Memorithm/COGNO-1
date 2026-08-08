# Native model-persistence interlock

Reviewed neural model generations are serialized at the public mutation boundary by a root-scoped OS file lock on `.MODEL-COMMIT.lock`.

## Invariants

- The raw persistence module is crate-private; external callers cannot bypass the interlock by importing its commit function directly.
- `commit_reviewed_model_generation` opens the persistent lock file read/write and calls `File::try_lock` before predecessor replay, staging, generation publication, crash recovery, or `MODEL_CURRENT` mutation.
- Contention never waits. `TryLockError::WouldBlock` is surfaced fail-closed as `ModelPersistenceError::Io` with `ErrorKind::WouldBlock`.
- The lock handle remains alive for the complete transaction. The OS releases ownership when the handle/process closes, so a process crash does not leave a logically owned stale lock behind.
- The lock file may remain on disk permanently; its pathname is only a rendezvous object and its bytes carry no authority.
- Loading and replaying persisted model generations remains read-only and does not require the exclusive mutation lock.

This interlock does not grant model authority, activate Meta, install weights into a live runtime, enable tools, or mutate policy.

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

## Durability state after this phase

The model-generation path now covers both inter-process serialization and the narrow pre-`MODEL_CURRENT` crash window. Remaining filesystem hardening is separate: symlink/unexpected-object rejection and broader TOCTOU-resistant pathname handling should be addressed without weakening the current identity and authority checks.
