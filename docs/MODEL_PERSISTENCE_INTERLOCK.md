# Native model-persistence interlock

Reviewed neural model generations are serialized at the public mutation boundary by a root-scoped OS file lock on `.MODEL-COMMIT.lock`.

## Invariants

- The raw persistence module is crate-private; external callers cannot bypass the interlock by importing its commit function directly.
- `commit_reviewed_model_generation` opens the persistent lock file read/write and calls `File::try_lock` before predecessor replay, staging, generation publication or `MODEL_CURRENT` mutation.
- Contention never waits. `TryLockError::WouldBlock` is surfaced fail-closed as `ModelPersistenceError::Io` with `ErrorKind::WouldBlock`.
- The lock handle remains alive for the complete transaction. The OS releases ownership when the handle/process closes, so a process crash does not leave a logically owned stale lock behind.
- The lock file may remain on disk permanently; its pathname is only a rendezvous object and its bytes carry no authority.
- Loading and replaying persisted model generations remains read-only and does not require the exclusive mutation lock.

This interlock does not grant model authority, activate Meta, install weights into a live runtime, enable tools, or mutate policy.

## Remaining durability work

The narrow crash window after `model-generation-N/` is renamed into place but before `MODEL_CURRENT` advances remains a separate recovery concern. Recovery must verify exact generation-manifest and artifact identity before selecting an already-published generation.
