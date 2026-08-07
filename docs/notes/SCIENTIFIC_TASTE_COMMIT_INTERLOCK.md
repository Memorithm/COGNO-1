# Scientific taste commit interlock

`commit_taste_cycle` serializes all mutation of the persisted scientific-taste generation chain with a root-scoped `.TASTE-COMMIT.lock` file created using atomic create-new semantics.

The interlock is acquired only after all in-memory artifact digests have been validated and before reading or mutating `CURRENT`, generation directories, staging directories, or crash-recovery state.

If the interlock already exists, the operation fails closed with `TasteCycleError::CommitInterlockHeld`; it does not wait, steal, truncate, or reinterpret the lock. Normal returns release the interlock through RAII. A process crash can intentionally leave a stale lock: automatic stale-lock stealing is not permitted because process liveness and lock ownership cannot be proven portably with the Rust standard library alone. Recovery of a stale interlock therefore remains an explicit administrative operation and must be preceded by persisted-chain verification.

This design prevents concurrent writers from both validating the same predecessor and racing to publish different successor generations or to rewrite `.CURRENT.tmp` concurrently.
