# Scientific taste commit interlock

`commit_taste_cycle` serializes all mutation of the persisted scientific-taste generation chain with a root-scoped `.TASTE-COMMIT.lock` file created using atomic create-new semantics.

The interlock is acquired only after all in-memory artifact digests have been validated and before reading or mutating `CURRENT`, generation directories, staging directories, or crash-recovery state.

If the interlock already exists, the operation never waits and never blindly steals it. On Linux, version-2 lock records contain the kernel boot ID, owner PID, and `/proc/<pid>/stat` start ticks. A residual lock is removed and acquisition is retried only when `/proc` proves that the recorded owner cannot still be the same process: the boot ID changed, the PID no longer exists, or the PID now has different start ticks. Malformed, legacy, inaccessible, or otherwise unverifiable locks remain fail-closed as `TasteCycleError::CommitInterlockHeld`.

On non-Linux targets, automatic stale-lock recovery is disabled because the Rust 1.75 standard library cannot provide a portable owner-liveness proof. Normal returns release the interlock through RAII on every supported target.

This design prevents concurrent writers from both validating the same predecessor and racing to publish different successor generations or to rewrite `.CURRENT.tmp` concurrently, while allowing provably stale Linux locks to recover without weakening the repository MSRV.
