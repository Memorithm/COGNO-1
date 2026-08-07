# Scientific taste commit interlock

`commit_taste_cycle` serializes all mutation of the persisted scientific-taste generation chain with a root-scoped `.TASTE-COMMIT.lock` file created using atomic create-new semantics.

The interlock is acquired only after all in-memory artifact digests have been validated and before reading or mutating `CURRENT`, generation directories, staging directories, or crash-recovery state.

If the interlock already exists, the operation never waits and never blindly steals it. On Linux, version-2 lock records contain the kernel boot ID, owner PID, and `/proc/<pid>/stat` start ticks. A residual lock is removed and acquisition is retried only when `/proc` proves that the recorded owner cannot still be the same process: the boot ID changed, the PID no longer exists, or the PID now has different start ticks. Malformed, legacy, inaccessible, or otherwise unverifiable locks remain fail-closed as `TasteCycleError::CommitInterlockHeld`.

On non-Linux targets, automatic stale-lock recovery remains disabled because the current implementation deliberately avoids platform-specific FFI for owner-liveness proofs. Normal returns release the interlock through RAII on every supported target.

COGNO-1 now targets Rust 1.97.1 and edition 2024. Although this toolchain includes newer standard-library locking primitives, the persisted scientific-taste lock retains its explicit owner identity and conservative recovery protocol because crash recovery requires stronger persisted ownership evidence than an advisory process-local file lock alone provides.

This design prevents concurrent writers from both validating the same predecessor and racing to publish different successor generations or to rewrite `.CURRENT.tmp` concurrently, while allowing provably stale Linux locks to recover without speculative lock stealing.
