# Scientific taste commit interlock verification

The commit interlock is covered at both unit and integration level.

The unit tests prove that a held interlock rejects a commit before `CURRENT`, a generation directory, or a staging directory can become visible, and that ordinary fail-closed errors release the RAII interlock so a valid retry can proceed.

The integration test independently creates the root-scoped lock file and verifies that the public `commit_taste_cycle` entry point returns `TasteCycleError::CommitInterlockHeld` without mutating persisted state. After releasing the lock, the same transaction succeeds and selects generation 1.
