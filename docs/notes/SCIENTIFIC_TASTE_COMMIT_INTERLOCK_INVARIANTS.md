# Commit interlock invariants

1. Artifact digest validation may run before interlock acquisition because it does not access persisted mutable state.
2. The interlock is acquired before reading `CURRENT`, verifying the persisted predecessor, inspecting recovery state, creating staging directories, renaming a generation, or writing `.CURRENT.tmp`.
3. A second writer never waits or steals the interlock; it fails closed.
4. Normal success and ordinary error returns release the interlock through RAII.
5. Process-crash residue is treated as a stale interlock and blocks mutation until explicit verified recovery.
