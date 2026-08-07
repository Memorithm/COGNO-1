# Stale interlock safety policy

A `.TASTE-COMMIT.lock` is never removed merely because it is old or because a caller asks to override it.

On Linux, version-2 locks record three owner facts: kernel boot ID, PID, and process start ticks from `/proc/<pid>/stat`. Automatic recovery is allowed only when those facts prove the recorded owner cannot still be the same process: a different boot ID, a missing PID, or different start ticks for that PID. PID reuse alone therefore cannot authorize stale-lock recovery.

Malformed locks, legacy locks without owner identity, `/proc` read failures, and every non-Linux target remain fail-closed. COGNO-1 targets Rust 1.97.1 and edition 2024, but the implementation intentionally keeps its persisted owner-identity protocol rather than weakening crash recovery to advisory file locking alone.

This policy preserves automatic recovery where owner death can be demonstrated without FFI while refusing speculative lock stealing everywhere else.
