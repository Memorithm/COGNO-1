# Stale interlock safety policy

A stale `.TASTE-COMMIT.lock` is intentionally not removed automatically. Automatic lock stealing would require a trustworthy, portable proof that the original owner can no longer mutate the persistence root. The Rust standard library does not provide such a proof across supported platforms.

Therefore a stale lock causes future commits and crash recovery attempts to fail closed. Administrative recovery must first validate the persisted generation chain and transaction state before removing the stale lock.
