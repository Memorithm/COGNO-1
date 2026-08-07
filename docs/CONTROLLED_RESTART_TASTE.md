# Controlled restart scientific taste boundary

This phase consumes the deterministic restart-review manifest produced after campaign review and compares it against an independently verified `VerifiedTasteProfile`.

The restart gate is fail-closed. It requires:

- a valid restart manifest;
- canonical strictly increasing preference identifiers;
- an exact preference count match;
- an exact preference identifier match;
- exact confidence agreement between reviewed and verified state.

The result is `ControlledRestartTasteProfile`, whose authority is explicitly `RestartInitializationOnly`. The wrapper is consuming when converted back into a `VerifiedTasteProfile`, preventing safe Rust code from reusing the same restart seal twice.

This phase does not authorize hot-swapping, live replacement, tool execution, policy elevation, or kernel authority. Runtime replacement remains forbidden by the existing one-time installation gate.
