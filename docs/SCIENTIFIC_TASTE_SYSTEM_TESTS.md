# Scientific Taste System Adversarial Tests

The autonomous scientific-taste phase includes system-level fail-closed coverage for the following classes of failure:

- model self-promotion attempts remain quarantined;
- deterministic evidence is required before activation;
- contradictory evidence blocks activation;
- duplicate validation identifiers are rejected;
- artifact digest mismatch prevents a generation from becoming visible;
- committing an already-visible generation is rejected;
- forked predecessor hashes are rejected;
- rollback to an unknown generation is rejected;
- corrupted selection metadata is never treated as evidence of a valid generation;
- longitudinal generation history must be strictly monotonic;
- benchmark case identifiers must be unique;
- bounded payload and corpus limits fail closed.

The transactional generation writer remains responsible for fsync-before-visibility and atomic `CURRENT` replacement. The tests verify that failures before commit do not advance `CURRENT` or expose an incomplete generation.
