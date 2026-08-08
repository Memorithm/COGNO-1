# Controlled neural model generation persistence

COGNO-1 persists reviewed neural candidates in a namespace that is independent from Scientific Taste state.

## Authority boundary

Persistence requires both:

1. an `EligibleMetaModelReview`, minted only after held-out, anti-poisoning and artifact verification;
2. an explicit `HostModelPromotionAttestation`.

Persisting a candidate does **not** activate Meta, install weights into a running process, enable tools, change policy, or authorize network/filesystem/kernel capabilities for model code.

## On-disk namespace

A model persistence root contains:

- `MODEL_CURRENT` — strict canonical selected generation number;
- `model-generation-N/model.bin` — canonical neural artifact;
- `model-generation-N/model.manifest` — fixed-width external `ModelManifest` encoding;
- `model-generation-N/generation.manifest` — immutable hash-chain manifest.

Staging uses `.model-stage-N` and is renamed atomically before `MODEL_CURRENT` advances.

## Generation chain

Each `ModelGenerationManifest` binds:

- monotonically increasing generation number;
- SHA-256 of the previous generation manifest;
- exact SHA-256 of the canonical neural artifact;
- SHA-256 of the persisted external model manifest;
- held-out validation accuracy in basis points;
- held-out test accuracy in basis points.

Generation 1 must link to `MODEL_GENESIS_DIGEST`. Every successor must extend the exact generation selected by `MODEL_CURRENT`.

## Commit ordering

The commit path is fail-closed:

1. re-verify the reviewed candidate with the hostile artifact loader;
2. replay and verify the persisted predecessor chain;
3. build the next immutable generation manifest;
4. create a fresh staging directory;
5. write and `fsync` the model artifact, model manifest, and generation manifest;
6. `fsync` the staging directory;
7. atomically rename staging to `model-generation-N`;
8. `fsync` the persistence root;
9. write and `fsync` `.MODEL_CURRENT.tmp`;
10. atomically rename it to `MODEL_CURRENT` and `fsync` the root again.

No live model is replaced by this operation.

## Restart replay

`load_persisted_model_generation_selection` reconstructs the chain from generation 1 through `MODEL_CURRENT`. For every generation it verifies:

- strict fixed-width generation-manifest decoding;
- predecessor hash linkage;
- SHA-256 of `model.manifest`;
- exact agreement between generation manifest and `ModelManifest.weights_hash`;
- bounded model artifact size;
- SHA-256 of the full artifact;
- full hostile artifact decoding, dimensions and finite weights.

Any missing, truncated, forked, tampered, oversized or malformed state fails closed.

## Deliberately deferred hardening

This phase establishes the canonical model-generation format, atomic commit and verified replay. Inter-process commit serialization and narrow crash-window recovery will be added as separate hardening phases, mirroring the already-audited Scientific Taste progression rather than combining authority, format and concurrency changes in one patch.
