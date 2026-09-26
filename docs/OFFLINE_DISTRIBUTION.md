# Autonomous source distribution

COGNO's **distribution** can carry its committed sources and all locked Cargo
dependencies, including the pinned SciRust agent protocol. This does not remove
third-party authorship, licenses, or operating-system/toolchain requirements.

## Prepare once, then transfer

Prerequisites: Git, Python 3.11.8+ (tar extraction filters and tomllib), rustup,
the installed Rust 1.97.1 toolchain and a native linker/C build toolchain.
Preparation can access the network to fetch the exact Cargo.lock dependencies:

```sh
python3 scripts/offline_bundle.py pack /absolute/new-directory/cogno-offline
```

The parent directory must exist; the final destination must not exist. Only Git
HEAD is exported: uncommitted files, local credentials, caches and external
datasets are not included. Failures leave an incomplete directory for diagnosis,
without a completed manifest. There is no overwrite or automatic cleanup.

The directory contains `source/`, vendored dependency sources and their shipped
notices, local Cargo source replacement configuration, and `manifest.json` with
the source revision, lockfile dependency identities and SHA-256 of every file.
Preserve all upstream licenses; packaging does not relicense third-party code.
The manifest is an integrity inventory, **not an authenticated signature**.

Transfer the whole directory to a compatible host, then run:

```sh
python3 /path/to/cogno-offline/source/scripts/offline_bundle.py verify /path/to/cogno-offline
```

Verification creates a disposable copy with empty Cargo home and target directory,
uses the already installed pinned compiler directly, and runs release build,
workspace tests, the existing Rust classification training/inference pilot,
and the experimental byte/BPE training comparison
with `--frozen`. Original bundle files and checkpoints remain unchanged.

Cargo offline mode blocks Cargo downloads; it is **not a network sandbox for
arbitrary build scripts or executables**. Linux CI additionally runs verification
inside an isolated network namespace. No Rust toolchain, system libraries, GPU
driver, Hugging Face corpus or expert-quality checkpoint is bundled. These must
be provisioned separately; the pilot's negative generalization result is unchanged.

## Integration sequence

1. Offline packaging and fresh-cache qualification (this slice).
2. Versioned, bounded Rust BPE integration, preserving byte-tokenizer artifacts.
   Reuse reviewed SciRust semantics with explicit upstream license/provenance;
   special IDs, pair framing, vocabulary limits and embeddings must be adapted.
3. Independent corpus admission, deduplication, project-disjoint splits and local
   compiler/test validation before scaling training.
4. Optional FLAT attention with trained projections and verified gradients;
   compare at matched budgets before promotion.
5. Optional compressed memory only after task-level evidence. Development tools
   (Forge, TDI, Verify, RemoteOps) remain outside the model's inference dependency
   graph. No remote service may be required for an activated model.

No BPE, FLAT or SLHA integration is claimed by this packaging milestone.
