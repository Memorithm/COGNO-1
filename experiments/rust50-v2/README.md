# Rust50 v2: completed CPU qualification

The fixed protocol was published in COGNO-1 #193 before integrated experiments.
The batch consists of COGNO-1 #193–241 and RemoteOps #50: 50 pull requests.
Incremental component reviews cover native training/gradients, tokenizer,
optimizer/losses, evaluation, and external source admission. Final integrated
evidence is attached to COGNO-1 #235.

## Reproduction

Thor execution **36296752085**, job **108556928364**, completed successfully.
It used COGNO commit `3624917ae1f98843adeb8920799b1c3487b5248e`, tree
`9074a29c5857757808f709bdf8954db22b9acaac`, and Rust 1.97.1. RemoteOps #50
merged at `d5f9a2d5bf71d97b3449db5cabdfeb230a980cd8`. Full weights, source,
compiler reports and logs remain in
`/srv/cogno-rust50-v2/run-36296752085-1` on Thor.

The final local run used the same implementation; a Rustdoc comment was corrected
while it ran. Its recorded source tree differs only by that documentation line.
Both machines produced exactly the same:

- 12 historical checkpoint identities and 3,456 predictions;
- six configurable-trainer checkpoints (two arms × three seeds × 24 epochs);
- complete configurable training bundle, epoch journals and predictions;
- validation-only selection and selected-arm test bundle;
- admitted external corpus from 18 sources in three actual upstream repositories.

The new configurable trainer performs 27,648 updates. Verified completed-run
replay produces an identical bundle. This is bounded CPU qualification, not
large-scale GPU training. The 288 synthetic examples and existing test split were
already observed. The external panel is curated and observed, and qualifies the
acquisition pipeline rather than establishing expert Rust performance.

`cycle_mix` remains selected. Its three test scores are 29/48, 28/48 and 30/48
(60.42% mean). They repeat the prior result: no new accuracy improvement is claimed.
The trainable system remains a compiler-acceptance classifier, not a Rust code
generator. No model is promoted.

## Measured implementation performance on Thor

| Operation and workload | Reference median | New path median | Interpretation |
|---|---:|---:|---|
| Sequence loss + backward, vocab384 / tokens192 / embed8 / hidden16 | 4.065 ms | 0.090 ms | 45.3× for this shape; exact gradients and 12 updates |
| BPE fit, three Rust files / 28,041 bytes / vocab384 | 130.799 ms | 7.623 ms | 17.2× against the independent full-recount/rebuild oracle |
| Encode three 128-byte prefixes, 100 passes | 3.193 ms | 1.721 ms | Heap faster for these admitted prefixes |
| Encode three full files, 100 passes | 217.018 ms | 315.132 ms | Heap slower; all three exceed context and are refused |
| AdamW, 16,384 elements / 256 updates | 52.744 ms | 52.662 ms | Approximately equal; no meaningful speedup claim |

Raw samples and input hashes are retained in `thor-*.txt` and `local-*.txt`.
The sequence probe currently retains seven-round medians, not each individual
round. These are shared-machine microbenchmarks, not universal speed guarantees
or end-to-end trading latency. The default tokenizer encoder remains rank-scan.
Gather/context/mask training paths are explicit options; legacy artifacts do not
store contextual strength or masks, so callers must retain those experiment
settings. Production runtime authority is unchanged.

## Evidence and checks

`trainer/` retains every non-weight training artifact, including all six seeds'
train/validation predictions, journals and completion manifests. Checkpoint
bytes are reproducibly regenerated and remain on Thor; their identities appear
in both summaries. `selected-test/` retains only the frozen selected arm's three
test predictions. The historical four-arm predictions are retained separately
in `experiments/domain-holdout-v1`.

Local integration passed 99 Python tests before the three final evidence tests
were added; those three also pass. Model/SciRust all-target tests passed 392 test
executions, with strict Clippy, formatting and documentation checks. GitHub's
workspace and offline checks cover the runtime in its supported PID environment.
An initial local Python invocation lacked `rustup` on PATH; it was corrected and
the entire 99-test suite rerun successfully. Rustdoc rejected a literal range
link; the documentation was fixed and rechecked.

The evidence verifier rejects changed predictions, completion manifests,
benchmarks or unsuccessful/mismatched Thor execution. Integration CI repeats
all 18 training runs and compares the full deterministic bundle against these
saved results. To reproduce:

```sh
cargo build --release --locked -p cogno-model --example rust_project_split --example bpe_curriculum_probe --example rust_train_v2 --example bpe_tokenizer_bench
cargo build --release --locked -p cogno-scirust --example sequence_paths_probe --example optimizer_training_probe
python3 scripts/qualify_rust50_v2.py /tmp/new-rust50-run --binaries target/release/examples --rustc "$(rustup which rustc)"
python3 scripts/verify_rust50_v2_evidence.py /tmp/new-rust50-run
```

## Rust50 v3 dense/gather comparison on Thor

RemoteOps run `36303253691` (job `108574837971`) qualified the explicit v3 dense/gather full-training comparison at COGNO commit `80cf940b1165b16a721185d27b63a55e8e6b4a7c`, tree `ffdd8918dacd80f0412099ef5bd3d764f5c939ee`, using RemoteOps commit `4ca2f5d3e8ca38b46da05f3d4820cc0290cdc1dc` and Rust 1.97.1.

The fixed v2 protocol permits one dense/gather pair under its 100,000-update budget: 55,296 updates, 18 byte-matched run artifacts per round, and `model_promoted=false`. Thor recorded dense at 28,715,786,569 ns and gather at 4,851,739,391 ns on `aarch64`. These are one-host CPU training timings for this declared shape; they do not establish a general speedup or model-quality change.

The completion marker binds protocol, corpus and provenance hashes. The raw marker and both bundle identities are retained in `thor-v3-graph-summary.json`; failed attempts before this run were not accepted as evidence.

The committed record is checked by `scripts/verify_rust50_v3_graph_evidence.py`; the same verifier and its tamper tests run in Rust50 v2 integration CI. To validate only this record:

```sh
python3 scripts/verify_rust50_v3_graph_evidence.py
```

## Rust50 v3 gated rerun on Thor

After COGNO-1 #249, RemoteOps #55 ran the same bounded comparison as run `36303993254` (job `108576919803`) at RemoteOps commit `11549073787d47ae7f7e3a1e5fd1cb124cb2600b`. It checked the committed v3 record before compiling COGNO-1 main `baabdc226f42221c1cb64a0ccccd75afa67c8d2b`, tree `8b75bb781a1869b6177722755fda5e953cce81a6`, with Rust 1.97.1.

The gate marker and successful Thor job produced the same 55,296 updates, 18 matched files per round, and the same dense/gather bundle identities. The one-host aarch64 CPU timings were dense `28,898,623,619 ns` and gather `4,956,827,068 ns`; this rerun remains a declared shape check and makes no general speedup or model-quality claim. No GPU training or model promotion occurred.

The complete gated record is retained in `thor-v3-gated-graph-summary.json`. The verifier now checks both the original v3 evidence and this post-gate rerun, while the RemoteOps log retains the `COGNO_COMMITTED_V3_EVIDENCE_VERIFIED` marker.

## Rust50 v3 runtime-verified rerun on Thor

RemoteOps run `36327304698` (job `108642451598`) completed successfully at RemoteOps commit `8a50d46723b4566cdaf04c08d32704659f349549`, after checking the committed record and the freshly generated summary. It ran COGNO-1 commit `9989cf5edd2f9d34a037877d2af746762715a84e`, tree `702e5e90135820027b66d93caef63a1f71cbb5ef`, on Rust 1.97.1 and `aarch64`.

The runtime verifier passed before the final completion marker: 55,296 updates, 18 matched files, dense `28,714,368,040 ns`, gather `4,841,435,851 ns`, and the previously recorded bundle identities. This is a bounded one-host CPU shape check with no GPU training, no promotion, and no general speedup or model-quality claim.

The exact record is retained in `thor-v3-runtime-verified-summary.json`; the failed pre-fix run was not accepted as evidence.

The same verifier can check a fresh RemoteOps summary after the run has produced it:

```sh
python3 scripts/verify_rust50_v3_graph_evidence.py --actual /path/to/summary.json --source-commit <commit> --source-tree <tree>
```
