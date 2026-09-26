# Fixed order/context ablation — 2026-09-26

Protocol and trainer were published before execution in PR #117, commit
`72cc528c3f6950d288635e5e41e251504e9b0976`. See
[protocol](../../docs/RUST_TRAINING_ABLATION.md). Rust 1.97.1, local CPU;
no Thor or remote training was used. All twelve checkpoint identities and
288 diagnostic predictions are retained; binaries are reproducible, not committed.

| Context | Order | Mean validation NLL | Parameters |
|---|---|---:|---:|
| 128 | ordered | 0.742285408 | 4326 |
| 128 | shuffled | 0.750157769 | 4326 |
| 256 | ordered | 0.664080295 | 5350 |
| 256 | shuffled | 0.667807191 | 5350 |

The predeclared validation-only rule selected **256 / ordered**, across all
three seeds. Its diagnostic validation accuracy was 3/4, 3/4, 2/4; its diagnostic
test accuracy was 3/4, 1/4, 2/4 (mean 50%, identical to baseline mean).
Context changes parameter count and initialization offsets, so the observed
validation difference cannot be attributed solely to added context capacity.
Shuffling did not improve validation NLL at either context.

The selected configuration was then run, without training, on the already
observed [external panel](../external-rust-panel/README.md). This is regression
data, not an untouched holdout. All eight sources now fit; the formerly refused
source requires 158 tokens. Confusion matrices use compile success as positive.

| Seed | Accepted | Correct | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 2/8 | 0 | 5 | 1 | 2 |
| 7 | 8/8 | 1/8 | 1 | 4 | 3 | 0 |
| 42 | 8/8 | 5/8 | 5 | 0 | 3 | 0 |

The fixed all-negative majority baseline scores 5/8 on the full panel. These
candidates do not beat it. The previous 128-token checkpoints scored 5/7 with
one refusal, so denominators differ. On the seven mutually accepted sources,
the selected candidates score 2/7, 1/7 and 5/7. **Coverage improved; predictive
quality did not demonstrate an improvement. No candidate is promoted.**

The 128/ordered checkpoint hashes reproduce all three original baseline hashes
exactly. Existing baseline artifacts are unchanged. All data are tiny and
previously examined; expert Rust capability, code generation, financial skill
and execution speed are not established by this experiment.
