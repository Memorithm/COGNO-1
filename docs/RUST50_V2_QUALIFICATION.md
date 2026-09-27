# Rust50 v2 qualification

This batch improves the bounded Rust compiler-acceptance research classifier and
its training tools. It does not establish code generation, expert Rust ability,
financial competence or production activation.

The protocol in `experiments/rust50-v2/protocol.json` is committed before the
new implementation's integrated experiments. It deliberately reuses the known
domain holdout for engineering regression checks. Those observations are not a
new blind benchmark and must not be used to advertise generalization gains.

Completed execution, raw measurements, cross-architecture reproduction and scope
limitations are documented in `experiments/rust50-v2/README.md`.

Qualification has five parts:

1. Preserve all 12 frozen domain-holdout checkpoint hashes and all 3,456
   predictions while improving implementation efficiency.
2. Exercise the new configurable trainer with six fixed runs: two arms, three
   seeds, 24 epochs. Fit the tokenizer only on training sources and select the
   arm using mean validation NLL. Retain every seed and checkpoint.
3. Admit a small pinned panel from at least three actual upstream repositories;
   preserve licensing and source hashes, compile metadata only, and distinguish
   compiler infrastructure failures from Rust rejection. The panel is observed
   during curation and is not an expert-level evaluation dataset.
4. Measure optimizer, encoder and tokenizer changes only after numerical and
   representation equivalence checks. Report the actual shape, hardware, raw
   timings and any regressions. No fixed speedup is assumed.
5. Run a bounded CPU qualification job on Thor through RemoteOps, with an exact
   source commit and separate run directory. Publish the actual completion and
   reproduction evidence; a queued job is not a completed experiment.

No model is promoted by this protocol. The repository's deterministic runtime
authority and existing artifact formats remain the production boundary.
