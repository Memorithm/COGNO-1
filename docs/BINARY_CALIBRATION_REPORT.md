# Frozen binary calibration report

Run `cargo run --release -p cogno-model --example bpe_calibration_report -- CHECKPOINT CHECKPOINT_SHA CORPUS CORPUS_SHA test` (also accepts train or validation).
Both artifacts must match their externally supplied SHA256. The corpus uses the admitted CRUST001 full three-partition format. No training, probability fitting, code compilation or source execution occurs. Every selected record must infer successfully; any capacity refusal or invalid probability aborts before printing a report.

The first CSV table reports mean binary negative log likelihood (natural logarithm, probability clipped to [1e-7, 1-1e-7] for this metric only), Brier score `mean((p_compile-label)^2)`, and positive-probability expected calibration error. ECE uses ten fixed equal-width bins in [0,1], left-inclusive/right-exclusive except the last includes 1. It equals `sum(bin_count / n * abs(mean_probability - positive_fraction))`. This is positive-class calibration, not confidence calibration. The second table includes every bin; empty means are `NA`.

Do not interpret a small diagnostic corpus or low ECE alone as generalization or expert Rust capability. The selected split, checkpoint and corpus hashes are in the report; calibration is descriptive and does not modify any model or baseline artifact. The CLI adds no dependency. Unit tests cover independently calculated scores, bin boundaries, empty bins, invalid and nonfinite inputs, and clipped certainty errors.
