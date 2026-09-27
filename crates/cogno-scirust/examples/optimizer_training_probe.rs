//! Offline numerical/throughput probe. Synthetic data, not a Rust skill benchmark.
use cogno_scirust::losses::{weighted_mean_loss, SoftTargetCrossEntropy};
use cogno_scirust::optim::{AdamWWorkspace, GradientAccumulator, WarmupCosine};
use cogno_scirust::{AdamW, Optimizer, Shape, Tape, Tensor};
use std::time::Instant;

fn scalar(tape: &mut Tape, value: f32) -> Result<cogno_scirust::Var, Box<dyn std::error::Error>> {
    Ok(tape.variable(Tensor::try_scalar(value)?)?)
}
fn train() -> Result<(), Box<dyn std::error::Error>> {
    let objective = SoftTargetCrossEntropy::try_new(2)?;
    let schedule = WarmupCosine::try_new(0.05, 0.005, 5, 120)?;
    let mut optimizer = AdamW::try_new(0.05, 2)?;
    let mut workspace = AdamWWorkspace::try_new(2)?;
    let mut accumulator = GradientAccumulator::try_new(2)?;
    let mut parameters = vec![0.0, 0.0];
    let mut first_loss = 0.0;
    let mut last_loss = 0.0;
    for step in 0..120 {
        accumulator.clear();
        let mut mean_loss = 0.0;
        for (x, label, weight) in [(-2.0, 0, 2.0), (-1.0, 0, 1.0), (1.0, 1, 1.0), (2.0, 1, 2.0)] {
            let mut tape = Tape::new(32, 16);
            let w = scalar(&mut tape, parameters[0])?;
            let b = scalar(&mut tape, parameters[1])?;
            let wx = tape.scale(w, x)?;
            let score = tape.add(wx, b)?;
            let zero = scalar(&mut tape, 0.0)?;
            let logits = tape.stack_scalars(&[zero, score])?;
            let target = objective.smoothed_target(2, label, 0.05)?;
            let loss = objective.loss(&mut tape, logits, &target)?;
            let loss = weighted_mean_loss(&mut tape, &[loss], &[1.0])?;
            tape.backward(loss)?;
            mean_loss += f64::from(tape.value_of(loss).data[0]) * weight / 6.0;
            accumulator.add(&[tape.grad_of(w)[0], tape.grad_of(b)[0]], weight)?;
        }
        if step == 0 {
            first_loss = mean_loss;
        }
        last_loss = mean_loss;
        let mut gradient = [0.0; 2];
        accumulator.mean_into(&mut gradient)?;
        optimizer.lr = schedule.learning_rate(optimizer.state.step);
        optimizer.step_with_workspace(&mut parameters, &gradient, &mut workspace)?;
        if step == 59 {
            let wire = optimizer.checkpoint(&parameters)?;
            (optimizer, parameters) = AdamW::from_checkpoint(&wire)?;
        }
    }
    if !(last_loss < first_loss && parameters[0] > 0.0 && optimizer.state.step == 120) {
        return Err("synthetic learning/resume check failed".into());
    }
    println!(
        "synthetic_initial_loss={first_loss:.9} synthetic_final_loss={last_loss:.9} steps={}",
        optimizer.state.step
    );
    Ok(())
}
fn timed(reusable: bool) -> Result<(u128, Vec<u8>), Box<dyn std::error::Error>> {
    let n = 16_384;
    let mut optimizer = AdamW::try_new(0.003, n)?;
    let mut workspace = AdamWWorkspace::try_new(n)?;
    let mut parameters = vec![0.2; n];
    let gradient = Tensor::try_new(
        Shape::try_new(&[n])?,
        (0..n).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect(),
        n,
    )?;
    let start = Instant::now();
    for _ in 0..256 {
        if reusable {
            optimizer.step_with_workspace(&mut parameters, &gradient.data, &mut workspace)?;
        } else {
            optimizer.step(&mut parameters, &gradient.data)?;
        }
    }
    let elapsed = start.elapsed().as_nanos();
    Ok((elapsed, optimizer.checkpoint(&parameters)?))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    train()?;
    // Alternate order between paired repetitions to reduce a systematic warmup
    // advantage. Ratios are observations, never CI pass/fail criteria.
    let mut legacy = Vec::new();
    let mut reused = Vec::new();
    for repetition in 0..7 {
        let (a, b) = if repetition % 2 == 0 {
            (timed(false)?, timed(true)?)
        } else {
            let b = timed(true)?;
            (timed(false)?, b)
        };
        if a.1 != b.1 {
            return Err("optimizer bit parity failed".into());
        }
        legacy.push(a.0);
        reused.push(b.0);
        println!(
            "repeat={repetition} legacy_ns={} workspace_ns={} checkpoint_equal=true",
            a.0, b.0
        );
    }
    legacy.sort_unstable();
    reused.sort_unstable();
    println!("elements=16384 updates=256 median_legacy_ns={} median_workspace_ns={} observed_ratio={:.4}", legacy[3], reused[3], legacy[3] as f64 / reused[3] as f64);
    Ok(())
}
