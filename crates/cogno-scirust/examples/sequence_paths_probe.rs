//! CPU microbenchmark, parity qualification and contextual training smoke.
//! No corpus accuracy, Rust expertise or guaranteed speedup is inferred.
use cogno_scirust::sequence::SequenceWorkspace;
use cogno_scirust::{
    SequenceClassifier, SequenceClassifierAdamW, SequenceClassifierConfig, SequenceEncoderConfig,
};
use std::{hint::black_box, time::Instant};

fn median_ns<F>(iterations: usize, mut operation: F) -> Result<u128, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..3 {
        operation()?;
    }
    let mut samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iterations {
            operation()?;
        }
        samples.push(start.elapsed().as_nanos() / iterations as u128);
    }
    samples.sort_unstable();
    Ok(samples[3])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: sequence_paths_probe [iterations:1..1000]".into());
    }
    let iterations = args
        .first()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    if !(1..=1000).contains(&iterations) {
        return Err("iterations must be 1..1000".into());
    }
    let config = SequenceClassifierConfig {
        encoder: SequenceEncoderConfig {
            vocab_size: 384,
            max_tokens: 512,
            embedding_dim: 8,
            hidden_dim: 16,
            seed: 42,
        },
        num_classes: 2,
        head_seed: 7,
    };
    let tokens: Vec<u16> = (0..192).map(|i| (i % 127 + 3) as u16).collect();
    let model = SequenceClassifier::try_new(config)?;
    if model.loss_and_gradients(&tokens, 1)? != model.loss_and_gradients_gather(&tokens, 1)? {
        return Err("dense/gather gradient parity failed".into());
    }
    let mut dense_model = model.clone();
    let mut gather_model = model.clone();
    let mut dense_optimizer = SequenceClassifierAdamW::try_new(0.003, &dense_model)?;
    let mut gather_optimizer = SequenceClassifierAdamW::try_new(0.003, &gather_model)?;
    for step in 0..12 {
        dense_model.train_step(&mut dense_optimizer, &tokens, step % 2)?;
        gather_model.train_step_gather(&mut gather_optimizer, &tokens, step % 2)?;
        if dense_model != gather_model {
            return Err("multi-step weight parity failed".into());
        }
    }
    let dense = median_ns(iterations, || {
        black_box(model.loss_and_gradients(black_box(&tokens), 1)?);
        Ok(())
    })?;
    let gather = median_ns(iterations, || {
        black_box(model.loss_and_gradients_gather(black_box(&tokens), 1)?);
        Ok(())
    })?;
    let encoder = model.encoder();
    let mut scratch = SequenceWorkspace::try_new(config.encoder)?;
    let direct = median_ns(iterations, || {
        black_box(encoder.forward(black_box(&tokens))?);
        Ok(())
    })?;
    let reused = median_ns(iterations, || {
        black_box(encoder.forward_with_workspace(black_box(&tokens), &mut scratch)?);
        Ok(())
    })?;
    let contextual = median_ns(iterations, || {
        black_box(model.loss_and_gradients_contextual(black_box(&tokens), 0.5, 1)?);
        Ok(())
    })?;
    // A one-example optimizer smoke demonstrates connection, not generalization.
    let mut context_model = model.clone();
    let before = context_model
        .loss_and_gradients_contextual(&tokens, 0.5, 1)?
        .0;
    let mut optimizer = SequenceClassifierAdamW::try_new(0.003, &context_model)?;
    for _ in 0..24 {
        context_model.train_step_contextual(&mut optimizer, &tokens, 0.5, 1)?;
    }
    let after = context_model
        .loss_and_gradients_contextual(&tokens, 0.5, 1)?
        .0;
    if after >= before {
        return Err("contextual training smoke did not reduce example NLL".into());
    }
    println!("metric\tvalue");
    println!("architecture\t{}", std::env::consts::ARCH);
    println!("iterations_per_round\t{iterations}");
    println!("timing_rounds\t7");
    println!("gradient_and_12_update_parity\ttrue");
    println!(
        "dense_peak_tensor_elements\t{}",
        encoder.required_max_elements(tokens.len())?
    );
    println!(
        "gather_peak_tensor_elements\t{}",
        encoder.required_gather_max_elements(tokens.len())?
    );
    println!("dense_loss_backward_median_ns\t{dense}");
    println!("gather_loss_backward_median_ns\t{gather}");
    println!("contextual_loss_backward_median_ns\t{contextual}");
    println!("direct_encoder_median_ns\t{direct}");
    println!("workspace_encoder_median_ns\t{reused}");
    println!("contextual_one_example_nll_before\t{before}");
    println!("contextual_one_example_nll_after\t{after}");
    println!("promoted\tfalse");
    Ok(())
}
