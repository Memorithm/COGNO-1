//! Optimizer tests (AdamW / AMSGrad convergence on a quadratic).

use cogno_scirust::{error::SciRustResult, Optimizer};

#[test]
fn adamw_converges_on_quadratic() -> SciRustResult<()> {
    let mut opt = cogno_scirust::optim::AdamW::try_new(0.1, 1)?;
    let mut x = [10.0f32];
    for _ in 0..500 {
        let grad = [x[0]];
        opt.step(&mut x, &grad)?;
    }
    assert!(x[0].abs() < 0.05);
    Ok(())
}

#[test]
fn amsgrad_converges_on_quadratic() -> SciRustResult<()> {
    let mut opt = cogno_scirust::optim::AmsGrad::try_new(0.1, 1)?;
    let mut x = [10.0f32];
    for _ in 0..500 {
        let grad = [x[0]];
        opt.step(&mut x, &grad)?;
    }
    assert!(x[0].abs() < 0.05);
    Ok(())
}

#[test]
fn adamw_weight_decay_pushes_to_zero() -> SciRustResult<()> {
    let mut opt = cogno_scirust::optim::AdamW::try_new(0.1, 1)?;
    let mut x = [5.0f32];
    for _ in 0..2000 {
        let grad = [0.0f32];
        opt.step(&mut x, &grad)?;
    }
    assert!(x[0].abs() < 1.0);
    Ok(())
}
