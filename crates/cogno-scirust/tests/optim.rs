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

#[test]
fn adamw_rejects_out_of_domain_hyperparams_at_construction_and_step() {
    use cogno_scirust::{error::SciRustError, Optimizer};
    // beta1 == 1.0 zeroes the bias-correction denominator and used to
    // silently produce NaN parameters. Fields are plain data, so the step
    // re-validates: mutation after construction fails closed.
    let mut opt = cogno_scirust::optim::AdamW::try_new(0.1, 1).unwrap();
    opt.beta1 = 1.0;
    let mut x = [1.0f32];
    let err = opt.step(&mut x, &[1.0]).unwrap_err();
    assert!(matches!(err, SciRustError::NonFinite));
    assert_eq!(x[0], 1.0, "parameter untouched on rejected step");
    // Same story for a NaN learning rate injected later.
    let mut opt = cogno_scirust::optim::AmsGrad::try_new(0.1, 1).unwrap();
    opt.lr = f32::NAN;
    let mut x = [1.0f32];
    assert!(opt.step(&mut x, &[1.0]).is_err());
    assert_eq!(x[0], 1.0);
}
