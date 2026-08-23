//! Loss function tests (COGNO-1 §SciRust #2/#3/#4).

use cogno_scirust::{
    engine::Tape,
    error::SciRustResult,
    losses::{InfoNCE, PairwiseLoss, SymbolicSatisfaction},
    tensor::{Shape, Tensor},
};

#[test]
fn pairwise_loss_basic() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let loss_fn = PairwiseLoss::try_new(1.0, 8, 256)?;
    let pref = Tensor {
        shape: Shape::try_new(&[4])?,
        data: vec![10.0, 8.0, 6.0, 4.0],
    };
    let disp = Tensor {
        shape: Shape::try_new(&[4])?,
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let loss_var = loss_fn.loss(&mut tape, pref, disp)?;
    assert!(tape.value_of(loss_var).data[0].is_finite());
    Ok(())
}

#[test]
fn pairwise_loss_zero_when_margin_satisfied() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let loss_fn = PairwiseLoss::try_new(1.0, 8, 256)?;
    let pref = Tensor {
        shape: Shape::try_new(&[2])?,
        data: vec![10.0, 10.0],
    };
    let disp = Tensor {
        shape: Shape::try_new(&[2])?,
        data: vec![5.0, 5.0],
    };
    let loss_var = loss_fn.loss(&mut tape, pref, disp)?;
    let loss = tape.value_of(loss_var).data[0];
    assert!(loss.abs() < 1e-5);
    Ok(())
}

#[test]
fn symbolic_satisfaction_product() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let sats = Tensor {
        shape: Shape::try_new(&[4])?,
        data: vec![1.0, 0.8, 0.9, 0.7],
    };
    let loss_var =
        SymbolicSatisfaction::loss(&SymbolicSatisfaction::try_new(8, 256)?, &mut tape, sats)?;
    assert_eq!(tape.value_of(loss_var).len(), 1);
    Ok(())
}

#[test]
fn infonce_loss() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let d = 4;
    let n = 5;
    let query = Tensor {
        shape: Shape::try_new(&[d])?,
        data: vec![1.0, 0.0, 0.0, 0.0],
    };
    let mut keys_data = vec![0.0f32; 5 * 4];
    keys_data[0] = 1.0;
    let keys = Tensor {
        shape: Shape::try_new(&[n, 4])?,
        data: keys_data,
    };
    let loss_var = InfoNCE::loss(&InfoNCE::try_new(0.1, 8, 256)?, &mut tape, query, keys, 0)?;
    assert_eq!(tape.value_of(loss_var).len(), 1);
    Ok(())
}

#[test]
fn calibration_surfaces_non_finite_scores_in_batch() {
    use cogno_scirust::calib::Calibration;
    let cal = Calibration::try_new(2.0).unwrap();
    // Scalar compat path: non-finite input maps to the most conservative
    // confidence (0 bps), never a plausible-looking value.
    assert_eq!(cal.calibrate_bps(f32::NAN), 0);
    // Batch path: the corruption is surfaced instead of published (§2).
    let z = cogno_scirust::tensor::Tensor::try_new(
        cogno_scirust::tensor::Shape::try_new(&[3]).unwrap(),
        vec![0.5, f32::NAN, -1.0],
        16,
    )
    .unwrap();
    assert!(cal.try_calibrate_batch(&z).is_err());
    let ok = cogno_scirust::tensor::Tensor::try_new(
        cogno_scirust::tensor::Shape::try_new(&[1]).unwrap(),
        vec![0.5],
        16,
    )
    .unwrap();
    let out = cal.try_calibrate_batch(&ok).unwrap();
    assert!(out.bps[0] <= cal.max_bps);
}
