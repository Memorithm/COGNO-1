use cogno_scirust::optim::GradientAccumulator;
#[test]
fn uneven_microbatches_equal_the_sample_mean_and_clear_reuses_shape() {
    let mut acc = GradientAccumulator::try_new(2).unwrap();
    acc.add(&[2.0, -4.0], 3.0).unwrap();
    acc.add(&[10.0, 8.0], 1.0).unwrap();
    let mut out = [0.0; 2];
    acc.mean_into(&mut out).unwrap();
    assert_eq!(out, [4.0, -1.0]);
    assert_eq!(acc.batches(), 2);
    acc.clear();
    assert!(acc.mean_into(&mut out).is_err());
    assert_eq!(out, [4.0, -1.0]);
    acc.add(&[1.0, 3.0], 1.0).unwrap();
    acc.mean_into(&mut out).unwrap();
    assert_eq!(out, [1.0, 3.0]);
}
#[test]
fn cancellation_uses_f64_and_late_invalid_microbatch_is_atomic() {
    let mut acc = GradientAccumulator::try_new(2).unwrap();
    acc.add(&[16_777_216.0, 0.0], 1.0).unwrap();
    acc.add(&[1.0, 1.0], 1.0).unwrap();
    acc.add(&[-16_777_216.0, 2.0], 1.0).unwrap();
    assert!(acc.add(&[2.0, f32::NAN], 1.0).is_err());
    assert!(acc.add(&[f32::MAX; 2], f64::MAX).is_err());
    assert_eq!(acc.batches(), 3);
    assert_eq!(acc.total_weight(), 3.0);
    let mut out = [0.0; 2];
    acc.mean_into(&mut out).unwrap();
    assert_eq!(out, [1.0 / 3.0, 1.0]);
}
