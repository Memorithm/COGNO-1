use cogno_scirust::optim::WarmupCosine;
#[test]
fn warmup_peak_decay_and_floor_have_unambiguous_indices() {
    let schedule = WarmupCosine::try_new(1.0, 0.1, 2, 6).unwrap();
    assert_eq!(schedule.learning_rate(0), 0.55);
    assert_eq!(schedule.learning_rate(1), 1.0);
    let values: Vec<_> = (1..6).map(|i| schedule.learning_rate(i)).collect();
    assert!(values.windows(2).all(|w| w[0] >= w[1]));
    assert_eq!(schedule.learning_rate(5), 0.1);
    assert_eq!(schedule.learning_rate(u64::MAX), 0.1);
    let no_warmup = WarmupCosine::try_new(1.0, 0.1, 0, 2).unwrap();
    assert_eq!(no_warmup.learning_rate(0), 1.0);
    assert_eq!(no_warmup.learning_rate(1), 0.1);
}
#[test]
fn hostile_configuration_rejected_and_tiny_floors_remain_positive() {
    for (peak, floor, warmup, total) in [
        (1.0, 0.0, 0, 2),
        (1.0, 2.0, 0, 2),
        (f32::NAN, 0.1, 0, 2),
        (1.0, 0.1, 2, 2),
        (1.0, 0.1, 0, 1),
    ] {
        assert!(WarmupCosine::try_new(peak, floor, warmup, total).is_err());
    }
    let schedule = WarmupCosine::try_new(1e-30, f32::from_bits(1), u64::MAX - 1, u64::MAX).unwrap();
    assert!(schedule.learning_rate(0) > 0.0);
}
