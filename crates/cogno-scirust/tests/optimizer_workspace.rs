use cogno_scirust::optim::{AdamW, AdamWWorkspace, Optimizer};

#[test]
fn reusable_adam_matches_legacy_every_bit_and_retries_after_overflow() {
    let mut original = AdamW::try_new(0.003, 3).unwrap();
    let mut reusable = original.clone();
    let mut workspace = AdamWWorkspace::try_new(3).unwrap();
    let mut left = [1.0, -0.2, 0.3];
    let mut right = left;
    for step in 0..100 {
        let grad = [step as f32 / 99.0, -0.4, 0.0];
        original.step(&mut left, &grad).unwrap();
        reusable.step_with_workspace(&mut right, &grad, &mut workspace).unwrap();
        assert_eq!(left, right);
        assert_eq!(original.state.m, reusable.state.m);
        assert_eq!(original.state.v, reusable.state.v);
        assert_eq!(original.state.step, reusable.state.step);
    }
    let before = reusable.clone();
    let values = right;
    assert!(reusable.step_with_workspace(&mut right, &[0.1, 0.1, f32::MAX], &mut workspace).is_err());
    assert_eq!(right, values);
    assert_eq!(reusable.state.m, before.state.m);
    assert_eq!(reusable.state.v, before.state.v);
    assert_eq!(reusable.state.step, before.state.step);
    original.step(&mut left, &[0.1; 3]).unwrap();
    reusable.step_with_workspace(&mut right, &[0.1; 3], &mut workspace).unwrap();
    assert_eq!(left, right);
}

#[test]
fn workspace_shape_and_capacity_are_checked() {
    assert!(AdamWWorkspace::try_new(0).is_err());
    assert!(AdamWWorkspace::try_new(1_048_577).is_err());
    let mut opt = AdamW::try_new(0.1, 2).unwrap();
    let mut workspace = AdamWWorkspace::try_new(1).unwrap();
    let mut param = [1.0; 2];
    assert!(opt.step_with_workspace(&mut param, &[1.0; 2], &mut workspace).is_err());
    assert_eq!(param, [1.0; 2]);
    assert_eq!(opt.state.step, 0);
}
