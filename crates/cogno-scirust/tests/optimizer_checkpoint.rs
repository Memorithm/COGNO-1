use cogno_scirust::{AdamW, Optimizer};
#[test]
fn uninterrupted_and_resumed_training_are_identical() {
    let mut a = AdamW::try_new(0.02, 2).unwrap();
    let mut x = vec![0.2, -0.5];
    for _ in 0..7 {
        a.step(&mut x, &[0.1, -0.8]).unwrap();
    }
    let wire = a.checkpoint(&x).unwrap();
    let (mut b, mut y) = AdamW::from_checkpoint(&wire).unwrap();
    assert_eq!(wire, b.checkpoint(&y).unwrap());
    for _ in 0..23 {
        a.step(&mut x, &[0.4, -0.2]).unwrap();
        b.step(&mut y, &[0.4, -0.2]).unwrap();
    }
    assert_eq!(a.checkpoint(&x).unwrap(), b.checkpoint(&y).unwrap());
}
#[test]
fn rejects_truncation_trailing_bytes_oversize_and_invalid_moments() {
    let a = AdamW::try_new(0.02, 2).unwrap();
    let wire = a.checkpoint(&[0.2, -0.5]).unwrap();
    for end in 0..wire.len() {
        assert!(AdamW::from_checkpoint(&wire[..end]).is_err());
    }
    let mut bad = wire.clone();
    bad.push(0);
    assert!(AdamW::from_checkpoint(&bad).is_err());
    let mut bad = wire.clone();
    bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(AdamW::from_checkpoint(&bad).is_err());
    for (offset, value) in [(20, f32::NAN), (40, f32::INFINITY), (56, -1.0)] {
        let mut bad = wire.clone();
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(AdamW::from_checkpoint(&bad).is_err());
    }
}
