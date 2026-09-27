use cogno_scirust::losses::SoftTargetCrossEntropy;
use cogno_scirust::{Shape, Tape, Tensor};
fn run(logits: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let mut tape = Tape::new(16, 16);
    let var = tape
        .variable(
            Tensor::try_new(
                Shape::try_new(&[logits.len()]).unwrap(),
                logits.to_vec(),
                16,
            )
            .unwrap(),
        )
        .unwrap();
    let loss = SoftTargetCrossEntropy::try_new(16)
        .unwrap()
        .loss(&mut tape, var, target)
        .unwrap();
    tape.backward(loss).unwrap();
    (tape.value_of(loss).data[0], tape.grad_of(var).to_vec())
}
#[test]
fn smoothed_gradients_match_finite_differences() {
    let objective = SoftTargetCrossEntropy::try_new(3).unwrap();
    let target = objective.smoothed_target(3, 1, 0.2).unwrap();
    let logits = [0.8, -0.2, 0.1];
    let (_, gradient) = run(&logits, &target);
    for i in 0..3 {
        let mut plus = logits;
        let mut minus = logits;
        plus[i] += 0.001;
        minus[i] -= 0.001;
        let finite = (run(&plus, &target).0 - run(&minus, &target).0) / 0.002;
        assert!((gradient[i] - finite).abs() < 0.0002);
    }
    assert_eq!(
        objective.smoothed_target(3, 1, 0.0).unwrap(),
        [0.0, 1.0, 0.0]
    );
}
#[test]
fn rejects_invalid_distribution_and_preserves_shift_invariance() {
    let target = [0.0, 1.0];
    assert!((run(&[1001.0, 1000.0], &target).0 - run(&[1.0, 0.0], &target).0).abs() < 1e-5);
    let mut tape = Tape::new(16, 16);
    let v = tape
        .variable(Tensor::try_new(Shape::try_new(&[2]).unwrap(), vec![0.0; 2], 16).unwrap())
        .unwrap();
    let objective = SoftTargetCrossEntropy::try_new(2).unwrap();
    for bad in [[0.0, 0.0], [-1.0, 2.0], [f32::NAN, 1.0], [0.2, 0.9]] {
        assert!(objective.loss(&mut tape, v, &bad).is_err());
    }
}
