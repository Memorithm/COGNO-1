use cogno_scirust::losses::weighted_mean_loss;
use cogno_scirust::{Tape, Tensor};
#[test]
fn uneven_weights_propagate_normalized_gradients_and_zero_mask() {
    let mut tape = Tape::new(16, 16);
    let vars: Vec<_> = [2.0, 10.0, 100.0]
        .into_iter()
        .map(|x| tape.variable(Tensor::try_scalar(x).unwrap()).unwrap())
        .collect();
    let loss = weighted_mean_loss(&mut tape, &vars, &[3e30, 1e30, 0.0]).unwrap();
    tape.backward(loss).unwrap();
    assert_eq!(tape.value_of(loss).data[0], 4.0);
    assert_eq!(tape.grad_of(vars[0]), [0.75]);
    assert_eq!(tape.grad_of(vars[1]), [0.25]);
    assert_eq!(tape.grad_of(vars[2]), [0.0]);
}
#[test]
fn all_masked_nonfinite_and_negative_weights_are_rejected() {
    let mut tape = Tape::new(16, 16);
    let v = tape.variable(Tensor::try_scalar(2.0).unwrap()).unwrap();
    for weights in [[0.0, 0.0], [-1.0, 2.0], [1.0, f32::INFINITY]] {
        assert!(weighted_mean_loss(&mut tape, &[v, v], &weights).is_err());
    }
    assert!(weighted_mean_loss(&mut tape, &[v], &[]).is_err());
    let good = weighted_mean_loss(&mut tape, &[v], &[1.0]).unwrap();
    tape.backward(good).unwrap();
    assert_eq!(tape.grad_of(v), [1.0]);
}
