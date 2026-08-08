use cogno_scirust::{InfoNCE, PairwiseLoss, Shape, Tape, Tensor};

#[test]
fn pairwise_var_loss_backpropagates_through_upstream_scoring_op() {
    let mut tape = Tape::new(16, 16);
    let base = tape
        .variable(
            Tensor::try_new(
                Shape::try_new(&[2]).expect("shape"),
                vec![0.0, 0.0],
                16,
            )
            .expect("base tensor"),
        )
        .expect("base var");
    let preferred = tape.scale(base, 2.0).expect("upstream scoring op");
    let dispreferred = tape
        .variable(
            Tensor::try_new(
                Shape::try_new(&[2]).expect("shape"),
                vec![0.0, 0.0],
                16,
            )
            .expect("dispreferred tensor"),
        )
        .expect("dispreferred var");
    let loss = PairwiseLoss::try_new(1.0, 4, 16)
        .expect("pairwise")
        .loss_vars(&mut tape, preferred, dispreferred)
        .expect("connected loss");

    tape.backward(loss).expect("backward");
    let gradient = tape.grad_of(base);
    assert_eq!(gradient.len(), 2);
    assert!(gradient.iter().all(|value| *value < 0.0));
}

#[test]
fn infonce_similarity_loss_backpropagates_to_similarity_producer() {
    let mut tape = Tape::new(16, 16);
    let base = tape
        .variable(
            Tensor::try_new(
                Shape::try_new(&[3]).expect("shape"),
                vec![0.2, 0.1, -0.1],
                16,
            )
            .expect("similarities"),
        )
        .expect("base var");
    let produced = tape.scale(base, 2.0).expect("similarity producer");
    let loss = InfoNCE::try_new(0.5, 8, 16)
        .expect("InfoNCE")
        .loss_similarities(&mut tape, produced, 1)
        .expect("connected InfoNCE");

    tape.backward(loss).expect("backward");
    let gradient = tape.grad_of(base);
    assert_eq!(gradient.len(), 3);
    assert!(gradient[1] < 0.0, "positive candidate must be pulled up");
    assert!(gradient[0] > 0.0, "negative candidate must be pushed down");
    assert!(gradient[2] > 0.0, "negative candidate must be pushed down");
}
