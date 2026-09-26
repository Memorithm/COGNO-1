use cogno_scirust::{InfoNCE, PairwiseLoss, Tape, Tensor};

#[test]
fn infonce_rejects_unrepresentable_inverse_and_mutated_temperature() {
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        assert!(InfoNCE::try_new(temperature, 2, 2).is_err());
        let mut loss = InfoNCE::try_new(1.0, 2, 2).unwrap();
        loss.temperature = temperature;
        let mut tape = Tape::new(3, 2);
        let a = tape.variable(Tensor::try_scalar(0.2).unwrap()).unwrap();
        let b = tape.variable(Tensor::try_scalar(0.1).unwrap()).unwrap();
        assert!(loss.loss_similarity_vars(&mut tape, &[a, b], 0).is_err());
        // Exactly one node remains: rejection must precede stack allocation.
        assert!(tape.variable(Tensor::try_scalar(1.0).unwrap()).is_ok());
    }
}

#[test]
fn infonce_checks_scalar_list_bounds_before_stacking() {
    for (count, index) in [(3, 0), (2, 2), (1, 0)] {
        let mut tape = Tape::new(2, 4);
        let a = tape.variable(Tensor::try_scalar(0.2).unwrap()).unwrap();
        let loss = InfoNCE::try_new(1.0, 2, 4).unwrap();
        assert!(loss
            .loss_similarity_vars(&mut tape, &vec![a; count], index)
            .is_err());
        assert!(tape.variable(Tensor::try_scalar(1.0).unwrap()).is_ok());
    }
}

#[test]
fn pairwise_revalidates_mutable_margin_before_graph_changes() {
    for margin in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(PairwiseLoss::try_new(margin, 1, 1).is_err());
        let mut loss = PairwiseLoss::try_new(0.0, 1, 1).unwrap();
        loss.margin = margin;
        let mut tape = Tape::new(3, 1);
        let a = tape.variable(Tensor::try_scalar(0.2).unwrap()).unwrap();
        let b = tape.variable(Tensor::try_scalar(0.1).unwrap()).unwrap();
        assert!(loss.loss_vars(&mut tape, a, b).is_err());
        assert!(tape.variable(Tensor::try_scalar(1.0).unwrap()).is_ok());
    }
}
