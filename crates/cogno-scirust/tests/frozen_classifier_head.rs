use cogno_scirust::{
    sequence_classifier::FrozenClassifierHeadAdamW, SequenceClassifier, SequenceClassifierConfig,
    SequenceEncoderConfig,
};

fn model(seed: u64) -> SequenceClassifier {
    SequenceClassifier::try_new(SequenceClassifierConfig {
        encoder: SequenceEncoderConfig {
            vocab_size: 16,
            max_tokens: 8,
            embedding_dim: 4,
            hidden_dim: 5,
            seed,
        },
        num_classes: 2,
        head_seed: 7,
    })
    .unwrap()
}

#[test]
fn frozen_head_gradients_match_end_to_end_head_gradients_exactly() {
    for seed in [1, 7, 42] {
        let model = model(seed);
        for tokens in [&[1][..], &[1, 2, 1, 3][..], &[7; 8][..]] {
            for target in 0..2 {
                let (full_loss, full) = model.loss_and_gradients(tokens, target).unwrap();
                let (loss, head) = model.loss_and_head_gradients(tokens, target).unwrap();
                assert_eq!(loss, full_loss);
                assert_eq!(head.weights(), full.head_weights());
                assert_eq!(head.bias(), full.head_bias());
            }
        }
    }
}

#[test]
fn head_training_reduces_fit_loss_without_touching_encoder() {
    let mut left = model(1);
    let mut right = left.clone();
    let encoder = left.encoder().clone();
    let mut a = FrozenClassifierHeadAdamW::try_new(0.02, &left).unwrap();
    let mut b = FrozenClassifierHeadAdamW::try_new(0.02, &right).unwrap();
    let initial = left.loss_and_head_gradients(&[1, 2, 3], 1).unwrap().0;
    for _ in 0..64 {
        left.train_head_step(&mut a, &[1, 2, 3], 1).unwrap();
        right.train_head_step(&mut b, &[1, 2, 3], 1).unwrap();
    }
    assert!(left.loss_and_head_gradients(&[1, 2, 3], 1).unwrap().0 < initial);
    assert_eq!(left.encoder(), &encoder);
    assert_eq!(left, right);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn rejected_head_training_preserves_model_and_optimizer() {
    let mut model = model(1);
    let before = model.clone();
    let mut opt = FrozenClassifierHeadAdamW::try_new(0.02, &model).unwrap();
    let state = format!("{opt:?}");
    for (tokens, target) in [
        (&[][..], 0),
        (&[16][..], 0),
        (&[1; 9][..], 0),
        (&[1][..], 2),
    ] {
        assert!(model.train_head_step(&mut opt, tokens, target).is_err());
        assert_eq!(model, before);
        assert_eq!(format!("{opt:?}"), state);
    }
}

#[test]
fn optimizer_from_different_head_shape_is_rejected() {
    let mut large = model(1);
    let mut config = large.config();
    config.encoder.hidden_dim = 2;
    let small = SequenceClassifier::try_new(config).unwrap();
    let mut opt = FrozenClassifierHeadAdamW::try_new(0.02, &small).unwrap();
    let initial = large.clone();
    let state = format!("{opt:?}");
    assert!(large.train_head_step(&mut opt, &[1, 2], 0).is_err());
    assert_eq!(large, initial);
    assert_eq!(format!("{opt:?}"), state);
}
