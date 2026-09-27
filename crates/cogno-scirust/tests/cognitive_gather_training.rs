use cogno_scirust::{
    CognitiveClassification, SequenceCognitiveAdamW, SequenceCognitiveConfig,
    SequenceCognitiveHeads, SequenceEncoderConfig,
};

fn model(seed: u64) -> SequenceCognitiveHeads {
    SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
        encoder: SequenceEncoderConfig {
            vocab_size: 32,
            max_tokens: 16,
            embedding_dim: 8,
            hidden_dim: 12,
            seed,
        },
        num_classes: 3,
        num_rules: 2,
        classification_seed: 1,
        preference_seed: 2,
        symbolic_seed: 3,
        contradiction_seed: 4,
    })
    .unwrap()
}

#[test]
fn gather_classification_matches_dense_gradients_and_repeated_updates() {
    for seed in [0, 7, 42] {
        for tokens in [&[0][..], &[1, 3, 1, 31, 0][..], &[2; 16][..]] {
            let mut dense = model(seed);
            let mut gather = dense.clone();
            let mut dense_optimizer = SequenceCognitiveAdamW::try_new(0.003, &dense).unwrap();
            let mut gather_optimizer = dense_optimizer.clone();
            for step in 0..9 {
                let row = CognitiveClassification {
                    token_ids: tokens,
                    target_class: step % 3,
                };
                assert_eq!(
                    dense.classification_loss_and_gradients(row).unwrap(),
                    gather
                        .classification_loss_and_gradients_gather(row)
                        .unwrap()
                );
                assert_eq!(
                    dense
                        .train_classification_step(&mut dense_optimizer, row)
                        .unwrap(),
                    gather
                        .train_classification_step_gather(&mut gather_optimizer, row)
                        .unwrap()
                );
                assert_eq!(dense, gather);
                assert_eq!(
                    format!("{dense_optimizer:?}"),
                    format!("{gather_optimizer:?}")
                );
            }
        }
    }
}

#[test]
fn rejected_gather_observations_preserve_model_and_optimizer() {
    let original = model(7);
    for tokens in [&[][..], &[32][..], &[0; 17][..], &[1][..]] {
        let mut candidate = original.clone();
        let mut optimizer = SequenceCognitiveAdamW::try_new(0.003, &candidate).unwrap();
        let before = format!("{optimizer:?}");
        let row = CognitiveClassification {
            token_ids: tokens,
            target_class: if tokens == [1] { 3 } else { 0 },
        };
        assert!(candidate
            .train_classification_step_gather(&mut optimizer, row)
            .is_err());
        assert_eq!(candidate, original);
        assert_eq!(format!("{optimizer:?}"), before);
    }
}

#[test]
fn gather_minibatches_match_dense_mean_and_repeated_updates() {
    let rows = [
        CognitiveClassification {
            token_ids: &[0],
            target_class: 0,
        },
        CognitiveClassification {
            token_ids: &[1, 31, 1, 3],
            target_class: 1,
        },
        CognitiveClassification {
            token_ids: &[2; 16],
            target_class: 2,
        },
    ];
    for seed in [0, 7, 42] {
        for size in 1..=rows.len() {
            let mut dense = model(seed);
            let mut gather = dense.clone();
            let mut dense_optimizer = SequenceCognitiveAdamW::try_new(0.003, &dense).unwrap();
            let mut gather_optimizer = dense_optimizer.clone();
            for _ in 0..9 {
                assert_eq!(
                    dense
                        .classification_minibatch_loss_and_gradients(&rows[..size])
                        .unwrap(),
                    gather
                        .classification_minibatch_loss_and_gradients_gather(&rows[..size])
                        .unwrap()
                );
                assert_eq!(
                    dense
                        .train_classification_minibatch_step(&mut dense_optimizer, &rows[..size])
                        .unwrap(),
                    gather
                        .train_classification_minibatch_step_gather(
                            &mut gather_optimizer,
                            &rows[..size]
                        )
                        .unwrap()
                );
                assert_eq!(dense, gather);
                assert_eq!(
                    format!("{dense_optimizer:?}"),
                    format!("{gather_optimizer:?}")
                );
            }
        }
    }
}

#[test]
fn gather_minibatch_late_rejection_and_capacity_are_atomic() {
    let valid = CognitiveClassification {
        token_ids: &[1, 1],
        target_class: 0,
    };
    let invalid = CognitiveClassification {
        token_ids: &[32],
        target_class: 0,
    };
    let original = model(42);
    for rows in [vec![], vec![valid, invalid], vec![valid; 257]] {
        let mut candidate = original.clone();
        let mut optimizer = SequenceCognitiveAdamW::try_new(0.003, &candidate).unwrap();
        let before = format!("{optimizer:?}");
        assert!(candidate
            .train_classification_minibatch_step_gather(&mut optimizer, &rows)
            .is_err());
        assert_eq!(candidate, original);
        assert_eq!(format!("{optimizer:?}"), before);
    }
    assert_eq!(
        original
            .classification_minibatch_loss_and_gradients_gather(&[valid; 256])
            .unwrap(),
        original
            .classification_minibatch_loss_and_gradients(&[valid; 256])
            .unwrap()
    );
}
