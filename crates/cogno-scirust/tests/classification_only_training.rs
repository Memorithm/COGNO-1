use cogno_scirust::*;

fn model(seed: u64) -> SequenceCognitiveHeads {
    SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
        encoder: SequenceEncoderConfig {
            vocab_size: 32,
            max_tokens: 16,
            embedding_dim: 8,
            hidden_dim: 16,
            seed,
        },
        num_classes: 2,
        num_rules: 1,
        classification_seed: 1,
        preference_seed: 2,
        symbolic_seed: 3,
        contradiction_seed: 4,
    })
    .unwrap()
}

fn batch<'a>(
    tokens: &'a [u16],
    candidates: &'a [&'a [u16]],
    target: usize,
) -> SequenceCognitiveBatch<'a> {
    SequenceCognitiveBatch {
        classification: CognitiveClassification {
            token_ids: tokens,
            target_class: target,
        },
        preference: CognitivePreference {
            preferred: tokens,
            dispreferred: tokens,
            margin: 1.0,
        },
        symbolic: CognitiveSymbolic {
            token_ids: tokens,
            targets: &[1.0],
        },
        contradiction: CognitiveContradiction {
            pair_token_ids: tokens,
            contradicts: false,
        },
        retrieval: CognitiveRetrieval {
            query: tokens,
            candidates,
            positive_idx: 0,
            temperature: 0.2,
        },
    }
}
fn weights() -> SequenceCognitiveLossWeights {
    SequenceCognitiveLossWeights {
        classification: 1.0,
        preference: 0.0,
        symbolic: 0.0,
        contradiction: 0.0,
        retrieval: 0.0,
    }
}

#[test]
fn classification_gradients_and_active_updates_match_joint_exactly() {
    for seed in [1, 7, 42] {
        let mut fast = model(seed);
        let mut joint = fast.clone();
        let initial = fast.clone();
        let mut fo = SequenceCognitiveAdamW::try_new(0.003, &fast).unwrap();
        let mut jo = SequenceCognitiveAdamW::try_new(0.003, &joint).unwrap();
        for step in 0..8 {
            let tokens = [1, 2, 3, 4, 2];
            let candidates: [&[u16]; 2] = [&tokens, &tokens];
            let b = batch(&tokens, &candidates, step % 2);
            let (fl, fg) = fast
                .classification_loss_and_gradients(b.classification)
                .unwrap();
            let (jl, jg) = joint.joint_loss_and_gradients(b, weights()).unwrap();
            assert_eq!(fl, jl.classification);
            assert_eq!(fg, jg);
            fast.train_classification_step(&mut fo, b.classification)
                .unwrap();
            joint.train_joint_step(&mut jo, b, weights()).unwrap();
            assert_eq!(fast.encoder(), joint.encoder());
            assert_eq!(
                fast.classification_weights(),
                joint.classification_weights()
            );
            assert_eq!(fast.classification_bias(), joint.classification_bias());
            assert_eq!(fast.preference_weights(), initial.preference_weights());
            assert_eq!(fast.preference_bias(), initial.preference_bias());
            assert_eq!(fast.symbolic_weights(), initial.symbolic_weights());
            assert_eq!(fast.symbolic_bias(), initial.symbolic_bias());
            assert_eq!(
                fast.contradiction_weights(),
                initial.contradiction_weights()
            );
            assert_eq!(fast.contradiction_bias(), initial.contradiction_bias());
        }
    }
}

#[test]
fn invalid_classification_is_atomic() {
    let mut m = model(1);
    let initial = m.clone();
    let mut opt = SequenceCognitiveAdamW::try_new(0.003, &m).unwrap();
    let state = format!("{opt:?}");
    for (tokens, target) in [(&[][..], 0), (&[32][..], 0), (&[1][..], 2)] {
        assert!(m
            .train_classification_step(
                &mut opt,
                CognitiveClassification {
                    token_ids: tokens,
                    target_class: target
                }
            )
            .is_err());
        assert_eq!(m, initial);
        assert_eq!(format!("{opt:?}"), state);
    }
}

#[test]
#[ignore = "local CPU diagnostic; run release with --ignored --nocapture"]
fn classification_only_cpu_probe() {
    use std::{hint::black_box, time::Instant};
    let m = model(1);
    let tokens = [1u16; 16];
    let candidates: [&[u16]; 2] = [&tokens, &tokens];
    let b = batch(&tokens, &candidates, 1);
    let n = 1000;
    let start = Instant::now();
    for _ in 0..n {
        black_box(m.joint_loss_and_gradients(b, weights()).unwrap());
    }
    let joint = start.elapsed();
    let start = Instant::now();
    for _ in 0..n {
        black_box(
            m.classification_loss_and_gradients(b.classification)
                .unwrap(),
        );
    }
    let fast = start.elapsed();
    println!(
        "iterations={n} joint_us={} classification_us={} ratio={:.3}",
        joint.as_micros(),
        fast.as_micros(),
        joint.as_secs_f64() / fast.as_secs_f64()
    );
}

#[test]
fn mismatched_optimizer_shape_is_rejected_without_mutation() {
    let mut large = model(1);
    let mut small_config = large.config();
    small_config.encoder.vocab_size = 8;
    let small = SequenceCognitiveHeads::try_new(small_config).unwrap();
    let mut optimizer = SequenceCognitiveAdamW::try_new(0.003, &small).unwrap();
    let initial = large.clone();
    let state = format!("{optimizer:?}");
    assert!(large
        .train_classification_step(
            &mut optimizer,
            CognitiveClassification {
                token_ids: &[1, 2],
                target_class: 0
            }
        )
        .is_err());
    assert_eq!(large, initial);
    assert_eq!(format!("{optimizer:?}"), state);
}
