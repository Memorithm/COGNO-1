#![forbid(unsafe_code)]
use cogno_scirust::{
    SciRustError, SequenceEncoder, SequenceEncoderConfig, Tape, SEQUENCE_ENCODER_TAPE_NODES,
};

fn reference(model: &SequenceEncoder, tokens: &[u16]) -> Result<Vec<f32>, SciRustError> {
    let mut tape = Tape::new(
        SEQUENCE_ENCODER_TAPE_NODES,
        model.required_max_elements(tokens.len())?,
    );
    let graph = model.append_to_tape(&mut tape, tokens)?;
    Ok(tape.value_of(graph.pooled()).as_slice().to_vec())
}

#[test]
fn inference_matches_unchanged_tape_across_shapes_seeds_and_repeated_tokens() {
    for (vocab, max_tokens, width, hidden) in [(3, 7, 3, 5), (259, 128, 32, 64), (512, 512, 4, 7)] {
        for seed in [0, 1, 42] {
            let model = SequenceEncoder::try_new(SequenceEncoderConfig {
                vocab_size: vocab,
                max_tokens,
                embedding_dim: width,
                hidden_dim: hidden,
                seed,
            })
            .unwrap();
            for len in [1, 3, max_tokens] {
                for repeated in [false, true] {
                    let tokens: Vec<u16> = (0..len)
                        .map(|i| {
                            if repeated {
                                1
                            } else {
                                ((i * 17) % vocab) as u16
                            }
                        })
                        .collect();
                    assert_eq!(
                        model.forward(&tokens).unwrap(),
                        reference(&model, &tokens).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn input_rejection_and_arithmetic_overflow_match_reference() {
    let config = SequenceEncoderConfig {
        vocab_size: 3,
        max_tokens: 4,
        embedding_dim: 2,
        hidden_dim: 2,
        seed: 0,
    };
    let model = SequenceEncoder::try_new(config).unwrap();
    for tokens in [vec![], vec![3], vec![0; 5]] {
        assert_eq!(model.forward(&tokens), reference(&model, &tokens));
    }
    let model =
        SequenceEncoder::from_parts(config, vec![f32::MAX; 6], vec![f32::MAX; 8], vec![1.0; 4])
            .unwrap();
    assert!(model.forward(&[0]).is_err());
    assert_eq!(model.forward(&[0]), reference(&model, &[0]));
}

#[test]
fn signed_zero_and_extreme_finite_weights_preserve_reference() {
    let config = SequenceEncoderConfig {
        vocab_size: 3,
        max_tokens: 4,
        embedding_dim: 2,
        hidden_dim: 2,
        seed: 0,
    };
    let model = SequenceEncoder::from_parts(
        config,
        vec![-0.0, 0.0, 1e-30, -1e-30, 1e20, -1e20],
        vec![-0.0; 8],
        vec![1e-10, -1e-10, -1e-10, 1e-10],
    )
    .unwrap();
    let actual = model.forward(&[0, 1, 2, 0]).unwrap();
    let expected = reference(&model, &[0, 1, 2, 0]).unwrap();
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
