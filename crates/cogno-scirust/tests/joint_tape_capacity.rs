use cogno_scirust::{
    CognitiveClassification, CognitiveContradiction, CognitivePreference, CognitiveRetrieval,
    CognitiveSymbolic, SequenceCognitiveBatch, SequenceCognitiveConfig, SequenceCognitiveHeads,
    SequenceCognitiveLossWeights, SequenceEncoderConfig,
};

#[test]
fn joint_objective_accepts_byte_scale_sequence_lengths() {
    let model = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
        encoder: SequenceEncoderConfig {
            vocab_size: 259,
            max_tokens: 48,
            embedding_dim: 8,
            hidden_dim: 8,
            seed: 1601,
        },
        num_classes: 3,
        num_rules: 2,
        classification_seed: 1607,
        preference_seed: 1609,
        symbolic_seed: 1613,
        contradiction_seed: 1619,
    })
    .expect("model");

    let classification = vec![1u16; 19];
    let preferred = vec![2u16; 18];
    let dispreferred = vec![3u16; 17];
    let symbolic = vec![4u16; 15];
    let contradiction = vec![5u16; 26];
    let query = vec![6u16; 13];
    let candidate_a = vec![7u16; 14];
    let candidate_b = vec![8u16; 14];
    let candidates: [&[u16]; 2] = [&candidate_a, &candidate_b];

    let result = model.joint_loss_and_gradients(
        SequenceCognitiveBatch {
            classification: CognitiveClassification {
                token_ids: &classification,
                target_class: 1,
            },
            preference: CognitivePreference {
                preferred: &preferred,
                dispreferred: &dispreferred,
                margin: 2.0,
            },
            symbolic: CognitiveSymbolic {
                token_ids: &symbolic,
                targets: &[1.0, 0.0],
            },
            contradiction: CognitiveContradiction {
                pair_token_ids: &contradiction,
                contradicts: true,
            },
            retrieval: CognitiveRetrieval {
                query: &query,
                candidates: &candidates,
                positive_idx: 0,
                temperature: 0.5,
            },
        },
        SequenceCognitiveLossWeights::default(),
    );

    assert!(
        result.is_ok(),
        "joint tape rejected bounded byte-scale views: {result:?}"
    );
}
