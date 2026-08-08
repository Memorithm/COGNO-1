use cogno_scirust::{
    SequenceEncoder, SequenceEncoderConfig, SequenceRetriever, SequenceRetrieverConfig,
};

#[test]
fn public_retriever_keeps_first_candidate_on_exact_similarity_tie() {
    let encoder_config = SequenceEncoderConfig {
        vocab_size: 3,
        max_tokens: 1,
        embedding_dim: 2,
        hidden_dim: 2,
        seed: 0,
    };
    let encoder = SequenceEncoder::from_parts(
        encoder_config,
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .expect("controlled encoder");
    let retriever = SequenceRetriever::from_encoder(
        SequenceRetrieverConfig {
            encoder: encoder_config,
            temperature: 0.5,
            max_candidates: 2,
        },
        encoder,
    )
    .expect("retriever");
    let first = [1u16];
    let second = [2u16];
    let candidates = [first.as_slice(), second.as_slice()];

    assert_eq!(
        retriever.similarities(&[0], &candidates).expect("scores"),
        vec![1.0, 1.0]
    );
    assert_eq!(retriever.select_best(&[0], &candidates), Ok(0));
}
