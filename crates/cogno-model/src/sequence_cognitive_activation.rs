//! Read-only activation bridge for an already hostile-validated V4 state.
//!
//! This module deliberately exposes no constructor from raw heads or tensors.
//! The only activation input is [`SequenceCognitiveArtifactState`], which can be
//! minted only by the V4 hostile decoder. Runtime authority remains outside the
//! model crate; this bridge only reconstructs the byte-tokenized read-only
//! facade from verified state.

use crate::{
    SciRustSequenceCognitiveReadOnlyModel, SequenceCognitiveArtifactState, SequenceCognitiveModel,
};

impl SequenceCognitiveArtifactState {
    /// Consume a hostile-validated V4 state and reconstruct its frozen
    /// byte-tokenized facade. This grants no runtime installation authority.
    pub fn into_read_only_model(
        self,
    ) -> Result<SciRustSequenceCognitiveReadOnlyModel, crate::SequenceCognitiveModelError> {
        let (heads, max_retrieval_candidates) = self.into_parts();
        let model = SequenceCognitiveModel::from_heads(heads, max_retrieval_candidates)?;
        Ok(SciRustSequenceCognitiveReadOnlyModel::from_trained(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encode_sequence_cognitive_artifact, load_sequence_cognitive_artifact,
        BYTE_TOKENIZER_VOCAB_SIZE,
    };
    use cogno_scirust::{SequenceCognitiveConfig, SequenceCognitiveHeads, SequenceEncoderConfig};

    #[test]
    fn only_verified_v4_state_reconstructs_exact_read_only_geometry() {
        let heads = SequenceCognitiveHeads::try_new(SequenceCognitiveConfig {
            encoder: SequenceEncoderConfig {
                vocab_size: BYTE_TOKENIZER_VOCAB_SIZE,
                max_tokens: 32,
                embedding_dim: 8,
                hidden_dim: 8,
                seed: 2309,
            },
            num_classes: 3,
            num_rules: 4,
            classification_seed: 2311,
            preference_seed: 2333,
            symbolic_seed: 2339,
            contradiction_seed: 2341,
        })
        .expect("heads");
        let artifact = encode_sequence_cognitive_artifact(&heads, 8).expect("artifact");
        let state = load_sequence_cognitive_artifact(&artifact.manifest, &artifact.bytes)
            .expect("verified state");
        let readonly = state.into_read_only_model().expect("read-only model");
        assert_eq!(readonly.model.heads(), &heads);
        assert_eq!(readonly.model.max_retrieval_candidates(), 8);
        assert_eq!(readonly.model.max_tokens(), 32);
    }
}
