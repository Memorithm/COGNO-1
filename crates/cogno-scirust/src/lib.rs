//! # cogno-scirust — optional differentiable, neuro-symbolic, contrastive backend for COGNO-1.
//!
//! This crate implements, in pure safe Rust and behind a fallible, bounded
//! COGNO API, the SciRust-style capabilities required by the spec:
//!
//! 1. differentiable tensor computation for the COGNO objective
//!    ([`engine`] — a Wengert-tape reverse-mode autograd engine);
//! 2. pairwise learning over accepted/rejected/edited outputs ([`losses`]);
//! 3. a differentiable symbolic satisfaction loss ([`losses`]);
//! 4. an InfoNCE objective for memory/rule selection ([`losses`]);
//! 5. a confidence calibration head ([`calib`]);
//! 6. a memory/context/latency cost measure ([`cost`]);
//! 7. AdamW / AMSGrad optimizers with checked arithmetic ([`optim`]);
//! 8. a bounded, pre-allocated, fallible KV cache ([`kv`]);
//! 9. bounded deterministic one-hidden-layer neural networks ([`nn`]);
//! 10. a bounded trainable positional sequence encoder ([`sequence`]);
//! 11. an end-to-end trainable sequence classifier ([`sequence_classifier`]);
//! 12. a connected sequence preference scorer ([`sequence_scorer`]);
//! 13. a connected sequence InfoNCE retriever ([`sequence_retriever`]);
//! 14. a connected sequence symbolic-satisfaction head ([`sequence_symbolic`]);
//! 15. one shared sequence encoder parameterization for cognitive heads
//!     ([`sequence_cognitive`]).
//!
//! ## Authority boundary (COGNO-1 V2 §3, §4, §8, §23)
//!
//! **Hard constraints remain applied by `cogno-core`** and are never converted
//! into compensable reward terms (S4). The differentiable objective produced
//! here is a *soft* score component consumed by the reward engine **after**
//! the lexicographic hard/capability/privacy gates. The
//! [`cogno_core::MetaObjective`] gate (§27 Phase 4) controls activation: the
//! backend's [`backend::SciRustBackend::objective`] refuses to run unless all
//! six preconditions are attested by the host, so adding this crate never
//! silently downgrades the MVP.
//!
//! ## Scalar oracle (COGNO-1 V2 §"comparé à cet oracle")
//!
//! `cogno_core::RewardEngine` (integer scalar, Phase 0) is the numerical
//! oracle. The SciRust backend is compared to it on deterministic batches:
//! see `tests/oracle.rs`. Because the backend is float and the oracle is
//! integer, the comparison asserts **ranking consistency** on deterministic
//! batches (order-preserving), not float-equality — matching the §8
//! lexicographic tie-break which is order-based, not magnitude-based.
//!
//! ## Why self-contained (no external `scirust` dependency)
//!
//! The external `scirust` crate (v0.0.5) does not compile on stable Rust 1.97
//! (37 errors: `#[feature]` not allowed on stable, missing `Self: Sized`
//! bounds, edition-incompatible code), has no autograd, pulls 20 transitive
//! crates including `rdrand` (RDRAND `unsafe`), uses `panic!` extensively, and
//! is unmaintained. The spec explicitly requires adapting such components
//! behind a fallible, bounded COGNO API (§"L'intégration ne doit pas réutiliser
//! aveuglément les prototypes existants"). §24 (minimal/pinned/audited deps)
//! and §23 (`forbid(unsafe_code)`) reinforce this. See
//! `docs/DEPENDENCIES.md` for the recorded decision.
//!
//! ## SIMD
//!
//! `std::simd` (`portable_simd`) requires nightly; COGNO-1 targets stable
//! Rust. This crate therefore uses autovectorization-friendly SoA layouts and
//! `#[inline(always)]` inner loops that LLVM auto-vectorizes. Explicit
//! `portable_simd` paths are left as a future nightly feature.
//!
//! ## Crate lint policy (§23)
//!
//! ```ignore
//! #![forbid(unsafe_code)]
//! #![deny(warnings, missing_debug_implementations, unreachable_pub)]
//! ```
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]

pub mod backend;
pub mod calib;
pub mod cost;
pub mod engine;
pub mod error;
pub mod kv;
pub mod losses;
pub mod nn;
pub mod optim;
pub mod sequence;
pub mod sequence_classifier;
pub mod sequence_cognitive;
pub mod sequence_retriever;
pub mod sequence_scorer;
pub mod sequence_symbolic;
pub mod tensor;

pub use backend::{BackendReport, Config, SciRustBackend};
pub use calib::{
    CalibratedConfidence, Calibration, CalibrationFitConfig, CalibrationFitReport,
    MAX_CALIBRATION_EPOCHS, MAX_CALIBRATION_EXAMPLES,
};
pub use cost::{Cost, CostBreakdown};
pub use engine::{Op, Tape, Var};
pub use error::{SciRustError, SciRustResult};
pub use kv::{BoundedKvCache, KvCachePolicy, KvPushError};
pub use losses::{InfoNCE, PairwiseLoss, SymbolicSatisfaction};
pub use nn::{
    Mlp, MlpAdamW, MlpConfig, MlpGradients, MAX_MLP_DIM, MAX_MLP_PARAMETERS, MAX_MLP_TAPE_NODES,
};
pub use optim::{AdamW, AmsGrad, Optimizer};
pub use sequence::{
    SequenceEncoder, SequenceEncoderAdamW, SequenceEncoderConfig, SequenceEncoderGradients,
    SequenceEncoderGraph, MAX_SEQUENCE_ACTIVATION_ELEMENTS, MAX_SEQUENCE_EMBEDDING_DIM,
    MAX_SEQUENCE_HIDDEN_DIM, MAX_SEQUENCE_PARAMETERS, MAX_SEQUENCE_TOKENS, MAX_SEQUENCE_VOCAB,
    SEQUENCE_ENCODER_TAPE_NODES, SEQUENCE_TRAINING_TAPE_NODES,
};
pub use sequence_classifier::{
    SequenceClassifier, SequenceClassifierAdamW, SequenceClassifierConfig,
    SequenceClassifierGradients, MAX_SEQUENCE_CLASSES, MAX_SEQUENCE_CLASSIFIER_PARAMETERS,
    SEQUENCE_CLASSIFIER_TAPE_NODES, SEQUENCE_CLASSIFIER_TRAINING_TAPE_NODES,
};
pub use sequence_cognitive::{
    SequenceCognitiveConfig, SequenceCognitiveHeads, COGNITIVE_CONTRADICTION_CLASSES,
    MAX_SEQUENCE_COGNITIVE_PARAMETERS,
};
pub use sequence_retriever::{
    SequenceRetriever, SequenceRetrieverAdamW, SequenceRetrieverConfig, SequenceRetrieverGradients,
    MAX_SEQUENCE_RETRIEVAL_CANDIDATES, MAX_SEQUENCE_RETRIEVER_PARAMETERS,
    SEQUENCE_RETRIEVER_TAPE_NODES,
};
pub use sequence_scorer::{
    SequenceScorer, SequenceScorerAdamW, SequenceScorerConfig, SequenceScorerGradients,
    MAX_SEQUENCE_SCORER_PARAMETERS, SEQUENCE_PAIRWISE_TAPE_NODES, SEQUENCE_SCORER_TAPE_NODES,
};
pub use sequence_symbolic::{
    SequenceSymbolicAdamW, SequenceSymbolicConfig, SequenceSymbolicGradients, SequenceSymbolicHead,
    MAX_SEQUENCE_SYMBOLIC_PARAMETERS, MAX_SEQUENCE_SYMBOLIC_RULES,
    SEQUENCE_SYMBOLIC_SUPERVISED_TAPE_NODES, SEQUENCE_SYMBOLIC_TAPE_NODES,
};
pub use tensor::{Shape, Tensor};
