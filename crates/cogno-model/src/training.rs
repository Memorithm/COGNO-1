//! Phase 3 — supervised training corpus and toy trainer (COGNO-1 V2 §27 Phase 3).
//!
//! Honest placeholder for a real differentiable backend. Phase 4 requires a
//! true tensor engine, log-probabilities and held-out tests; that is
//! intentionally not present here. This trainer builds a small hashed-feature
//! integer perceptron that is good enough to exercise the full
//! provenance/splits/adversarial-example pipeline and to give Phase 2's
//! `ReadOnlyModel` a non-trivial, deterministic classifier to load.
//!
//! Guarantees honored from the spec:
//!  - examples carry provenance (§9, S6) and explicit data classification (§20);
//!  - `Secret` and unauthorized `Confidential` data are rejected before storage;
//!  - stored examples are exposed read-only outside `cogno-model`;
//!  - splits are deterministic (train/val/test, seeded shuffle);
//!  - the trainer accepts contradictory, adversarial, malformed and negative
//!    examples so downstream tests can poison it deterministically;
//!  - no `unsafe`, no network, no FS, no floats (integer perceptron).

use crate::training_data_policy::{
    HostConfidentialTrainingAttestation, TrainingDataAdmissionPolicy, TrainingDataGovernanceError,
};
use cogno_core::DataClassification;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Opaque label id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub u16);

impl Label {
    #[must_use]
    pub const fn from_u16(v: u16) -> Self {
        Self(v)
    }
}

/// Provenance of a single example (§9, S6). The fingerprint covers the label
/// and the payload bytes; the runtime computes it once and the trainer stores
/// it. Duplicates are detected by fingerprint (§9).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance {
    pub origin: cogno_core::InputOrigin,
    pub evidence_origin: cogno_core::EvidenceOrigin,
    pub fingerprint: cogno_core::Fingerprint,
}

/// A labeled training example with explicit §20 data classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabeledExample {
    pub label: Label,
    pub payload: Vec<u8>,
    pub provenance: Provenance,
    data_class: DataClassification,
}

impl LabeledExample {
    /// Convenience constructor that computes the canonical fingerprint from the
    /// payload and label while requiring an explicit data classification.
    ///
    /// Classification intentionally does not alter the fingerprint, so an
    /// identical example cannot bypass duplicate detection by being relabelled.
    #[must_use]
    pub fn new(
        label: Label,
        payload: Vec<u8>,
        origin: cogno_core::InputOrigin,
        evidence_origin: cogno_core::EvidenceOrigin,
        data_class: DataClassification,
    ) -> Self {
        let mut h = DefaultHasher::new();
        h.write_u16(label.0);
        h.write(&payload);
        let hash = h.finish();
        let mut fp = [0u8; 32];
        fp[..8].copy_from_slice(&hash.to_le_bytes());
        Self {
            label,
            payload,
            provenance: Provenance {
                origin,
                evidence_origin,
                fingerprint: cogno_core::Fingerprint(fp),
            },
            data_class,
        }
    }

    /// Host-owned §20 classification attached before corpus insertion.
    #[must_use]
    pub const fn data_class(&self) -> DataClassification {
        self.data_class
    }
}

/// Which split an example belongs to. Deterministic, seeded shuffle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SplitKind {
    Train,
    Validation,
    Test,
}

/// A split view over a corpus. No copies: holds indices into the source `Corpus`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusSplit {
    pub kind: SplitKind,
    pub indices: Vec<usize>,
}

/// In-memory corpus carrying provenance and classification per example.
///
/// The public insertion boundary validates §20 before the example reaches the
/// corpus and then deduplicates by canonical fingerprint (§9). Stored examples
/// cannot be mutated by external crates after admission.
#[derive(Debug)]
pub struct Corpus {
    pub(crate) examples: Vec<LabeledExample>,
    seen: std::collections::HashSet<cogno_core::Fingerprint>,
    pub seed: u64,
    policy: TrainingDataAdmissionPolicy,
}

impl Default for Corpus {
    fn default() -> Self {
        Self::with_seed(0)
    }
}

impl Corpus {
    /// Construct with a deterministic seed. Only `Public` and `Internal` data
    /// are admitted by default.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            examples: Vec::new(),
            seen: std::collections::HashSet::new(),
            seed,
            policy: TrainingDataAdmissionPolicy::DEFAULT,
        }
    }

    /// Construct a corpus whose host explicitly authorizes `Confidential`
    /// training data. `Secret` remains forbidden unconditionally.
    #[must_use]
    pub fn with_seed_and_confidential_authorization(
        seed: u64,
        attestation: HostConfidentialTrainingAttestation,
    ) -> Self {
        Self {
            examples: Vec::new(),
            seen: std::collections::HashSet::new(),
            seed,
            policy: TrainingDataAdmissionPolicy::with_confidential_authorization(attestation),
        }
    }

    /// Validate classification before storage, then deduplicate by fingerprint.
    ///
    /// Governance rejection is an `Err`; a canonical duplicate is `Ok(false)`;
    /// a newly stored example is `Ok(true)`.
    pub fn add(&mut self, ex: LabeledExample) -> Result<bool, TrainingDataGovernanceError> {
        let index = self.examples.len();
        self.policy.validate(ex.data_class(), index)?;
        if !self.seen.insert(ex.provenance.fingerprint) {
            return Ok(false);
        }
        self.examples.push(ex);
        Ok(true)
    }

    /// Read-only view of admitted examples.
    #[must_use]
    pub fn examples(&self) -> &[LabeledExample] {
        &self.examples
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub(crate) fn validate_data_classifications(
        &self,
    ) -> Result<(), TrainingDataGovernanceError> {
        for (index, example) in self.examples.iter().enumerate() {
            self.policy.validate(example.data_class(), index)?;
        }
        Ok(())
    }

    /// Deterministically split into train/validation/test by ratio. Uses a
    /// simple seeded linear-congruential shuffle — no external `rand`.
    /// Ratios sum should be 1.0; the test split gets the remainder to avoid
    /// floating-point boundary drift.
    #[must_use]
    pub fn split(
        &self,
        train_ratio: f32,
        val_ratio: f32,
    ) -> (CorpusSplit, CorpusSplit, CorpusSplit) {
        let n = self.examples.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = self.seed.wrapping_add(0x9E3779B97F4A7C15);
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
        let train_end = (n as f32 * train_ratio).round() as usize;
        let val_end = train_end + (n as f32 * val_ratio).round() as usize;
        let train_end = train_end.min(n);
        let val_end = val_end.min(n);
        (
            CorpusSplit {
                kind: SplitKind::Train,
                indices: order[..train_end].to_vec(),
            },
            CorpusSplit {
                kind: SplitKind::Validation,
                indices: order[train_end..val_end].to_vec(),
            },
            CorpusSplit {
                kind: SplitKind::Test,
                indices: order[val_end..].to_vec(),
            },
        )
    }
}

/// Hashed-feature integer perceptron. Trained by [`ToyTrainer::train`]; frozen
/// and handed to [`crate::readonly::ReadOnlyModel`] for Phase 2 inference.
#[derive(Clone, Debug)]
pub struct TrainedModel {
    pub num_features: usize,
    pub weights: Vec<i32>,
    pub num_labels: u16,
    pub bias: Vec<i32>,
}

impl TrainedModel {
    /// Feature indices for a payload. 4-byte rolling hash into `num_features`.
    #[must_use]
    pub fn features(num_features: usize, payload: &[u8]) -> Vec<usize> {
        let mut out = Vec::with_capacity(payload.len().saturating_add(1) / 4 + 1);
        let mut h: u64 = 0xCBF29CE484222325;
        for &b in payload {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001B3);
            if h.trailing_zeros() >= 5 {
                out.push((h as usize) % num_features.max(1));
            }
        }
        out.push(0); // bias feature
        out
    }

    /// Score for a single label.
    #[must_use]
    pub fn score(&self, label: u16, payload: &[u8]) -> i64 {
        if label >= self.num_labels {
            return i64::MIN;
        }
        let feats = Self::features(self.num_features, payload);
        let mut s = i64::from(self.bias[label as usize]);
        let base = label as usize * self.num_features;
        for &f in &feats {
            s = s.saturating_add(i64::from(self.weights[base + f]));
        }
        s
    }

    /// Argmax over labels.
    #[must_use]
    pub fn classify(&self, payload: &[u8]) -> Label {
        let mut best = Label(0);
        let mut best_score = i64::MIN;
        for l in 0..self.num_labels {
            let s = self.score(l, payload);
            if s > best_score {
                best_score = s;
                best = Label(l);
            }
        }
        best
    }
}

/// Honest placeholder trainer for Phase 3.
#[derive(Debug)]
pub struct ToyTrainer {
    pub num_features: usize,
    pub epochs: u32,
}

impl ToyTrainer {
    /// Construct a trainer. `num_features` should be small for tests (e.g. 256).
    #[must_use]
    pub fn new(num_features: usize, epochs: u32) -> Self {
        Self {
            num_features: num_features.max(64),
            epochs,
        }
    }

    /// Train an integer perceptron on a split. Returns the trained model and
    /// the final train accuracy (correct / total). Stops saturating on weight
    /// overflow; keeps training to obey the epoch budget.
    pub fn train(&self, corpus: &Corpus, split: &CorpusSplit) -> (TrainedModel, f32) {
        let num_labels = corpus
            .examples
            .iter()
            .map(|e| e.label.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut m = TrainedModel {
            num_features: self.num_features,
            weights: vec![0i32; (num_labels as usize) * self.num_features],
            num_labels,
            bias: vec![0i32; num_labels as usize],
        };

        let mut correct: usize = 0;
        let mut total: usize = 0;
        for epoch in 0..self.epochs {
            for &i in &split.indices {
                let ex = &corpus.examples[i];
                total = total.saturating_add(1);
                let pred = m.classify(&ex.payload);
                if pred == ex.label {
                    correct = correct.saturating_add(1);
                } else {
                    let feats = TrainedModel::features(m.num_features, &ex.payload);
                    let true_lbl = ex.label.0 as usize;
                    let pred_lbl = pred.0 as usize;
                    let base_true = true_lbl * m.num_features;
                    let base_pred = pred_lbl * m.num_features;
                    for &f in &feats {
                        m.weights[base_true + f] = m.weights[base_true + f].saturating_add(1);
                        m.weights[base_pred + f] = m.weights[base_pred + f].saturating_sub(1);
                    }
                    m.bias[true_lbl] = m.bias[true_lbl].saturating_add(1);
                    m.bias[pred_lbl] = m.bias[pred_lbl].saturating_sub(1);
                }
            }
            if epoch == self.epochs - 1 {
                correct = 0;
                total = 0;
                for &i in &split.indices {
                    let ex = &corpus.examples[i];
                    total = total.saturating_add(1);
                    if m.classify(&ex.payload) == ex.label {
                        correct = correct.saturating_add(1);
                    }
                }
            }
        }
        let acc = if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        };
        (m, acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{EvidenceOrigin, InputOrigin};

    fn example(data_class: DataClassification) -> LabeledExample {
        LabeledExample::new(
            Label::from_u16(0),
            b"classified training payload".to_vec(),
            InputOrigin::TrainingCorpus,
            EvidenceOrigin::TestResult,
            data_class,
        )
    }

    #[test]
    fn secret_never_enters_historical_training_corpus() {
        let mut corpus = Corpus::with_seed_and_confidential_authorization(
            1,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            corpus.add(example(DataClassification::Secret)),
            Err(TrainingDataGovernanceError::SecretTrainingData { index: 0 })
        );
        assert!(corpus.is_empty());
    }

    #[test]
    fn confidential_requires_attestation_before_storage() {
        let mut denied = Corpus::with_seed(2);
        assert_eq!(
            denied.add(example(DataClassification::Confidential)),
            Err(TrainingDataGovernanceError::ConfidentialTrainingDataRequiresAuthorization {
                index: 0,
            })
        );
        assert!(denied.is_empty());

        let mut allowed = Corpus::with_seed_and_confidential_authorization(
            3,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(
            allowed.add(example(DataClassification::Confidential)),
            Ok(true)
        );
        assert_eq!(allowed.len(), 1);
    }

    #[test]
    fn relabelling_classification_does_not_bypass_deduplication() {
        let mut corpus = Corpus::with_seed_and_confidential_authorization(
            4,
            HostConfidentialTrainingAttestation::authorize_confidential_training_data(),
        );
        assert_eq!(corpus.add(example(DataClassification::Internal)), Ok(true));
        assert_eq!(
            corpus.add(example(DataClassification::Confidential)),
            Ok(false)
        );
        assert_eq!(corpus.len(), 1);
    }

    #[test]
    fn defensive_revalidation_detects_internal_post_storage_tampering() {
        let mut corpus = Corpus::with_seed(5);
        assert_eq!(corpus.add(example(DataClassification::Internal)), Ok(true));
        corpus.examples[0].data_class = DataClassification::Secret;
        assert_eq!(
            corpus.validate_data_classifications(),
            Err(TrainingDataGovernanceError::SecretTrainingData { index: 0 })
        );
    }
}
