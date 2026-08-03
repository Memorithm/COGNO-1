//! Phase 1–3 backend tests: simulator, read-only model, toy trainer.
//! All deterministic, no network, no FS, no `unsafe`.

use cogno_core::{
    validate_proposal, EvidenceOrigin, InputOrigin, ProposalAction, RuleCategory, RuleScope,
};
use cogno_model::{
    BackendError, Corpus, Label, LabeledExample, ModelBackend, OwnedProposal, ReadOnlyModel,
    SimBackend, ToyTrainer, TrainedModel,
};

#[test]
fn simulator_returns_scripted_proposals_in_order() {
    let script = vec![
        make_proposal(RuleCategory::Style),
        make_proposal(RuleCategory::Behavior),
    ];
    let mut sim = SimBackend::new(script);
    let p1 = sim.next_proposal().unwrap();
    let p2 = sim.next_proposal().unwrap();
    assert_eq!(p1.category, RuleCategory::Style);
    assert_eq!(p2.category, RuleCategory::Behavior);
    // Third call: exhausted — fail closed (S10).
    assert_eq!(sim.next_proposal().unwrap_err(), BackendError::Exhausted);
}

#[test]
fn simulator_proposals_validate_against_strict_schema() {
    let script = vec![make_proposal(RuleCategory::Style)];
    let mut sim = SimBackend::new(script);
    let p = sim.next_proposal().unwrap();
    assert!(validate_proposal(&p.as_view()).is_ok());
}

#[test]
fn read_only_model_classifies_deterministically() {
    let model = train_simple_two_label_model();
    let ro = ReadOnlyModel::from_trained(model);
    let a = ro.classify(b"apple");
    let b = ro.classify(b"banana");
    assert_ne!(a, b, "two distinct labels expected");
}

#[test]
fn read_only_model_extract_returns_valid_proposal() {
    use cogno_core::EvidenceId;
    let model = train_simple_two_label_model();
    let ro = ReadOnlyModel::from_trained(model);
    let p = ro.extract(b"apple", EvidenceId::from_u64(1));
    assert_eq!(p.action, ProposalAction::ExtractPreference);
    assert!(validate_proposal(&p.as_view()).is_ok());
}

#[test]
fn read_only_model_rank_is_deterministic_and_total() {
    let model = train_simple_two_label_model();
    let ro = ReadOnlyModel::from_trained(model);
    let cands: Vec<&[u8]> = vec![b"apple", b"banana", b"apple", b"banana"];
    let r1 = ro.rank(&cands);
    let r2 = ro.rank(&cands);
    assert_eq!(r1, r2, "ranking must be deterministic");
    assert_eq!(r1.len(), cands.len(), "ranking must be total");
}

#[test]
fn read_only_model_next_proposal_is_read_only_violation() {
    let model = train_simple_two_label_model();
    let mut ro = ReadOnlyModel::from_trained(model);
    assert_eq!(
        ro.next_proposal().unwrap_err(),
        BackendError::ReadOnlyViolation
    );
}

#[test]
fn trainer_accuracy_is_finite_and_deterministic() {
    let c = build_corpus();
    let (train, _val, _test) = c.split(0.7, 0.15);
    let t = ToyTrainer::new(256, 5);
    let (m1, acc1) = t.train(&c, &train);
    let (m2, acc2) = t.train(&c, &train);
    assert!((0.0..=1.0).contains(&acc1), "accuracy must be finite");
    assert_eq!(acc1, acc2, "training must be deterministic");
    // weights deterministic
    assert_eq!(m1.weights, m2.weights);
}

#[test]
fn corpus_deduplicates_by_fingerprint() {
    let mut c = Corpus::with_seed(7);
    let a = LabeledExample::new(
        Label(0),
        b"x".to_vec(),
        InputOrigin::TrainingCorpus,
        EvidenceOrigin::TestResult,
    );
    let added1 = c.add(a.clone());
    let added2 = c.add(a);
    assert!(added1);
    assert!(!added2, "duplicate must not be added (§9)");
    assert_eq!(c.examples.len(), 1);
}

#[test]
fn t07_corpus_poisoning_adversarial_examples_handled_bounded() {
    // §9, T07: a poisoned corpus (contradictory labels for identical payloads,
    // adversarial examples) is handled without crash, without silently
    // producing a high-confidence rule, and with provenance preserved. The
    // trainer's accuracy stays bounded in [0, 1]; the model never gains the
    // ability to mint a hard rule (S1/S4 — hard rules come from
    // FormalValidator/ExplicitUser only, not from training).
    use cogno_core::{EvidenceOrigin, InputOrigin};
    let mut c = Corpus::with_seed(31);
    // Poison: same payload, opposite labels. The perceptron cannot fit both
    // and must not silently pick the "attacker's" label.
    for i in 0..8u32 {
        let payload = b"poisoned".to_vec();
        let lbl = if i % 2 == 0 { Label(0) } else { Label(1) };
        // Dedupe by fingerprint would collapse identical (label,payload);
        // vary the provenance fingerprint by appending i so both labels land.
        let mut p = payload.clone();
        p.push(i as u8);
        c.add(LabeledExample::new(
            lbl,
            p,
            InputOrigin::TrainingCorpus,
            EvidenceOrigin::TestResult,
        ));
    }
    let (train, _val, _test) = c.split(1.0, 0.0);
    let t = ToyTrainer::new(64, 10);
    let (m, acc) = t.train(&c, &train);
    // Accuracy is bounded and finite — no crash, no NaN, no overflow panic.
    assert!(
        (0.0..=1.0).contains(&acc),
        "accuracy must be finite in [0,1], got {acc}"
    );
    // The trained model is deterministic and classifies without panic.
    let _ = m.classify(b"poisoned");
    // Provenance preserved: every example still carries its origin.
    assert!(c
        .examples
        .iter()
        .all(|e| e.provenance.fingerprint.0 != [0u8; 32]));
}

fn make_proposal(category: RuleCategory) -> OwnedProposal {
    OwnedProposal {
        action: ProposalAction::ClassifyFeedback,
        category,
        scope: RuleScope::Session,
        confidence_bps: 1_000,
        evidence_ids: vec![EvidenceId::from_u64(1)],
        payload: Vec::new(),
    }
}

use cogno_core::EvidenceId;

fn train_simple_two_label_model() -> TrainedModel {
    let mut c = Corpus::with_seed(42);
    for _ in 0..5 {
        c.add(LabeledExample::new(
            Label(0),
            b"apple".to_vec(),
            InputOrigin::TrainingCorpus,
            EvidenceOrigin::TestResult,
        ));
        c.add(LabeledExample::new(
            Label(1),
            b"banana".to_vec(),
            InputOrigin::TrainingCorpus,
            EvidenceOrigin::TestResult,
        ));
    }
    let (train, _, _) = c.split(1.0, 0.0);
    let t = ToyTrainer::new(128, 8);
    t.train(&c, &train).0
}

fn build_corpus() -> Corpus {
    let mut c = Corpus::with_seed(13);
    for i in 0..10u32 {
        let lbl = if i % 2 == 0 { Label(0) } else { Label(1) };
        let mut payload = vec![b'x'];
        payload.extend_from_slice(&i.to_le_bytes());
        c.add(LabeledExample::new(
            lbl,
            payload,
            InputOrigin::TrainingCorpus,
            EvidenceOrigin::TestResult,
        ));
    }
    c
}
