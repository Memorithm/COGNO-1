//! Deterministic pipeline orchestrator (COGNO-1 V2 §3, §8).
//!
//! Wires the §3 stages together. The pipeline is **borrowed-only** on the hot
//! path: scratch buffers come pre-allocated; no allocation happens between
//! trust-classification and the final decision. Side effects are forbidden
//! until the chain completes.

use cogno_core::{
    decide, symbolic, validate_proposal, Candidate, CandidateScore, CognoProposalView,
    DataClassification, Decision, RejectReason, Reward, RewardParams, SafetyPolicy, TrustClass,
};

/// Surface parameters for one run of the pipeline. Borrowed or `Copy`; the
/// runtime owns the underlying storage.
#[derive(Clone, Debug)]
pub struct PipelineParams<'a> {
    pub policy: SafetyPolicy,
    pub reward_params: RewardParams,
    pub data_class: DataClassification,
    pub trust: TrustClass,
    pub reward_signal: Reward,
    pub proposal: Option<CognoProposalView<'a>>,
}

/// Outcome of one pipeline run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineOutcome {
    /// Stage 1 — trust/size ok, parsing rejected.
    Rejected {
        stage: &'static str,
        reason: RejectReason,
    },
    /// Stage 5 — admissible; rewarded with the deterministic score.
    Eligible { score: CandidateScore },
}

/// The pipeline struct is stateless on the hot path; a policy snapshot suffices.
#[derive(Debug, Default)]
pub struct Pipeline;

impl Pipeline {
    /// Run one proposal through the §3 stages. Returns the deterministic
    /// decision. Every rejection carries the stage name so the audit layer
    /// can record it without re-deriving.
    pub fn run(&self, p: &PipelineParams<'_>) -> PipelineOutcome {
        // Stage 0 — trust classification (§6) is data carried in `p.trust`;
        // an untrusted trust class can never mint privileges past the hard
        // gates below.

        // Stage 1a — size control. Handled upstream by admission; this stage
        // does the cheap field-bound checks of the strict schema.

        let Some(proposal) = p.proposal else {
            return PipelineOutcome::Rejected {
                stage: "size",
                reason: RejectReason::Malformed,
            };
        };

        // Stage 1b — strict parsing + structural validation (§2).
        if let Err(reason) = validate_proposal(&proposal) {
            return PipelineOutcome::Rejected {
                stage: "structural",
                reason,
            };
        }

        // Stage 2 — symbolic validation.
        let sym = symbolic(&proposal);
        if !sym.passed {
            return PipelineOutcome::Rejected {
                stage: "symbolic",
                reason: sym.reason.unwrap_or(RejectReason::Malformed),
            };
        }

        // Stage 3 — hard safety rules (lexicographic step 2, §8).
        // Data classification gate: this is a privacy gate (step 4) AND a
        // hard gate (Secret is always hard-rejected, §20).
        match p.policy.check_data(p.data_class) {
            cogno_core::HardVerdict::Reject(reason) => {
                return PipelineOutcome::Rejected {
                    stage: "hard",
                    reason,
                };
            }
            cogno_core::HardVerdict::Allow => {}
        }

        // Stage 4 — reward engine (scalar). Runs ONLY after the hard gates
        // (S4: a hard violation cannot be compensated).
        let engine = cogno_core::RewardEngine;
        let score = match engine.score(
            proposal.confidence_bps,
            proposal.evidence_ids.len(),
            p.reward_signal,
            p.reward_params,
        ) {
            Ok(s) => s,
            Err(_) => {
                return PipelineOutcome::Rejected {
                    stage: "reward",
                    reason: RejectReason::Overflow,
                };
            }
        };

        // Stage 5 — lexicographic decision. The local candidate adapter
        // encodes the §8 order: structural already passed, hard already
        // passed, capability/privacy always allowed here (we are inside the
        // MVP read-only path with no tool side effect).
        struct AdmittedCandidate;
        impl Candidate for AdmittedCandidate {
            fn is_structurally_valid(&self) -> bool {
                true
            }
            fn hard_constraint_ok(&self) -> bool {
                true
            }
            fn capability_allowed(&self) -> bool {
                true
            }
            fn privacy_rejection(&self) -> Option<RejectReason> {
                None
            }
        }
        match decide(&AdmittedCandidate, score) {
            Decision::Reject(reason) => PipelineOutcome::Rejected {
                stage: "decision",
                reason,
            },
            Decision::Eligible(score) => PipelineOutcome::Eligible { score },
        }
    }
}
