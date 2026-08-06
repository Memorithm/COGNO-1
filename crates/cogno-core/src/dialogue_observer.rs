//! Silent dialogue observation for COGNO-1.
//!
//! This module converts externally validated dialogue envelopes into bounded,
//! append-only journal events. It never generates a reply, executes a tool, or
//! promotes model output into trusted policy.

use crate::evidence::{EvidenceId, EvidenceOrigin};
use crate::journal::{AppendOutcome, EventId, Fingerprint, Journal, JournalEvent};
use crate::origin::InputOrigin;

pub const DIALOGUE_OBSERVATION_CATEGORY_TAG: u16 = 0x444f;
pub const MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedDialogueKind {
    Question,
    Hypothesis,
    Critique,
    Counterexample,
    ExperimentRequest,
    ExperimentResult,
    PreferenceObservation,
    Contradiction,
    Explanation,
    Decision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedSenderClass {
    Human,
    ExternalModel,
    SciAgent,
    RuntimeDiscovery,
    DeterministicKernel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialogueObservation {
    pub fingerprint: [u8; 32],
    pub evidence_id: u64,
    pub sender_class: ObservedSenderClass,
    pub message_kind: ObservedDialogueKind,
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogueObservationError {
    ZeroFingerprint,
    PayloadTooLarge,
    EmptyPayload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogueObservationOutcome {
    Appended { id: EventId },
    Duplicate { id: EventId },
}

impl DialogueObservation {
    pub fn validate(&self) -> Result<(), DialogueObservationError> {
        if self.fingerprint.iter().all(|byte| *byte == 0) {
            return Err(DialogueObservationError::ZeroFingerprint);
        }
        if self.payload.is_empty() {
            return Err(DialogueObservationError::EmptyPayload);
        }
        if self.payload.len() > MAX_DIALOGUE_OBSERVATION_PAYLOAD_BYTES {
            return Err(DialogueObservationError::PayloadTooLarge);
        }
        Ok(())
    }

    fn input_origin(&self) -> InputOrigin {
        match self.sender_class {
            ObservedSenderClass::Human => InputOrigin::ExplicitUserInstruction,
            ObservedSenderClass::ExternalModel | ObservedSenderClass::SciAgent => {
                InputOrigin::ModelOutput
            }
            ObservedSenderClass::RuntimeDiscovery => InputOrigin::ToolOutput,
            ObservedSenderClass::DeterministicKernel => InputOrigin::SystemPolicy,
        }
    }

    fn evidence_origin(&self) -> EvidenceOrigin {
        match self.sender_class {
            ObservedSenderClass::Human => EvidenceOrigin::ExplicitUserStatement,
            ObservedSenderClass::ExternalModel | ObservedSenderClass::SciAgent => {
                EvidenceOrigin::ModelInference
            }
            ObservedSenderClass::RuntimeDiscovery => EvidenceOrigin::TestResult,
            ObservedSenderClass::DeterministicKernel => EvidenceOrigin::FormalValidator,
        }
    }
}

pub fn observe_dialogue(
    journal: &mut Journal,
    observation: DialogueObservation,
) -> Result<DialogueObservationOutcome, DialogueObservationError> {
    observation.validate()?;
    let event = JournalEvent {
        id: EventId(0),
        fingerprint: Fingerprint(observation.fingerprint),
        origin: observation.input_origin(),
        evidence_origin: observation.evidence_origin(),
        evidence_id: EvidenceId::from_u64(observation.evidence_id),
        category_tag: DIALOGUE_OBSERVATION_CATEGORY_TAG,
        payload: observation.payload,
    };
    Ok(match journal.append(event) {
        AppendOutcome::Appended { id } => DialogueObservationOutcome::Appended { id },
        AppendOutcome::Duplicate { id } => DialogueObservationOutcome::Duplicate { id },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> DialogueObservation {
        DialogueObservation {
            fingerprint: [7; 32],
            evidence_id: 11,
            sender_class: ObservedSenderClass::SciAgent,
            message_kind: ObservedDialogueKind::Hypothesis,
            payload: br#"{"message_id":"m1","kind":"hypothesis"}"#.to_vec(),
        }
    }

    #[test]
    fn model_dialogue_is_journaled_as_model_inference() {
        let mut journal = Journal::default();
        let outcome = observe_dialogue(&mut journal, observation()).expect("valid observation");
        assert_eq!(
            outcome,
            DialogueObservationOutcome::Appended { id: EventId(0) }
        );
        let event = journal.iter().next().expect("event");
        assert_eq!(event.origin, InputOrigin::ModelOutput);
        assert_eq!(event.evidence_origin, EvidenceOrigin::ModelInference);
        assert_eq!(event.category_tag, DIALOGUE_OBSERVATION_CATEGORY_TAG);
    }

    #[test]
    fn repeated_dialogue_is_deduplicated() {
        let mut journal = Journal::default();
        observe_dialogue(&mut journal, observation()).expect("first observation");
        let outcome = observe_dialogue(&mut journal, observation()).expect("duplicate observation");
        assert_eq!(
            outcome,
            DialogueObservationOutcome::Duplicate { id: EventId(0) }
        );
        assert_eq!(journal.len(), 1);
    }

    #[test]
    fn zero_fingerprint_fails_closed() {
        let mut item = observation();
        item.fingerprint = [0; 32];
        assert_eq!(
            item.validate(),
            Err(DialogueObservationError::ZeroFingerprint)
        );
    }
}
