# AXCOG4 — reconcile ambiguous effects before re-proposal

COGNO-1 binds a RemoteOps receipt to the exact task attempt, task binding,
workspace snapshot, model artifact and typed proposal. If the remote outcome is
unknown, the receipt is unauthenticated or any binding differs, the result
requires external reconciliation. A receipt that confirms an effect blocks
another attempt.

Only an authenticated receipt that confirms no effect can produce a
digest-only ReproposalCandidate. The candidate starts a fresh deterministic
authorization cycle; it is not permission to execute, retry or trade. The
current MVP executor remains disabled.

## Ownership boundary

RemoteOps owns durable host evidence, the remote execution identity, target
host identity and authentication of its receipt. SciRust Hub owns task identity
and lifecycle. COGNO-1 owns comparison against the current deterministic task
scope and exact model proposal. The RemoteOpsEffectReceiptVerifier trait is an
adapter boundary; structural validation inside COGNO-1 does not authenticate
host evidence.

The receipt retains only SHA-256 references for attempt and execution
identifiers, host identity, workspace, model, task binding, proposal and
evidence. Raw task bindings and proposal arguments are not copied into the
receipt or candidate.

## Fail-closed cases

- Unknown outcome requires reconciliation and cannot yield a candidate.
- Confirmed effect blocks another attempt.
- Wrong attempt, task, workspace, model or proposal cannot be reused.
- Missing provenance, non-effect capability, scope mismatch, malformed
  identifiers, unsupported schema or failed authentication cannot yield a
  candidate.
- A successful decision never invokes the tool executor.

This contract defines the deterministic COGNO consumer boundary. A production
adapter must verify a durable RemoteOps receipt before this API can produce a
candidate. No host-side receipt verifier or trading executor is enabled here.
