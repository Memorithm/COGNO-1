# Scientific Taste Campaign Review Phase

This phase follows the bounded campaign runner and adds a deterministic, non-authoritative review gate over completed campaign reports.

## Goal

Convert a successful multi-generation campaign report into a reproducible recommendation for controlled restart review without granting the campaign, SciAgent, or any model authority to persist, install, replace, or hot-swap a verified runtime taste profile.

## Gate inputs

The gate consumes only a completed `TasteCampaignReport` plus an explicit `TasteCampaignReviewPolicy`.

It checks:

- campaign generation count and monotonicity;
- aggregate benchmark regret and regressed cases;
- final active state for every longitudinally observed preference;
- latest confidence against the configured threshold;
- longitudinal contradiction, weakening, and staleness;
- duplicate longitudinal preference identifiers.

## Output

The result is either:

- `Hold`; or
- `EligibleForControlledRestartReview`.

The result always carries `TasteCampaignReviewAuthority::ReviewOnly`.

Eligibility is not profile installation. Existing transactional generation persistence, artifact digest verification, restart-time loading, and sealed runtime profile installation remain the only authority-bearing path.

## Default policy

- minimum successful generations: `2`;
- minimum latest confidence: `7000` basis points;
- maximum regressed benchmark cases: `0`.

The aggregate taste regret must also be no greater than aggregate baseline regret.

## Fail-closed behavior

Malformed campaign shape, generation-count mismatches, invalid generation ranges, non-monotonic autonomous reports, invalid policy values, and duplicate longitudinal preferences are rejected before a review report is returned.

Quarantined model observations are counted for provenance visibility but cannot validate promotion and do not receive authority through this phase.
