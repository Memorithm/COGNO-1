# `tests/adversarial/` — tests de sécurité obligatoires (§25)

Chaque menace de `docs/THREAT_MODEL.md` possède **au moins un test**. Les tests
vivants résident dans les crates (pratique Rust : tests d'intégration par
crate), Churchill du cahier ci-dessous. Un test **réussit** quand le système
**rejette** (fail-closed, S10).

## Fichiers de tests exécutables

- `crates/cogno-core/tests/adversarial.rs` — 28 tests couvrant §2 (validation
  schéma), §11 (budget), §15 (bounded), §18 (journal/profile), §9
  (quarantaine modèle), §8 (décision avant récompense), §20 (secret/confidential),
  §7 (proposition shell), §22 (chemin `..` / hors-racine), §21 (manifeste).
- `crates/cogno-runtime/tests/runtime_integration.rs` — 26 tests couvrant
  admission (§12), KV (§17), queue/backpressure (§16), exécuteur outils
  Phase 5 (§7), pipeline §3, méta-objectif §4.

## Mapping menaces → tests

| Menace | Fichier | Fonction |
|--------|---------|----------|
| T01 injection directe | `cogno-core/tests/adversarial.rs` | `direct_prompt_injection_is_untrusted` |
| T02 injection indirecte | `cogno-core/tests/adversarial.rs` | `confidence_above_max_is_rejected` (schéma) ; `direct_prompt_injection_is_untrusted` (origine) |
| T03 injection base | `cogno-runtime/tests/runtime_integration.rs` | `pipeline_rejects_secret_at_hard_stage_ignoring_reward` (privacy gate) |
| T04 sortie d'outil malveillante | `cogno-runtime/tests/runtime_integration.rs` | `mvp_tool_executor_refuses_everything` |
| T05 empoisonnement feedback | `cogno-model/tests/backends.rs` | `corpus_deduplicates_by_fingerprint` |
| T06 empoisonnement profil | `cogno-runtime/tests/runtime_integration.rs` | `runtime_run_pipeline_rejects_secret_and_audits` |
| T07 corpus empoisonné | `cogno-model/tests/backends.rs` | `trainer_accuracy_is_finite_and_deterministic` |
| T08 poids malveillants | `cogno-core/tests/adversarial.rs` | `manifest_corrupted_or_unknown_version_rejected` |
| T09 tokenizer incompatible | `cogno-core/tests/adversarial.rs` | `manifest_truncated_artifact_rejected` (schema) |
| T10 artefact tronqué | `cogno-core/tests/adversarial.rs` | `manifest_truncated_artifact_rejected` |
| T11 traversée | `cogno-core/tests/adversarial.rs` | `parent_component_path_rejected` |
| T12 symlink | `cogno-runtime/tests/runtime_integration.rs` | `root_policy_rejects_parent_component_and_outside_root` |
| T13 hors-racine | `cogno-core/tests/adversarial.rs` | `outside_root_rejected` ; `cogno-runtime/.../root_policy_rejects_...` |
| T14 commande shell | `cogno-runtime/tests/runtime_integration.rs` | `phase5_executor_when_enabled_still_rejects_shell_shape` |
| T15 exfiltration secret | `cogno-core/tests/adversarial.rs` | `secret_data_rejected_by_policy` |
| T16 DoS mémoire | `cogno-runtime/tests/runtime_integration.rs` | `admission_rejects_oversized_input` |
| T17 batch surdimensionné | `cogno-runtime/tests/runtime_integration.rs` | `admission_rejects_context_too_large` |
| T18 récursion profonde | `cogno-core/tests/adversarial.rs` | `too_many_evidence_ids_rejected` |
| T19 saturation file | `cogno-runtime/tests/runtime_integration.rs` | `queue_rejects_newest_when_full` ; `runtime_enqueue_applies_backpressure_and_counts_rejections` |
| T20 overflow arithmétique | `cogno-core/tests/adversarial.rs` | `kv_cache_arithmetic_overflow_detected` |
| T21 rejeu/duplication | `cogno-core/tests/adversarial.rs` | `journal_deduplicates_by_fingerprint` |
| T22 rollback version | `cogno-core/tests/adversarial.rs` | `unknown_schema_version_rejected` |
| T23 fausse provenance | `cogno-core/tests/adversarial.rs` | `zero_evidence_id_rejected` ; `evidence_required_for_extraction` |
| T24 récompense manipulée | `cogno-core/tests/adversarial.rs` | `lexicographic_decision_rejects_hard_before_reward` ; `cogno-runtime/.../pipeline_rejects_secret_at_hard_stage_ignoring_reward` |
| T25 contournement validateurs | `cogno-runtime/tests/runtime_integration.rs` | `pipeline_rejects_malformed_proposal_at_structural_stage` |
| T26 confusion donnée/instruction | `cogno-core/tests/adversarial.rs` | `direct_prompt_injection_is_untrusted` ; `unsupported_action_rejected` ; `duplicate_evidence_rejected` |