# Critères d'acceptation — COGNO-1 (§28)

État au commit courant. Chaque item pointe vers le test ou le module qui le
démontre. Un item **refusé par défaut** signifie que la valeur sûre est la
valeur initiale ; toute activation demande un acte explicite de l'hôte.

## Sécurité & autorité

- [x] Le modèle ne possède aucune autorité directe.
      `cogno_model::{SimBackend, ReadOnlyModel}` ne peut que proposer/classer ;
      la décision est `cogno_core::decide` (§8) exécutée par le noyau.
- [x] Toute sortie du modèle est traitée comme non fiable.
      `cogno_core::validate_proposal` ; `TrustClass::UntrustedModelData` pour
      `InputOrigin::ModelOutput` (test `direct_prompt_injection_is_untrusted`,
      `t04_tool_output_never_elevated_to_policy`).
- [x] Les contraintes dures sont appliquées avant la récompense.
      `cogno_runtime::Pipeline::run` : `hard` avant `reward` (§8).
- [x] Une violation dure ne peut pas être compensée.
      Test `lexicographic_decision_rejects_hard_before_reward`,
      `pipeline_rejects_secret_at_hard_stage_ignoring_reward` (reward=1_000_000
      ne sauve pas un `Secret`).
- [x] Aucun outil n'est exécuté par le MVP.
      `ToolExecutor::mvp()` ; test `mvp_tool_executor_refuses_everything`.
- [x] Toute capacité est refusée par défaut.
      `SafetyPolicy::MVP` ; `MVP_TOOLS_ENABLED = false` ; `QueueFullPolicy` par
      défaut `RejectNewest` ; `MetaObjective::new()` désactivé.
- [x] Les instructions et les données restent séparées.
      `InputOrigin`/`TrustClass` typés (§6) ;
      `t03_kb_injection_retrieved_document_is_untrusted_and_distinct_from_policy`.
- [x] Toutes les tailles influencées par l'extérieur sont bornées.
      `MAX_EVIDENCE_IDS`, `MAX_PAYLOAD_BYTES`, `MAX_CONFIDENCE_BPS`,
      `MemoryBudget.max_*`, `BoundedVec`, `BoundedQueue`, `KvController.capacity`.
- [x] Toutes les opérations de taille utilisent une arithmétique contrôlée.
      `checked_kv_cache_bytes`, `MemoryBudget::validate`,
      `RequestEstimate::try_new` ; tests `kv_cache_arithmetic_overflow_detected`,
      `kv_cache_ok_for_small_dims`.
- [x] Un budget mémoire global est défini.
      `MemoryBudget::try_new` ; test `valid_budget`,
      `budget_subbudget_sum_exceeds_hard_rejected`.
- [x] Chaque requête passe par un contrôle d'admission.
      `cogno_runtime::Admission::admit` ; `Runtime::admit`.
- [x] Le KV cache possède une limite stricte.
      `KvController` ; `KvCachePolicy::RejectOnOverflow` par défaut dans le CLI.
- [x] Les files de messages sont bornées.
      `BoundedQueue::try_new` (capacité 0 rejetée) ;
      `queue_zero_capacity_rejected`.
- [x] La backpressure est testée.
      `queue_rejects_newest_when_full`, `queue_reject_oldest_drops_oldest_not_silently`,
      `queue_block_with_deadline_refuses_synchronously`,
      `runtime_enqueue_applies_backpressure_and_counts_rejections`.
- [x] Les allocations du régime stable sont nulles.
      `Pipeline::run` n'alloue pas entre la classification de confiance et la
      décision ; `RequestScratch` est emprunté. `admission_latency_is_constant_time_in_budget_fields`.
- [x] Les échecs d'allocation sont testés.
      `init_allocation_impossible_when_budget_invalid`,
      `buffer_grow_impossible_past_max_len`,
      `kv_reservation_impossible_on_overflow`,
      `snapshot_payload_above_limit_is_skipped_during_derivation`,
      `queue_saturated_returns_structured_error_no_partial_state`.
- [x] Aucun secret n'est enregistré.
      `DataClassification::Secret` interdit partout (§20) ;
      `secret_data_rejected_by_policy`, `pipeline_rejects_secret_at_hard_stage_ignoring_reward`.
- [x] Les modèles et tokenizers sont vérifiés avant chargement.
      `ModelManifest::validate` ; `manifest_*` tests ; `t09_tokenizer_hash_mismatch_rejected`.
- [x] Le format de poids ne peut pas exécuter de code.
      Aucun format de poids n'est chargé en Phase 0–3 (simulateur + perceptron
      entier). Le chargeur réel (Phase 2+) appliquera `ModelManifest` + hash +
      dimensions avant allocation ; aucune exécution de code pendant le
      chargement (§21).
- [x] Les règles inférées commencent en quarantaine.
      `Profile::derive` met `Quarantined` par défaut ; test
      `model_only_rule_stays_quarantined_and_never_hard`.
- [x] Une inférence du modèle ne crée jamais une règle dure.
      `EvidenceOrigin::can_create_hard_rule` exclut `ModelInference` ;
      même test.
- [x] Les contradictions restent conservées.
      `Profile` conserver toutes les règles ; `RuleState::Conflicted` est un
      état terminal non supprimé (§9). Test authentique :
      `t06_profile_poisoning_contradictory_evidence_becomes_conflicted` (un
      `UserRejection` contre un `UserApproval` existant -> `Conflicted`,
      jamais `Active`, les 3 preuves restent dans le journal). Le profil dérivé
      ne supprime pas silencieusement.
- [x] Toute règle possède une provenance.
      `JournalEvent` porte `origin`/`evidence_origin`/`evidence_id` (S6).
- [x] Tout état persistant est reconstructible.
      `Profile::derive(&Journal, …)` est pure ; `cmd_replay` ;
      `trainer_accuracy_is_finite_and_deterministic`.
- [x] Toutes les modifications sont réversibles.
      Le journal est append-only ; `RuleState::Revoked`/`Disabled` sont des
      transitions reversibles ; aucune suppression silencieuse de preuve.
- [x] Chaque menace possède un test.
      Voir `tests/adversarial/README.md` (mapping T01–T26 → tests).
- [x] Les dépendances sont verrouillées et auditées.
      `Cargo.lock` committé ; CI `--locked` + `--frozen` ;
      `docs/DEPENDENCIES.md` ; zéro dépendance externe aujourd'hui.

## Critère supplémentaire de Phase 4 (gated)

- [x] L'objectif Meta-NeuroSymbolic n'est activé qu'avec les 6 préconditions
      attestées par l'hôte. `MetaObjective::activate` ;
      `runtime_meta_objective_refuses_without_preconditions`,
      `runtime_meta_objective_activates_with_all_preconditions`.

## Critère supplémentaire de Phase 5 (gated)

- [x] Les outils ne s'exécutent qu'après audit et derrière une liste positive.
      `ToolExecutor::phase5(true, positive_tools)` ;
      `phase5_executor_when_enabled_still_rejects_shell_shape`,
      `phase5_executor_when_enabled_authorizes_known_non_shell`,
      `phase5_executor_rejects_unknown_tool_even_when_enabled`.