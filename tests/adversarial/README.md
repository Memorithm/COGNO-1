# `tests/adversarial/` — tests de sécurité obligatoires (§25)

Chaque menace de `docs/THREAT_MODEL.md` possède **au moins un test**. En Phase
0, ce répertoire est un **cahier de test** : les fichiers `.rs` seront créés au
fur et à mesure que l'API ciblée (validation, budget, reward, manifeste,
racine fichiers, queue) est implémentée dans `cogno-core` /
`cogno-runtime`. Les tests adversariaux font partie de la *définition du finite* :
ils échouent en mode fermé (S10) — un test valide est un rejet, jamais une
exécution.

## Cas obligatoires et leur correspondance

| Test | Menace | Fichier prévu | Fonction |
|------|--------|---------------|----------|
| Injection directe | T01 | `prompt_injection.rs` | `direct_prompt_injection` |
| Injection indirecte dans un document | T02 | `indirect_injection.rs` | `indirect_injection_from_file` |
| Faux message système dans une donnée | T03/T26 | `kb_injection.rs` | `injection_from_knowledge_base` |
| Faux résultat d'outil | T04 | `tool_output.rs` | `malicious_tool_output` |
| Empoisonnement feedback | T05 | `feedback_poisoning.rs` | `feedback_event_poisoning` |
| Empoisonnement profil | T06 | `profile_poisoning.rs` | `corrupted_profile` |
| Empoisonnement corpus | T07 | `corpus_poisoning.rs` | `training_corpus_poisoning` |
| Profil corrompu | T06 | `profile_poisoning.rs` | `corrupted_profile` |
| Manifeste corrompu | T08 | `model_weights.rs` | `manifest_corrupted` |
| Poids tronqués | T10 | `artifact.rs` | `truncated_artifact` |
| Dimensions → overflow | T20 | `size_overflow.rs` | `arithmetic_size_overflow` |
| Nombre de tenseurs excessif | T20 | `model_weights.rs` | `excess_tensor_count` |
| Poids malveillants/corrompus | T08 | `model_weights.rs` | `malicious_or_corrupt_weights` |
| Tokenizer incompatible | T09 | `tokenizer.rs` | `incompatible_tokenizer` |
| Contexte surdimensionné | T16 | `memory_dos.rs` | `memory_denial_of_service` |
| Sortie surdimensionnée | T17 | `oversized.rs` | `oversized_output` |
| Batch/séquence surdimensionné | T17 | `oversized.rs` | `oversized_batch_or_sequence` |
| Récursion/structure profonde | T18 | `deep_recursion.rs` | `deeply_nested_structure` |
| Proposition commande shell | T14 | `command_injection.rs` | `shell_command_proposal` |
| Écriture hors racine | T13 | `write_root.rs` | `write_outside_allowed_root` |
| Chemin contenant `..` | T11 | `path_traversal.rs` | `directory_traversal` |
| Lien symbolique | T12 | `symlink.rs` | `symlink_escape` |
| File pleine | T19 | `queue_saturation.rs` | `queue_saturation` |
| Délai dépassé | T19 | `queue_saturation.rs` | `deadline_exceeded` |
| Allocation refusée | T16 | (cf. `tests/allocation/`) | — |
| Limite KV atteinte | T17 | `oversized.rs` | `kv_limit_reached` |
| Événement dupliqué | T21 | `replay.rs` (cf. `tests/replay/`) | `duplicate_event` |
| Événement provenance absente | T23 | `provenance.rs` | `missing_provenance` |
| Règle générée uniquement par le modèle | S1/§9 | `provenance.rs` | `model_only_rule_rejected` |
| Souple → dure | §9 | `validator_bypass.rs` | `soft_to_hard_rule_rejected` |
| Compensation dure par récompense | T24 | `reward.rs` | `manipulated_reward_compensates_hard_violation` |
| Contournement validateurs | T25 | `validator_bypass.rs` | `validator_bypass` |
| Confusion donnée/instruction | T26 | `data_injection_confusion.rs` | `data_instruction_confusion` |
| Rollback version vulnérable | T22 | `rollback.rs` | `rollback_to_vulnerable_version` |
| Falsification provenance | T23 | `provenance.rs` | `provenance_forgery` |
| Secret dans une trace | T15 | `secret_exfil.rs` | `secret_in_trace` |
| Secret dans un message d'erreur | T15 | `secret_exfil.rs` | `secret_in_error_message` |

## Conventions de test

- Un test **réussit** quand le système **rejette** (fail-closed, S10).
- Aucun test ne déclenche d'effet de bord réel (réseau, FS hors racine temporaire
  de test, exécution de processus).
- Les chemins de test utilisent un répertoire temporaire dédié et borné.