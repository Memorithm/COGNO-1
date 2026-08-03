# `tests/replay/` — tests de rejeu et régression (S6/S7)

## Tests exécutables

| Test | Fichier | Objet |
|------|---------|-------|
| Rejouer le journal → profil identique | `cogno-cli` `replay` (`crates/cogno-cli/src/main.rs::cmd_replay`) | `Journal` → `Profile::derive` déterministe |
| Rejeu d'un événement dupliqué | `cogno-core/tests/adversarial.rs::journal_deduplicates_by_fingerprint` | compté une seule fois (§9) |
| Quarantaine d'une règle model-only | `cogno-core/tests/adversarial.rs::model_only_rule_stays_quarantined_and_never_hard` | règle inférée jamais active ni dure |
| Promotion formelle après min_evidence | `cogno-core/tests/adversarial.rs::formal_validator_creates_hard_rule_after_min_evidence` | `Candidate -> Active` par politique déterministe |
| Entraînement déterministe (poids stables) | `cogno-model/tests/backends.rs::trainer_accuracy_is_finite_and_deterministic` | même corpus ⇒ même modèle ⇒ même profil |

## Propriétés auditées

- Déterminisme : même journal ⇒ même profil dérivé (`Profile::derive` pure).
- Provenance conservée à travers compaction (`Provenance.fingerprint`).
- Aucune preuve nécessaire à l'audit supprimée silencieusement (§18).
- Toute modification est **réversible** (S7) — pas de suppression silencieuse.
- Le simulateur est **exhausted** fermé (`BackendError::Exhausted`) :
  aucun repli improvisé (S10).