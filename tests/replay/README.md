# `tests/replay/` — tests de rejeu et régression

But : garantir la **déterminisme** du noyau et la **reconstruction** de l'état
persistant depuis le journal (S6/S7).

## Cas prévus

| Test | Fichier prévu | Objet |
|------|---------------|-------|
| Rejouer le journal → profil identique | `replay-events.rs` | `events.log` → snapshot dérivé égal à la référence |
| Rejeu d'un événement dupliqué | `replay-duplicate.rs` | compté une seule fois (§9) |
| Rollback + re-replay | `replay-rollback.rs` | `RuleState` restauré ; pas de règle dure créée par inférence seule |
| Régénérer `profile.md` | `replay-profile.rs` | vue humaine non canonique régénérée depuis le journal |

## Propriétés auditées

- Déterminisme : même journal ⇒ même profil dérivé.
- Provenance conservée à travers compaction.
- Aucune preuve nécessaire à l'audit supprimée silencieusement (§18).
- Toute modification est **réversible** (S7).