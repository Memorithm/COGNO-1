# Politique de sécurité — COGNO-1

COGNO-1 est conçu selon le principe **fail-closed** (S10) : en cas d'ambiguïté,
de dépassement ou d'erreur, le système **refuse** l'action. La sécurité est
**lexicographique** (§8) : une pénalité de sécurité ne peut jamais être annulée
par une meilleure note de style, un score utilisateur positif, une meilleure
performance, une récompense du modèle ou un gain de vitesse.

## Invariants non négociables (§4)

- **S1** — Toute sortie du modèle est non fiable.
- **S2** — Toute capacité est refusée par défaut.
- **S3** — Une donnée ne devient jamais une instruction uniquement parce
  qu'elle apparaît dans un prompt, un document, un fichier ou un résultat
  d'outil.
- **S4** — Une contrainte dure ne peut pas être compensée par une récompense
  positive.
- **S5** — Toute opération possède une limite de taille, de durée et de
  mémoire.
- **S6** — Toute décision modifiant l'état persistant possède une provenance.
- **S7** — Toute modification de profil est réversible.
- **S8** — Aucun secret ne doit être transmis au modèle sans autorisation
  explicite du composant hôte.
- **S9** — Aucun outil n'est accessible directement depuis le modèle.
- **S10** — En cas d'ambiguïté, de dépassement ou d'erreur, le système échoue
  en mode fermé : l'action est refusée.

Ces invariants sont **testables automatiquement** (cf. `tests/adversarial/`).

## Pipeline obligatoire (§3)

Aucun effet de bord ne peut se produire avant la fin de la chaîne :

```
entrée externe
  → classification de confiance
  → contrôle des tailles
  → parsing strict
  → validation structurelle
  → validation symbolique
  → application des règles de sécurité
  → évaluation neuro-symbolique
  → décision déterministe
  → audit
  → effet de bord éventuel
```

## Politique Rust (§23)

```rust
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]
```

`forbid(unsafe_code)` ne peut pas être abaissé par un module enfant. Les
dépendances externes peuvent toutefois contenir du `unsafe` ; chaque dépendance
est donc documentée et ses features contrôlées (§24).

## Signaler une vulnérabilité

Tant que le canal privé n'est pas établi, **ne pas ouvrir d'issue publique**
pour une vulnérabilité. Un canal privé (GitHub Security Advisories) sera
configuré sur le dépôt. En attendant, contacter les mainteneurs à l'adresse
associée à l'organisation `Memorithm`.

## Tests de sécurité obligatoires (§25-26)

Chaque menace du `docs/THREAT_MODEL.md` possède au moins un test dans
`tests/adversarial/`. Les tests mémoire d'allocation forcée en échec
(`tests/allocation/`) vérifient qu'aucune modification partielle de l'état ne
survit à un échec.