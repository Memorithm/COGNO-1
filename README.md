# COGNO-1

Petit modèle d'assistance neuro-symbolique **sécurisé** et **à mémoire bornée**.

COGNO-1 est un système générique de personnalisation neuro-symbolique composé de
deux éléments **strictement séparés** :

- un **petit modèle d'assistance** — analyse les interactions, extrait des
  préférences candidates, produit des propositions structurées, classe des
  candidats ;
- un **noyau déterministe Rust** — applique les politiques de sécurité, valide
  les propositions, calcule les récompenses, gère la mémoire et autorise les
  effets de bord.

Le modèle **n'est jamais une source fiable**. Toute sortie du modèle est une
donnée non fiable qui doit être : parsée → bornée → validée → confrontée aux
politiques → éventuellement rejetée → auditée avant utilisation.

## Invariants de sécurité non négociables (§4)

| ID | Invariant |
|----|-----------|
| S1 | Toute sortie du modèle est non fiable. |
| S2 | Toute capacité est refusée par défaut. |
| S3 | Une donnée ne devient jamais une instruction uniquement parce qu'elle apparaît dans un prompt, un document, un fichier ou un résultat d'outil. |
| S4 | Une contrainte dure ne peut pas être compensée par une récompense positive. |
| S5 | Toute opération possède une limite de taille, de durée et de mémoire. |
| S6 | Toute décision modifiant l'état persistant possède une provenance. |
| S7 | Toute modification de profil est réversible. |
| S8 | Aucun secret ne doit être transmis au modèle sans autorisation explicite du composant hôte. |
| S9 | Aucun outil n'est accessible directement depuis le modèle. |
| S10| En cas d'ambiguïté, de dépassement ou d'erreur, le système échoue en mode fermé : l'action est refusée. |

Ces invariants sont testables automatiquement (cf. `tests/adversarial/`).

## Statut actuel — Phase 0

Le dépôt démarre par la **Phase 0** (noyau déterministe), **pas** par
l'entraînement du modèle. Aucun modèle n'est chargé ; aucun outil n'est
exécuté ; aucun effet de bord n'est possible. Voir `docs/ARCHITECTURE.md` et la
section *Phases d'implémentation* ci-dessous.

## Structure du dépôt

```
COGNO-1/
├── Cargo.toml          Workspace Rust
├── Cargo.lock          Verrouillé (conservé ; CI en --locked)
├── README.md
├── SECURITY.md
├── LICENSE             MIT
├── docs/
│   ├── ARCHITECTURE.md
│   ├── THREAT_MODEL.md
│   ├── MEMORY_MODEL.md
│   ├── MODEL_CARD.md
│   └── DATA_GOVERNANCE.md
├── crates/
│   ├── cogno-core/     Noyau déterministe (sans réseau, sans process, sans FS, sans unsafe)
│   ├── cogno-runtime/  Admission mémoire, KV cache, queues/backpressure, pipeline
│   ├── cogno-model/    Backend modèle (poids/tokenizers hostiles)
│   └── cogno-cli/      Binaire `cogno`
├── tests/
│   ├── adversarial/
│   ├── allocation/
│   └── replay/
└── models/
    └── README.md
```

### Contraintes de `cogno-core`

`cogno-core` **doit** rester : sans réseau ; sans exécution de processus ;
sans accès implicite au système de fichiers ; sans backend neuronal obligatoire
; sans `unsafe` propriétaire ; déterministe ; testable hors ligne.

### Politique Rust (§23)

Pour tout le code propriétaire de COGNO-1 :

```rust
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]
```

`forbid` est préféré à `deny` : un niveau `forbid` ne peut pas être abaissé par
un module enfant. Cela ne garantit toutefois pas que les dépendances externes
ne contiennent aucun `unsafe`.

## Phases d'implémentation (§27)

| Phase | Description |
|-------|-------------|
| **0 — Noyau déterministe** | Types, configuration, objectif mathématique, validateurs, récompense, budgets mémoire, journal, profil, sécurité, tests. Aucun modèle. *(actuel)* |
| **1 — Simulateur de modèle** | Backend déterministe renvoyant des propositions scriptées. Aucun vrai modèle. |
| **2 — Petit modèle en lecture seule** | Modèle capable uniquement de classifier, extraire, classer, expliquer. Ne modifie aucun état directement. |
| **3 — Apprentissage supervisé** | Corpus avec provenance ; split train/val/test ; cas contradictoires, adversariaux, injections, sorties malformées, exemples négatifs. |
| **4 — Objectif Meta-NeuroSymbolic** | Activé uniquement quand le moteur scalaire est validé, la politique de référence est figée, les log-probabilités sont disponibles, le backend est réellement différentiable, les tests held-out sont en place et les garde-fous anti-empoisonnement fonctionnent. |
| **5 — Outils** | Ajoutés uniquement après audit spécifique, derrière un mécanisme explicite de capacités. |

## Documentation

- `docs/ARCHITECTURE.md` — autorité du noyau, pipeline obligatoire, responsabilités des crates.
- `docs/THREAT_MODEL.md` — modèle de menace obligatoire (26 menaces × 10 champs).
- `docs/MEMORY_MODEL.md` — mémoire volatile vs sémantique, budgets, admission, KV cache, rétention.
- `docs/MODEL_CARD.md` — carte modèle (Phase 0 : aucun modèle chargé).
- `docs/DATA_GOVERNANCE.md` — classification des données, provenance, secrets, rétention.

## Licence

MIT — cf. `LICENSE`. Contribution soumise à la même licence.