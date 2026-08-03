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

## Statut actuel — Phases 0–5 implémentées (gate fermée)

Toutes les phases ont un squelette fonctionnel, avec le **fail-closed**
préservé partout : aucune exécution d'outil, aucun secret, aucun effet de bord,
lexicographique appliqué avant la récompense.

| Phase | État | Implémentation |
|-------|------|----------------|
| **0 — Noyau déterministe** | ✅ | `cogno-core` : types (§6/§9), `MemoryBudget` + `checked_kv_cache_bytes` (§11), `BoundedVec` (§15), `Journal`/`Profile` (§18), `SafetyPolicy`/`PathPolicy` (§22), validateurs (§8), `RewardEngine` (scalaire), `ModelManifest` (§21), `ToolProposalView` (§7), `MetaObjective` (§4) |
| **1 — Simulateur** | ✅ | `cogno_model::SimBackend` : propositions scriptées FIFO ; `Exhausted` fermé |
| **2 — Lecture seule** | ✅ | `cogno_model::ReadOnlyModel` sur `TrainedModel` (frozen, `Arc`-partagé) : `classify` / `extract` / `rank` / `explain` |
| **3 — Apprentissage supervisé** | ✅ | `cogno_model::{Corpus, ToyTrainer}` : perceptron entier hashé (placeholder réel), provenance, dedup par empreinte (§9), splits déterministes |
| **4 — Méta-objectif** | ✅ (gated) | `cogno_core::MetaObjective` : `activate()` renvoie `PreconditionMissing` tant que les 6 préconditions ne sont pas attestées |
| **5 — Outils** | ✅ (gated) | `cogno_runtime::ToolExecutor` : MVP refuse tout ; `phase5(true, positive_tools)` + contrôle `sh -c` après audit |

Le dépôt démarre par la **Phase 0** et refuse toute activation réelle tant que
les préconditions §27 ne sont pas satisfaites (S10).

### Tests

```bash
cargo test --all-targets          # 62 tests (28 core + 8 model + 26 runtime)
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Tous les tests **réussissent en mode fermé** : un test passe quand le système
**rejette** (S10), échoue quand le système laisserait passer une attaque.

### CLI

```bash
cargo run -q -- phase           # phase + état méta-objectif
cargo run -q -- doctor          # validation budget/KV/queue/outils
cargo run -q -- validate 4      # validation schéma strict (fail-closed)
cargo run -q -- simulate        # simulateur Phase 1
cargo run -q -- demo-pipeline   # proposition à travers le pipeline §3
cargo run -q -- replay          # journal → profil dérivé (déterministe, S6/S7)
```

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
- `docs/MODEL_CARD.md` — carte modèle (statut Phases 0–5, gate fermée).
- `docs/DATA_GOVERNANCE.md` — classification des données, provenance, secrets, rétention.
- `docs/DEPENDENCIES.md` — inventaire des dépendances (§24), zéro dépendance externe, CI `--locked`/`--frozen`.
- `docs/ACCEPTANCE.md` — critères d'acceptation §28, chaque item pointé vers son test.

## Licence

MIT — cf. `LICENSE`. Contribution soumise à la même licence.