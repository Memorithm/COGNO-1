# COGNO-1

COGNO-1 est un système de personnalisation neuro-symbolique **sécurisé,
borné et déterministe dans ses décisions d'autorité**.

Le projet sépare strictement :

- un petit modèle neuronal optionnel, traité comme une source de données non
  fiable ;
- un noyau Rust déterministe, seule autorité pour les contraintes dures, la
  mémoire, la provenance, la persistance, les promotions, les décisions et les
  capacités d'outil.

Le modèle ne peut jamais transformer seul une probabilité en règle, en
permission ou en effet de bord.

## État actuel

COGNO-1 possède maintenant une chaîne cognitive V4 complète et contrôlée :

```text
bytes
  -> tokenizer déterministe
  -> SequenceEncoder partagé
     -> classification
     -> préférence pairwise
     -> satisfaction symbolique soft
     -> contradiction
     -> retrieval InfoNCE
  -> objectif joint / autograd
  -> revue Meta held-out weakest-link
  -> artefact hostile V4
  -> persistance générationnelle + replay
  -> installation read-only au redémarrage contrôlé
  -> hard gates déterministes
  -> observation V4 non autoritaire
  -> delta soft borné
  -> reward entier checked
  -> décision multi-candidats déterministe
  -> audit
```

Le runtime/CLI par défaut reste en mode fermé : **aucun V4 installé, Meta
inactif, outils désactivés**.

### Composants principaux

| Composant | État |
|-----------|------|
| Noyau déterministe `cogno-core` | ✅ contraintes dures, sécurité, budgets, reward entier, décisions |
| Backend différentiable `cogno-scirust` | ✅ autograd borné, optimisateurs, encodeur séquence, heads cognitifs |
| Tokenizer byte déterministe | ✅ vocabulaire 259, BOS/EOS/SEP, maximum global 512 tokens |
| Encodeur cognitif partagé | ✅ classification + préférence + symbolique + contradiction + retrieval |
| Objectif joint cinq tâches | ✅ une représentation partagée et une backward pass connectée |
| Revue Meta V4 | ✅ held-out, anti-empoisonnement, weakest-link cinq tâches |
| Artefact hostile V4 | ✅ `COG4`, 11 tenseurs, hash tokenizer, SHA-256, taille exacte |
| Persistance / replay | ✅ générations immuables et `MODEL_CURRENT` vérifié |
| Installation runtime V4 | ✅ read-only, one-shot, redémarrage contrôlé |
| Reward cognitif | ✅ uniquement après hard gates, delta global plafonné à ±100 |
| Décision multi-candidats | ✅ même provenance V4/Meta, score entier, tie-break stable |
| Outils | 🔒 refusés par défaut |

## Architecture du modèle V4

Le modèle actuel n'est **pas un Transformer** et n'utilise pas d'attention.
Il repose sur :

1. embeddings de tokens ;
2. embeddings de position ;
3. projection/mélange puis ReLU ;
4. pooling ;
5. heads cognitifs bornés.

Un seul `SequenceEncoder` est partagé entre les tâches. Le retrieval utilise la
représentation partagée directement et n'ajoute pas de tenseur de head propre.

### Tokenizer

Le tokenizer est fixe et versionné :

```text
raw bytes 0..255
BOS = 256
EOS = 257
SEP = 258
vocab = 259
max = 512 tokens
```

Des entrées pairwise utilisent le framing `[BOS] left [SEP] right [EOS]`.
L'empreinte SHA-256 de la définition du tokenizer est vérifiée dans les
artefacts.

## Autorité : hard d'abord, neural soft ensuite

La règle centrale est non négociable : **une contrainte dure ne peut pas être
compensée par une récompense neuronale**.

Le chemin runtime respecte l'ordre :

```text
validation structurelle
  -> validation symbolique dure
  -> sécurité / confidentialité
  -> score scalaire admissible
  -> seulement si Eligible : vérification Meta + digest V4
  -> observation neuronale
  -> ajustement soft borné
  -> décision déterministe
```

Un rejet hard n'invoque donc ni l'observation V4 ni le reward cognitif. Cet
invariant est couvert par un test end-to-end utilisant un vrai V4 revu,
persisté, rejoué et installé.

## Invariants de sécurité

| ID | Invariant |
|----|-----------|
| S1 | Toute sortie du modèle est non fiable. |
| S2 | Toute capacité est refusée par défaut. |
| S3 | Une donnée ne devient jamais une instruction privilégiée par simple présence dans un prompt/document/résultat d'outil. |
| S4 | Une contrainte dure ne peut pas être compensée par une récompense positive. |
| S5 | Toute opération influencée par l'extérieur possède des bornes explicites. |
| S6 | Toute mutation persistante possède une provenance. |
| S7 | Les modifications de profil restent réversibles. |
| S8 | Aucun secret n'est transmis au modèle sans autorisation explicite du composant hôte. |
| S9 | Aucun outil n'est accessible directement depuis le modèle. |
| S10 | Ambiguïté, dépassement ou erreur → échec fermé. |

## Revue Meta et anti-empoisonnement

La revue Meta du V4 évalue séparément sur validation/test : classification,
préférence, satisfaction symbolique, contradiction et retrieval.

L'éligibilité utilise une métrique **weakest-link** : une tâche forte ne peut
pas masquer un autre head défaillant. Les empreintes de corpus couvrent les
cinq signaux et leurs cibles, et les splits train/validation/test doivent être
disjoints.

Le proof de revue est scellé. Un artefact de poids seul ne suffit ni à la
persistance ni à l'activation Meta.

## Artefacts et persistance

Le format V4 :

- architecture `COG4` ;
- version binaire `4` ;
- `11` tenseurs ;
- header borné ;
- nombre de paramètres vérifié ;
- dimensions et caps vérifiés ;
- hash tokenizer ;
- SHA-256 du fichier ;
- rejet des poids non finis ;
- taille exacte obligatoire.

Les générations persistées forment une chaîne immuable. Au redémarrage, le
runtime rejoue et vérifie la chaîne jusqu'à `MODEL_CURRENT`, puis peut installer
le V4 via un sceau one-shot. Aucun hot-swap implicite n'est autorisé.

## Reward et décision cognitive

Après les hard gates, l'hôte fournit explicitement la sémantique attendue des
signaux (classe, préférence, vérités symboliques, contradiction, cible de
retrieval). Le modèle ne choisit donc pas lui-même ce qu'un identifiant de
classe ou de règle signifie.

Les alignements sont convertis en entier et normalisés. Le delta cognitif est
plafonné globalement à **±100 points**, puis ajouté au reward via arithmétique
checked.

La comparaison multi-candidats n'accepte que des rewards déjà scellés par le
chemin post-hard. Tous les candidats doivent provenir de la même génération,
du même artefact SHA-256 et du même candidat Meta. Le score final le plus élevé
gagne ; une égalité exacte utilise le plus petit identifiant candidat stable.

## CLI

```bash
cargo run -q -- phase
cargo run -q -- doctor
cargo run -q -- validate 4
cargo run -q -- simulate
cargo run -q -- demo-pipeline
cargo run -q -- replay
cargo run -q -- observe-jsonl <root>
```

`phase` et `doctor` exposent l'état Meta/V4, la génération et les digests sans
activer ni installer un modèle.

## Validation

Chaque PR doit fermer les six gates :

```bash
cargo test --all-targets --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo doc --no-deps --locked
cargo build --release --frozen --all-targets
# + inventaire verrouillé/documenté des dépendances externes (§24)
```

Les tests couvrent notamment les entrées hostiles, l'ordre hard-before-soft,
la revue Meta, les artefacts V4, la persistance/replay, le redémarrage contrôlé,
le reward borné et la décision multi-candidats.

## Structure du dépôt

```text
COGNO-1/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── SECURITY.md
├── crates/
│   ├── cogno-core/       autorité déterministe
│   ├── cogno-scirust/    calcul différentiable et heads cognitifs
│   ├── cogno-model/      tokenizer, modèles, revue et artefacts
│   ├── cogno-runtime/    pipeline, persistence, activation et décisions
│   └── cogno-cli/        binaire `cogno`
├── docs/
└── tests/
```

## Politique Rust

Le code propriétaire applique :

```rust
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]
```

Les dépendances externes sont minimisées, verrouillées dans `Cargo.lock` et
contrôlées par le gate §24.

## Documentation

- `docs/ARCHITECTURE.md` — séparation d'autorité et pipeline obligatoire ;
- `docs/MODEL_CARD.md` — architecture neuronale, V4, Meta et limites ;
- `docs/THREAT_MODEL.md` — modèle de menace ;
- `docs/MEMORY_MODEL.md` — budgets et mémoire ;
- `docs/DATA_GOVERNANCE.md` — provenance, classification et secrets ;
- `docs/DEPENDENCIES.md` — inventaire des dépendances ;
- `docs/ACCEPTANCE.md` — critères d'acceptation et tests.

## Licence

MIT — voir `LICENSE`.
