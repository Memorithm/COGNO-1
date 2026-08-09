# Architecture — COGNO-1

COGNO-1 est un système générique de personnalisation neuro-symbolique composé de
deux domaines **strictement séparés** :

1. un **petit modèle neuronal d'assistance** — non fiable par construction ;
2. un **noyau déterministe Rust** — seule autorité pour la sûreté, la mémoire,
   les formats, la provenance, les décisions d'adoption, la persistance et les
   effets de bord.

Le modèle n'est jamais une source d'autorité. Toute sortie neuronale reste une
donnée soft : bornée, validée, liée à une provenance et utilisable seulement à
l'endroit explicitement prévu par le runtime.

L'indépendance couvre le fournisseur du modèle, le tokenizer, le moteur
tensoriel, le format de poids, le matériel, le système d'exploitation et le
projet hôte.

## 1. Pipeline obligatoire (§3)

Le chemin d'autorité respecte l'ordre suivant :

```text
entrée externe
  ↓ classification de confiance
  ↓ contrôle des tailles
  ↓ parsing strict
  ↓ validation structurelle
  ↓ validation symbolique déterministe
  ↓ règles de sécurité / confidentialité       ← hard gates
  ↓ reward scalaire déterministe de base
  ↓ décision d'admissibilité
  ├─ rejet : fin, aucun signal V4
  └─ Eligible
       ↓ vérification Meta + digest V4 exact
       ↓ observation cognitive V4 non autoritaire
       ↓ cibles sémantiques explicites de l'hôte
       ↓ ajustement soft borné
       ↓ addition entière checked
       ↓ classement déterministe des candidats admissibles
       ↓ audit complet
       ↓ effet de bord éventuel derrière capacités explicites
```

Le contexte nécessaire au reward cognitif **n'existe pas** avant
`PipelineOutcome::Eligible`. Une contrainte dure ne peut donc pas être
compensée numériquement.

## 2. Autorité du noyau Rust (§3)

Le noyau est l'unique autorité pour : les règles de sûreté ; les politiques
d'accès ; les limites mémoire ; les formats acceptés ; la provenance ; les
décisions d'adoption ; les effets de bord ; les outils ; les écritures ; les
suppressions ; les migrations ; les mises à jour du profil ; les promotions de
modèle ; l'activation Meta ; l'installation d'un modèle persisté.

Le modèle ne produit jamais une ligne de shell directement exécutée. Dans la
configuration par défaut, **aucun outil n'est exécuté** (§7).

## 3. Décision lexicographique (§8)

La sûreté n'est **pas** une composante numérique compensable. L'ordre est :

1. admissibilité structurelle ;
2. conformité symbolique dure ;
3. conformité aux contraintes de sécurité ;
4. conformité aux capacités et à la confidentialité ;
5. score scalaire de base ;
6. éventuellement, signal V4 soft borné pour un candidat déjà admissible ;
7. classement des candidats admissibles ;
8. tie-break déterministe.

Le delta cognitif actuel est plafonné globalement à `±100` points et appliqué
par arithmétique entière checked. Cette borne n'est pas la protection primaire
contre la compensation d'une violation dure : la protection primaire est que
le delta n'est calculable **qu'après** les hard gates.

## 4. Origine typée et classe de confiance (§6)

Toute entrée porte une origine et une classe de confiance indépendantes. Une
chaîne de caractères ne contient jamais implicitement son niveau de privilège.
Instructions système, données récupérées, résultats d'outils et sorties du
modèle restent dans des champs distincts jusqu'à leur consommation par une API
typée.

Il est interdit de construire une autorité en concaténant simplement une donnée
externe à une instruction privilégiée. Une délimitation textuelle seule n'est
pas une frontière de sécurité.

## 5. Architecture cognitive V4

Le V4 utilise un `SequenceEncoder` borné partagé par plusieurs heads :

- classification ;
- préférence pairwise ;
- satisfaction symbolique soft ;
- contradiction binaire ;
- retrieval contrastif sur la représentation partagée.

Le modèle n'est pas un Transformer et n'utilise pas d'attention. L'encodeur
utilise des embeddings token+position, une projection/mélange, ReLU et pooling.

Le tokenizer est déterministe : octets `0..255`, `BOS=256`, `EOS=257`,
`SEP=258`, vocabulaire `259`, maximum global `512` tokens.

## 6. Objectif joint et frontière symbolique

Les cinq tâches participent à un objectif différentiable joint dans
`cogno-scirust`. Les gradients de chaque vue sont accumulés vers le même
encodeur avant une mise à jour AdamW.

La satisfaction symbolique neuronale reste un **signal soft**. Les vérités de
règles utilisées pour la supervision sont fournies par l'hôte ; les règles
dures et validateurs symboliques du core restent hors du graphe différentiable.

Le coût runtime/mémoire reste une mesure déterministe et n'est pas transformé
en prédiction neuronale.

## 7. Revue Meta et activation

Un candidat V4 doit passer une revue held-out multi-signal avant de devenir un
`MetaReviewedCandidate` scellé. Les cinq tâches sont mesurées séparément et la
métrique retenue suit une logique **weakest-link** : un head fort ne peut pas
masquer un autre head insuffisant.

L'activation Meta exige ensuite :

- moteur scalaire validé ;
- politique de référence figée ;
- backend différentiable ;
- log-probabilités/signaux nécessaires disponibles ;
- tests held-out en place ;
- anti-empoisonnement attesté.

Le runtime conserve le digest du candidat ayant activé Meta. Avant toute
observation V4 post-hard, ce digest doit être exactement égal au digest de
l'artefact installé.

## 8. Artefacts, persistance et redémarrage

Les formats neuronaux sont versionnés. V4 (`COG4`) sérialise exactement
`11` tenseurs : encodeur `(3)`, classification `(2)`, préférence `(2)`,
symbolique `(2)` et contradiction `(2)`. Le retrieval n'a pas de tenseur propre.

Le hostile loader vérifie le manifeste, la version, l'architecture, le hash du
tokenizer, les dimensions, le nombre de paramètres, la taille exacte, le
SHA-256 de l'artefact et la finitude des poids.

Les modèles revus sont persistés sous forme de générations immuables. Le replay
revalide la chaîne jusqu'à `MODEL_CURRENT`. L'installation V4 est one-shot et
nécessite un sceau de redémarrage contrôlé ; elle n'est pas un hot-swap.

## 9. Reward cognitif et décision multi-candidats

Après `Eligible`, le runtime peut produire une observation V4 liée à :

- la génération persistée ;
- le SHA-256 de l'artefact ;
- le digest du candidat Meta actif.

Les probabilités sont quantifiées en points de base avant audit. Pour convertir
ces observations en influence soft, l'hôte fournit explicitement les cibles
attendues pour les tâches pondérées. Le modèle ne décide donc pas de la
sémantique des identifiants de classe, règles ou slots de retrieval.

Un `AppliedCognitiveReward` est scellé par le chemin post-hard. La décision
multi-candidats n'accepte que ces valeurs scellées, refuse les IDs dupliqués et
exige la même génération, le même artefact et le même digest Meta pour tous les
candidats. Le score final entier le plus élevé gagne ; une égalité exacte est
départagée par le plus petit ID candidat stable.

## 10. Responsabilités des crates

| Crate | Rôle | Contraintes clés |
|-------|------|------------------|
| `cogno-core` | Types, hard validators, reward entier, budgets, mémoire, sécurité | déterministe, sans backend neuronal obligatoire, sans `unsafe` propriétaire |
| `cogno-scirust` | autograd borné, optimisateurs, encodeur partagé, heads et objectif joint | Rust sûr, bornes strictes, erreurs fallibles |
| `cogno-model` | tokenizer, bridges d'entraînement, revue Meta, artefacts hostiles | aucune exécution de code au chargement, provenance contrôlée |
| `cogno-runtime` | admission, pipeline, audit, persistance/replay, activation, reward/decision V4 | hard-before-soft, fail-closed, installation one-shot |
| `cogno-cli` | binaire `cogno`, observabilité et ingestion explicite | aucune autorité implicite, outils gated |

## 11. État des phases (§27)

Les briques des phases 0→5 coexistent désormais dans le dépôt, mais leurs
**autorités restent gated** :

- **Phase 0 — noyau déterministe :** active ;
- **Phase 1 — simulateur :** disponible ;
- **Phase 2 — modèles read-only :** disponibles ;
- **Phase 3 — apprentissage supervisé :** disponible avec provenance et splits ;
- **Phase 4 — Meta-NeuroSymbolic :** implémenté, activation contrôlée et inactif
  par défaut ;
- **Phase 5 — outils :** surface présente, refusée par défaut et soumise à des
  capacités explicites.

Le CLI par défaut construit un runtime sans V4 installé, Meta inactif et outils
désactivés.

## 12. Observabilité

`RuntimeReport` expose l'état Meta/V4 : digest Meta, présence du modèle,
génération, digest de l'artefact et liaison exacte entre Meta et V4.

`cognitive_model_meta_bound` ne peut être vrai que si Meta est actif, qu'un V4
est réellement installé et que les deux digests concordent. Deux absences ne
forment jamais une liaison valide.

Les commandes `phase` et `doctor` rendent ces informations visibles sans
charger ni activer un modèle.

## 13. Politique Rust (§23)

Pour tout le code propriétaire :

```rust
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]
```

Les dépendances externes sont minimisées, documentées et verrouillées. La CI
utilise Rust `1.97.1`, `Cargo.lock`, `--locked`, et un build release `--frozen`.

## 14. Gates permanents

Chaque PR doit fermer :

```text
cargo test --all-targets --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo doc --no-deps --locked
cargo build --release --frozen --all-targets
inventaire documenté des dépendances externes (§24)
```

Des tests end-to-end couvrent notamment le rejet hard avant toute observation
V4, le reward cognitif borné, la persistance/replay, le redémarrage contrôlé,
la liaison digest Meta↔artefact et la décision multi-candidats réelle.
