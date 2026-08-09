# Model Card — COGNO-1

Ce document décrit le modèle neuronal optionnel intégré à COGNO-1 et, surtout,
sa frontière d'autorité avec le noyau déterministe Rust.

Le runtime par défaut reste **sans modèle installé**. Un modèle neuronal ne peut
être utilisé qu'après revue, persistance vérifiée et redémarrage contrôlé. Même
installé, il reste une source de signaux **soft et non autoritaires**.

## 1. État actuel

| Champ | Valeur |
|-------|--------|
| Phases | **0–5 implémentées et gated** ; Meta et outils désactivés par défaut |
| Modèle neuronal actuel | `SequenceCognitiveHeads` / `SequenceCognitiveModel` V4, optionnel et read-only au runtime |
| Backend différentiable | `cogno-scirust`, Rust sûr, autograd reverse-mode borné, sans FFI |
| Architecture | un `SequenceEncoder` partagé + heads classification, préférence, satisfaction symbolique et contradiction ; retrieval directement sur la représentation partagée |
| Attention / Transformer | **non** — embeddings token+position, projection/ReLU, pooling puis heads bornés |
| Tokenizer | byte tokenizer déterministe : octets `0..255`, BOS `256`, EOS `257`, SEP `258`, vocabulaire `259` |
| Contexte tokenizer | maximum global `512` tokens ; chaque artefact encode son propre `max_tokens` borné |
| Retrieval | jusqu'à `32` candidats côté SciRust ; le cap effectif est persisté dans V4 |
| Artefact V4 | architecture `COG4`, version binaire `4`, `11` tenseurs, checksum SHA-256, hash tokenizer et taille exacte |
| Persistance | générations immuables, `MODEL_CURRENT`, chaîne de manifests, replay intégral et contrôlé |
| Activation | Meta revu + attestation hôte + digest exact du même artefact V4 + installation one-shot au redémarrage |
| Outils | refusés par défaut ; aucune autorité d'outil n'est accordée par le modèle |
| Licence | MIT pour le code du projet |

Aucune donnée n'est envoyée à un service de modèle externe par cette
architecture. Le noyau COGNO reste l'autorité sur les politiques, la mémoire,
la persistance, les promotions, les outils et les contraintes dures.

## 2. Architecture neuronale V4

Le V4 possède **un seul encodeur séquence partagé**. Les différentes tâches ne
maintiennent donc plus cinq copies indépendantes de représentation.

Le paramétrage partagé comprend :

1. embeddings de tokens ;
2. embeddings de position ;
3. poids de mélange/projection de l'encodeur ;
4. head de classification ;
5. scorer de préférence pairwise ;
6. head de satisfaction symbolique ;
7. head binaire de contradiction ;
8. retrieval InfoNCE utilisant directement la représentation partagée, sans
   tenseur de head supplémentaire.

Le format hostile V4 sérialise exactement `11` tenseurs : encodeur `(3)`,
classification `(2)`, préférence `(2)`, symbolique `(2)` et contradiction
`(2)`. Le retrieval n'ajoute pas de tenseur propre.

L'architecture n'est pas un Transformer et ne contient pas de mécanisme
d'attention. Elle est volontairement petite, déterministe, bornée et testable.

## 3. Objectif d'entraînement partagé

L'entraînement joint connecte cinq signaux à une **même backward pass** :

- classification supervisée ;
- préférence pairwise ;
- satisfaction symbolique supervisée par des vérités de règles fournies par
  l'hôte ;
- contradiction binaire ;
- retrieval contrastif InfoNCE.

Les pertes sont pondérées mais restent des objectifs **soft**. Une vérité de
règle, un verdict de sécurité ou une autorisation n'est jamais dérivé du fait
qu'un head neuronal possède une bonne probabilité.

Avant construction de l'optimiseur, les bridges `cogno-model` prévalident les
longueurs tokenizer, classes, tailles des vecteurs de règles, candidats de
retrieval et indices positifs. Les entrées hostiles échouent en mode fermé.

## 4. Rôle limité du modèle

Le modèle peut fournir des signaux pour :

- classifier un événement de feedback ;
- comparer deux contenus par préférence ;
- estimer des satisfactions symboliques soft ;
- estimer une probabilité de contradiction ;
- sélectionner ou scorer des candidats de retrieval ;
- contribuer, après les hard gates, à un ajustement de récompense borné ;
- classer plusieurs candidats déjà admissibles.

Le modèle ne peut **jamais**, à lui seul : créer une règle de sécurité
obligatoire ; supprimer une règle existante ; exécuter une commande ; écrire
dans un dépôt ; accéder au réseau ; ouvrir un fichier arbitraire ; promouvoir
ses propres poids ; modifier un budget mémoire ; modifier les validateurs ;
contourner une contrainte dure ; activer Meta ; installer un artefact ; ni
accorder une capacité d'outil.

## 5. Tokenizer déterministe

Le tokenizer V4 est fixe et versionné :

- octets bruts : `0..=255` ;
- `BOS = 256` ;
- `EOS = 257` ;
- `SEP = 258` ;
- vocabulaire : `259` ;
- maximum global : `512` tokens ;
- framing pairwise : `[BOS] left [SEP] right [EOS]`.

Son descripteur canonique est :

```text
cogno-byte-tokenizer-v2;raw=0..255;bos=256;eos=257;sep=258;max=512
```

Le SHA-256 de cette définition est incorporé au manifeste. Un artefact portant
un hash tokenizer incompatible est rejeté avant activation.

## 6. Artefact hostile et manifeste

Tout artefact est accompagné d'un `ModelManifest` :

```rust
pub struct ModelManifest {
    pub schema_version: u16,
    pub model_family: ModelFamily,
    pub architecture_id: ArchitectureId,
    pub tensor_count: u32,
    pub parameter_count: u64,
    pub max_context_tokens: u32,
    pub tokenizer_hash: [u8; 32],
    pub weights_hash: [u8; 32],
    pub expected_file_bytes: u64,
}
```

Le loader V4 vérifie avant activation :

- schéma et architecture attendus ;
- magic/version binaires ;
- `11` tenseurs exactement ;
- vocabulaire, dimensions, classes, règles et cap retrieval ;
- nombre de paramètres et arithmétique de taille contrôlée ;
- taille de fichier exacte ;
- hash tokenizer ;
- SHA-256 de l'artefact ;
- poids `f32` tous finis ;
- bornes globales de contexte et de paramètres.

Le décodage produit un `SequenceCognitiveArtifactState` vérifié. **Décoder un
artefact ne l'installe pas** : l'activation runtime est une étape d'autorité
séparée.

## 7. Données d'entraînement et anti-empoisonnement

La revue Meta V4 travaille sur un corpus multi-signal possédant une provenance
explicite via `InputOrigin` et `EvidenceOrigin`. Les splits train/validation/test
sont distincts et les empreintes canoniques couvrent les cinq tâches et leurs
cibles, afin qu'un même exemple ne puisse pas être injecté discrètement dans
plusieurs splits.

Les vérités symboliques d'entraînement restent des labels **host-owned**. Le
réseau apprend à les approximer mais ne devient jamais l'autorité qui décide
qu'une règle est vraie.

La classification de confidentialité des données d'entraînement relève de la
gouvernance hôte ; elle n'est pas implicitement déduite du corpus neuronal.

## 8. Revue Meta weakest-link

La revue V4 évalue séparément sur held-out :

1. classification ;
2. préférence ;
3. satisfaction symbolique ;
4. contradiction ;
5. retrieval.

La métrique d'éligibilité conservée est de type **weakest-link** : une tâche
forte ne peut pas masquer un head défaillant. La classification est également
comparée à une référence figée afin de borner la régression.

Un candidat éligible produit un proof scellé `MetaReviewedCandidate`. Le proof
est requis pour la persistance et pour l'activation Meta ; un simple fichier de
poids ne suffit pas.

## 9. Persistance, replay et redémarrage contrôlé

Les modèles revus sont persistés sous forme de générations immuables. Le replay
vérifie depuis la genèse jusqu'à `MODEL_CURRENT` : liens de chaîne, manifests,
digests, artefact et hostile loader.

Un V4 rejoué ne peut être installé que via un sceau de redémarrage contrôlé. Le
runtime refuse un second install : il n'existe pas de hot-swap implicite.

Le modèle installé conserve :

- la génération sélectionnée ;
- le SHA-256 de l'artefact ;
- la façade read-only ;
- l'absence d'autorité d'outil.

## 10. Utilisation runtime et ordre des hard gates

Le chemin cognitif respecte l'ordre suivant :

```text
pipeline déterministe
  -> validation structurelle
  -> validation symbolique dure
  -> règles de sécurité / confidentialité
  -> Eligible + score scalaire de base
  -> vérification Meta actif
  -> vérification digest Meta == digest V4 installé
  -> observation V4 non autoritaire
  -> cibles sémantiques explicites fournies par l'hôte
  -> delta soft borné
  -> addition entière checked
  -> décision déterministe entre candidats déjà admissibles
  -> audit complet
```

Un candidat hard-rejeté ne peut donc pas obtenir d'observation V4 ni de reward
cognitif. Les tests end-to-end construisent un vrai V4, l'activent, le
persistent, le rejouent, l'installent et vérifient ce comportement.

Le delta cognitif est normalisé et plafonné globalement à **±100 points**. Il
ne peut pas compenser une violation dure, puisque le contexte nécessaire pour
le calculer n'existe qu'après `PipelineOutcome::Eligible`.

Lors d'une comparaison multi-candidats, seuls des `AppliedCognitiveReward`
scellés sont acceptés. Tous doivent partager la même génération, le même digest
d'artefact et le même digest Meta. Les scores finaux sont comparés en entier ;
un tie exact est départagé par le plus petit identifiant candidat stable.

## 11. Observabilité

`RuntimeReport` expose explicitement :

- Meta actif ou non ;
- digest du candidat Meta ;
- V4 installé ou non ;
- génération persistée ;
- digest de l'artefact installé ;
- liaison exacte Meta ↔ V4.

La liaison n'est vraie que si Meta est actif, qu'un artefact est réellement
installé et que les deux digests sont identiques. Deux valeurs absentes
(`None == None`) ne constituent jamais une liaison valide.

## 12. Limites connues

- Le runtime/CLI par défaut démarre sans V4 installé et sans Meta actif.
- L'encodeur actuel est volontairement simple : pas de Transformer, pas
  d'attention et pas de génération autoregressive.
- Les cinq heads partagent une représentation, mais leurs signaux restent des
  approximations statistiques et ne remplacent jamais les validateurs du core.
- La calibration disponible est post-hoc ; elle ne transforme pas une
  probabilité en autorité.
- Les outils sont refusés par défaut (`ToolExecutor::mvp`).
- Les formats V1/V2/V3 restent supportés par le dispatcher versionné, mais le
  runtime cognitif multi-head contrôlé décrit ici repose sur V4.

## 13. Tests et gates permanents

Le dépôt impose sur chaque PR :

```text
cargo test --all-targets --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo doc --no-deps --locked
cargo build --release --frozen --all-targets
inventaire verrouillé/documenté des dépendances externes (§24)
```

Des tests système couvrent notamment : hard rejection avant V4, reward borné,
liaison génération/digest, persistence/replay, redémarrage contrôlé et décision
multi-candidats déterministe.
