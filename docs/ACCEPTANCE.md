# Critères d'acceptation — COGNO-1 (§28)

État au commit courant. Un item **refusé par défaut** signifie que la valeur
sûre est la valeur initiale et que toute activation demande un acte explicite
de l'hôte.

## 1. Sécurité et autorité

- [x] **Le modèle ne possède aucune autorité directe.**
      Les façades neuronales sont read-only ; hard validators, politiques,
      persistance, activation Meta, outils et effets de bord restent dans le
      core/runtime.
- [x] **Toute sortie du modèle est traitée comme non fiable.**
      `InputOrigin::ModelOutput` reste non privilégiée ; une observation V4
      est explicitement `authoritative: false`.
- [x] **Les contraintes dures sont appliquées avant toute influence neuronale.**
      `Pipeline::run` valide structure, symbolique et sécurité avant
      `PipelineOutcome::Eligible`.
- [x] **Une violation dure ne peut pas être compensée.**
      Le contexte cognitif V4 ne peut être minté qu'après `Eligible`. Le test
      end-to-end `hard_rejection_never_mints_cognitive_observation_or_reward`
      vérifie qu'un `Secret` produit uniquement un rejet, sans observation ni
      reward cognitif.
- [x] **Les capacités et outils sont refusés par défaut.**
      `SafetyPolicy::MVP`, `ToolExecutor::mvp()` et Meta inactif au démarrage.
- [x] **Instructions et données restent typées séparément.**
      `InputOrigin`, `EvidenceOrigin`, `TrustClass` et les types de proposition
      empêchent qu'une simple concaténation de texte ne crée un privilège.
- [x] **Les tailles influencées par l'extérieur sont bornées.**
      Propositions, buffers, queues, KV, tokenizer, dimensions neuronales,
      candidats retrieval, corpus et artefacts ont des caps explicites.
- [x] **L'arithmétique critique est contrôlée.**
      Budgets, tailles d'artefact, nombre de paramètres, tape autograd et reward
      utilisent des opérations checked/fallibles.
- [x] **La backpressure et les dépassements échouent en mode fermé.**
      Queue, KV, admission mémoire, tokenizer et graph/tape refusent les caps
      dépassés sans état partiel silencieux.
- [x] **Les secrets sont rejetés par le hard gate runtime.**
      `DataClassification::Secret` est rejeté avant reward et avant V4.
- [x] **Les règles inférées ne deviennent pas des règles dures par le modèle.**
      `EvidenceOrigin::ModelInference` ne peut pas créer l'autorité requise pour
      une règle hard.
- [x] **Les contradictions et preuves restent reconstructibles.**
      Le journal conserve la provenance et `Profile::derive` reconstruit l'état.
- [x] **Le code propriétaire interdit `unsafe`.**
      Les crates appliquent `#![forbid(unsafe_code)]` et des warnings stricts.

## 2. Backend différentiable et tokenizer

- [x] **Un backend différentiable réel existe.**
      `cogno-scirust` fournit un tape reverse-mode borné, des opérations
      connectées, AdamW/AMSGrad et les gradients exacts utilisés par les bridges
      séquence.
- [x] **Le tokenizer est déterministe et versionné.**
      Octets `0..255`, `BOS=256`, `EOS=257`, `SEP=258`, vocabulaire `259`, cap
      global `512`.
- [x] **Les entrées pairwise ont un framing explicite.**
      `[BOS] left [SEP] right [EOS]` ; dépassement de capacité → erreur.
- [x] **Le hash tokenizer est vérifié dans les artefacts.**
      Un manifeste avec tokenizer incompatible est rejeté avant activation.

## 3. Modèle cognitif V4 partagé

- [x] **Une seule représentation est partagée.**
      `SequenceCognitiveHeads` possède un seul `SequenceEncoder` utilisé par
      classification, préférence, symbolique, contradiction et retrieval.
- [x] **Les cinq pertes sont réellement connectées au graphe.**
      L'objectif joint effectue une backward pass pondérée et accumule les
      gradients de toutes les vues vers le même encodeur.
- [x] **La supervision symbolique reste host-owned.**
      Les vérités de règles deviennent des cibles `0/1` uniquement à la
      frontière numérique ; le réseau ne devient pas autorité symbolique.
- [x] **Le coût runtime reste déterministe.**
      Il n'est pas remplacé par une prédiction neuronale compensable.
- [x] **Les configurations hostiles échouent avant update.**
      Classes, règles, longueurs, candidats retrieval et indices positifs sont
      prévalidés avant construction/utilisation de l'optimiseur.

## 4. Artefact hostile V4

- [x] **Le V4 est versionné et sélectionné par architecture.**
      `COG4`, version binaire `4` ; les loaders V1/V2/V3 restent distincts.
- [x] **Le V4 contient exactement 11 tenseurs.**
      Encodeur `(3)` + classification `(2)` + préférence `(2)` + symbolique
      `(2)` + contradiction `(2)` ; retrieval réutilise l'encodeur.
- [x] **Le loader vérifie avant activation.**
      Manifeste, magic, version, architecture, tensor count, dimensions, cap
      retrieval, parameter count, taille exacte, tokenizer hash, SHA-256 et
      finitude des poids.
- [x] **Le format de poids n'exécute aucun code.**
      Le loader décode uniquement des métadonnées et scalaires `f32` bornés.
- [x] **Le décodage ne confère pas d'autorité.**
      Il produit un `SequenceCognitiveArtifactState`; installation runtime et
      activation Meta restent des étapes séparées et attestées.

## 5. Revue Meta V4

- [x] **Train/validation/test sont distincts et validés.**
      Mauvais type de split, chevauchement ou index invalide → rejet.
- [x] **Le corpus rejette les doublons canoniques.**
      L'empreinte SHA-256 couvre les cinq tâches et leurs cibles.
- [x] **La provenance de revue est contrôlée.**
      Les exemples portent `InputOrigin` et `EvidenceOrigin`; une provenance
      non admise provoque `UntrustedReviewProvenance`.
- [x] **Les cinq tâches sont évaluées held-out.**
      Classification, préférence, symbolique, contradiction et retrieval.
- [x] **La promotion est weakest-link.**
      La métrique persistée est le minimum des cinq accuracies ; un head fort ne
      peut pas masquer un head faible.
- [x] **La classification est comparée à une référence figée.**
      La politique borne explicitement la régression autorisée.
- [x] **L'éligibilité produit un proof scellé.**
      Un fichier de poids seul ne peut pas prétendre avoir passé la revue.

## 6. Persistance, replay et redémarrage contrôlé

- [x] **Seul un candidat revu peut être persisté.**
      `commit_reviewed_model_generation` exige `MetaReviewedCandidate` et une
      attestation hôte explicite.
- [x] **Les générations sont immuables et chaînées.**
      Chaque génération contient manifests + artefact et `MODEL_CURRENT` avance
      atomiquement.
- [x] **Le replay revalide la chaîne complète.**
      Liens de génération, digests, manifests et hostile loader sont vérifiés
      jusqu'à la sélection courante.
- [x] **L'installation V4 est one-shot au redémarrage.**
      Un second install est refusé ; aucun hot-swap implicite.
- [x] **La génération et le digest restent attachés au modèle installé.**
      Le runtime expose ces bindings en lecture seule.

## 7. Activation Meta et identité du modèle

- [x] **Meta reste inactif par défaut.**
      `MetaObjective::new()` est quarantiné.
- [x] **Les six préconditions sont requises.**
      Moteur scalaire validé, politique figée, signaux requis, backend
      différentiable, held-out et anti-empoisonnement.
- [x] **Le candidat Meta est lié par digest.**
      Le runtime conserve le SHA-256 du candidat ayant activé Meta.
- [x] **Le V4 utilisé doit être exactement ce candidat.**
      Avant observation post-hard : `meta_candidate_digest ==
      cognitive_model_artifact_sha256`.
- [x] **Une absence de modèle n'est jamais interprétée comme une liaison.**
      `RuntimeReport::cognitive_model_meta_bound` exige Meta actif, artefact
      présent et digests égaux ; `None == None` ne suffit pas.

## 8. Observation et reward cognitif

- [x] **Une observation V4 est generation-bound et non autoritaire.**
      Elle contient génération + digest et quantifie les probabilités en basis
      points `0..=10_000` avant audit.
- [x] **Les cibles sémantiques sont fournies par l'hôte.**
      Classe, préférence, vérités symboliques, contradiction et cible retrieval
      ne sont pas inventées par le réseau.
- [x] **Le delta est normalisé et borné.**
      `MAX_COGNITIVE_SOFT_DELTA = 100` ; les poids relatifs ne peuvent pas
      contourner ce cap global.
- [x] **Le reward final utilise une addition entière checked.**
      Overflow → `CognitiveRewardError::ArithmeticOverflow`.
- [x] **L'audit conserve base, final, delta et provenance.**
      Aucun ajustement ne devient invisible après application.

## 9. Décision multi-candidats

- [x] **Seuls des rewards post-hard scellés sont comparables.**
      L'API publique prend des références à `AppliedCognitiveReward`, pas un
      booléen `hard_ok` fourni par l'appelant.
- [x] **Le nombre de candidats est borné.**
      Maximum `256`, liste vide et IDs dupliqués rejetés.
- [x] **La provenance doit être identique.**
      Même génération, même digest d'artefact et même digest Meta pour tous les
      candidats.
- [x] **Le classement est déterministe.**
      Score final entier maximum ; égalité exacte → plus petit ID candidat.
- [x] **Le chemin réel est testé end-to-end.**
      Deux propositions passent pipeline+V4, obtiennent des rewards scellés,
      sont comparées puis auditées.

## 10. Outils Phase 5

- [x] **Les outils restent refusés par défaut.**
      `ToolExecutor::mvp()`.
- [x] **Une surface Phase 5 positive existe derrière un gate explicite.**
      Les tests refusent les formes shell et les outils inconnus même lorsque
      la phase est activée.
- [x] **Le modèle ne peut pas activer Phase 5.**
      L'autorité d'outil reste une décision hôte/runtime séparée.

## 11. Observabilité opérationnelle

- [x] `RuntimeReport` expose Meta actif, digest Meta, V4 chargé, génération,
      digest V4 et état de liaison exacte.
- [x] `phase` affiche ces informations sans charger de modèle.
- [x] `doctor` conserve un état MVP sûr lorsque Meta/V4/outils sont absents et
      signale une configuration non-MVP sinon.

## 12. Dépendances et gates CI

- [x] **Les dépendances externes sont verrouillées et documentées.**
      `Cargo.lock`, `docs/DEPENDENCIES.md` et gate §24. Le projet n'affirme pas
      qu'il n'existe aucune dépendance externe.
- [x] **Rust est figé par le dépôt.**
      CI sur Rust `1.97.1`.
- [x] **Chaque PR doit fermer six gates.**

```text
cargo test --all-targets --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo doc --no-deps --locked
cargo build --release --frozen --all-targets
inventaire documenté des dépendances externes (§24)
```

Aucune PR ne doit être considérée validée simplement parce que son code a été
poussé : le head final doit fermer ces six gates.
