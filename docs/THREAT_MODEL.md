# Modèle de menace — COGNO-1

Document obligatoire (COGNO-1 V2 §5). Couvre les 26 menaces listées par la
spec. Chaque menace est documentée avec :

- **Actif protégé**
- **Attaquant supposé**
- **Vecteur d'entrée**
- **Impact**
- **Contrôle préventif**
- **Contrôle de détection**
- **Réponse**
- **Test correspondant** (`tests/adversarial/…`)
- **Risque résiduel**

Référence : invariants S1–S10 (§4), séparation instruction/donnée (§6), politique
d'outils (§7), sûreté lexicographique (§8), résistance à l'empoisonnement (§9),
politique Rust (§23), chaîne d'approvisionnement (§24).

Légende des tests : en Phase 0 les fichiers de test sont des **cahiers de test**
(documentés dans `tests/adversarial/README.md`) ; les tests exécutables sont
ajoutés au fur et à mesure que l'API correspondante est implémentée dans
`cogno-core` / `cogno-runtime`. Chaque menace a **au moins** un test prévu.

---

## T01 — Injection directe de prompt

- **Actif protégé** : instructions système, politiques, capacités.
- **Attaquant supposé** : utilisateur authentifié malveillant ou compromis.
- **Vecteur d'entrée** : instruction utilisateur explicite tentant d'outrepasser
  la politique (« ignore les instructions précédentes… »).
- **Impact** : contournement de règles de sûreté, élevation de privilège.
- **Contrôle préventif** : `InputOrigin::ExplicitUserInstruction` ≠
  `SystemPolicy` (§6) ; instructions et données dans des champs distincts ;
  rejet de tout champ hors schéma ; S2 (refus par défaut), S3.
- **Contrôle de détection** : validation symbolique + audit du journal
  d'événements ; détection de mots-clés d'injection typés.
- **Réponse** : `Decision::Reject(RejectReason::HardConstraint)` ; échec en
  mode fermé (S10).
- **Test correspondant** : `tests/adversarial/prompt_injection.rs::direct_prompt_injection`.
- **Risque résiduel** : modèle pouvant être influencé par le contenu (atténué
  par le fait qu'aucune sortie du modèle n'a d'autorité directe, S1).

## T02 — Injection indirecte depuis un fichier

- **Actif protégé** : instructions système, préférences dérivées.
- **Attaquant supposé** : auteur d'un document lu par le système.
- **Vecteur d'entrée** : document parcouru contenant une fausse instruction
  système (« SYSTEM: … »).
- **Impact** : extraction/écriture de règle non voulue, corruption du profil.
- **Contrôle préventif** : `InputOrigin::RetrievedDocument` classé
  `UntrustedExternalData` ; délimitation structurelle contrôlée par le runtime
  (mais non suffisante à elle seule, §6) ; S3.
- **Contrôle de détection** : comparaison proposée vs. version retenue ;
  validateur de cohérence instruction/donnée.
- **Réponse** : rejet + journalisation de la provenance.
- **Test correspondant** : `tests/adversarial/indirect_injection.rs::indirect_injection_from_file`.
- **Risque résiduel** : faux négatifs possibles ; mitigé par quarantaine des
  règles inférées (§9).

## T03 — Injection depuis une page ou une base de connaissance

- **Actif protégé** : base de connaissance, profil de préférences.
- **Attaquant supposé** : contributeur de la base / page externe.
- **Vecteur d'entrée** : contenu récupéré (RAG-like) porteur d'instructions ou
  de règles contradictoires.
- **Impact** : profil empoisonné par connaissance non fiable.
- **Contrôle préventif** : `InputOrigin::RetrievedDocument` ;
  `TrustClass::UntrustedExternalData` ; preuves externes à faible autorité (§9).
- **Contrôle de détection** : détection de contradiction,
  `RuleState::Conflicted`.
- **Réponse** : quarantaine, conservation de la preuve contradictoire.
- **Test correspondant** : `tests/adversarial/kb_injection.rs::injection_from_knowledge_base`.
- **Risque résiduel** : lente détection d'empoisonnement progressif.

## T04 — Sortie d'outil malveillante

- **Actif protégé** : pipeline, profil, système de fichiers.
- **Attaquant supposé** : outil externe compromis ou malveillant.
- **Vecteur d'entrée** : résultat d'outil porteur d'instructions ou de données
  piégées (chemin, payload).
- **Impact** : exécution non prévue, écriture hors racine, exfiltration.
- **Contrôle préventif** : `InputOrigin::ToolOutput` non fiable ; MVP n'exécute
  **aucun** outil (§7) ; liste positive d'outils/arguments ; racine de fichiers
  ; limites durée/mémoire/sortie/processus/réseau ; S9.
- **Contrôle de détection** : audit des sorties d'outils ; comparaison contre
  schéma attendu.
- **Réponse** : rejet, journal, escalade au composant hôte.
- **Test correspondant** : `tests/adversarial/tool_output.rs::malicious_tool_output`.
- **Risque résiduel** : nul en Phase 0 (aucun outil) ; résiduel après Phase 5.

## T05 — Empoisonnement des événements de feedback

- **Actif protégé** : journal d'événements, règles dérivées.
- **Attaquant supposé** : utilisateur ou script automatisé injectant des
  feedbacks massifs.
- **Vecteur d'entrée** : événements de feedback répétés/contradictoires.
- **Impact** : règle stable induite par une seule acceptation ; profil biaisé.
- **Contrôle préventif** : nombre minimal d'évidences (§9) ; événements
  dupliqués comptés une seule fois ; limites par session/projet ;
  `EvidenceOrigin` différencié.
- **Contrôle de détection** : empreinte + identifiant par preuve ; détection de
  doublons.
- **Réponse** : dédoublonnage, plafonnement, quarantaine.
- **Test correspondant** : `tests/adversarial/feedback_poisoning.rs::feedback_event_poisoning`.
- **Risque résiduel** : empoisonnement lent sous le seuil minimal d'évidences.

## T06 — Empoisonnement du profil de préférences

- **Actif protégé** : profil de préférences (vue dérivée).
- **Attaquant supposé** : profil importé malveillant ou attaque par stockage.
- **Vecteur d'entrée** : `ImportedProfile` altéré.
- **Impact** : préférence contradictoire remplaçant silencieusement la
  précédente.
- **Contrôle préventif** : `EvidenceOrigin::ImportedProfile` faible autorité ;
  une préférence contradictoire ne remplace pas silencieusement (§9) ;
  `RuleState::Conflicted` ; provenance obligatoire (S6).
- **Contrôle de détection** : checksum du profil, comparaison
  proposée/rettenue.
- **Réponse** : rejet de l'import, conservation des preuves contradictoires.
- **Test correspondant** : `tests/adversarial/profile_poisoning.rs::corrupted_profile`.
- **Risque résiduel** : import légitime mais biaisé.

## T07 — Empoisonnement du corpus d'entraînement

- **Actif protégé** : modèle (poids), qualité d'inférence.
- **Attaquant supposé** : contributeur du corpus, acteur de la chaîne
  d'approvisionnement.
- **Vecteur d'entrée** : exemples entraînés porteurs d'instructions ou de
  biais.
- **Impact** : modèle produisant des propositions manipulées.
- **Contrôle préventif** : corpus avec provenance ; splits
  train/val/test ; cas contradictoires, adversariaux, injections, sorties
  malformées, exemples négatifs (Phase 3) ; S1 (sortie toujours non fiable).
- **Contrôle de détection** : tests held-out, métriques anti-empoisonnement.
- **Réponse** : retrait des exemples, réentraînement signé.
- **Test correspondant** : `tests/adversarial/corpus_poisoning.rs::training_corpus_poisoning`.
- **Risque résiduel** : pas applicable en Phase 0/1 (aucun entraînement).

## T08 — Poids de modèle malveillants ou corrompus

- **Actif protégé** : runtime, mémoire, intégrité du modèle.
- **Attaquant supposé** : distributeur de poids malveillant.
- **Vecteur d'entrée** : fichier de poids falsifié altérant le comportement.
- **Impact** : exécution de code si le format le permet, comportement
  imprévisible.
- **Contrôle préventif** : poids considérés hostiles (§21) ; `ModelManifest`
  avec `weights_hash` ; pas d'exécution de code au chargement ; noyau ne
  télécharge rien.
- **Contrôle de détection** : vérifications pré-allocation (taille, empreinte,
  n tenseurs, dims, types, doublons).
- **Réponse** : refus de chargement, erreur structurée.
- **Test correspondant** : `tests/adversarial/model_weights.rs::malicious_or_corrupt_weights`.
- **Risque résiduel** : collision de hash (atténué par SHA-256 + provenance).

## T09 — Tokenizer incompatible

- **Actif protégé** : segmentation, comptage de tokens, budget KV.
- **Attaquant supposé** : fournisseur de tokenizer incohérent avec les poids.
- **Vecteur d'entrée** : tokenizer dont les unités ne correspondent pas au
  modèle.
- **Impact** : dépassement de contexte silencieux, calculs erronés.
- **Contrôle préventif** : `tokenizer_hash` dans le manifeste (§21) ; pas de
  tokenizer non vérifié.
- **Contrôle de détection** : test de cohérence tokenizer/poids.
- **Réponse** : refus de chargement.
- **Test correspondant** : `tests/adversarial/tokenizer.rs::incompatible_tokenizer`.
- **Risque résiduel** : subtil mismatch non détecté par le hash seul.

## T10 — Artefact tronqué

- **Actif protégé** : intégrité des fichiers de modèle.
- **Attaquant supposé** : canal de transfert défectueux/attaquant.
- **Vecteur d'entrée** : fichier coupé avant `expected_file_bytes`.
- **Impact** : parsing partiel, panic ou mémoire inconsistante.
- **Contrôle préventif** : `expected_file_bytes` ; vérification de taille avant
  allocation.

- **Contrôle de détection** : empreinte recalculée, taille 对照 manifeste.
- **Réponse** : rejet, aucune allocation majeure.
- **Test correspondant** : `tests/adversarial/artifact.rs::truncated_artifact`.
- **Risque résiduel** : troncature à la limite exacte de la taille.

## T11 — Traversée de répertoire

- **Actif protégé** : système de fichiers du système hôte.
- **Attaquant supposé** : utilisateur proposant un chemin malveillant.
- **Vecteur d'entrée** : chemin contenant `..` ou composants parents.
- **Impact** : écriture/lecture hors racine autorisée.
- **Contrôle préventif** : racine configurée (§22) ; refus des composants
  parents non autorisés ; politique de liens symboliques.
- **Contrôle de détection** : audit des chemins résolus.
- **Réponse** : rejet, journalisation.
- **Test correspondant** : `tests/adversarial/path_traversal.rs::directory_traversal`.
- **Risque résiduel** :TOCTOU entre résolution et écriture (mitigé par écriture
  dans la même racine + remplacement atomique).

## T12 — Liens symboliques

- **Actif protégé** : racine de fichiers.
- **Attaquant supposé** : attaquant créant un symlink pointant hors racine.
- **Vecteur d'entrée** : lien symbolique dans la cible d'écriture/lecture.
- **Impact** : contournement de la racine.
- **Contrôle préventif** : `canonicalize` contrôle le FS (§22) ; politique de
  racine complète (pas seulement suppression textuelle de `..`).
- **Contrôle de détection** : résolution effective avant écriture ; audit.
- **Réponse** : rejet si cible hors racine.
- **Test correspondant** : `tests/adversarial/symlink.rs::symlink_escape`.
- **Risque résiduel** : lien créé entre canonicalize et write (mitigé par
  écriture en temporaire dans la racine + rename atomique).

## T13 — Écriture hors du répertoire autorisé

- **Actif protégé** : système de fichiers hôte.
- **Attaquant supposé** : attaquant manipulant une proposition d'écriture.
- **Vecteur d'entrée** : proposition d'outil ou d'événement visant un chemin
  absolu hors racine.
- **Impact** : modification arbitraire de fichiers.
- **Contrôle préventif** : contrôle de capacité (§7) ; racine unique ; pas
  d'écriture hors espace temporaire sans autorisation hôte.
- **Contrôle de détection** : audit des écritures ; recalcul d'empreinte.
- **Réponse** : rejet.
- **Test correspondant** : `tests/adversarial/write_root.rs::write_outside_allowed_root`.
- **Risque résiduel** : bug dans la politique de racine lui-même.

## T14 — Injection de commande

- **Actif protégé** : shell/processus hôte.
- **Attaquant supposé** : attaquant via une proposition modèle.
- **Vecteur d'entrée** : proposition produisant du texte shell exécuté.
- **Impact** : exécution arbitraire de commandes.
- **Contrôle préventif** : interdiction explicite (§7) de
  `Command::new("sh").arg("-c").arg(model_text)` ; arguments comme éléments
  séparés ; MVP n'exécute aucun outil.
- **Contrôle de détection** : schéma `ToolProposalView` typé ; liste positive
  d'outils/arguments.
- **Réponse** : rejet `RejectReason::Unauthorized`.
- **Test correspondant** : `tests/adversarial/command_injection.rs::shell_command_proposal`.
- **Risque résiduel** : nul en MVP ; résiduel après Phase 5 si exécuteur mal
  câblé.

## T15 — Exfiltration de secrets

- **Actif protégé** : secrets, données `Secret`.
- **Attaquant supposé** : attaquant induisant une sortie contenant un secret.
- **Vecteur d'entrée** : prompt/événement forçant un log ou une sortie
  contenant un secret.
- **Impact** : fuite d'informations sensibles.
- **Contrôle préventif** : `DataClassification::Secret` interdite partout
  (§20) ;首 stratégie = ne pas charger le secret ; pas de logging de secret ;
  S8.
- **Contrôle de détection** : tests « secret dans une trace » et « secret dans
  un message d'erreur ».
- **Réponse** : échec en mode fermé ; révision du flux.
- **Test correspondant** : `tests/adversarial/secret_exfil.rs::secret_exfiltration`.
- **Risque résiduel** : représentation indirecte d'un secret non détectée.

## T16 — Déni de service mémoire

- **Actif protégé** : RAM, disponibilité.
- **Attaquant supposé** : client envoyant une entrée massique.
- **Vecteur d'entrée** : entrée ou contexte gigantesque.
- **Impact** : OOM, crash.
- **Contrôle préventif** : `MemoryBudget` global (§11) ; contrôle d'admission
  avant allocation (§12) ; S5.
- **Contrôle de détection** : compteurs de refus et pic mémoire.
- **Réponse** : rejet avant chargement complet si possible.
- **Test correspondant** : `tests/adversarial/memory_dos.rs::memory_denial_of_service`.
- **Risque résiduel** : requête sous la limite mais combinée à charge
  concurrente (mitigé par `max_concurrent_requests`).

## T17 — Séquence ou batch surdimensionné

- **Actif protégé** : budget `max_batch_size`, `max_context_tokens`.
- **Attaquant supposé** : client contournant la borne batch.
- **Vecteur d'entrée** : batch ou séquence dépassant les limites.
- **Impact** : saturation mémoire latente.
- **Contrôle préventif** : `max_batch_size`, `max_context_tokens`,
  `max_output_tokens` ; `checked_kv_cache_bytes` (§11).
- **Contrôle de détection** : `RequestEstimate.total_bytes` vs budget.
- **Réponse** : `MemoryError::CapacityExceeded`.
- **Test correspondant** : `tests/adversarial/oversized.rs::oversized_batch_or_sequence`.
- **Risque résiduel** : estimation pessimiste consommant trop de budget.

## T18 — Récursion ou structure trop profonde

- **Actif protégé** : parsing, pile.
- **Attaquant supposé** : entrée profondément imbriquée.
- **Vecteur d'entrée** : JSON/structure à profondeur excessive.
- **Impact** : débordement de pile, lenteur.
- **Contrôle préventif** : parsing strict avec profondeur maximale ; S5.
- **Contrôle de détection** : métrique de profondeur de parsing.
- **Réponse** : rejet `RejectReason::Malformed`.
- **Test correspondant** : `tests/adversarial/deep_recursion.rs::deeply_nested_structure`.
- **Risque résiduel** : profondeur juste sous le seuil répétée.

## T19 — Saturation des files d'attente

- **Actif protégé** : queues de requêtes, backpressure.
- **Attaquant supposé** : client inondant le runtime.
- **Vecteur d'entrée** : flux de requêtes > capacité.
- **Impact** : latence, famine, OOM léger.
- **Contrôle préventif** : files bornées (§15-16) ; `QueueFullPolicy`
  (`RejectNewest` par défaut en interactif) ; `max_queue_depth`.
- **Contrôle de détection** : métrique de saturation des files.
- **Réponse** : rejet du plus récent, jamais suppression silencieuse.
- **Test correspondant** : `tests/adversarial/queue_saturation.rs::queue_saturation`.
- **Risque résiduel** : famine des requêtes plus anciennes sous `RejectNewest`.

## T20 — Erreur arithmétique de calcul de taille

- **Actif protégié** : intégrité des bornes mémoire.
- **Attaquant supposé** : entrée provoquant un overflow multiplicateur
  (dims).
- **Vecteur d'entrée** : dimensions telles que `layers*tokens*heads*…`
  déborde.
- **Impact** : allocation minuscule alors que besoin énorme.
- **Contrôle préventif** : `checked_add/checked_mul/checked_sub` (§11) ;
  `checked_kv_cache_bytes`.
- **Contrôle de détection** : tests « dimensions provoquant un overflow »,
  « nombre de tenseurs excessif ».
- **Réponse** : `MemoryError::ArithmeticOverflow`.
- **Test correspondant** : `tests/adversarial/size_overflow.rs::arithmetic_size_overflow`.
- **Risque résiduel** : backend avec layout différent non couvert par
  l'estimation standard (mitigé : le backend vérifie sa disposition réelle).

## T21 — Duplication ou rejeu d'événement

- **Actif protégé** : journal d'événements, comptes d'évidences.
- **Attaquant supposé** : attaquant rejouant un événement légitime.
- **Vecteur d'entrée** : même événement soumis plusieurs fois.
- **Impact** : fausse preuve de répétition, règle induite artificiellement.
- **Contrôle préventif** : identifiant + empreinte par preuve (§9) ;
  dédoublonnage.
- **Contrôle de détection** : détection de doublon par empreinte.
- **Réponse** : un seul comptage, journal d'audit.
- **Test correspondant** : `tests/adversarial/replay.rs::duplicate_or_replayed_event`.
- **Risque résiduel** : variantes légères non identiques par empreinte.

## T22 — Rollback vers une version vulnérable

- **Actif protégé** : politique, profil, journals.
- **Attaquant supposé** : attaquant réinstallant une version antérieure.
- **Vecteur d'entrée** : déploiement d'une version vulnérable.
- **Impact** : contournement des contrôles ajoutés.
- **Contrôle préventif** : `schema_version` (§2, §21) ; rejet des schémas
  inconnus ; provenance des migrations (§3).
- **Contrôle de détection** : audit des versions de schéma.
- **Réponse** : refus de charger/charger en lecture seule, alerte.
- **Test correspondant** : `tests/adversarial/rollback.rs::rollback_to_vulnerable_version`.
- **Risque résiduel** : downgrade au sein d'une même `schema_version`.

## T23 — Falsification de provenance

- **Actif protégé** : chaîne de causalité des décisions (S6).
- **Attaquant supposé** : attaque tentative d'attribuer une décision à une
  fausse source.
- **Vecteur d'entrée** : événement avec `InputOrigin`/`EvidenceOrigin` falsifié
  ou absent.
- **Impact** : décision non auditable, règle à tort fiable.
- **Contrôle préventif** : provenance obligatoire (S6) ; champs d'origine typés
  et immuables côté runtime.
- **Contrôle de détection** : test « événement avec provenance absente ».
- **Réponse** : rejet `RejectReason::Malformed`/`HardConstraint`.
- **Test correspondant** : `tests/adversarial/provenance.rs::provenance_forgery`.
- **Risque résiduel** : composant interne compromis forgeant une provenance
  valide.

## T24 — Récompense manipulée

- **Actif protégé** : `reward_engine`, décision lexicographique.
- **Attaquant supposé** : attaquant tentant de surpondérer une candidate
  violente.
- **Vecteur d'entrée** : proposition visant à augmenter le score pour compenser
  une violation dure.
- **Impact** : adoption d'une action non sûre.
- **Contrôle préventif** : S4 (dure non compensable) ; ordre lexicographique
  (§8) ; reward appliqué **après** les contraintes dures.
- **Contrôle de détection** : test « compensation d'une violation dure par une
  récompense ».
- **Réponse** : rejet `RejectReason::HardConstraint`.
- **Test correspondant** : `tests/adversarial/reward.rs::manipulated_reward_compensates_hard_violation`.
- **Risque résiduel** : bug dans l'ordre lexicographique.

## T25 — Contournement des validateurs

- **Actif protégé** : chaîne de validation (structurelle, symbolique, dure).
- **Attaquant supposé** : proposition cherchant un chemin court-circuitant la
  chaîne.
- **Vecteur d'entrée** : proposition rejetée en structurelle mais présentée
  comme déjà validée.
- **Impact** : entrée d'une candidate non validée.
- **Contrôle préventif** : pipeline obligatoire (§3) ; aucun effet de bord avant
  fin de chaîne ; S10.
- **Contrôle de détection** : test « tentative de transformer une règle souple
  en règle dure » + « contournement des validateurs ».
- **Réponse** : rejet ; échec en mode fermé.
- **Test correspondant** : `tests/adversarial/validator_bypass.rs::validator_bypass`.
- **Risque résiduel** : bug d'ordonnancement du pipeline.

## T26 — Confusion entre donnée et instruction

- **Actif protégé** : la distinction instruction/donnée (S3).
- **Attaquant supposé** : entrée récupérée transformée en instruction
  privilégiée.
- **Vecteur d'entrée** : donnée contenant un fragment de politique/gestion de
  capacité.
- **Impact** : élevation de privilège de la donnée.
- **Contrôle préventif** : `InputOrigin`/`TrustClass` typés (§6) ; champs
  distincts ; interdiction de concaténer toutes les sources non typées ; S3.
- **Contrôle de détection** : tests « faux message système dans une donnée »,
  « confusion entre donnée et instruction ».
- **Réponse** : rejet.
- **Test correspondant** : `tests/adversarial/data_injection_confusion.rs::data_instruction_confusion`.
- **Risque résiduel** : délimitation structurelle insuffisante à elle seule
  (reconnu par la spec§6).

---

## Récapitulatif des tests

| Menace | Test |
|--------|------|
| T01 | `prompt_injection.rs::direct_prompt_injection` |
| T02 | `indirect_injection.rs::indirect_injection_from_file` |
| T03 | `kb_injection.rs::injection_from_knowledge_base` |
| T04 | `tool_output.rs::malicious_tool_output` |
| T05 | `feedback_poisoning.rs::feedback_event_poisoning` |
| T06 | `profile_poisoning.rs::corrupted_profile` |
| T07 | `corpus_poisoning.rs::training_corpus_poisoning` |
| T08 | `model_weights.rs::malicious_or_corrupt_weights` |
| T09 | `tokenizer.rs::incompatible_tokenizer` |
| T10 | `artifact.rs::truncated_artifact` |
| T11 | `path_traversal.rs::directory_traversal` |
| T12 | `symlink.rs::symlink_escape` |
| T13 | `write_root.rs::write_outside_allowed_root` |
| T14 | `command_injection.rs::shell_command_proposal` |
| T15 | `secret_exfil.rs::secret_exfiltration` |
| T16 | `memory_dos.rs::memory_denial_of_service` |
| T17 | `oversized.rs::oversized_batch_or_sequence` |
| T18 | `deep_recursion.rs::deeply_nested_structure` |
| T19 | `queue_saturation.rs::queue_saturation` |
| T20 | `size_overflow.rs::arithmetic_size_overflow` |
| T21 | `replay.rs::duplicate_or_replayed_event` |
| T22 | `rollback.rs::rollback_to_vulnerable_version` |
| T23 | `provenance.rs::provenance_forgery` |
| T24 | `reward.rs::manipulated_reward_compensates_hard_violation` |
| T25 | `validator_bypass.rs::validator_bypass` |
| T26 | `data_injection_confusion.rs::data_instruction_confusion` |

Chaque test est listé dans `tests/adversarial/README.md` et sera implémenté en
Rust dans son crate correspondant au fur et à mesure que l'API ciblée est
disponible.