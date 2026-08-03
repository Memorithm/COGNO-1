# Architecture — COGNO-1

COGNO-1 est un système générique de personnalisation neuro-symbolique composé de
deux éléments **strictement séparés** :

1. un **petit modèle d'assistance** — non fiable par construction ; il propose,
   classe, extrait, explique, signale ;
2. un **noyau déterministe Rust** — seule autorité pour la sûreté, la mémoire,
   les formats, la provenance, les décisions d'adoption, les effets de bord.

Le modèle n'est jamais une source fiable. Toute sortie du modèle est de la
**donnée non fiable** : parsée → bornée → validée → confrontée aux politiques →
éventuellement rejetée → auditée avant utilisation. L'indépendance annoncée
(§1) couvre le fournisseur du modèle, le tokenizer, le moteur tensoriel, le
format de poids, le matériel, le système d'exploitation et le projet hôte.

## 1. Pipeline obligatoire (§3)

Aucun effet de bord ne se produit avant la fin de la chaîne :

```
entrée externe
  ↓ classification de confiance        (TrustClass, InputOrigin)
  ↓ contrôle des tailles              (checked arithmetic, MemoryBudget)
  ↓ parsing strict                    (schéma versionné, rejet des champs inconnus/dupliqués)
  ↓ validation structurelle           (admissibilité)
  ↓ validation symbolique             (validateurs formels)
  ↓ application des règles de sécurité (contraintes dures)
  ↓ évaluation neuro-symbolique       (reward engine)
  ↓ décision déterministe             (lexicographique, fail-closed)
  ↓ audit                             (journal append-only + provenance)
  ↓ effet de bord éventuel            (derrière capacités explicites)
```

## 2. Autorité du noyau Rust (§3)

Le noyau est l'unique autorité pour : les règles de sûreté ; les politiques
d'accès ; les limites mémoire ; les formats acceptés ; la provenance ; les
décisions d'adoption ; les effets de bord ; les outils ; les écritures ; les
suppressions ; les migrations ; les mises à jour du profil ; les promotions de
modèle.

Le modèle ne produit jamais une ligne de shell directement exécutée. Dans le
MVP, **aucun outil n'est exécuté** (§7).

## 3. Décision lexicographique (§8)

La sûreté n'est **pas** une simple composante numérique du score. L'ordre est :

1. admissibilité structurelle ;
2. conformité aux contraintes dures ;
3. conformité aux capacités ;
4. conformité aux politiques de confidentialité ;
5. score neuro-symbolique ;
6. classement des candidats admissibles ;
7. tie-break déterministe.

```rust
if !candidate.is_structurally_valid() {
    return Decision::Reject(RejectReason::Malformed);
}
if hard_validators.reject(candidate) {
    return Decision::Reject(RejectReason::HardConstraint);
}
if !capability_policy.allows(candidate) {
    return Decision::Reject(RejectReason::Unauthorized);
}
let score = reward_engine.score(candidate)?;
Decision::Eligible(score)
```

Une pénalité de sécurité ne peut pas être annulée par une meilleure note de
style, un score utilisateur positif, une meilleure performance, une récompense
du modèle ou un gain de vitesse.

## 4. Origine typée et classe de confiance (§6)

Toute entrée porte une `InputOrigin` et une `TrustClass` indépendantes :

```rust
pub enum InputOrigin {
    SystemPolicy,
    ExplicitUserInstruction,
    UserData,
    RetrievedDocument,
    ToolOutput,
    ModelOutput,
    ImportedProfile,
    TrainingCorpus,
}

pub enum TrustClass {
    TrustedPolicy,
    AuthenticatedUser,
    ValidatedLocalData,
    UntrustedExternalData,
    UntrustedModelData,
}
```

Une chaîne de caractères ne contient jamais implicitement son niveau de
privilège. Les instructions système, données récupérées, résultats d'outils et
sorties du modèle restent dans des **champs distincts** jusqu'à l'inférence. Il
est **interdit** de construire un unique prompt par simple concaténation non
typée de toutes les sources. Une délimitation structurelle contrôlée par le
runtime précède les données externes, mais **n'est pas** à elle seule une
protection suffisante (S3).

## 5. Propositions du modèle (§2, §7)

Le modèle émet uniquement des propositions structurées et versionnées. Schéma
de référence (Vue ; forme à durée de vie bornée) :

```rust
pub struct CognoProposalView<'a> {
    pub schema_version: u16,
    pub action: ProposalAction,
    pub category: RuleCategory,
    pub scope: RuleScope,
    pub confidence_bps: u16,         // 0..=10_000 ; >10_000 -> rejet
    pub evidence_ids: &'a [EvidenceId],
    pub payload: &'a [u8],
}

pub struct ToolProposalView<'a> {
    pub tool_id: ToolId,
    pub capability_id: CapabilityId,
    pub arguments: &'a [TypedArgument<'a>],
    pub justification_code: ReasonCode,
}
```

`confidence_bps` est en points de base : `0` = confiance nulle,
`10_000` = confiance maximale. Toute valeur supérieure est rejetée. Les champs
inconnus, dupliqués ou hors limites provoquent un rejet explicite.

## 6. Responsabilités des crates

| Crate | Rôle | Phase | contraintes clés |
|-------|------|-------|------------------|
| `cogno-core` | Types, validateurs, reward, budgets mémoire, journal, profil, sécurité | 0 | sans réseau, sans process, sans FS, sans `unsafe` propriétaire, déterministe, hors-ligne |
| `cogno-runtime` | Admission mémoire, KV cache, queues/backpressure, pipeline, exécuteur d'outils (Phase 5) | 0→ | orchestre le pipeline ; MVP n'exécute aucun outil |
| `cogno-model` | Backend modèle (poids/tokenizers hostiles), simulateur Phase 1, modèle Phase 2 | 1→ | validation manifeste ; pas d'exécution de code au chargement |
| `cogno-cli` | Binaire `cogno` | 0→ | point d'entrée ; capacités derrière autorisation hôte |

### Contraintes de `cogno-core`

`cogno-core` **doit** rester : sans réseau ; sans exécution de processus ;
sans accès implicite au système de fichiers ; sans backend neuronal obligatoire
; sans `unsafe` propriétaire ; déterministe ; testable hors ligne.

## 7. État des phases (§27)

- **Phase 0 — actuelle.** Noyau déterministe. Aucun modèle, aucun outil, aucun
  effet de bord. Le code présent est un squelette : le workspace compile, les
  directives de lint (`forbid(unsafe_code)`, `deny(warnings, ...)`) sont en
  place, les documents d'architecture, de menace et de mémoire existent.
- **Phase 1 — Simulateur.** Backend déterministe renvoyant des propositions
  scriptées.
- **Phase 2 — Modèle en lecture seule.** Classifier, extraire, classer,
  expliquer ; aucune modification directe d'état.
- **Phase 3 — Apprentissage supervisé.** Corpus avec provenance ; splits ;
  cas contradictoires, adversariaux, injections, sorties malformées, négatifs.
- **Phase 4 — Objectif Meta-NeuroSymbolic.** Activé uniquement sous garde-fous
  (moteur scalaire validé, politique figée, log-probabilités disponibles,
  backend différentiable, tests held-out, anti-empoisonnement).
- **Phase 5 — Outils.** Après audit spécifique, derrière capacités explicites.

## 8. Politique de Rust (§23)

Pour tout le code propriétaire :

```rust
#![forbid(unsafe_code)]
#![deny(warnings, missing_debug_implementations, unreachable_pub)]
```

`forbid` est préféré à `deny` car il ne peut être abaissé par un module enfant.
Les dépendances externes peuvent contenir du `unsafe` : chaque dépendance est
donc minimisée, documentée, épinglée (Cargo.lock conservé, CI en `--locked`),
ses features contrôlées, ses scripts de build et proc-macros audités. Aucune
dépendance Git non épinglée n'est autorisée (§24).

## 9. Critères d'acceptation (§28 extraits)

- Le modèle ne possède aucune autorité directe.
- Toute sortie du modèle est traitée comme non fiable.
- Les contraintes dures sont appliquées **avant** la récompense ; elles ne
  peuvent pas être compensées.
- Aucun outil n'est exécuté par le MVP ; toute capacité est refusée par défaut.
- Instructions et données restent séparées.
- Toutes les tailles influencées par l'extérieur sont bornées et calculées en
  arithmétique contrôlée.
- Budget mémoire global défini ; contrôle d'admission par requête ; KV cache à
  limite stricte ; files bornées ; backpressure testée.
- Aucun secret enregistré ; modèles/tokenizers vérifiés avant chargement ;
  format de poids incapable d'exécuter du code.
- Les règles inférées commencent en quarantaine ; une inférence du modèle ne
  crée jamais une règledure ; contradictions conservées ; provenance obligatoire
  ; état persistant reconstructible ; modifications réversibles.
- Chaque menace possède un test. Dépendances verrouillées et auditées.