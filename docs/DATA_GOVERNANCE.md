# Data Governance — COGNO-1

## 1. Classification des données (§20)

```rust
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Secret,
}
```

| Classe | Modèle reçoit par défaut ? | Notes |
|--------|---------------------------|-------|
| `Public` | Oui | — |
| `Internal` | Oui | — |
| `Confidential` | Non |Autorisation explicite du composant hôte requise. |
| `Secret` | **Interdit** | Interdit dans : prompts, traces, rapports, événements de préférence, jeux d'entraînement, messages d'erreur, métriques, noms de fichiers exportés. |

### Secrets

- **Ne jamais** considérer `Drop` comme une preuve que les octets sensibles ont
  été immédiatement effacés de toutes les copies mémoire (le compilateur peut
  introduire des copies, le swap et les crash dumps peuvent persister).
- La **stratégie première est de ne pas charger le secret**.
- Lorsqu'un secret est indispensable : minimiser sa durée de vie ; interdire
  les copies ; interdire le logging ; utiliser une abstraction dédiée ;
  effectuer une remise à zéro au mieux des garanties du backend ; documenter
  les limites. (S8)

## 2. Origine typée et confiance (§6)

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

Règles :

- Une chaîne de caractères ne doit jamais contenir implicitement son niveau de
  privilège.
- Les instructions système, données récupérées, résultats d'outils et sorties
  du modèle restent dans des **champs distincts** jusqu'à l'inférence.
- Il est interdit de construire un unique prompt par simple concaténation non
  typée de toutes les sources.
- Une délimitation structurelle contrôlée par le runtime précède les données
  externes, mais **n'est pas** à elle seule une protection suffisante (S3).

## 3. Provenance (§9, S6)

Toute décision modifiant l'état persistant possède une provenance. Les preuves
ont une autorité différenciée :

```rust
pub enum EvidenceOrigin {
    ExplicitUserStatement,
    ExplicitUserApproval,
    UserEdit,
    UserRejection,
    TestResult,
    FormalValidator,
    ToolObservation,
    ImportedProfile,
    ModelInference,
}
```

Ordre de confiance par défaut (décroissant) :

```
politique signée
  > instruction utilisateur explicite
  > validation formelle
  > édition utilisateur répétée
  > acceptation implicite
  > résultat d'outil
  > profil importé
  > inférence du modèle
```

Contraintes :

- Une inférence du modèle ne crée **jamais** une règle dure.
- Une seule acceptation ne crée pas une préférence stable.
- Une règle inférée commence en **quarantaine** (`RuleState::Quarantined`).
- Une préférence contradictoire ne remplace pas silencieusement la précédente.
- Les preuves contradictoires sont **conservées**.
- Chaque règle possède un nombre minimal d'évidences.
- Les mises à jour sont limitées par session et par projet.
- Les événements dupliqués ne sont comptés qu'une seule fois.
- Chaque preuve possède un identifiant et une empreinte.
- Une règle peut être annulée et reconstruite depuis son journal.

```rust
pub enum RuleState {
    Quarantined,
    Candidate,
    Active,
    Conflicted,
    Disabled,
    Revoked,
}
```

Une règle ne passe de `Candidate` à `Active` que par une politique
déterministe.

## 4. Rétention (§19)

```rust
pub enum MemoryClass {
    EphemeralRequest,
    Session,
    Project,
    UserGlobal,
    SecurityAudit,
}

pub struct RetentionPolicy {
    pub max_session_events: usize,
    pub max_project_events: usize,
    pub max_global_rules: usize,
    pub retain_raw_prompts: bool,
    pub retain_model_outputs: bool,
    pub retain_diffs: bool,
}
```

Valeurs de sécurité recommandées pour le MVP :

```
retain_raw_prompts    = false
retain_model_outputs  = false
retain_diffs          = false
```

Conserver de préférence : identifiants ; catégories ; empreintes ; statistiques
; règles extraites ; provenance minimale ; résultats des validateurs. Les
contenus bruts ne sont conservés que lorsqu'une fonctionnalité explicite
l'exige.

## 5. Réversibilité (S7)

Toute modification de profil est réversible. L'état persistant est
reconstructible depuis le journal d'événements (`events.log`) ; le profil
actif est une **vue dérivée** (`profile.md` est une vue humaine non
canonique). Cf. `docs/MEMORY_MODEL.md` §18.

## 6. Fichiers (§22)

Toutes les opérations sont limitées à une racine configurée. Avant une
écriture : résoudre la racine autorisée ; vérifier le chemin cible ; refuser
les composants parents non autorisés ; politique sur les liens symboliques ;
écrire dans un fichier temporaire situé dans la **même racine** ; synchroniser
si le contrat le demande ; remplacer atomiquement lorsque le système le permet
; recalculer l'empreinte ; journaliser le résultat. La normalisation
lexicale d'un chemin ne résout pas nécessairement les liens symboliques ; la
sécurité repose sur une **politique de racine complète**, pas sur la seule
suppression textuelle de `..`.

## 7. Inventaire des dépendances (§24)

Tenu à jour dans le dépôt (voir `Cargo.lock`). Chaque dépendance est
documentée, ses licences vérifiées, ses vulnérabilités connues contrôlées, ses
features maîtrisées, ses scripts de build et proc-macros audités. Aucune
dépendance Git non épinglée. CI en `--locked`, builds hors-réseau testées en
`--frozen` lorsque possible.