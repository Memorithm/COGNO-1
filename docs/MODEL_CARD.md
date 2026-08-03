# Model Card — COGNO-1

> Conformément à l'usage des *model cards*, ce document décrit le modèle
> intégré à COGNO-1. **En Phase 0, aucun modèle n'est chargé.**

## 1. État actuel

| Champ | Valeur |
|-------|--------|
| Phase | **0 — Noyau déterministe** |
| Modèle chargé | **Aucun** |
| Backend | N/A (simulateur déterministe prévu en Phase 1) |
| Paramètres | 0 |
| Contexte max | N/A |
| Tokenizer | N/A |
| Origine des poids | N/A |
| Licence du modèle | N/A |

Aucune inférence n'a lieu. Aucune donnée n'est envoyée à un modèle.

## 2. Rôle limité du modèle (§2)

Une fois intégré (Phase 2+), le petit modèle peut **uniquement** :

- classifier un événement de feedback ;
- comparer une proposition et une version retenue ;
- extraire une préférence candidate ;
- associer une préférence à une catégorie ;
- estimer la pertinence contextuelle d'une règle ;
- classer plusieurs sorties déjà produites ;
- générer une explication ;
- signaler une contradiction possible.

Le modèle ne peut **jamais**, à lui seul : créer une règle de sécurité
obligatoire ; supprimer une règle existante ; exécuter une commande ; écrire
dans un dépôt ; accéder au réseau ; ouvrir un fichier arbitraire ; modifier
ses propres poids ; promouvoir un modèle ; modifier un budget mémoire ;
modifier les validateurs ; décider qu'une contrainte dure peut être ignorée ;
transformer une donnée récupérée en instruction privilégiée.

## 3. Outputs strictement typés (§2)

```rust
pub struct CognoProposalView<'a> {
    pub schema_version: u16,
    pub action: ProposalAction,
    pub category: RuleCategory,
    pub scope: RuleScope,
    pub confidence_bps: u16,  // 0..=10_000
    pub evidence_ids: &'a [EvidenceId],
    pub payload: &'a [u8],
}
```

`confidence_bps` est en points de base (`0`..=`10_000`). Toute valeur
supérieure est rejetée. Champs inconnus, dupliqués ou hors limites → rejet
explicite.

## 4. Modèle de menace associé

Voir `docs/THREAT_MODEL.md` : poids malveillants/corrompus, tokenizer
incompatible, artefact tronqué, dimensions provoquant un overflow, nombre de
tenseurs excessif, etc. Chaque menace possède un test dans
`tests/adversarial/`.

## 5. Manifeste d'artefact (§21)

Tout artefact de modèle devra être accompagné d'un `ModelManifest` :

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

Avant toute allocation majeure, chargeur vérifie : taille du fichier, empreinte,
version, nombre de tenseurs, dimensions, types numériques, multiplications de
dimensions (arithmétique contrôlée), doublons de noms, données hors limites,
champs inconnus, architectures non prises en charge. Le format ne permet pas
l'exécution de code pendant le chargement.

## 6. Données d'entraînement (Phase 3)

Prévu : corpus avec provenance ; séparation train/validation/test ; cas
contradictoires, adversariaux, injections, sorties malformées, exemples
négatifs. Aucun secret (donnée `Secret`, §20) dans les jeux d'entraînement.

## 7. Évaluation (Phase 4)

L'objectif Meta-NeuroSymbolic n'est activé que lorsque :
le moteur scalaire est validé ; la politique de référence est figée ; les
log-probabilités sont disponibles ; le backend est réellement différentiable ;
les tests held-out sont en place ; les garde-fous contre l'empoisonnement
fonctionnent.

## 8. Limites connues

- En Phase 0, **aucune** capacité n'est active.
- Le noyau Rust reste la seule autorité, independamment du modèle_intégré.
- Le modèle n'est jamais une source fiable (S1).