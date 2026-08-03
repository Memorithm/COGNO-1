# Modèle mémoire — COGNO-1

Couvre les sections §10–19 de la spec (mémoire renforcée). Les aspects secrets
(§20), chargement modèle (§21) et fichiers (§22) sont traités dans
`docs/DATA_GOVERNANCE.md` et `docs/MODEL_CARD.md` ; ce document les rappelle
brièvement et pointe vers eux.

## 1. Deux formes de mémoire (§10)

COGNO-1 distingue :

### Mémoire volatile

- poids du modèle ; tokenizer ; KV cache ; tenseurs intermédiaires ; buffers
  d'entrée / sortie ; espace de travail des validateurs ; queues de requêtes ;
  caches temporaires.

### Mémoire sémantique persistante

- événements ; préférences ; règles ; contradictions ; profils ; provenance ;
  snapshots ; métriques ; manifestes.

> Une limite de RAM **ne remplace pas** une politique de rétention persistante.
> Une politique de rétention **ne remplace pas** un budget d'allocation en RAM.

## 2. Budget mémoire obligatoire (§11)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub hard_limit_bytes: usize,

    pub model_weights_limit_bytes: usize,
    pub tokenizer_limit_bytes: usize,
    pub kv_cache_limit_bytes: usize,
    pub tensor_workspace_limit_bytes: usize,
    pub request_scratch_limit_bytes: usize,
    pub profile_cache_limit_bytes: usize,
    pub event_buffer_limit_bytes: usize,

    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub max_batch_size: usize,
    pub max_candidates: usize,
    pub max_concurrent_requests: usize,
    pub max_queue_depth: usize,
}
```

Le constructeur vérifie (arithmétique contrôlée) :

- somme des sous-budgets ≤ `hard_limit_bytes` ;
- aucune multiplication ne déborde ;
- aucune addition ne déborde ;
- aucune limite obligatoire n'est nulle.

API : `checked_add`, `checked_mul`, `checked_sub`. **Interdit** d'écrire
directement `layers * tokens * heads * head_dim * element_size`. À la place :

```rust
pub fn checked_kv_cache_bytes(
    layers: usize,
    tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    element_bytes: usize,
    batch_size: usize,
) -> Result<usize, MemoryError>;
```

Pour un Transformer autoregressif standard, le cache couvre **clés et valeurs** :

```
KV bytes =
    2
    × layers
    × tokens
    × kv_heads
    × head_dim
    × bytes_per_element
    × batch_size
```

> Le backend doit vérifier sa **propre disposition réelle** avant d'utiliser
> cette estimation.

## 3. Contrôle d'admission avant allocation (§12)

```rust
pub struct RequestEstimate {
    pub input_bytes: usize,
    pub estimated_tokens: usize,
    pub kv_cache_bytes: usize,
    pub workspace_bytes: usize,
    pub output_reserve_bytes: usize,
    pub total_bytes: usize,
}
```

Pipeline :

```
lecture en-tête / métadonnées
  → validation des limites
  → calcul contrôlé de la taille
  → comparaison au budget disponible
  → admission ou rejet
  → allocation
```

Une requête non admissible est rejetée **avant** le chargement complet de son
contenu lorsque le protocole le permet.

## 4. Politique d'allocation (§13)

Trois phases :

### Initialisation
bornées ; contrôlées ; fallibles ; mesurées ; cohérentes avec le budget.

### Échauffement
pré-allocation des caches, espaces de travail et buffers.

### Régime stable
**aucune** nouvelle allocation dans les chemins critiques.

Les allocations fallibles utilisent des API retournant une erreur :

```rust
buffer
    .try_reserve_exact(required_capacity)
    .map_err(|_| MemoryError::AllocationFailed)?;
```

Interdit d'augmenter automatiquement la capacité au-delà de la limite
configurée.

```rust
pub enum MemoryError {
    ArithmeticOverflow,
    BudgetExceeded { requested: usize, available: usize },
    CapacityExceeded { requested: usize, maximum: usize },
    AllocationFailed,
    QueueFull,
    ContextTooLarge,
    OutputTooLarge,
}
```

## 5. Buffers préalloués (§14)

```rust
pub struct RequestScratch<'a> {
    pub token_ids: &'a mut [TokenId],
    pub logits: &'a mut [f32],
    pub validator_results: &'a mut [ValidationResult],
    pub rule_refs: &'a mut [RuleRef],
    pub candidate_scores: &'a mut [CandidateScore],
    pub output_bytes: &'a mut [u8],
}
```

Les fonctions ne conservent pas ces références au-delà de la requête.

Interdit **dans les chemins critiques** : `Vec::new`, `Vec::with_capacity`,
`vec![…]`, `String::new`, `String::with_capacity`, `Box::new`, `format!`,
`to_string`, `to_owned`, `collect::<Vec<_>>`, `collect::<String>`. Cette
interdiction **ne s'applique pas** au chargement initial, aux outils hors ligne
ou à la construction de rapports humains.

## 6. Conteneurs bornés (§15)

Toute collection influencée par une entrée externe possède une capacité
maximale :

```rust
pub struct BoundedVec<T> {
    inner: Vec<T>,
    max_len: usize,
}

impl<T> BoundedVec<T> {
    pub fn try_push(&mut self, value: T) -> Result<(), CapacityError>;
    pub fn try_extend_from_slice(&mut self, values: &[T])
        -> Result<(), CapacityError>
    where T: Clone;
}
```

Interdites (conceptuellement illimitées) pour : événements en attente ;
messages ; candidats ; règles sélectionnées ; tokens ; résultats de
validation ; sorties d'outils ; traces ; métriques.

Les canaux asynchrones **non bornés** accumulent sans limite pratique : utiliser
des files **bornées** avec stratégie de backpressure explicite. La std Rust
distingue elle-même canaux non bornés (`fn unbounded`-like) et canaux
synchrones bornés.

## 7. Backpressure et concurrence (§16)

Toute file possède : une capacité ; un comportement à file pleine ;
un délai maximal ; une métrique de saturation.

```rust
pub enum QueueFullPolicy {
    RejectNewest,
    RejectOldest,
    BlockWithDeadline,
}
```

Pour les requêtes interactives : `RejectNewest` par défaut. Ne **pas supprimer
silencieusement** de requête existante.

Limiter : requêtes simultanées ; générations parallèles ; candidats ;
validateurs actifs ; tâches de fond ; profondeur des files. Toute tâche est
**annulable**.

## 8. Cache KV (§17)

Le cache KV a une capacité **fixe ou explicitement réservée** ; il ne croît
jamais sans contrôle. Politique explicite :

```rust
pub enum KvCachePolicy {
    RejectOnOverflow,
    SlidingWindow { window_tokens: usize },
    PrefixPinnedSlidingWindow {
        prefix_tokens: usize,
        window_tokens: usize,
    },
}
```

Chaque politique documente : tokens conservés ; tokens évincés ; effet sur le
contexte ; déterminisme de l'éviction ; mémoire maximale ; impact qualité.

Une troncature de contexte **n'est jamais silencieuse** :

```rust
pub struct ContextReport {
    pub requested_tokens: usize,
    pub admitted_tokens: usize,
    pub dropped_tokens: usize,
    pub policy: KvCachePolicy,
}
```

## 9. Mémoire sémantique persistante (§18)

Source de vérité = **journal d'événements**. Le profil actif est une **vue
dérivée**.

Architecture :

```
events.log       source append-only
snapshot.bin     état dérivé compact
manifest.json    versions et empreintes
profile.md       vue humaine non canonique
```

```rust
pub struct SemanticMemoryBudget {
    pub max_log_bytes: u64,
    pub max_events: u64,
    pub max_rules: usize,
    pub max_evidence_per_rule: usize,
    pub max_payload_bytes: usize,
    pub max_snapshot_bytes: u64,
}
```

À la limite : arrêter les écritures automatiques ; créer un snapshot vérifié ;
compacter selon politique déterministe ; conserver règles actives + provenance
minimale ; **ne jamais supprimer silencieusement** une preuve nécessaire à
l'audit. L'état persistant reste **reconstructible** depuis le journal
(S6/S7).

## 10. Durée de vie et rétention (§19)

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
l'exige. Cf. `docs/DATA_GOVERNANCE.md`.

## 11. Cross-références

- Secrets & classification des données (§20) → `docs/DATA_GOVERNANCE.md`.
- Chargement sécurisé du modèle (§21) → `docs/MODEL_CARD.md`,
  `docs/ARCHITECTURE.md`.
- Sécurité des fichiers & racine (§22) → `docs/DATA_GOVERNANCE.md`.
- Menaces mémoire (T10, T16–T20) → `docs/THREAT_MODEL.md`.
- Tests mémoire d'allocation forcée en échec (§26) → `tests/allocation/` :
  allocation initiale impossible, croissance du buffer impossible, réservation
  KV impossible, snapshot impossible, queue saturée. Le comportement attendu
  est une **erreur structurée** et **aucune modification partielle** de l'état.