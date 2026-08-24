# Échange scientifique de goûts (packages, consentement, transport, boucle)

Ce document couvre l'infrastructure d'échange et d'apprentissage continu des
préférences (« scientific taste ») introduite par les étapes 3→7 de la vision
TASTE-1 : format de package, identité de scope, consentement hôte,
composition déterministe, transport réseau durci et boucle de feedback.

Principe constant : **le modèle ne détient jamais d'autorité**. Chaque
préférence active repose sur des confirmations non-model vérifiables, et
chaque couche décrite ici échoue fermée (`fail-closed`) face au hostile.

## 1. Consentement hôte (`taste.settings.json`)

Chaque racine de store porte un fichier de consentement :

```json
{"schema_version":1,"learning_enabled":true,"export_allowed":true,"import_allowed":false}
```

| Clé | Défaut | Effet |
|-----|--------|-------|
| `learning_enabled` | `true` | sans lui : aucun enregistrement ni dérivation |
| `export_allowed` | `false` | requis pour écrire/partager `taste.md` ou pousser sur le réseau |
| `import_allowed` | `false` | requis pour ingérer un package ou tirer (`GET`) |

Défauts préservant la vie privée : la machine apprend localement mais
**n'exporte et n'importe rien** sans décision explicite. Fichier > 4 Kio ou
JSON invalide ⇒ erreur fatale (jamais de fallback silencieux).

## 2. Format package (`taste.md`)

Un package = coquille markdown + bloc JSON canonique scellé :

```
# COGNO Taste Package

- digest: sha256:<64 hex>

```json
{...corps canonique...}
```

## Preferences

- #7 project[project:cogno-1] confidence=8000bps sources=…
```

- le digest couvre **tous les octets** du corps canonique ; toute édition,
  même d'un chiffre, casse la vérification (`DigestMismatch`) ;
- le ré-encodage canonique doit reproduire les octets exacts (ordre des clés
  et espacement non négociables) ;
- bornes : 256 préférences, 16 sources/entrée, 16 parents, 256 Kio ;
- n'exporte que l'état `Active` — états quarantaine/conflit/compteurs ne
  quittent jamais l'hôte ;
- à l'import, tout devient une proposition liée à la quarantaine
  (`ImportedProfile`) : elle ne peut **jamais** s'activer seule.

Identité de scope : chaque entrée porte `(scope, scope_key)` où `scope_key`
est une instance bornée (≤ 128 octets imprimables). Une préférence apprised
dans `project:cogno-1` n'est pas celle de `project:autre` : le premier
binding gagne, toute divergence crée un état `Conflicted` qu'un arbitrage
humain seul peut résoudre.

## 3. Composition déterministe (`compose`)

`TastePackage::compose(&parts, policy)` fusionne N packages :

- union par clé `(preference_id, scope, scope_key)` ;
- confiance maximale retenue ; toute divergence est inscrite dans
  `conflicts` (traçabilité) ;
- sources : union triée dédupliquée ;
- ancêtres : union des digests (bornée à 16) ;
- **indépendant de l'ordre** : mêmes entrées ⇒ octets identiques, quel que
  soit l'ordre d'appel (vérifié par tests et smoke CLI).

CLI : `cogno-taste-compose STORE_ROOT OUTPUT_DIR PKG.md [PKG.md…]`
(gated export).

## 4. Transport (`cogno-transport`, protocole `COGNO-TASTE/1`)

Framing borné sur tout flux `Read + Write` (std only, TCP via CLIs) :

```
COGNO-TASTE/1 PUSH <len> [AUTH <token>]\n<payload>   ->  OK|DUP|ERR <code>\n
COGNO-TASTE/1 GET <digest> [AUTH <token>]\n          ->  PKG <len>\n<payload> | ERR not_found\n
```

Durcissements :

- **auth** : secret partagé via variable d'environnement
  `COGNO_TASTE_TOKEN` (jamais en argv), comparaison en temps constant ;
  refus ⇒ `ERR auth_failed` après drainage complet de la trame ;
- **sessions** : jusqu'à `--max-pushes` requêtes par connexion puis
  `ERR too_many_requests` ;
- **idempotence** : digest déjà accepté ⇒ `DUP` (mémoire par session côté
  bibliothèque ; persistée dans `taste.accepted.log` côté CLI serveur) ;
- **pull** : récupération par digest avec re-vérification locale du digest
  demandé (`DigestMismatch` sinon) ;
- **retry** : `push_with_retry` ne retente que les erreurs d'i/o ; refus de
  consentement et rejets pairs sont définitifs ;
- **consentement aux deux bords** : exporter pour envoyer, importer pour
  accepter ou tirer.

Vérification systématique : chaque payload reçu est intégralement lu puis
re-parse et re-vérifié (digest + bornes) avant remise à l'appelant.

CLIs : `cogno-taste-send`, `cogno-taste-serve`, `cogno-taste-fetch`.

## 5. Boucle de feedback continue

Journal append-only `<root>/taste.feedback.jsonl` d'issues du travail réel :

```json
{"schema_version":1,"event_id":2,"preference_id":7,"scope":"project",
 "scope_key":"project:cogno-1","kind":"confirmed","origin":"deterministic_evaluation",
 "source_kind":"deterministic_kernel","source_id":"scirust@gen7","confidence_bps":6500,
 "observed_at_unix":1787600000}
```

Règles :

- tags fermés (`kind`: proposed/accepted/confirmed/edited/rejected/
  contradicted) ; origine inconnue ou `model_inference` hors `proposed` ⇒
  refus d'enregistrement ; les enregistrements sont horodatés automatiquement ;
- itération : journal → validation → déduplication par `event_id` → ordre
  canonique → dérivation déterministe ;
- **oubli déterministe** : `observed_at_unix` + fenêtre de rétention excluent
  les preuves expirées du calcul, sans jamais réécrire l'historique ;
- **décroissance graduée** (`--graduated-decay`) : dans la fenêtre, la
  confiance décroît linéairement jusqu'à zéro à l'horizon (arithmétique
  entière) — la fraîcheur d'une preuve module son poids ;
- **bandit borné** (`--bandit-step-bps B`) : le bilan des issues d'une
  préférence (confirmations+acceptations moins rejets+contradictions)
  décale la confiance effective de ses preuves, borné à ±5 000 bps — aucun
  journal ne peut fabriquer une confiance illimitée ;
- **compaction** : `compact_feedback_journal` déduplique et purge
  atomiquement le journal (échec fermé sur ligne corrompue) ;
- snapshots `taste.profile.json` (audit/redémarrage) : toujours recalculables
  depuis le journal, jamais autoritaires ; le CLI affiche le delta d'état et
  de confiance entre itérations.

Conditionnement du modèle : `active_preference_views()` expose les seules
préférences `Active` (id, scope, clé, confiance) à la couche de génération,
sous consentement `learning_enabled` — jamais le graphe de provenance.

Audit : `cogno-taste-verify STORE_ROOT [--retention-secs S]` contrôle
lecture-seule settings, journal (corruption, doublons, expiration), snapshot,
journal d'acceptations et package local.

Chaîne opérationnelle complète :

```
validations SciRust ──▶ cogno-taste-ingest ──┐
packages distants  ──▶ cogno-taste-import ───┤
                                             ▼
                                  taste.feedback.jsonl
                                             │ cogno-taste-loop (--retention-secs
                                             │   --graduated-decay --bandit-step-bps)
                                             ▼
                              profil dérivé + snapshot + delta
                                             │ cogno-taste-export (consent export)
                                             ▼
                                     taste.md signé ──▶ compose / send / serve ⇄ pairs
```

Le serveur accepte `--daemon` (boucle d'accept bornée par
`--max-connections`, backoff sur erreurs, arrêt gracieux au budget) pour un
déploiement persistant ; les mémoires d'idempotence survivent aux
connexions via `taste.accepted.log`.

## 6. Notes adversariales

- Toute trame est consommée avant décision (pas de connexion à moitié lue).
- En-tête > 256 octets, longueur 0 ou > 256 Kio ⇒ rejet immédiat.
- Un corps altéré d'un octet est détecté par le digest, pas par la confiance.
- Les imports et pulls restent liés à la quarantaine : le consentement donne
  le droit de lire, jamais celui de faire confiance.
