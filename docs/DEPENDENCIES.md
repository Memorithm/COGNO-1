# Inventaire des dépendances — COGNO-1

Conformément à COGNO-1 V2 §24 (chaîne d'approvisionnement).

## Vue d'ensemble

Le noyau `cogno-core` et le moteur différentiable propriétaire `cogno-scirust`
restent intentionnellement sans dépendance externe. Les couches modèle, runtime
et CLI utilisent un ensemble réduit et verrouillé de dépendances pour les
empreintes cryptographiques et la sérialisation de contrôle.

| Crate | Dépendances externes directes | Dépendances internes |
|-------|-------------------------------|----------------------|
| `cogno-core` | 0 | 0 |
| `cogno-scirust` | 0 | `cogno-core` |
| `cogno-model` | `sha2` | `cogno-core`, `cogno-scirust` |
| `cogno-runtime` | `serde`, `serde_json`, `sha2`, `scirust-agent-protocol` | `cogno-core`, `cogno-model` |
| `cogno-cli` | `serde`, `serde_json`, `sha2` | `cogno-core`, `cogno-runtime`, `cogno-model` |

## Justification des dépendances directes

### `serde` 1.x

- usage : structures sérialisables et désérialisables pour les profils,
  validations, enveloppes d'observation et rapports vérifiés ;
- feature activée : `derive` ;
- licence : MIT OR Apache-2.0 ;
- proc-macro transitif : `serde_derive` ;
- aucune autorité réseau, processus, outil ou système de fichiers n'est
  accordée par cette dépendance.

### `serde_json` 1.x

- usage : lecture et écriture déterministes des artefacts JSON de contrôle et
  ingestion des enveloppes JSONL ;
- features activées : défaut uniquement ;
- licence : MIT OR Apache-2.0 ;
- les octets persistés sont liés à leurs empreintes et revérifiés avant leur
  exposition au runtime lorsqu'ils participent à un état persistant vérifié.

### `sha2` 0.10.x

- usage dans `cogno-model` : hash du tokenizer, empreintes de corpus/revue et
  SHA-256 des artefacts neuronaux hostiles ;
- usage dans `cogno-runtime` : manifests et chaînes de génération, profils et
  état persisté vérifié ;
- usage dans `cogno-cli` : contrôles/empreintes nécessaires aux flux exposés par
  le binaire ;
- features activées : défaut uniquement ;
- licence : MIT OR Apache-2.0 ;
- le hash sert à l'intégrité et à la provenance, jamais à créer à lui seul une
  autorité ou une preuve de confiance.

### `scirust-agent-protocol` 0.1.0

- usage dans `cogno-runtime` : validation d'un `AgentMessage` contre l'identité
  de l'émetteur authentifiée par le transport et vérification canonique de
  `ExecutionAttestation` avant création de l'autorité hôte COGNO ;
- source : dépôt `Memorithm/scirust`, révision exacte
  `154e61677511c859086c39141933103b4a844c64`, verrouillée dans `Cargo.lock` ;
- dépendances propres : `serde` et `serde_json`, déjà présentes dans l'inventaire ;
- aucune confiance n'est accordée à une identité, un booléen ou un digest fourni
  librement par l'appelant : l'API protocolaire vérifie l'identité authentifiée
  et recalcule l'empreinte du profil d'exécution ;
- toute modification de la révision nécessite une revue explicite de ce document
  et du lockfile.

## Inventaire verrouillé transitif

La CI compare l'ensemble exact des crates externes présentes dans `Cargo.lock`
à l'inventaire suivant :

```text
block-buffer
cfg-if
cpufeatures
crypto-common
digest
generic-array
itoa
libc
memchr
proc-macro2
quote
scirust-agent-protocol
serde
serde_core
serde_derive
serde_json
sha2
syn
typenum
unicode-ident
version_check
zmij
```

Toute apparition, disparition ou modification de cet ensemble fait échouer le
job `documented external dependencies (§24)` jusqu'à mise à jour explicite de
ce document et de la politique CI.

## Vérification

```bash
cargo metadata --locked --format-version 1 --no-deps >/dev/null
grep -E '^name = "' Cargo.lock \
  | sed 's/name = "//; s/"//' \
  | grep -v '^cogno-' \
  | sort -u
```

## Politique (§24)

- **Cargo.lock conservé** dans le dépôt.
- **CI en `--locked`** pour les tests, Clippy et la documentation.
- **Build hors réseau en `--frozen`** après préchargement des dépendances
  verrouillées.
- **Dépendances Git interdites par défaut** ; l'unique exception actuelle est
  `scirust-agent-protocol`, épinglée sur une révision SciRust exacte et utilisée
  uniquement pour le contrat de transport/attestation entre les deux projets.
- **Aucun script de build propriétaire**.
- **Aucun `unsafe` propriétaire** : chaque crate COGNO conserve
  `#![forbid(unsafe_code)]`.
- **Dépendances minimales et justifiées** : `cogno-core` et `cogno-scirust`
  restent sans dépendance externe ; `sha2` est utilisé par la couche modèle et
  les dépendances de sérialisation/hachage restent confinées aux couches qui en
  ont explicitement besoin.
- **Vulnérabilités connues** : l'inventaire doit être audité lors de chaque
  changement de version ou ajout de crate.

## Ajout ou mise à jour d'une dépendance externe

1. Vérifier que l'objectif ne peut pas être atteint raisonnablement avec les
   abstractions déjà présentes.
2. Documenter le nom, la version, la licence, la justification, les features,
   les proc-macros, scripts de build et surfaces `unsafe` connues.
3. Régénérer et vérifier `Cargo.lock`.
4. Mettre à jour l'inventaire transitif exact ci-dessus et le job CI.
5. Exécuter `cargo fmt`, `cargo test --locked`, `cargo clippy --locked` et
   `cargo build --frozen --release`.

## Qualification séparée du SciRust canonique

Le nom historique `cogno-scirust` ne signifie pas que le backend tensoriel de
`Memorithm/scirust` est utilisé. `LocalDifferentiableBackend` est une API alias
de l'implémentation COGNO existante, sans nouvelle dépendance Cargo.

Le workflow `canonical scalar qualification` compile séparément deux modules
originaux de `Memorithm/scirust` à la révision
`68d544679df465d3b85539c80926ed2644fbbba1`, sans features BLAS/rayon/SIMD et
sans les copier dans les sources ou la distribution MIT de COGNO. Ce programme
de qualification uniquement utilise `MatrixView` canonique (contenant du
`unsafe` revu dans son dépôt propriétaire) et `ScalarBackend`; les crates COGNO
conservent `forbid(unsafe_code)`. Les notices/licences SciRust sont conservées
dans le checkout et leurs URL dans les preuves. Aucun binaire ou source SciRust
n'est redistribué par l'artefact de résultats. La licence PolyForm Noncommercial
1.0.0 de SciRust n'est ni remplacée ni étendue par la licence MIT de COGNO ;
une adoption/distribution ultérieure doit respecter ses propres droits.

Le gate vérifie le commit et les deux identités blob avant compilation ; tout
repin impose une nouvelle revue et qualification. Cette source de test hors
workspace ne change ni `Cargo.lock`, ni l'inventaire transitif ci-dessus.
Le protocole et les limites sont dans [SCIRUST_QUALIFICATION.md](SCIRUST_QUALIFICATION.md).

## Rappel sur l'invariant `unsafe`

`#![forbid(unsafe_code)]` protège le code propriétaire du workspace. Les crates
transitives peuvent contenir du code `unsafe`; leur présence est donc rendue
explicite, verrouillée et soumise à revue de chaîne d'approvisionnement.
