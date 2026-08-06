# Inventaire des dépendances — COGNO-1

Conformément à COGNO-1 V2 §24 (chaîne d'approvisionnement).

## Vue d'ensemble

Le noyau `cogno-core` reste intentionnellement sans dépendance externe. Les
crates d'intégration utilisent un ensemble réduit et verrouillé de dépendances
pour la sérialisation déterministe des artefacts et le calcul de leurs
empreintes cryptographiques.

| Crate | Dépendances externes directes | Dépendances internes |
|-------|-------------------------------|----------------------|
| `cogno-core` | 0 | 0 |
| `cogno-runtime` | `serde`, `serde_json`, `sha2` | `cogno-core` |
| `cogno-model` | 0 | `cogno-core` |
| `cogno-cli` | `serde`, `serde_json`, `sha2` | `cogno-core`, `cogno-runtime`, `cogno-model` |

## Justification des dépendances directes

### `serde` 1.x

- usage : structures sérialisables et désérialisables pour les profils,
  validations et rapports vérifiés ;
- feature activée : `derive` ;
- licence : MIT OR Apache-2.0 ;
- proc-macro transitif : `serde_derive` ;
- aucune autorité réseau, processus, outil ou système de fichiers n'est
  accordée par cette dépendance.

### `serde_json` 1.x

- usage : lecture et écriture déterministes des artefacts JSON de contrôle ;
- features activées : défaut uniquement ;
- licence : MIT OR Apache-2.0 ;
- les octets persistés sont ensuite liés à leurs empreintes SHA-256 et revérifiés
  avant exposition au runtime.

### `sha2` 0.10.x

- usage : empreintes SHA-256 des journaux, rapports, profils et manifests de
  génération ;
- features activées : défaut uniquement ;
- licence : MIT OR Apache-2.0 ;
- la fonction de hachage sert à l'intégrité et à la provenance, pas à créer une
  autorité ou une preuve de confiance autonome.

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
- **Build hors réseau en `--frozen`** pour la construction release.
- **Aucune dépendance Git** : toutes les dépendances externes viennent du
  registre crates.io et sont verrouillées par checksum.
- **Aucun script de build propriétaire**.
- **Aucun `unsafe` propriétaire** : chaque crate COGNO conserve
  `#![forbid(unsafe_code)]`.
- **Dépendances minimales et justifiées** : `cogno-core` reste sans dépendance ;
  les dépendances externes sont confinées aux couches runtime/CLI.
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

## Note sur l'invariant `unsafe`

`#![forbid(unsafe_code)]` protège le code propriétaire du workspace. Les crates
transitives peuvent contenir du code `unsafe`; leur présence est donc rendue
explicite, verrouillée et soumise à revue de chaîne d'approvisionnement.
