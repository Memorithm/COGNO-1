# Inventaire des dépendances — COGNO-1

Conformément à COGNO-1 V2 §24 (chaîne d'approvisionnement).

## Vue d'ensemble

À ce jour, **COGNO-1 n'a aucune dépendance externe**. Le workspace ne déclare
aucun crate hors `cogno-*`. `cogno-core` est intentionnellement sans
dépendances (§23 : sans réseau, sans process, sans FS, sans `unsafe`
propriétaire). Les autres crates ne dépendent que de `cogno-core` (chemin
local).

| Crate | Dépendances externes | Dépendances internes |
|-------|----------------------|----------------------|
| `cogno-core` | 0 | 0 |
| `cogno-runtime` | 0 | `cogno-core` |
| `cogno-model` | 0 | `cogno-core` |
| `cogno-cli` | 0 | `cogno-core`, `cogno-runtime`, `cogno-model` |

## Vérification

```bash
# La CI (.github/workflows/ci.yml, job no-external-deps) échoue si une
# dépendance externe apparaît sans être documentée ici.
cargo generate-lockfile
grep -E '^name = "' Cargo.lock | sed 's/name = "//; s/"//' | grep -v '^cogno-'
```

## Politique (§24)

- **Cargo.lock conservé** dans le dépôt (committé dès l'init).
- **CI en `--locked`** pour `clippy` et `test` : la résolution utilisée est
  celle enregistrée dans `Cargo.lock`.
- **Build hors réseau en `--frozen`** pour le job `frozen` : empêche Cargo
  d'accéder au réseau.
- **Aucune dépendance Git non épinglée**. Aucune dépendance Git dans le
  workspace.
- **Aucun script de build** ni **macro procédurale** dans le workspace.
- **Licences** : N/A (code propriétaire uniquement, sous MIT).
- **Vulnérabilités connues** : à auditer via `cargo audit` lorsqu'une
  dépendance externe sera ajoutée. Aucune dépendance externe aujourd'hui donc
  aucune surface d'audit aujourd'hui.
- **Features** : aucune feature activée (rien à désactiver).

## Ajout d'une dépendance externe

Avant d'ajouter une dépendance externe :

1. Évaluer si l'objectif peut être atteint sans elle (le noyau doit rester
   déterministe, sans réseau, sans `unsafe` propriétaire — §23).
2. Si oui, l'ajouter ici avec : nom, version **épinglée**, licence, justification
   courte, features activées, surface `unsafe` connue, sponsors/mainteneurs.
3. Vérifier les vulnérabilités connues (`cargo audit`).
4. Auditer les scripts de build et proc-macros de la dépendance.
5. Commettre le `Cargo.lock` mis à jour.
6. Mettre à jour le job `no-external-deps` si la politique change.

## Note sur l'invariant `unsafe`

`#![forbid(unsafe_code)]` dans tout le code propriétaire n'empêche pas les
dépendances externes de contenir du `unsafe`. C'est pourquoi chaque ajout de
dépendance externalise l'audit `unsafe` et le documente dans cet inventaire.