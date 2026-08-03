# `models/`

Répertoire réservé aux **artefacts de modèle** (poids, tokenizer, manifeste)
utilisés par COGNO-1.

## État Phase 0

**Aucun artefact n'est présent.** Aucun modèle n'est chargé en Phase 0. Le
dépôt démarre par le noyau déterministe.

## Règles (§21)

- Le chargeur considère **toujours** les poids et le tokenizer comme
  **hostiles**.
- Avant toute allocation majeure, le chargeur vérifie : taille du fichier,
  empreinte, version du schéma, nombre de tenseurs, dimensions, types
  numériques, multiplications de dimensions (arithmétique contrôlée), doublons
  de noms, données hors limites, champs inconnus, architectures non prises en
  charge.
- Le format de modèle **ne doit pas permettre l'exécution de code** pendant le
  chargement.
- Le noyau **ne télécharge aucun poids** depuis Internet. Tout téléchargement
  éventuel appartient à un outil séparé, soumis à vérification d'intégrité.
- Un `ModelManifest` (cf. `docs/MEMORY_MODEL.md` / `docs/MODEL_CARD.md`) doit
  accompagner tout artefact.

## Format attendu (planned)

```
models/
└── <model_id>/
    ├── manifest.json        ModelManifest (schema_version, hashes, dims, ...)
    ├── tokenizer.*           hash vérifié contre tokenizer_hash
    └── weights.*             hash vérifié contre weights_hash
```

Ne **jamais** committer d'artefact de modèle dans ce dépôt. Les artefacts sont
gérés hors dépôt (ou via Git LFS dans un dépôt séparé) afin de garder le dépôt
principal léger et auditable.