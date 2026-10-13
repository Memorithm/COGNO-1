# Identité du backend et qualification canonique — COG-COGNO-003

Le runtime différentiable est **local et propriétaire de COGNO**. Le package
`cogno-scirust` et `SciRustBackend` sont des noms historiques conservés pour
compatibilité. L'alias `LocalDifferentiableBackend` expose cette identité sans
migration cassante. Le protocole `scirust-agent-protocol` réellement consommé
par le transport n'est pas un backend tensoriel.

Une ancienne évaluation de crates.io `scirust` 0.0.5 ne portait pas sur le
SciRust canonique **Memorithm/scirust** et ne prouve rien sur son état actuel.

## Protocole figé avant exécution

Le workflow `canonical scalar qualification` exécute séparément :

1. le `Tape` COGNO courant, avec 8 nœuds / 128 éléments maximum par tenseur ;
2. les **modules originaux** `scirust-core/src/matrix/{backend,view}.rs` du
   dépôt canonique au commit `68d544679df465d3b85539c80926ed2644fbbba1`.

Le SHA Git et les deux blobs sont vérifiés avant compilation. Ce petit binaire
compile les modules sans features avec `rustc`, pas le workspace complet ou un
backend automatique. Il appelle réellement `ScalarBackend::{sdot_f32,relu_f32,
sgemm_f32}` ; aucune réimplémentation locale ne tient lieu d'oracle canonique.

Le corpus commun ne contient que les entrées : 16 graines × longueurs
1, 2, 7, 8, 17, 33 × 3 opérations = **288 cas**. Les valeurs dyadiques finies
et bornées rendent ces cas exactement représentables. On compare tous les
bits f32 produits pour le dot, la ReLU et les matrices (2×k)@(k×3), alpha=1,
beta=0. Les gradients locaux du dot et de la ReLU sont aussi vérifiés contre
leurs dérivées analytiques ; ce n'est pas une comparaison d'autograd SciRust.

Commande dans un environnement de qualification isolé, après checkout
canonique exact dans `qualification/canonical` : `bash qualification/run.sh`.
Les deux programmes et la compilation ont des délais indépendants ; la CI
utilise un runner GitHub hébergé éphémère, jamais un runner interne privilégié.

## Preuves et acceptation

Les artefacts `canonical-scalar-<COGNO SHA>` conservent les identités COGNO
commit/tree, SciRust commit/blobs, toolchain/target, digest du corpus/harness,
les deux sorties effectivement produites et le diff. Un résultat différent,
une sortie manquante, un repin ou un dépassement échoue. Aucun résultat n'est
annoncé avant la fin du job. Les six gates COGNO habituels restent obligatoires
sur le HEAD exact ; la qualification doit être revérifiée sur `main` intégré.
Les preuves synthétiques n'activent ni Meta ni modèle, et ne contournent aucun
hard gate. Les holdouts/populations existants ne sont pas modifiés.

## Limites et prochaine convergence

Cette tranche qualifie uniquement les **primitives forward scalaires finies**
décrites ci-dessus. Elle ne qualifie ni l'autograd canonique complet, ni les
optimisateurs, InfoNCE, les modèles réels, WGPU/CUDA, SIMD, performance ou un
remplacement du runtime. La convergence reste conditionnelle à une couverture
des méthodes utilisées, gradients, erreurs et budgets avec résultats conservés
avant migration. Les refus COGNO (non-fini, forme, budget) ne doivent pas être
remplacés par des primitives permissives ou paniqueuses sans adaptateur borné.

Le code SciRust reste dans son checkout sous PolyForm Noncommercial 1.0.0,
Copyright 2026 Tarek Zekriti ; consulter sa [licence exacte](https://github.com/Memorithm/scirust/blob/68d544679df465d3b85539c80926ed2644fbbba1/LICENSE.md).
Ce workflow ne copie pas ces modules dans le code COGNO, n'en change pas les
notices, ne distribue pas leurs binaires et ne confère pas de droits commerciaux.
L'artefact publié est limité aux identités et sorties textuelles de qualification.
