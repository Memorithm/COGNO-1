# AXCOG3 — provenance d'une autorisation par tâche

Cette étape lie chaque autorisation dry-run à un snapshot de workspace, à
l'empreinte du modèle installé, à la liaison de tâche et au résultat typé exact
de la proposition. Elle n'exécute aucun outil et ne confère aucune capacité
financière.

## Propriétaires

SciRust Hub conserve l'identité, l'admission et le cycle de vie des tâches.
RemoteOps reste le plan de contrôle distant : il matérialise et vérifie le
workspace, applique l'enveloppe d'exécution et collecte les preuves. COGNO-1
vérifie les listes et classes de capacités et lie l'autorisation aux
références fournies par l'hôte. La sortie du modèle demeure une donnée non
fiable.

## Entrées requises

`TaskCapabilityScope::new_with_provenance` reçoit :

- le binding opaque de tâche, fourni après admission ;
- le SHA-256 du manifeste de snapshot complet du workspace ;
- le SHA-256 de l'artefact de modèle effectivement installé ;
- les listes positives de tools et les classes de capacités admises.

Le manifeste de workspace doit couvrir tous les dépôts, leurs identifiants
d'objet immuables (dont les commits Git exacts), les entrées générées et toute
autre source nécessaire à la matérialisation. Le producteur fixe une
sérialisation versionnée avant d'en calculer le SHA-256. Un nom de branche, un
seul commit qui omet des sources, ou une arborescence locale modifiée ne
remplace pas cette preuve. RemoteOps vérifie que le workspace matérialisé est
propre et correspond au manifeste. COGNO-1 reçoit les digests et ne prétend
pas vérifier lui-même le système de fichiers, le dépôt ou le modèle.

Les valeurs nulles sont rejetées comme sentinelles. Cette validation contrôle
la présence et la forme ; elle n'authentifie pas l'émetteur. Un scope construit
avec `TaskCapabilityScope::new` ne possède pas de provenance et échoue donc
fermé pour toute autorisation.

## Digest du résultat

Le résultat désigne la proposition typée du modèle, pas un résultat d'outil.
Son SHA-256 est calculé sur un encodage déterministe versionné et séparé par le
domaine `cogno-1:tool-proposal-result:v1\0`. L'encodage lie le snapshot de
workspace, le digest du modèle, le binding de tâche, les identifiants du tool
et de la capacité, le code de justification, le nombre d'arguments et chaque
argument typé.

Les textes, chemins et octets utilisent des tags distincts et une longueur
`u64` little-endian avant leurs octets. Les entiers utilisent un tag distinct
et un `i64` little-endian. COGNO refuse les propositions qui dépassent 128
arguments ou 64 Kio de données d'arguments avant de scanner ou hacher leurs
contenus. Ainsi, des valeurs identiques dans deux variantes de type différentes
produisent des digests distincts. La liaison de tâche est
aussi enregistrée sous forme de digest ; le binding brut et les arguments ne
sont pas copiés dans l'audit.

## Audit et limites

L'enregistrement structuré conserve la classe de capacité, le snapshot du
workspace, l'artefact du modèle, le digest du binding de tâche et le digest du
résultat. L'audit reste borné et en mémoire dans ce runtime ; sa persistance et
son transport relèvent du plan de contrôle distant et doivent être traités
comme des preuves avant toute promotion.

`DryRunAuthorized` signifie uniquement que la politique aurait admis la
proposition. Le MVP garde les outils désactivés et ne lance aucun processus,
n'écrit aucun fichier, n'ouvre aucun accès réseau et ne réalise aucune
opération de trading. Une permission de classe `Effect` reste un contrôle de
politique, pas une exécution. Les résultats ambigus d'un effet réel et leur
réconciliation avant retry restent à traiter dans AXCOG4.
