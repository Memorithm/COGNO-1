# AXCOG2 — classes de capacités par tâche

Cette étape distingue trois classes de capacité : **Read**, **Reason** et
**Effect**. Elle ne donne pas d’autorité au modèle et ne met aucun outil en
exécution.

## Propriétaires et données de confiance

SciRust Hub reste responsable de l’identité, de l’admission et du cycle de vie
d’une tâche. RemoteOps applique les contrôles du runtime distant. COGNO-1
vérifie la politique produit déterministe. Le champ de liaison de tâche reste
opaque dans COGNO-1 : sa validation de forme ne l’authentifie pas.

Le registre de classes du runtime et les classes du scope de tâche sont fournis
par l’hôte. Le modèle ne peut produire qu’un `CapabilityId`; il ne peut pas
indiquer que cette capacité est `Read`, `Reason` ou `Effect`.

## Décision

Une proposition task-scoped est admise uniquement si :

1. le binding et le scope ont une forme valide, sans doublons ;
2. le tool figure dans la liste positive du runtime **et** dans celle de la tâche ;
3. le runtime et la tâche attribuent la même classe à l’identifiant de capacité ;
4. les contrôles déterministes existants, dont le rejet des formes shell, passent.

Une classe absente, dupliquée ou différente entre runtime et tâche est refusée.
Le chemin `ToolExecutor::execute` sans binding refuse toute proposition,
même si les anciennes allowlists correspondent. Seul `execute_for_task` peut
produire une autorisation dry-run, après intersection des classes et des listes.

## Limite d’exécution

Le résultat `DryRunAuthorized` signifie seulement que la politique classée
permettrait la proposition. Le MVP garde `MVP_TOOLS_ENABLED = false` et ne
lance aucun processus, ne lit/écrit aucun fichier et n’ouvre aucun accès réseau.
La classe `Effect` n’autorise donc encore aucun effet réel. Tout runtime concret
et toute garantie de sandbox restent des étapes ultérieures, sous le contrat
RemoteOps. Aucune capacité de trading ni aucune décision financière n’est
qualifiée par cette étape.

## Exemple

    use cogno_core::{CapabilityClass, CapabilityClassification, CapabilityId};
    static RUNTIME_CAPS: &[CapabilityClassification] = &[
        CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
        CapabilityClassification::new(CapabilityId(2), CapabilityClass::Effect),
    ];
    static TASK_CAPS: &[CapabilityClassification] = &[
        CapabilityClassification::new(CapabilityId(1), CapabilityClass::Read),
    ];

La tâche ne reçoit pas la capacité `Effect`, même si le registre global la
connaît. Les identifiants et classes restent des déclarations hôte ; cette
étape n’authentifie pas leur source et ne transforme pas RemoteOps en autorité
sur la sémantique produit.
