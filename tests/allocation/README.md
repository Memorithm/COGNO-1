# `tests/allocation/` — tests mémoire obligatoires (§26)

Mesurer : allocations ; réallocations ; désallocations ; pic de mémoire ;
mémoire par requête ; mémoire du KV cache ; mémoire des poids ; mémoire du
tokenizer ; mémoire des buffers ; profondeur maximale des files ; nombre de
refus pour dépassement ; latence du contrôle d'admission.

## Tests d'allocation forcée en échec

| Test | Fichier prévu | Comportement attendu |
|------|---------------|----------------------|
| Allocation initiale impossible | `init_allocation_failed.rs` | `MemoryError::AllocationFailed` ; aucun état partiel |
| Croissance du buffer impossible | `buffer_grow_failed.rs` | `MemoryError::CapacityExceeded` ; buffer intact |
| Réservation KV impossible | `kv_reservation_failed.rs` | `MemoryError::BudgetExceeded` ; cache inchangé |
| Snapshot impossible | `snapshot_failed.rs` | `MemoryError::AllocationFailed` ; `events.log` intact |
| Queue saturée | `queue_full.rs` | `MemoryError::QueueFull` selon `QueueFullPolicy` ; aucune suppression silencieuse |

Principe directeur : **une erreur structurée et aucune modification partielle de
l'état**. Le régime stable ne doit effectuer **aucune** nouvelle allocation
(§13) ; ces tests valident également ce invariant sur les chemins critiques.