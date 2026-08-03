# `tests/allocation/` — tests mémoire obligatoires (§26)

Les tests mémoire d'allocation forcée en échec et de bornage vivent dans
`crates/cogno-runtime/tests/runtime_integration.rs` (admission/KV/queue) et
`crates/cogno-core/tests/adversarial.rs` (budget/kv overflow/bounded). Le
comportement attendu est une **erreur structurée** et **aucune modification
partielle** de l'état (§13).

## Tests exécutables

| Test | Fichier | Métrique/échec couvert |
|------|---------|-------------------------|
| `runtime_constructs_with_valid_mvp_budget` | `cogno-runtime/tests/runtime_integration.rs` | budget valide, état initial cohérent |
| `admission_admits_within_budget` | idem | `RequestEstimate.total_bytes ≤ hard_limit` |
| `admission_rejects_context_too_large` | idem | `MemoryError::ContextTooLarge` |
| `admission_rejects_oversized_input` | idem | `MemoryError::CapacityExceeded` |
| `admission_rejects_budget_exceeded` | idem | `MemoryError::BudgetExceeded` |
| `kv_reject_on_overflow_admits_within_capacity` | idem | KV strict |
| `kv_sliding_window_truncates_not_silently` | idem | troncature non silencieuse (§17) |
| `kv_prefix_pinned_admits_prefix_plus_window` | idem | prefix+window |
| `queue_rejects_newest_when_full` | idem | `QueueFull` + rejections++, queue intacte |
| `queue_reject_oldest_drops_oldest_not_silently` | idem | drop journalisé (non silencieux) |
| `queue_block_with_deadline_refuses_synchronously` | idem | `DeadlineExceeded` |
| `queue_zero_capacity_rejected` | idem | configuration nulle refusée |
| `kv_cache_arithmetic_overflow_detected` | `cogno-core/tests/adversarial.rs` | `MemoryError::ArithmeticOverflow` |
| `kv_cache_ok_for_small_dims` | idem | formule KV `2×layers×tokens×kv_heads×head_dim×bytes×batch` |
| `budget_subbudget_sum_exceeds_hard_rejected` | idem | somme > hard |
| `budget_mandatory_zero_rejected` | idem | limite nulle obligatoire |
| `valid_budget` | idem | budget sain accepté |
| `bounded_vec_refuses_overflow` | idem | `BoundedVec` pas de grow au-delà du plafond |
| `bounded_vec_zero_capacity_rejected` | idem | 0 interdit |

## Régime stable (§13)

Les chemins critiques notés « aucun `Vec::new` / `format!` / `Box::new` /
`to_string` » (§14) ne sont pas exécutés sur le hot path : le pipeline
`cogno_runtime::Pipeline::run` n'alloue pas entre la classification de confiance
et la décision. Les buffers scratch (`RequestScratch`) sont pré-alloués par le
runtime et passés par emprunt.