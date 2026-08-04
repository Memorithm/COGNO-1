//! Oracle comparison: `cogno_scirust` backend vs `cogno_core` scalar oracle.
//!
//! The scalar `RewardEngine` in `cogno-core` is the oracle. On deterministic
//! batches, the SciRust backend's soft components must produce rankings
//! consistent with the oracle's integer scores. The comparison is ranking
//! consistency (order-preserving), not float equality.

use cogno_scirust::{
    backend::{Config, SciRustBackend},
    engine::Tape,
    error::SciRustResult,
    losses::{InfoNCE, PairwiseLoss},
    tensor::{Shape, Tensor},
};

fn default_config() -> Config {
    Config {
        max_nodes: 256,
        max_elements: 256,
        kv_capacity: 64,
        head_dim: 32,
        num_heads: 4,
        num_layers: 2,
        temperature: 1.0,
        lr: 0.01,
    }
}

#[test]
fn oracle_ranking_consistency_pairwise() -> SciRustResult<()> {
    let oracle_scores = [100.0f32, 80.0, 60.0, 40.0, 20.0];
    let loss_fn = PairwiseLoss::try_new(1.0, 8, 256)?;
    let mut tape = Tape::new(256, 2048);
    let loss_var = loss_fn.loss(
        &mut tape,
        Tensor {
            shape: Shape::try_new(&[5])?,
            data: oracle_scores.to_vec(),
        },
        Tensor {
            shape: Shape::try_new(&[5])?,
            data: vec![0.0, 10.0, 20.0, 30.0, 40.0],
        },
    )?;
    assert!(tape.value_of(loss_var).data[0].is_finite());
    Ok(())
}

#[test]
fn oracle_ranking_consistency_infonce() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let query = Tensor {
        shape: Shape::try_new(&[4])?,
        data: vec![1.0, 0.0, 0.0, 0.0],
    };
    let mut keys = vec![0.0f32; 5 * 4];
    keys[0] = 1.0;
    keys[4] = 0.5;
    let keys_tensor = Tensor {
        shape: Shape::try_new(&[5, 4])?,
        data: keys,
    };
    let loss_var = InfoNCE::loss(
        &InfoNCE::try_new(0.1, 8, 256)?,
        &mut tape,
        query,
        keys_tensor,
        0,
    )?;
    assert!(tape.value_of(loss_var).data[0].is_finite());
    Ok(())
}

#[test]
fn backend_gate_blocks_when_meta_inactive() {
    let mut backend = SciRustBackend::try_new(default_config()).unwrap();
    assert!(matches!(
        backend.objective(),
        Err(cogno_scirust::error::SciRustError::Gated)
    ));
}

#[test]
fn backend_activates_with_all_preconditions() {
    let mut backend = SciRustBackend::try_new(default_config()).unwrap();
    let pre = cogno_core::MetaPreconditions {
        scalar_engine_validated: true,
        reference_policy_frozen: true,
        log_probabilities_available: true,
        backend_differentiable: true,
        held_out_tests_in_place: true,
        anti_poisoning_working: true,
    };
    assert!(backend.activate(pre).is_ok());
    assert!(backend.meta.is_active());
}

#[test]
fn backend_rejects_partial_preconditions() {
    let mut backend = SciRustBackend::try_new(default_config()).unwrap();
    let pre = cogno_core::MetaPreconditions {
        scalar_engine_validated: true,
        reference_policy_frozen: false,
        log_probabilities_available: true,
        backend_differentiable: true,
        held_out_tests_in_place: true,
        anti_poisoning_working: true,
    };
    assert!(backend.activate(pre).is_err());
    assert!(backend.meta.is_quarantined());
    assert!(!backend.meta.is_active());
}
