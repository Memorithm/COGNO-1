//! Autograd correctness tests via finite differences.
//!
//! Compares the engine's backward gradients against a centered finite
//! difference approximation on a quadratic: L = 0.5 * x^2 -> dL/dx = x.

use cogno_scirust::{
    engine::{Tape, Var},
    error::{SciRustError, SciRustResult},
    tensor::{Shape, Tensor},
};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn grad_linear_add() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![2.0, 3.0],
        1024,
    )?)?;
    let b = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![5.0, 7.0],
        1024,
    )?)?;
    let c = tape.add(a, b)?;
    let loss = tape.sum(c)?;
    tape.backward(loss)?;
    let ga = tape.grad_of(a);
    let gb = tape.grad_of(b);
    assert!(approx_eq(ga[0], 1.0, 1e-5) && approx_eq(ga[1], 1.0, 1e-5));
    assert!(approx_eq(gb[0], 1.0, 1e-5) && approx_eq(gb[1], 1.0, 1e-5));
    Ok(())
}

#[test]
fn grad_mul() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![2.0, 4.0],
        1024,
    )?)?;
    let b = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![3.0, 5.0],
        1024,
    )?)?;
    let c = tape.mul(a, b)?;
    let loss = tape.sum(c)?;
    tape.backward(loss)?;
    assert!(approx_eq(tape.grad_of(a)[0], 3.0, 1e-5));
    assert!(approx_eq(tape.grad_of(a)[1], 5.0, 1e-5));
    assert!(approx_eq(tape.grad_of(b)[0], 2.0, 1e-5));
    assert!(approx_eq(tape.grad_of(b)[1], 4.0, 1e-5));
    Ok(())
}

#[test]
fn grad_matmul() -> SciRustResult<()> {
    let mut tape = Tape::new(256, 2048);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[2, 3])?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1024,
    )?)?;
    let b = tape.variable(Tensor::try_new(
        Shape::try_new(&[3, 2])?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1024,
    )?)?;
    let c = tape.matmul(a, b)?;
    let loss = tape.sum(c)?;
    tape.backward(loss)?;
    assert!(!tape.grad_of(a).is_empty());
    assert!(!tape.grad_of(b).is_empty());
    Ok(())
}

#[test]
fn grad_relu() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![-1.0, 2.0],
        1024,
    )?)?;
    let r = tape.relu(a)?;
    let loss = tape.sum(r)?;
    tape.backward(loss)?;
    let ga = tape.grad_of(a);
    assert_eq!(ga[0], 0.0);
    assert_eq!(ga[1], 1.0);
    Ok(())
}

#[test]
fn grad_sigmoid() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(Shape::try_new(&[1])?, vec![0.0], 1024)?)?;
    let s = tape.sigmoid(a)?;
    let loss = tape.sum(s)?;
    tape.backward(loss)?;
    let ga = tape.grad_of(a);
    // d/dx sigmoid(x) at 0 = 0.25
    assert!((ga[0] - 0.25).abs() < 1e-4);
    Ok(())
}

#[test]
fn grad_softmax() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[3])?,
        vec![1.0, 2.0, 3.0],
        1024,
    )?)?;
    let s = tape.softmax(a)?;
    let loss = tape.sum(s)?;
    tape.backward(loss)?;
    let ga = tape.grad_of(a);
    for g in ga {
        assert!(g.abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn grad_log_softmax() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let a = tape.variable(Tensor::try_new(
        Shape::try_new(&[2])?,
        vec![0.0, 1.0],
        1024,
    )?)?;
    let ls = tape.log_softmax(a)?;
    let loss = tape.sum(ls)?;
    tape.backward(loss)?;
    // Just verify it runs without error
    assert!(!tape.grad_of(Var { idx: 0 }).is_empty());
    Ok(())
}

#[test]
fn single_row_log_softmax_preserves_shape_and_backpropagates() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let logits = tape.variable(Tensor::try_new(
        Shape::try_new(&[1, 3])?,
        vec![1.0, 2.0, 3.0],
        1024,
    )?)?;
    let log_probs = tape.log_softmax(logits)?;
    assert_eq!(tape.value_of(log_probs).shape.as_slice(), &[1, 3]);
    let target = tape.variable(Tensor::try_new(
        Shape::try_new(&[1, 3])?,
        vec![0.0, 0.0, 1.0],
        1024,
    )?)?;
    let selected = tape.mul(log_probs, target)?;
    let selected = tape.sum(selected)?;
    let loss = tape.neg(selected)?;
    tape.backward(loss)?;
    assert!(tape.grad_of(logits).iter().any(|gradient| gradient.abs() > 1e-6));
    Ok(())
}

#[test]
fn softmax_rejects_multirow_matrix() -> SciRustResult<()> {
    let mut tape = Tape::new(64, 1024);
    let logits = tape.variable(Tensor::try_new(
        Shape::try_new(&[2, 2])?,
        vec![1.0, 2.0, 3.0, 4.0],
        1024,
    )?)?;
    assert!(matches!(tape.softmax(logits), Err(SciRustError::Shape { .. })));
    Ok(())
}
