//! Bounded local execution side of the canonical scalar qualification.
#![forbid(unsafe_code)]
#![deny(warnings)]

#[path = "../../../qualification/common.rs"]
mod common;

use cogno_scirust::{SciRustError, Shape, Tape, Tensor};

const ELEMENTS: usize = 128;

fn input(
    tape: &mut Tape,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<cogno_scirust::Var, SciRustError> {
    tape.variable(Tensor::try_new(Shape::try_new(shape)?, values, ELEMENTS)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for seed in 0..common::SEEDS {
        for len in common::LENGTHS {
            let (left, right) = common::fixture(len, seed);
            let mut tape = Tape::new(8, ELEMENTS);
            let a = input(&mut tape, &[len], left.clone())?;
            let b = input(&mut tape, &[len], right.clone())?;
            let product = tape.mul(a, b)?;
            let dot = tape.sum(product)?;
            common::emit("dot", seed, len, tape.value_of(dot).as_slice());
            // Analytical local gradient check; NOT a canonical autodiff comparison.
            tape.backward(dot)?;
            assert_eq!(tape.grad_of(a), right);
            assert_eq!(tape.grad_of(b), left);

            let mut tape = Tape::new(8, ELEMENTS);
            let a = input(&mut tape, &[len], left.clone())?;
            let relu = tape.relu(a)?;
            common::emit("relu", seed, len, tape.value_of(relu).as_slice());
            let loss = tape.sum(relu)?;
            tape.backward(loss)?;
            for (&grad, &value) in tape.grad_of(a).iter().zip(&left) {
                assert_eq!(grad, if value > 0.0 { 1.0 } else { 0.0 });
            }

            let (left, _) = common::fixture(2 * len, seed);
            let (_, right) = common::fixture(len * 3, seed);
            let mut tape = Tape::new(4, ELEMENTS);
            let a = input(&mut tape, &[2, len], left)?;
            let b = input(&mut tape, &[len, 3], right)?;
            let matrix = tape.matmul(a, b)?;
            common::emit("matmul", seed, len, tape.value_of(matrix).as_slice());
        }
    }
    Ok(())
}
