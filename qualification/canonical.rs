//! Execute the original SciRust ScalarBackend, not a copied mock/oracle.
// Dead code is expected when compiling two original modules without their
// whole workspace. No BLAS, rayon or portable-simd feature is enabled.
#![allow(dead_code)]

mod common;
mod matrix;

use matrix::backend::{ScalarBackend, SimdBackend};
use matrix::view::{MatrixView, MatrixViewMut};

fn main() {
    for seed in 0..common::SEEDS {
        for len in common::LENGTHS {
            let (mut left, right) = common::fixture(len, seed);
            common::emit("dot", seed, len, &[ScalarBackend.sdot_f32(&left, &right)]);
            ScalarBackend.relu_f32(&mut left);
            common::emit("relu", seed, len, &left);

            let (left, _) = common::fixture(2 * len, seed);
            let (_, right) = common::fixture(len * 3, seed);
            let mut output = vec![0.0; 6];
            ScalarBackend.sgemm_f32(
                1.0,
                MatrixView::from_slice(&left, 2, len),
                MatrixView::from_slice(&right, len, 3),
                0.0,
                MatrixViewMut::from_slice(&mut output, 2, 3),
            );
            common::emit("matmul", seed, len, &output);
        }
    }
}
