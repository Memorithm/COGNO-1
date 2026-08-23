//! Tensors with autovectorization-friendly SoA layouts (COGNO-1 §SciRust #1).
//!
//! Tensors own their `f32` data in row-major order but expose elementwise ops
//! as flattened, contiguous slices so LLVM can auto-vectorize. The crate
//! targets stable Rust; explicit `std::simd` (`portable_simd`) is left for a
//! future nightly feature. All construction and ops are fallible: shape
//! mismatches return [`SciRustError::Shape`], overflows return
//! [`SciRustError::Overflow`], capacities return
//! [`SciRustError::CapacityExceeded`]. No `panic!`.

use crate::error::{SciRustError, SciRustResult};

/// Tensor shape. Stored as a `Vec<usize>` so shapes and ranks are dynamic; the
/// backing buffer length is always the product (checked).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Construct a shape from dims with checked product.
    pub fn try_new(dims: &[usize]) -> SciRustResult<Self> {
        if dims.is_empty() {
            return Err(SciRustError::Empty);
        }
        if dims.contains(&0) {
            return Err(SciRustError::Shape {
                lhs: dims.to_vec(),
                rhs: vec![0],
            });
        }
        let _ = Self::checked_len(dims)?;
        Ok(Self(dims.to_vec()))
    }

    /// Checked number of elements for a dim slice.
    pub fn checked_len(dims: &[usize]) -> SciRustResult<usize> {
        dims.iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or(SciRustError::Overflow)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Number of **dimensions** (rank), not the element count. For the
    /// element count use [`Shape::checked_len`] on [`Self::as_slice`]. Kept
    /// under this name for slice-like ergonomics; do not use it to size data
    /// buffers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.0.len() == 1 && self.0[0] == 1
    }
}

/// Owned tensor. Data is row-major contiguous; the shape constrains its
/// interpretation. Construction allocates only through `try_new`, bounded by
/// the caller-supplied max; ops reuse buffers where possible.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Construct from a shape and flat data. Validates the product matches
    /// the data length. `max_elements` caps hostile inputs (§12/§15 bounded).
    pub fn try_new(shape: Shape, data: Vec<f32>, max_elements: usize) -> SciRustResult<Self> {
        let expected = Shape::checked_len(shape.as_slice())?;
        if expected != data.len() {
            return Err(SciRustError::Shape {
                lhs: shape.as_slice().to_vec(),
                rhs: vec![data.len()],
            });
        }
        if data.len() > max_elements {
            return Err(SciRustError::CapacityExceeded {
                requested: data.len(),
                maximum: max_elements,
            });
        }
        Ok(Self { shape, data })
    }

    /// Scalar (rank-1, single element) tensor. Useful for zero-grad leaves.
    pub fn try_scalar(v: f32) -> SciRustResult<Self> {
        Ok(Self {
            shape: Shape::try_new(&[1])?,
            data: vec![v],
        })
    }

    /// Zeros tensor of a given shape, bounded by `max_elements`.
    pub fn try_zeros(shape: Shape, max_elements: usize) -> SciRustResult<Self> {
        // The element count is the checked *product* of the dims, not the
        // rank: allocating the rank silently produced shape/data inconsistency.
        let n = Shape::checked_len(shape.as_slice())?;
        if n > max_elements {
            return Err(SciRustError::CapacityExceeded {
                requested: n,
                maximum: max_elements,
            });
        }
        Ok(Self {
            shape,
            data: vec![0.0; n],
        })
    }

    /// Borrow the flat data.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable borrow of the flat data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Same shape as another tensor? Used by elementwise ops.
    #[must_use]
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.shape == other.shape
    }

    /// Element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
