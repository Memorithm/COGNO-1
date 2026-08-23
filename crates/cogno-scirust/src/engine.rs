//! Reverse-mode autograd on a Wengert tape (COGNO-1 §SciRust #1).
//!
//! Design choices honoring the COGNO spec:
//!
//! - **No `RefCell`**: a `RefCell::borrow_mut` collision would panic, which is
//!   forbidden here. The tape owns all nodes by value; `Var` is just an index
//!   into the tape. Backward sweeps the tape in reverse, accumulating into
//!   per-node gradient buffers indexed by node id. This is panic-free.
//! - **Bounded**: the tape length and per-node tensor size are capped by
//!   `max_nodes` and `max_elements`, checked at construction time. Hostile
//!   inputs cannot grow the tape without bound (§11/§15).
//! - **Fallible**: every op returns `SciRustResult<Var>`; shape mismatches and
//!   overflows surface as structured errors instead of `panic!`.
//! - **No `unsafe`**: pure safe Rust, `#![forbid(unsafe_code)]` enforced at the
//!   crate root.

use crate::error::{ensure_finite, SciRustError, SciRustResult};
use crate::tensor::{Shape, Tensor};

/// Operation kind recorded on the tape. Data-driven so the backward sweep
/// dispatches on this enum, avoiding `Box<dyn Fn>` and any closure-borrow
/// panic surface.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Input,
    Add,
    Sub,
    Mul, // elementwise
    MatMul,
    Sum,
    Stack,
    Scale(f32),
    Neg,
    Relu,
    Sigmoid,
    Softmax,
    LogSoftmax,
}

/// One tape node: op kind, consumer inputs, cached forward value, and
/// (during backward) accumulated gradient.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub value: Tensor,
    pub grad: Vec<f32>,
}

/// The Wengert tape. Owns nodes by value; `Var` is an index.
#[derive(Debug)]
pub struct Tape {
    pub nodes: Vec<Node>,
    pub max_nodes: usize,
    pub max_elements: usize,
}

impl Tape {
    /// Construct a tape with explicit bounds. `max_nodes` caps the tape
    /// length; `max_elements` caps the per-node tensor element count.
    #[must_use]
    pub fn new(max_nodes: usize, max_elements: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(max_nodes.min(4096)),
            max_nodes,
            max_elements,
        }
    }

    fn push(&mut self, node: Node) -> SciRustResult<Var> {
        if self.nodes.len() >= self.max_nodes {
            return Err(SciRustError::CapacityExceeded {
                requested: self.nodes.len() + 1,
                maximum: self.max_nodes,
            });
        }
        if node.value.len() > self.max_elements {
            return Err(SciRustError::CapacityExceeded {
                requested: node.value.len(),
                maximum: self.max_elements,
            });
        }
        let idx = self.nodes.len();
        self.nodes.push(node);
        Ok(Var { idx })
    }

    /// Add a leaf input/parameter tensor. Its gradient is accumulated into
    /// `nodes[idx].grad` during backward.
    pub fn variable(&mut self, value: Tensor) -> SciRustResult<Var> {
        let n = value.len();
        self.push(Node {
            op: Op::Input,
            inputs: vec![],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Two-operand elementwise op (broadcasting not supported — the spec
    /// requires bounded shapes, so same-shape is enforced).
    fn binop(&mut self, a: Var, b: Var, op: Op) -> SciRustResult<Var> {
        let na = &self.nodes[a.idx].value;
        let nb = &self.nodes[b.idx].value;
        if !na.same_shape(nb) {
            return Err(SciRustError::Shape {
                lhs: na.shape.as_slice().to_vec(),
                rhs: nb.shape.as_slice().to_vec(),
            });
        }
        let data: Vec<f32> = match op {
            Op::Add => na
                .data
                .iter()
                .zip(nb.data.iter())
                .map(|(x, y)| x + y)
                .collect(),
            Op::Sub => na
                .data
                .iter()
                .zip(nb.data.iter())
                .map(|(x, y)| x - y)
                .collect(),
            Op::Mul => na
                .data
                .iter()
                .zip(nb.data.iter())
                .map(|(x, y)| x * y)
                .collect(),
            _ => unreachable_binop(),
        };
        for &v in &data {
            ensure_finite(v)?;
        }
        let n = data.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data,
        };
        self.push(Node {
            op,
            inputs: vec![a.idx, b.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// `a + b`.
    pub fn add(&mut self, a: Var, b: Var) -> SciRustResult<Var> {
        self.binop(a, b, Op::Add)
    }

    /// `a - b`.
    pub fn sub(&mut self, a: Var, b: Var) -> SciRustResult<Var> {
        self.binop(a, b, Op::Sub)
    }

    /// `a * b` (elementwise; same shape required).
    pub fn mul(&mut self, a: Var, b: Var) -> SciRustResult<Var> {
        self.binop(a, b, Op::Mul)
    }

    /// Matrix multiplication `a @ b` for rank-2 tensors. Checked shapes.
    pub fn matmul(&mut self, a: Var, b: Var) -> SciRustResult<Var> {
        let na = &self.nodes[a.idx].value;
        let nb = &self.nodes[b.idx].value;
        let sa = na.shape.as_slice();
        let sb = nb.shape.as_slice();
        if sa.len() != 2 || sb.len() != 2 {
            return Err(SciRustError::Shape {
                lhs: sa.to_vec(),
                rhs: sb.to_vec(),
            });
        }
        if sa[1] != sb[0] {
            return Err(SciRustError::Shape {
                lhs: sa.to_vec(),
                rhs: sb.to_vec(),
            });
        }
        let (m, k) = (sa[0], sa[1]);
        let (_k2, n) = (sb[0], sb[1]);
        let out_len = m.checked_mul(n).ok_or(SciRustError::Overflow)?;
        if out_len > self.max_elements {
            return Err(SciRustError::CapacityExceeded {
                requested: out_len,
                maximum: self.max_elements,
            });
        }
        let mut data = vec![0.0; out_len];
        // Standard triple loop; LLVM auto-vectorizes the inner k-loop.
        for i in 0..m {
            let row_a = &na.data[i * k..(i + 1) * k];
            for j in 0..n {
                let mut acc = 0.0f32;
                #[allow(clippy::needless_range_loop)]
                for p in 0..k {
                    acc += row_a[p] * nb.data[p * n + j];
                }
                data[i * n + j] = acc;
            }
        }
        for &v in &data {
            ensure_finite(v)?;
        }
        let value = Tensor {
            shape: Shape::try_new(&[m, n])?,
            data,
        };
        self.push(Node {
            op: Op::MatMul,
            inputs: vec![a.idx, b.idx],
            value,
            grad: vec![0.0; out_len],
        })
    }

    /// Elementwise scale by a constant.
    pub fn scale(&mut self, a: Var, k: f32) -> SciRustResult<Var> {
        let na = self.nodes[a.idx].value.clone();
        let data: Vec<f32> = na.data.iter().map(|x| x * k).collect();
        for &v in &data {
            ensure_finite(v)?;
        }
        let n = data.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data,
        };
        self.push(Node {
            op: Op::Scale(k),
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Negation.
    pub fn neg(&mut self, a: Var) -> SciRustResult<Var> {
        let na = self.nodes[a.idx].value.clone();
        // Surface non-finite inputs like every other op: `neg` would silently
        // propagate NaN/Inf into downstream gradients.
        for &v in &na.data {
            ensure_finite(v)?;
        }
        let data: Vec<f32> = na.data.iter().map(|x| -x).collect();
        let n = data.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data,
        };
        self.push(Node {
            op: Op::Neg,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// ReLU (elementwise).
    pub fn relu(&mut self, a: Var) -> SciRustResult<Var> {
        let na = self.nodes[a.idx].value.clone();
        // `f32::max` swallows NaN (NaN.max(0.0) == 0.0), so a NaN gradient
        // would silently vanish here instead of surfacing as `NonFinite`.
        // Check the input explicitly, like every other op.
        for &v in &na.data {
            ensure_finite(v)?;
        }
        let data: Vec<f32> = na.data.iter().map(|x| x.max(0.0)).collect();
        let n = data.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data,
        };
        self.push(Node {
            op: Op::Relu,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Sigmoid (elementwise), numerically stable.
    pub fn sigmoid(&mut self, a: Var) -> SciRustResult<Var> {
        let na = self.nodes[a.idx].value.clone();
        let data: Vec<f32> = na
            .data
            .iter()
            .map(|&x| {
                if x >= 0.0 {
                    let z = (-x).exp();
                    1.0 / (1.0 + z)
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                }
            })
            .collect();
        for &v in &data {
            ensure_finite(v)?;
        }
        let n = data.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data,
        };
        self.push(Node {
            op: Op::Sigmoid,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Sum to a scalar (rank-1, length 1).
    pub fn sum(&mut self, a: Var) -> SciRustResult<Var> {
        let na = &self.nodes[a.idx].value;
        let mut acc = 0.0f32;
        for &v in &na.data {
            acc += v;
        }
        ensure_finite(acc)?;
        let value = Tensor {
            shape: Shape::try_new(&[1])?,
            data: vec![acc],
        };
        self.push(Node {
            op: Op::Sum,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; 1],
        })
    }

    /// Stack scalar Vars into one connected rank-1 vector.
    ///
    /// This is intentionally narrow: every input must be a scalar tensor and
    /// the output length is bounded by `max_elements`. The backward rule sends
    /// each output gradient element directly to the corresponding scalar input.
    pub fn stack_scalars(&mut self, values: &[Var]) -> SciRustResult<Var> {
        if values.is_empty() {
            return Err(SciRustError::Empty);
        }
        if values.len() > self.max_elements {
            return Err(SciRustError::CapacityExceeded {
                requested: values.len(),
                maximum: self.max_elements,
            });
        }
        let mut data = Vec::with_capacity(values.len());
        let mut inputs = Vec::with_capacity(values.len());
        for value in values {
            let tensor = &self.nodes[value.idx].value;
            if !tensor.shape.is_scalar() {
                return Err(SciRustError::Shape {
                    lhs: tensor.shape.as_slice().to_vec(),
                    rhs: vec![1],
                });
            }
            let scalar = tensor.data[0];
            ensure_finite(scalar)?;
            data.push(scalar);
            inputs.push(value.idx);
        }
        let n = data.len();
        self.push(Node {
            op: Op::Stack,
            inputs,
            value: Tensor {
                shape: Shape::try_new(&[n])?,
                data,
            },
            grad: vec![0.0; n],
        })
    }

    /// Softmax over a vector represented as `[N]` or a single row `[1, N]`.
    /// Numerically stable via max-subtraction. General matrices remain
    /// unsupported so normalization semantics cannot silently widen.
    pub fn softmax(&mut self, a: Var) -> SciRustResult<Var> {
        let na = &self.nodes[a.idx].value;
        if !is_vector_or_single_row(&na.shape) {
            return Err(SciRustError::Shape {
                lhs: na.shape.as_slice().to_vec(),
                rhs: vec![1],
            });
        }
        let max = na.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = na.data.iter().map(|x| (x - max).exp()).collect();
        let s: f32 = exps.iter().copied().sum();
        if s == 0.0 || !s.is_finite() {
            return Err(SciRustError::NonFinite);
        }
        for v in &mut exps {
            *v /= s;
            ensure_finite(*v)?;
        }
        let n = exps.len();
        let value = Tensor {
            shape: na.shape.clone(),
            data: exps,
        };
        self.push(Node {
            op: Op::Softmax,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Log-softmax over a vector represented as `[N]` or `[1, N]`.
    pub fn log_softmax(&mut self, a: Var) -> SciRustResult<Var> {
        let sm = self.softmax(a)?;
        let sm_val = self.nodes[sm.idx].value.clone();
        let data: Vec<f32> = sm_val.data.iter().map(|x| x.ln()).collect();
        for &v in &data {
            ensure_finite(v)?;
        }
        let n = data.len();
        // Replace softmax node's role: we record a new node referencing a
        // (log of softmax). For backward correctness we keep the softmax
        // node's data accessible via the inputs chain: the LogSoftmax
        // backward rule uses the softmax probabilities, recomputed from input.
        // To keep the tape data-driven, we store the original input index.
        let value = Tensor {
            shape: sm_val.shape.clone(),
            data,
        };
        self.push(Node {
            op: Op::LogSoftmax,
            inputs: vec![a.idx],
            value,
            grad: vec![0.0; n],
        })
    }

    /// Backward sweep from the tape's output node (scalar). Accumulates
    /// gradients into `nodes[i].grad`. Returns the gradient slice for a
    /// requested leaf (for optimizer consumption).
    pub fn backward(&mut self, loss_node: Var) -> SciRustResult<()> {
        if !self.nodes[loss_node.idx].value.shape.is_scalar() {
            return Err(SciRustError::Shape {
                lhs: self.nodes[loss_node.idx].value.shape.as_slice().to_vec(),
                rhs: vec![1],
            });
        }
        // Seed: dL/dL = 1.
        self.nodes[loss_node.idx].grad[0] = 1.0;
        // Reverse iteration; children always appear before their consumers in
        // a Wengert tape, so a single reverse pass topologically propagates.
        for i in (0..=loss_node.idx).rev() {
            let g = self.nodes[i].grad[0]; // only the scalar-output nodes are
                                           // summed to 1; intermediate grads
                                           // are elementwise arrays below.
            let _ = g;
            let node = &mut self.nodes[i];
            let op = node.op.clone();
            let inputs = node.inputs.clone();
            let out_val = node.value.clone();
            let out_grad = node.grad.clone();
            let _ = node;
            match op {
                Op::Input => {}
                Op::Add => {
                    let [a, b] = [inputs[0], inputs[1]];
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        self.nodes[a].grad[k] += ogk;
                        self.nodes[b].grad[k] += ogk;
                    }
                }
                Op::Sub => {
                    let [a, b] = [inputs[0], inputs[1]];
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        self.nodes[a].grad[k] += ogk;
                        self.nodes[b].grad[k] -= ogk;
                    }
                }
                Op::Mul => {
                    let [a, b] = [inputs[0], inputs[1]];
                    let av = self.nodes[a].value.clone();
                    let bv = self.nodes[b].value.clone();
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        self.nodes[a].grad[k] += ogk * bv.data[k];
                        self.nodes[b].grad[k] += ogk * av.data[k];
                    }
                }
                Op::MatMul => {
                    // out = a @ b, shape [m,n]; a:[m,k], b:[k,n]
                    let [a, b] = [inputs[0], inputs[1]];
                    let sa = self.nodes[a].value.shape.as_slice().to_vec();
                    let (m, k) = (sa[0], sa[1]);
                    let sb = self.nodes[b].value.shape.as_slice().to_vec();
                    let (_k2, n) = (sb[0], sb[1]);
                    // grad_a = out_grad @ b^T  -> [m,k]
                    let bv = self.nodes[b].value.clone();
                    let og = out_grad.clone();
                    for i in 0..m {
                        for p in 0..k {
                            let mut acc = 0.0f32;
                            for j in 0..n {
                                acc += og[i * n + j] * bv.data[p * n + j];
                            }
                            self.nodes[a].grad[i * k + p] += acc;
                        }
                    }
                    // grad_b = a^T @ out_grad  -> [k,n]
                    let av = self.nodes[a].value.clone();
                    for p in 0..k {
                        for j in 0..n {
                            let mut acc = 0.0f32;
                            for i in 0..m {
                                acc += av.data[i * k + p] * og[i * n + j];
                            }
                            self.nodes[b].grad[p * n + j] += acc;
                        }
                    }
                    let _ = out_val;
                }
                Op::Scale(c) => {
                    let a = inputs[0];
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        self.nodes[a].grad[k] += ogk * c;
                    }
                }
                Op::Neg => {
                    let a = inputs[0];
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        self.nodes[a].grad[k] -= ogk;
                    }
                }
                Op::Relu => {
                    let a = inputs[0];
                    let av = self.nodes[a].value.clone();
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        if av.data[k] > 0.0 {
                            self.nodes[a].grad[k] += ogk;
                        }
                    }
                }
                Op::Sigmoid => {
                    let a = inputs[0];
                    let ov = out_val.clone();
                    for (k, &ogk) in out_grad.iter().enumerate() {
                        let s = ov.data[k];
                        self.nodes[a].grad[k] += ogk * s * (1.0 - s);
                    }
                }
                Op::Sum => {
                    let a = inputs[0];
                    let g0 = out_grad[0];
                    for g in self.nodes[a].grad.iter_mut() {
                        *g += g0;
                    }
                }
                Op::Stack => {
                    for (index, input) in inputs.iter().copied().enumerate() {
                        self.nodes[input].grad[0] += out_grad[index];
                    }
                }
                Op::Softmax => {
                    let a = inputs[0];
                    let sv = out_val.clone();
                    let og = out_grad.clone();
                    let n = sv.data.len();
                    // dL/dx_i = s_i * (dL/dy_i - sum_j dL/dy_j * s_j)
                    let dot: f32 = og.iter().zip(sv.data.iter()).map(|(g, s)| g * s).sum();
                    for (k, &ogk) in og.iter().enumerate().take(n) {
                        let s = sv.data[k];
                        self.nodes[a].grad[k] += s * (ogk - dot);
                    }
                }
                Op::LogSoftmax => {
                    let a = inputs[0];
                    let av = self.nodes[a].value.clone();
                    let og = out_grad.clone();
                    let n = av.data.len();
                    // Recompute softmax for the original input.
                    let max = av.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps: Vec<f32> = av.data.iter().map(|x| (x - max).exp()).collect();
                    let s: f32 = exps.iter().copied().sum();
                    if s == 0.0 || !s.is_finite() {
                        return Err(SciRustError::NonFinite);
                    }
                    for v in &mut exps {
                        *v /= s;
                    }
                    let dot: f32 = og.iter().sum();
                    for k in 0..n {
                        self.nodes[a].grad[k] += og[k] - exps[k] * dot;
                    }
                }
            }
        }
        Ok(())
    }

    /// Borrow the gradient slice for a leaf variable (for the optimizer).
    #[must_use]
    pub fn grad_of(&self, v: Var) -> &[f32] {
        &self.nodes[v.idx].grad
    }

    /// Borrow the value of a variable.
    #[must_use]
    pub fn value_of(&self, v: Var) -> &Tensor {
        &self.nodes[v.idx].value
    }
}

/// A handle to a tape node. Cheap to copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var {
    pub idx: usize,
}

fn is_vector_or_single_row(shape: &Shape) -> bool {
    let dims = shape.as_slice();
    dims.len() == 1 || (dims.len() == 2 && dims[0] == 1)
}

#[inline(always)]
fn unreachable_binop() -> ! {
    // Internal invariant: binop is only called with Add/Sub/Mul. Reachable
    // only via a programming error, not via hostile input.
    unreachable!("binop called with non-elementwise op")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_scalars_preserves_exact_backward_connections() {
        let mut tape = Tape::new(8, 8);
        let left = tape
            .variable(Tensor::try_scalar(2.0).expect("left"))
            .expect("left var");
        let right = tape
            .variable(Tensor::try_scalar(-3.0).expect("right"))
            .expect("right var");
        let stacked = tape.stack_scalars(&[left, right]).expect("stack");
        assert_eq!(tape.value_of(stacked).as_slice(), &[2.0, -3.0]);
        let scaled = tape.scale(stacked, 2.0).expect("scale");
        let loss = tape.sum(scaled).expect("sum");
        tape.backward(loss).expect("backward");
        assert_eq!(tape.grad_of(left), &[2.0]);
        assert_eq!(tape.grad_of(right), &[2.0]);
    }

    #[test]
    fn stack_scalars_rejects_empty_and_non_scalar_inputs() {
        let mut tape = Tape::new(8, 8);
        assert!(matches!(tape.stack_scalars(&[]), Err(SciRustError::Empty)));
        let vector = tape
            .variable(
                Tensor::try_new(Shape::try_new(&[2]).expect("shape"), vec![1.0, 2.0], 8)
                    .expect("vector"),
            )
            .expect("vector var");
        assert!(matches!(
            tape.stack_scalars(&[vector]),
            Err(SciRustError::Shape { .. })
        ));
    }
}
