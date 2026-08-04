//! Differentiable losses (COGNO-1 §SciRust #2/#3/#4): pairwise ranking over
//! accepted/rejected/edited outputs, differentiable symbolic satisfaction,
//! and a bounded InfoNCE objective.
//!
//! All losses operate on the [`crate::engine::Tape`] and return scalar `Var`s
//! (rank-1, length 1). They are **soft** components: the §8 lexicographic
//! hard/capability/privacy gates still apply *before* the reward engine
//! consults any loss output, so a loss can never compensate a hard violation
//! (S4).

use crate::engine::{Tape, Var};
use crate::error::SciRustResult;
use crate::tensor::{Shape, Tensor};

/// Pairwise (contrastive) margin-ranking loss over the COGNO feedback
/// taxonomy (accepted / edited / rejected).
///
/// Given a score `s(p)` per candidate, the loss for a contrastive pair
/// `(preferred, dispreferred)` is `max(0, margin - (s_pref - s_disp))`.
/// Pairs are constructed from the §9 taxonomy:
///   - accepted  > rejected   (preferred = accepted,   dispreferred = rejected)
///   - edited    > rejected   (preferred = edited,     dispreferred = rejected)
///   - edited    < accepted   (preferred = accepted,   dispreferred = edited)
///     (an edit signals the accepted output was not quite right, so it ranks
///     above the rejected basin but the *accepted* version is the strongest
///     signal.)
#[derive(Clone, Debug)]
pub struct PairwiseLoss {
    pub margin: f32,
    pub max_pairs: usize,
    pub max_elements: usize,
}

impl PairwiseLoss {
    /// Construct with a margin, a max number of pairs (§15 bounded), and a
    /// per-tensor element cap.
    pub fn try_new(margin: f32, max_pairs: usize, max_elements: usize) -> SciRustResult<Self> {
        if max_pairs == 0 || max_elements == 0 {
            return Err(crate::error::SciRustError::Empty);
        }
        Ok(Self {
            margin,
            max_pairs,
            max_elements,
        })
    }

    /// Reduce a per-candidate scalar score tensor to a single number via
    /// sum (deterministic, no weighting).
    fn score(&self, tape: &mut Tape, v: Var) -> SciRustResult<Var> {
        tape.sum(v)
    }

    /// Compute the contrastive margin loss for a batch of pairs.
    /// `preferred_scores` and `dispreferred_scores` are 1-D tensors of equal
    /// length (one scalar per candidate per side). Bounded by `max_pairs`.
    pub fn loss(
        &self,
        tape: &mut Tape,
        preferred: Tensor,
        dispreferred: Tensor,
    ) -> SciRustResult<Var> {
        if preferred.data.len() != dispreferred.data.len() {
            return Err(crate::error::SciRustError::Shape {
                lhs: preferred.shape.as_slice().to_vec(),
                rhs: dispreferred.shape.as_slice().to_vec(),
            });
        }
        if preferred.data.len() > self.max_pairs {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: preferred.data.len(),
                maximum: self.max_pairs,
            });
        }
        let p = tape.variable(preferred)?;
        let d = tape.variable(dispreferred)?;
        // s_pref - s_disp, elementwise.
        let diff = tape.sub(p, d)?;
        // margin - diff, elementwise.
        let m_tensor = Tensor {
            shape: diff_shape_like(tape, diff),
            data: vec![self.margin; tape.value_of(diff).len()],
        };
        let m_var = tape.variable(m_tensor)?;
        let margin_minus = tape.sub(m_var, diff)?;
        // relu(margin - diff)
        let relu = tape.relu(margin_minus)?;
        // sum -> scalar loss
        self.score(tape, relu)
    }
}

fn diff_shape_like(tape: &Tape, v: Var) -> Shape {
    tape.value_of(v).shape.clone()
}

/// Differentiable symbolic satisfaction loss.
///
/// For each hard rule, the symbolic layer produces a satisfaction `s_i` in
/// `[0, 1]` (1 = satisfied). The differentiable surrogate aggregates them via
/// soft-conjunction (product) and the loss is `1 - Π s_i` so minimizing it
/// drives all rules toward satisfaction. This is **soft** and never replaces
/// the hard gate (§8): the lexicographic decider still rejects any hard
/// violation; this loss only shapes the neuro-symbolic score.
#[derive(Clone, Debug)]
pub struct SymbolicSatisfaction {
    pub max_rules: usize,
    pub max_elements: usize,
}

impl SymbolicSatisfaction {
    pub fn try_new(max_rules: usize, max_elements: usize) -> SciRustResult<Self> {
        if max_rules == 0 || max_elements == 0 {
            return Err(crate::error::SciRustError::Empty);
        }
        Ok(Self {
            max_rules,
            max_elements,
        })
    }

    /// Compute the satisfaction loss for a vector of `s_i` in `[0,1]`.
    /// Returns a scalar `Var` = `1 - Π s_i`.
    pub fn loss(&self, tape: &mut Tape, satisfactions: Tensor) -> SciRustResult<Var> {
        if satisfactions.data.len() > self.max_rules {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: satisfactions.data.len(),
                maximum: self.max_rules,
            });
        }
        if satisfactions
            .data
            .iter()
            .any(|&s| !(0.0..=1.0).contains(&s))
        {
            return Err(crate::error::SciRustError::Shape {
                lhs: satisfactions.shape.as_slice().to_vec(),
                rhs: vec![satisfactions.data.len()],
            });
        }
        // Compute the soft-conjunction product in plain Rust (deterministic) —
        // the satisfaction vector is data, not a differentiable input here
        // (the symbolic layer is the authority; this loss only shapes the
        // neuro-symbolic score). We expose the product as a tape leaf so a
        // downstream combiner can chain ops if needed.
        let product: f32 = satisfactions.data.iter().copied().product();
        let one = Tensor::try_scalar(1.0)?;
        let one_var = tape.variable(one)?;
        let prod_tensor = Tensor::try_scalar(product)?;
        let prod_var = tape.variable(prod_tensor)?;
        // loss = 1 - product
        let loss = tape.sub(one_var, prod_var)?;
        Ok(loss)
    }
}

/// Bounded InfoNCE objective for memory/rule selection.
///
/// `L = -log( exp(sim(q, k+) / τ) / Σ_i exp(sim(q, k_i) / τ) )`
///
/// Candidates are bounded by `max_candidates` (§11/§15). The denominator is
/// computed in log-space (log-sum-exp) for numerical stability. This is a
/// *selection* objective: it shapes which memory/rule embeddings are pulled
/// for a query; the hard selection gate still runs in `cogno-core` before any
/// rule is adopted.
#[derive(Clone, Debug)]
pub struct InfoNCE {
    pub temperature: f32,
    pub max_candidates: usize,
    pub max_elements: usize,
}

impl InfoNCE {
    pub fn try_new(
        temperature: f32,
        max_candidates: usize,
        max_elements: usize,
    ) -> SciRustResult<Self> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(crate::error::SciRustError::NonFinite);
        }
        if max_candidates < 2 || max_elements == 0 {
            return Err(crate::error::SciRustError::Empty);
        }
        Ok(Self {
            temperature,
            max_candidates,
            max_elements,
        })
    }

    /// Compute the InfoNCE loss.
    /// `query`: shape `\[d\]`; `keys`: shape `\[n, d\]`; `positive_idx`: index in `0..n`.
    /// Returns a scalar `Var`.
    pub fn loss(
        &self,
        tape: &mut Tape,
        query: Tensor,
        keys: Tensor,
        positive_idx: usize,
    ) -> SciRustResult<Var> {
        let ks = keys.shape.as_slice();
        if ks.len() != 2 {
            return Err(crate::error::SciRustError::Shape {
                lhs: ks.to_vec(),
                rhs: vec![2],
            });
        }
        let (n, d) = (ks[0], ks[1]);
        if query.shape.as_slice() != [d] {
            return Err(crate::error::SciRustError::Shape {
                lhs: query.shape.as_slice().to_vec(),
                rhs: vec![d],
            });
        }
        if n > self.max_candidates {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: n,
                maximum: self.max_candidates,
            });
        }
        if positive_idx >= n {
            return Err(crate::error::SciRustError::Index {
                idx: positive_idx,
                len: n,
            });
        }
        // Sim scores via dot products, scaled by 1/temperature. Compute as a
        // leaf tensor and let the tape backprop through log-softmax + NLL.
        let mut sims: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let row = &keys.data[i * d..(i + 1) * d];
            let mut acc = 0.0f32;
            for (j, &row_j) in row.iter().enumerate().take(d) {
                acc += query.data[j] * row_j;
            }
            sims.push(acc / self.temperature);
        }
        let sim_tensor = Tensor {
            shape: Shape::try_new(&[n])?,
            data: sims,
        };
        let sim_var = tape.variable(sim_tensor)?;
        let logsm = tape.log_softmax(sim_var)?;
        // loss = -logsm[positive_idx] ; represent as sub: zero - logsm[pos]
        // For a 1-D scalar pick, build a one-hot mask of length n, dot with
        // logsm, negate.
        let mut mask = vec![0.0f32; n];
        mask[positive_idx] = 1.0;
        let mask_tensor = Tensor {
            shape: Shape::try_new(&[n])?,
            data: mask,
        };
        let mask_var = tape.variable(mask_tensor)?;
        let picked = tape.mul(mask_var, logsm)?;
        let picked_sum = tape.sum(picked)?;
        // loss = -picked_sum
        tape.neg(picked_sum)
    }
}
