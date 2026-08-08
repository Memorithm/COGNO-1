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

    /// Compute the contrastive margin loss from already-connected score Vars.
    ///
    /// Unlike [`Self::loss`], this path does not create new leaf score tensors.
    /// Gradients therefore continue through the caller's scoring head and into
    /// any upstream encoder parameters.
    pub fn loss_vars(
        &self,
        tape: &mut Tape,
        preferred: Var,
        dispreferred: Var,
    ) -> SciRustResult<Var> {
        let preferred_len = tape.value_of(preferred).len();
        let dispreferred_len = tape.value_of(dispreferred).len();
        if preferred_len != dispreferred_len {
            return Err(crate::error::SciRustError::Shape {
                lhs: tape.value_of(preferred).shape.as_slice().to_vec(),
                rhs: tape.value_of(dispreferred).shape.as_slice().to_vec(),
            });
        }
        if preferred_len > self.max_pairs {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: preferred_len,
                maximum: self.max_pairs,
            });
        }
        if preferred_len > self.max_elements {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: preferred_len,
                maximum: self.max_elements,
            });
        }

        let diff = tape.sub(preferred, dispreferred)?;
        let shape = tape.value_of(diff).shape.clone();
        let margin = Tensor::try_new(shape, vec![self.margin; preferred_len], self.max_elements)?;
        let margin = tape.variable(margin)?;
        let margin_minus = tape.sub(margin, diff)?;
        let relu = tape.relu(margin_minus)?;
        tape.sum(relu)
    }

    /// Compute the contrastive margin loss for a batch of detached score data.
    ///
    /// This compatibility path preserves the original API. New model code that
    /// needs gradients to reach a scoring head should use [`Self::loss_vars`].
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
        let preferred = tape.variable(preferred)?;
        let dispreferred = tape.variable(dispreferred)?;
        self.loss_vars(tape, preferred, dispreferred)
    }
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

    /// Compute InfoNCE from a caller-owned similarity tensor already connected
    /// to the tape.
    ///
    /// This is the path trainable query/key heads should use: the temperature,
    /// log-softmax and NLL remain on the same autograd graph, so gradients can
    /// reach the similarity producer and its upstream encoder.
    pub fn loss_similarities(
        &self,
        tape: &mut Tape,
        similarities: Var,
        positive_idx: usize,
    ) -> SciRustResult<Var> {
        let n = tape.value_of(similarities).len();
        if n < 2 {
            return Err(crate::error::SciRustError::Empty);
        }
        if n > self.max_candidates {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: n,
                maximum: self.max_candidates,
            });
        }
        if n > self.max_elements {
            return Err(crate::error::SciRustError::CapacityExceeded {
                requested: n,
                maximum: self.max_elements,
            });
        }
        if positive_idx >= n {
            return Err(crate::error::SciRustError::Index {
                idx: positive_idx,
                len: n,
            });
        }

        let scaled = tape.scale(similarities, self.temperature.recip())?;
        let logsm = tape.log_softmax(scaled)?;
        let mut mask = vec![0.0f32; n];
        mask[positive_idx] = 1.0;
        let mask = Tensor::try_new(tape.value_of(logsm).shape.clone(), mask, self.max_elements)?;
        let mask = tape.variable(mask)?;
        let picked = tape.mul(mask, logsm)?;
        let picked_sum = tape.sum(picked)?;
        tape.neg(picked_sum)
    }

    /// Compute the original detached query/key compatibility objective.
    ///
    /// This API is retained for callers that already own concrete tensors. New
    /// trainable model paths should construct similarities on the tape and call
    /// [`Self::loss_similarities`].
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

        let mut sims = Vec::with_capacity(n);
        for i in 0..n {
            let row = &keys.data[i * d..(i + 1) * d];
            let mut acc = 0.0f32;
            for (j, &row_j) in row.iter().enumerate().take(d) {
                acc += query.data[j] * row_j;
            }
            sims.push(acc);
        }
        let sim_tensor = Tensor::try_new(Shape::try_new(&[n])?, sims, self.max_elements)?;
        let sim_var = tape.variable(sim_tensor)?;
        self.loss_similarities(tape, sim_var, positive_idx)
    }
}
