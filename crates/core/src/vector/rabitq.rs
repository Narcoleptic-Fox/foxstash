//! RaBitQ quantization — theoretically-grounded 1-bit quantization with an
//! unbiased distance estimator (Gao & Long, "RaBitQ", SIGMOD 2024).
//!
//! Unlike [`BinaryQuantizer`](super::quantize::BinaryQuantizer), which packs
//! sign bits and compares them with crude Hamming distance, RaBitQ:
//!
//! 1. Subtracts a shared **centroid** and works on unit residuals.
//! 2. Applies a shared random **orthonormal rotation** `R`, which spreads
//!    information evenly across coordinates (so no axis is privileged) and is
//!    what makes the 1-bit code informative.
//! 3. Stores, per vector, the sign bits plus two scalars that drive an
//!    **unbiased estimator** of the inner product — giving a real distance
//!    estimate (with a provable error bound) at the same 32x compression as
//!    binary, rather than a Hamming proxy.
//!
//! The estimate is good enough to use as the first stage of a two-phase
//! search: rank by the RaBitQ estimate, then rerank the top candidates with
//! exact full-precision distance (see [`RaBitQuantizer::prepare_query`]).
//!
//! # Derivation (folded for the hot path)
//!
//! Let `c` be the centroid, `o_res = o - c`, `dtc = ‖o_res‖`, and `ro = R·o_res`.
//! With sign bits `bᵢ = [roᵢ ≥ 0]` and `L1 = Σ|roᵢ|`, define the per-vector
//! `est_factor = dtc² / L1`. For a query with `q_res = q - c` and rotated
//! `rq = R·q_res`, let `S = Σ(2bᵢ − 1)·rqᵢ`. Then
//!
//! ```text
//! ⟨o_res, q_res⟩ ≈ est_factor · S
//! ‖o − q‖²       ≈ dtc² + ‖q_res‖² − 2 · est_factor · S
//! ```
//!
//! The query norm cancels in the cross term, so per-candidate cost is one O(D)
//! signed accumulation over `rq`; the only O(D²) work is rotating the query once.
//!
//! This module implements base (1-bit) RaBitQ. Extended RaBitQ (configurable
//! B-bit) builds on the same rotation + estimator and is a planned follow-up.

use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};


/// Default seed for the rotation, so builds are reproducible by default.
const DEFAULT_SEED: u64 = 0x5241_4249_5451_5121; // "RABITQ!" ish

/// RaBitQ 1-bit quantizer (32x compression, unbiased distance estimator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQuantizer {
    dim: usize,
    /// Shared centroid (mean of training data).
    centroid: Vec<f32>,
    /// Row-major `dim x dim` orthonormal rotation matrix `R`.
    rotation: Vec<f32>,
}

/// RaBitQ code for one vector: sign bits plus the two estimator scalars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitCode {
    /// Packed sign bits of the rotated residual (`ceil(dim/8)` bytes).
    pub bits: Vec<u8>,
    /// `dtc²` — squared distance from the centroid.
    pub dtc_sq: f32,
    /// `est_factor = dtc² / ‖R·(o−c)‖₁` — the inner-product rescale.
    pub est_factor: f32,
}

/// A query rotated into RaBitQ space, reusable across many candidate estimates.
pub struct PreparedQuery {
    /// `R·(q − c)` — rotated raw residual (length `dim`).
    rq: Vec<f32>,
    /// `‖q − c‖²`.
    qn_sq: f32,
}

impl PreparedQuery {
    /// `R·(q − c)` — the rotated query residual, borrowed.
    ///
    /// Exposed for callers that fold the estimator into their own SIMD kernel rather than
    /// going through [`RaBitQuantizer::estimate_dist_sq`] — that method takes `&RaBitCode`,
    /// which owns a `Vec<u8>`, so building one per candidate to call it would allocate on
    /// every distance computation in a hot graph-traversal loop.
    pub fn rq(&self) -> &[f32] {
        &self.rq
    }

    /// `‖q − c‖²`.
    pub fn qn_sq(&self) -> f32 {
        self.qn_sq
    }
}

impl RaBitQuantizer {
    /// Fit a quantizer from training vectors using the default rotation seed.
    ///
    /// Computes the centroid as the per-dimension mean and generates a seeded
    /// random orthonormal rotation. Reproducible across runs.
    ///
    /// # Panics
    /// Panics if `training_vectors` is empty or has inconsistent dimensions.
    pub fn fit(training_vectors: &[Vec<f32>]) -> Self {
        Self::fit_with_seed(training_vectors, DEFAULT_SEED)
    }

    /// Fit with an explicit rotation seed.
    pub fn fit_with_seed(training_vectors: &[Vec<f32>], seed: u64) -> Self {
        assert!(
            !training_vectors.is_empty(),
            "Need at least one training vector"
        );
        let dim = training_vectors[0].len();
        assert!(dim > 0, "Dimension must be positive");

        let mut centroid = vec![0.0f32; dim];
        for v in training_vectors {
            assert_eq!(v.len(), dim, "Inconsistent vector dimensions");
            for (c, &x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        let inv_n = 1.0 / training_vectors.len() as f32;
        for c in &mut centroid {
            *c *= inv_n;
        }

        let rotation = random_orthonormal(dim, seed);
        Self {
            dim,
            centroid,
            rotation,
        }
    }

    /// Construct directly from a known centroid (e.g. an IVF coarse centroid).
    pub fn with_centroid(centroid: Vec<f32>, seed: u64) -> Self {
        let dim = centroid.len();
        assert!(dim > 0, "Dimension must be positive");
        let rotation = random_orthonormal(dim, seed);
        Self {
            dim,
            centroid,
            rotation,
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a vector into a RaBitQ code.
    pub fn encode(&self, vector: &[f32]) -> RaBitCode {
        debug_assert_eq!(vector.len(), self.dim);

        // Residual from centroid, and its norm.
        let mut res = vec![0.0f32; self.dim];
        let mut dtc_sq = 0.0f32;
        for ((r, &v), &c) in res.iter_mut().zip(vector).zip(&self.centroid) {
            *r = v - c;
            dtc_sq += (v - c) * (v - c);
        }

        // Rotate the residual: ro = R · res.
        let ro = self.matvec(&res);

        // Sign bits + L1 norm of the rotated residual.
        let mut bits = vec![0u8; self.bytes()];
        let mut l1 = 0.0f32;
        for (i, &x) in ro.iter().enumerate() {
            l1 += x.abs();
            if x >= 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }

        // est_factor = dtc² / L1, guarded against the degenerate (vector == centroid) case.
        let est_factor = if l1 > f32::EPSILON { dtc_sq / l1 } else { 0.0 };

        RaBitCode {
            bits,
            dtc_sq,
            est_factor,
        }
    }

    /// Prepare a query once for repeated [`estimate_dist_sq`](Self::estimate_dist_sq) calls.
    pub fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        debug_assert_eq!(query.len(), self.dim);
        let mut res = vec![0.0f32; self.dim];
        let mut qn_sq = 0.0f32;
        for ((r, &q), &c) in res.iter_mut().zip(query).zip(&self.centroid) {
            *r = q - c;
            qn_sq += (q - c) * (q - c);
        }
        let rq = self.matvec(&res);
        PreparedQuery { rq, qn_sq }
    }

    /// Estimate squared L2 distance between a prepared query and a code.
    ///
    /// `S = Σ (2bᵢ − 1) · rqᵢ`, then `‖o−q‖² ≈ dtc² + ‖q−c‖² − 2·est_factor·S`.
    pub fn estimate_dist_sq(&self, query: &PreparedQuery, code: &RaBitCode) -> f32 {
        let mut s = 0.0f32;
        for (i, &rq) in query.rq.iter().enumerate() {
            let bit = (code.bits[i / 8] >> (i % 8)) & 1;
            // (2b - 1): +rq when bit set, -rq when clear.
            if bit == 1 {
                s += rq;
            } else {
                s -= rq;
            }
        }
        let dsq = code.dtc_sq + query.qn_sq - 2.0 * code.est_factor * s;
        dsq.max(0.0)
    }

    /// Number of bytes in a packed code.
    fn bytes(&self) -> usize {
        self.dim.div_ceil(8)
    }

    /// `R · v` (row-major matrix-vector product).
    fn matvec(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.rotation[r * d..(r + 1) * d];
            *o = super::simd::dot_product_simd(row, v);
        }
        out
    }

    /// `Rᵀ · v` — used only for the (lossy) dequantize path.
    fn matvec_transpose(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for (r, &vr) in v.iter().enumerate() {
            let row = &self.rotation[r * d..(r + 1) * d];
            for (o, &rc) in out.iter_mut().zip(row) {
                *o += rc * vr;
            }
        }
        out
    }
}

/// Inherent, not a trait impl.
///
/// These used to satisfy a `Quantizer` trait in `vector::quantize`, which had three implementors
/// and was used polymorphically by nothing -- no `dyn Quantizer`, no `T: Quantizer` bound anywhere
/// in the workspace. The other two implementors (`ScalarQuantizer`, `BinaryQuantizer`) were a
/// SECOND implementation of SQ8, which the index never called: `hnsw.rs` has its own SoA layout
/// and its own AVX2 kernels in `vector::simd`. Two copies of one idea, one of them shipped and
/// one of them merely benchmarked. That is the shape of every bug in the 1.0 audit, so the copy
/// nobody ran was deleted and the abstraction over it went with it.
impl RaBitQuantizer {
    /// Encode a vector to its 1-bit code. See [`Self::encode`].
    pub fn quantize(&self, vector: &[f32]) -> RaBitCode {
        self.encode(vector)
    }

    /// Reconstruct an approximate vector from its code. Lossy (sign-only) but directionally
    /// correct; used by [`Self::distance_symmetric`].
    pub fn dequantize(&self, quantized: &RaBitCode) -> Vec<f32> {
        // Reconstruct o ≈ c + dtc · Rᵀ · x̄, where x̄ᵢ = ±1/√D from the sign bits.
        // Lossy (sign-only), but directionally correct.
        let d = self.dim;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let mut xbar = vec![0.0f32; d];
        for (i, x) in xbar.iter_mut().enumerate() {
            let bit = (quantized.bits[i / 8] >> (i % 8)) & 1;
            *x = if bit == 1 { inv_sqrt_d } else { -inv_sqrt_d };
        }
        let dir = self.matvec_transpose(&xbar);
        let dtc = quantized.dtc_sq.sqrt();
        dir.iter()
            .zip(&self.centroid)
            .map(|(&u, &c)| c + dtc * u)
            .collect()
    }

    /// Distance between two codes, with one side dequantized -- RaBitQ's estimator is asymmetric,
    /// so there is no honest code-to-code distance.
    pub fn distance_quantized(&self, a: &RaBitCode, b: &RaBitCode) -> f32 {
        let a_full = self.dequantize(a);
        self.distance_asymmetric(&a_full, b)
    }

    /// Distance from a full-precision query to a code. This is the estimator the index uses,
    /// though the hot path goes through `simd::rabitq_asymmetric_l2_simd` rather than here.
    pub fn distance_asymmetric(&self, query: &[f32], quantized: &RaBitCode) -> f32 {
        let prepared = self.prepare_query(query);
        self.estimate_dist_sq(&prepared, quantized).sqrt()
    }
}

/// Generate a `dim x dim` orthonormal matrix (row-major) via Gram–Schmidt on
/// seeded Gaussian vectors. Deterministic for a given `(dim, seed)`.
fn random_orthonormal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(dim);

    for _ in 0..dim {
        // Fresh Gaussian row.
        let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();

        // Orthogonalize against previously accepted rows (modified Gram–Schmidt).
        for prev in &rows {
            let proj = dot(&v, prev);
            for (vi, &pi) in v.iter_mut().zip(prev) {
                *vi -= proj * pi;
            }
        }

        // Normalize; if it collapsed (near-dependent), resample.
        let mut norm = dot(&v, &v).sqrt();
        while norm < 1e-6 {
            v = (0..dim).map(|_| gaussian(&mut rng)).collect();
            for prev in &rows {
                let proj = dot(&v, prev);
                for (vi, &pi) in v.iter_mut().zip(prev) {
                    *vi -= proj * pi;
                }
            }
            norm = dot(&v, &v).sqrt();
        }
        let inv = 1.0 / norm;
        for vi in &mut v {
            *vi *= inv;
        }
        rows.push(v);
    }

    let mut flat = Vec::with_capacity(dim * dim);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    flat
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Standard-normal sample via Box–Muller.
#[inline]
fn gaussian(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-7);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn rotation_is_orthonormal() {
        let d = 64;
        let r = random_orthonormal(d, 123);
        // Rows should be unit-norm and mutually orthogonal: R·Rᵀ ≈ I.
        for i in 0..d {
            for j in 0..d {
                let ri = &r[i * d..(i + 1) * d];
                let rj = &r[j * d..(j + 1) * d];
                let prod = dot(ri, rj);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod - expected).abs() < 1e-3,
                    "R·Rᵀ[{i},{j}] = {prod}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rotation_is_deterministic() {
        assert_eq!(random_orthonormal(32, 42), random_orthonormal(32, 42));
    }

    #[test]
    fn estimator_is_approximately_unbiased() {
        // The mean estimated squared distance should track the true one closely.
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 128;
        let train: Vec<Vec<f32>> = (0..500).map(|_| rng_vec(&mut rng, dim)).collect();
        let q = RaBitQuantizer::fit(&train);

        let mut rel_errs = Vec::new();
        for _ in 0..200 {
            let o = rng_vec(&mut rng, dim);
            let query = rng_vec(&mut rng, dim);
            let code = q.encode(&o);
            let prep = q.prepare_query(&query);
            let est = q.estimate_dist_sq(&prep, &code);
            let truth: f32 = o.iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum();
            rel_errs.push((est - truth) / truth);
        }
        let mean_bias: f32 = rel_errs.iter().sum::<f32>() / rel_errs.len() as f32;
        // Unbiased ⇒ mean relative error near zero (loose bound for 1-bit).
        assert!(
            mean_bias.abs() < 0.10,
            "estimator mean relative bias too large: {mean_bias}"
        );
    }

    #[test]
    fn rerank_recall_beats_hamming_floor() {
        // End-to-end gate: rank by RaBitQ estimate, rerank top candidates by
        // exact distance, and require high recall@10. This is the property that
        // makes RaBitQ usable as a first-stage filter.
        let mut rng = StdRng::seed_from_u64(99);
        let dim = 128;
        let n = 2000;
        let base: Vec<Vec<f32>> = (0..n).map(|_| rng_vec(&mut rng, dim)).collect();
        let q = RaBitQuantizer::fit(&base);
        let codes: Vec<RaBitCode> = base.iter().map(|v| q.encode(v)).collect();

        let k = 10;
        let rerank = 100; // first-stage candidate pool
        let mut total_recall = 0.0;
        let trials = 50;
        for _ in 0..trials {
            let query = rng_vec(&mut rng, dim);

            // Ground truth top-k by exact L2².
            let mut exact: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        v.iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum(),
                        i,
                    )
                })
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: std::collections::HashSet<usize> =
                exact.iter().take(k).map(|(_, i)| *i).collect();

            // Stage 1: rank all by RaBitQ estimate, keep top `rerank`.
            let prep = q.prepare_query(&query);
            let mut est: Vec<(f32, usize)> = codes
                .iter()
                .enumerate()
                .map(|(i, c)| (q.estimate_dist_sq(&prep, c), i))
                .collect();
            est.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Stage 2: rerank the pool by exact distance, take top-k.
            let mut pool: Vec<(f32, usize)> = est
                .iter()
                .take(rerank)
                .map(|&(_, i)| {
                    let d: f32 = base[i]
                        .iter()
                        .zip(&query)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (d, i)
                })
                .collect();
            pool.sort_by(|a, b| a.0.total_cmp(&b.0));
            let got: std::collections::HashSet<usize> =
                pool.iter().take(k).map(|(_, i)| *i).collect();

            total_recall += truth.intersection(&got).count() as f32 / k as f32;
        }
        let recall = total_recall / trials as f32;
        assert!(recall > 0.80, "RaBitQ rerank recall@10 too low: {recall}");
    }

    #[test]
    fn quantizer_trait_roundtrip() {
        let mut rng = StdRng::seed_from_u64(5);
        let dim = 96;
        let train: Vec<Vec<f32>> = (0..200).map(|_| rng_vec(&mut rng, dim)).collect();
        let q = RaBitQuantizer::fit(&train);

        let v = rng_vec(&mut rng, dim);
        let code = q.quantize(&v);
        assert_eq!(code.bits.len(), dim.div_ceil(8));

        // Asymmetric self-distance should be small relative to a random pair.
        let self_d = q.distance_asymmetric(&v, &code);
        let other = rng_vec(&mut rng, dim);
        let other_d = q.distance_asymmetric(&other, &code);
        assert!(
            self_d < other_d,
            "self distance {self_d} should be < cross distance {other_d}"
        );

        // Dequantize returns the right shape.
        assert_eq!(q.dequantize(&code).len(), dim);
    }
}
