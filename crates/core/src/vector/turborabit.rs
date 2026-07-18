//! TurboRabit — Extended RaBitQ: multi-bit quantization with the RaBitQ
//! unbiased estimator (Gao & Long, arXiv:2409.09913).
//!
//! Deliberately **separate from and independent of** [`super::rabitq`]. RaBitQ
//! is the frozen 1-bit baseline; TurboRabit is its B-bit extension, implemented
//! from the paper + the authors' RaBitQ-Library and measured against both the
//! baseline and [`super::turboquant`]. Shared helpers (rotation, Gaussian
//! sampling) are duplicated on purpose so RaBitQ stays frozen; each copy
//! carries its own tests.
//!
//! # Construction (per vector, `B = total_bits` per dimension)
//!
//! 1. Residual from the fitted centroid, rotated: `r = R·(x − c)`, `ℓ = ‖r‖`.
//! 2. Codebook = the signed grid `{u − (2^B−1)/2 : u ∈ 0..2^B}^D`, normalized —
//!    each code represents a *direction*; scale lives in per-vector scalars.
//!    `B = 1` is exactly classic RaBitQ's `{±0.5}` grid.
//! 3. Encoding maximizes `⟨v/‖v‖, r/ℓ⟩` over grid points `v`. Lemma 3.1: the
//!    maximizer is a rounding of `t·r/ℓ` for some scale `t > 0`, so only the
//!    critical values of `t` need enumeration — an ascending min-heap sweep in
//!    `O(2^B · D log D)`, on the all-positive orthant with signs folded back in.
//! 4. Stored per vector: the B-bit codes `u`, `dtc_sq = ℓ²`, and
//!    `f_rescale = −2ℓ²/⟨r, v⟩`.
//! 5. Estimate against a query rotated the same way (`rq = R·(q − c)`):
//!    ```text
//!    ‖x − q‖² ≈ ℓ² + ‖q−c‖² + f_rescale · (⟨u, rq⟩ + c_B·Σrq)
//!    ```
//!    since `⟨v, rq⟩ = ⟨u, rq⟩ + c_B·Σrq` with `c_B = −(2^B−1)/2`. At `B = 1`
//!    this is algebraically identical to `rabitq::estimate_dist_sq`
//!    (`f_rescale·⟨v,rq⟩ = −2·est_factor·S`), which the tests assert.
//!
//! Encoding fixes four reference-implementation edge cases (initial `t_start`
//! candidate never evaluated; heap pushes past `max_code`/`t_end`; inconsistent
//! sign convention at exactly 0; unguarded zero residual) — see the spec notes
//! in the repo history for the file:line receipts.

use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Default seed for the rotation, so builds are reproducible by default.
const DEFAULT_SEED: u64 = 0x5455_5242_5242_5421; // "TURBRBT!" ish

/// Extra enumeration headroom past the nominal grid top (`kNEnum`): lets the
/// non-max coordinates keep refining after the largest coordinate saturates.
const N_ENUM: u32 = 10;

/// Absorbs FP round-off when flooring at a critical value (`kEpsilon`).
const FLOOR_EPS: f64 = 1e-5;

/// Extended-RaBitQ B-bit quantizer: centroid + shared random rotation.
///
/// Like [`super::rabitq::RaBitQuantizer`] it is fitted (centroid = data mean);
/// the codebook itself is data-oblivious (a rotated fixed grid).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboRabitQuantizer {
    dim: usize,
    /// Total bits per dimension `B` (sign bit + `B−1` magnitude bits).
    total_bits: usize,
    /// Shared centroid (mean of training data).
    centroid: Vec<f32>,
    /// Row-major `dim×dim` orthonormal rotation `R`.
    rotation: Vec<f32>,
}

/// TurboRabit code for one vector: B-bit codes plus the two estimator scalars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboRabitCode {
    /// One B-bit code per coordinate, one `u8` per coordinate for clarity
    /// (`< 2^B`; the arena packs to `B` bits later). MSB of the B bits is the
    /// sign; low `B−1` bits are the (sign-folded) magnitude.
    pub codes: Vec<u8>,
    /// `ℓ² = ‖x − c‖²` — squared distance from the centroid.
    pub dtc_sq: f32,
    /// `−2ℓ² / ⟨r, v⟩` — the folded inner-product rescale.
    pub f_rescale: f32,
}

/// A query rotated into TurboRabit space, reusable across candidates.
pub struct PreparedQuery {
    /// `R·(q − c)`.
    rq: Vec<f32>,
    /// `‖q − c‖²`.
    qn_sq: f32,
    /// `c_B · Σᵢ rqᵢ`, the grid-offset term folded once per query.
    cb_sum: f32,
}

impl PreparedQuery {
    /// `R·(q − c)` — the rotated query residual, borrowed (for callers folding
    /// the estimate into their own kernel; see the rabitq `PreparedQuery` note).
    pub fn rq(&self) -> &[f32] {
        &self.rq
    }
    /// `‖q − c‖²`.
    pub fn qn_sq(&self) -> f32 {
        self.qn_sq
    }
    /// `c_B · Σ rq` — precomputed grid-offset term.
    pub fn cb_sum(&self) -> f32 {
        self.cb_sum
    }
}

/// Recommend a TurboRabit bit budget (`rabit_bits`) from a corpus sample via **centroid
/// dominance** `‖μ‖ / E‖x − μ‖` — the cheap, build-free predictor of how hostile a distribution
/// is to sign-based (RaBitQ-family) codes.
///
/// TurboRabit centres on the corpus mean `μ` and codes the *residual* `x − μ`. When the corpus
/// sits in a narrow cone (`‖μ‖` large, residuals small → dominance ≳ 1) the ranking signal lives
/// in tiny residuals that few bits cannot resolve, so it needs a larger budget; a centred, spread
/// corpus (dominance ≪ 1) is well served by few bits.
///
/// Returns a value in `2..=4`:
/// - `dom < 0.3` → **2** (friendly; even 1-bit RaBitQ often suffices, 2 is a safe floor)
/// - `0.3 ≤ dom < 1.0` → **3**
/// - `dom ≥ 1.0` → **4** (hostile cone)
///
/// Calibrated on two embedders — nomic (`dom ≈ 1.9`, needs b4 for 0.9999 recall@100) and
/// distilroberta (`dom ≈ 0.07`, b2 already at 0.999). Treat the middle band as interpolated until
/// a third embedder lands. **Advisory**: pass the result as [`super::super::index::hnsw::HNSWConfig::rabit_bits`].
pub fn recommend_turborabit_bits(sample: &[Vec<f32>]) -> usize {
    if sample.is_empty() {
        return 3; // the HNSWConfig default
    }
    let dominance = centroid_dominance(sample);
    if dominance >= 1.0 {
        4
    } else if dominance >= 0.3 {
        3
    } else {
        2
    }
}

/// **Centroid dominance** `‖μ‖ / E‖x − μ‖` of a corpus sample — the cheap, build-free predictor of
/// how hostile a distribution is to sign-based (RaBitQ-family) codes. High (≳1) = a narrow cone
/// where the ranking signal lives in tiny residuals; low (≪1) = centred and spread. It drives both
/// [`recommend_turborabit_bits`] (how many bits) and
/// [`super::super::index::hnsw::HNSWConfig::with_auto_storage`] (whether f32 reranking is needed).
/// An empty sample returns 0 (treated as friendly).
pub fn centroid_dominance(sample: &[Vec<f32>]) -> f64 {
    if sample.is_empty() {
        return 0.0;
    }
    let dim = sample[0].len();
    let n = sample.len() as f64;
    let mut mu = vec![0f64; dim];
    for v in sample {
        for (m, &x) in mu.iter_mut().zip(v) {
            *m += x as f64;
        }
    }
    for m in &mut mu {
        *m /= n;
    }
    let mu_norm = mu.iter().map(|&m| m * m).sum::<f64>().sqrt();
    let mut res_sum = 0f64;
    for v in sample {
        let d: f64 = v
            .iter()
            .zip(&mu)
            .map(|(&x, &m)| {
                let e = x as f64 - m;
                e * e
            })
            .sum();
        res_sum += d.sqrt();
    }
    let e_res = res_sum / n;
    if e_res > 1e-12 {
        mu_norm / e_res
    } else {
        f64::INFINITY
    }
}

impl TurboRabitQuantizer {
    /// Fit from training vectors (centroid = per-dimension mean), default seed.
    ///
    /// # Panics
    /// Panics if `training_vectors` is empty, dimensions are inconsistent,
    /// or `total_bits` is not in `1..=8`.
    pub fn fit(training_vectors: &[Vec<f32>], total_bits: usize) -> Self {
        Self::fit_with_seed(training_vectors, total_bits, DEFAULT_SEED)
    }

    /// Fit with an explicit rotation seed.
    pub fn fit_with_seed(training_vectors: &[Vec<f32>], total_bits: usize, seed: u64) -> Self {
        assert!(
            !training_vectors.is_empty(),
            "Need at least one training vector"
        );
        assert!(
            (1..=8).contains(&total_bits),
            "total_bits must be in 1..=8, got {total_bits}"
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

        Self {
            dim,
            total_bits,
            centroid,
            rotation: random_orthonormal(dim, seed),
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total bits per dimension `B`.
    pub fn total_bits(&self) -> usize {
        self.total_bits
    }

    /// `c_B = −(2^B − 1)/2` — the signed-grid offset.
    #[inline]
    fn cb(&self) -> f32 {
        -(((1u32 << self.total_bits) - 1) as f32) / 2.0
    }

    /// Encode a vector into a TurboRabit code.
    pub fn encode(&self, vector: &[f32]) -> TurboRabitCode {
        debug_assert_eq!(vector.len(), self.dim);
        let d = self.dim;
        let ex_bits = (self.total_bits - 1) as u32;

        // Rotated residual r = R·(x − c) and ℓ².
        let mut res = vec![0.0f32; d];
        let mut dtc_sq = 0.0f32;
        for ((r, &v), &c) in res.iter_mut().zip(vector).zip(&self.centroid) {
            *r = v - c;
            dtc_sq += (v - c) * (v - c);
        }
        let r = self.matvec(&res);
        let l = dtc_sq.sqrt();

        // Degenerate: vector == centroid. Zero code, zero factors — the
        // estimator then returns dtc² + qn² = qn², which is correct (x = c).
        if l <= f32::EPSILON {
            return TurboRabitCode {
                codes: vec![0u8; d],
                dtc_sq,
                f_rescale: 0.0,
            };
        }

        // All-positive frame: a = |r| / ℓ (unit norm by construction).
        let a: Vec<f64> = r.iter().map(|&x| (x.abs() as f64) / l as f64).collect();

        // Magnitude codes m ∈ {0..2^ex−1}: round t*·a onto the half-offset grid.
        // ⟨r, v⟩ = ℓ · Σ (m+0.5)·a  (the sign-fold identity).
        let (m_pos, ip_over_l) = if ex_bits == 0 {
            // B = 1: no magnitude bits; v_abs ≡ 0.5, so ⟨r,v⟩/ℓ = 0.5·Σa = L1/(2ℓ).
            (vec![0u8; d], 0.5 * a.iter().sum::<f64>())
        } else {
            let t = best_rescale_factor(&a, ex_bits);
            quantize_ex(&a, ex_bits, t)
        };

        // Fold signs back in: sign bit (MSB) = (r[i] >= 0), magnitude
        // bit-flipped for negative coordinates. Grid value = u + c_B.
        let mask = if ex_bits == 0 {
            0
        } else {
            (1u8 << ex_bits) - 1
        };
        let mut codes = vec![0u8; d];
        for (i, (&ri, &mi)) in r.iter().zip(&m_pos).enumerate() {
            codes[i] = if ri >= 0.0 {
                (1u8 << ex_bits) | mi
            } else {
                (!mi) & mask
            };
        }

        // f_rescale = −2ℓ²/⟨r,v⟩ = −2ℓ/(⟨r,v⟩/ℓ), guarded like the reference.
        let ip_rv = (l as f64) * ip_over_l;
        let f_rescale = if ip_rv > f64::EPSILON {
            (-2.0 * (dtc_sq as f64) / ip_rv) as f32
        } else {
            0.0
        };

        TurboRabitCode {
            codes,
            dtc_sq,
            f_rescale,
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
        let cb_sum = self.cb() * rq.iter().sum::<f32>();
        PreparedQuery { rq, qn_sq, cb_sum }
    }

    /// Estimate squared L2 distance between a prepared query and a code.
    ///
    /// `‖x−q‖² ≈ dtc² + ‖q−c‖² + f_rescale·(⟨u, rq⟩ + c_B·Σrq)`.
    pub fn estimate_dist_sq(&self, query: &PreparedQuery, code: &TurboRabitCode) -> f32 {
        let mut ip = 0.0f32;
        for (&u, &rq) in code.codes.iter().zip(&query.rq) {
            ip += u as f32 * rq;
        }
        let dsq = code.dtc_sq + query.qn_sq + code.f_rescale * (ip + query.cb_sum);
        dsq.max(0.0)
    }
}

// ---- Encoding search: optimal rescale over grid critical values ------------

/// Find `t*` maximizing `⟨y(t)+0.5, a⟩ / ‖y(t)+0.5‖` where `y(t) = ⌊t·a⌋`
/// clamped to `2^ex_bits − 1` (paper Algorithm 1, floor parametrization).
///
/// `a` must be unit-norm with non-negative entries and at least one positive
/// entry. Ascending sweep over critical values `(y+1)/a[i]` via a min-heap;
/// the objective is maintained incrementally in O(1) per step, f64 throughout.
fn best_rescale_factor(a: &[f64], ex_bits: u32) -> f64 {
    let max_code = (1u64 << ex_bits) - 1;
    let max_a = a.iter().cloned().fold(0.0f64, f64::max);
    debug_assert!(max_a > 0.0, "zero residual must be guarded by the caller");
    let t_end = (max_code + N_ENUM as u64) as f64 / max_a;

    // Sweep from t = 0 — the FULL critical-value enumeration (paper Algorithm 1).
    // The reference library prunes with an empirical `kTightStart` lower bound
    // tuned at high D, where coordinates concentrate near 1/√D. At small D one
    // large coordinate drags t* below that bound and the prune cuts off the
    // true optimum (measured: d=32, ex=3 — pruned 0.9968 vs true 0.9980 cosine).
    // Encode is index-time only, so we pay the ~2x heap ops for correctness
    // across all dimensionalities instead.
    let d = a.len();
    let mut cur = vec![0u64; d];
    let mut sqr_den = d as f64 * 0.25; // ‖y+0.5‖² = Σ(y²+y) + D/4
    let mut num = 0.5 * a.iter().sum::<f64>(); // ⟨y+0.5, a⟩ at y = 0

    // Evaluate the all-zero state itself (the reference skips its initial
    // state entirely and can miss the optimum when it is the first interval).
    let mut best_ip = num / sqr_den.sqrt();
    let mut best_t = FLOOR_EPS;

    // Min-heap of (next critical value, dim). Skip dims already at max_code or
    // whose next step lies past t_end (the reference pushes unconditionally — F3).
    let mut heap: BinaryHeap<Reverse<(NotNanF64, usize)>> = BinaryHeap::with_capacity(d);
    for (i, &ai) in a.iter().enumerate() {
        if ai > 0.0 {
            let t_next = 1.0 / ai; // first critical value: cur[i] = 0 → 1
            if t_next < t_end {
                heap.push(Reverse((NotNanF64(t_next), i)));
            }
        }
    }

    // Ascending sweep.
    while let Some(Reverse((NotNanF64(t), i))) = heap.pop() {
        cur[i] += 1;
        let y = cur[i];
        sqr_den += (2 * y) as f64; // Δ(y²+y) for y−1 → y
        num += a[i];
        let ip = num / sqr_den.sqrt();
        if ip > best_ip {
            best_ip = ip;
            best_t = t;
        }
        if y < max_code {
            let t_next = (y + 1) as f64 / a[i];
            if t_next < t_end {
                heap.push(Reverse((NotNanF64(t_next), i)));
            }
        }
    }
    best_t
}

/// Quantize `a` (unit-norm, non-negative) onto the half-offset grid at scale
/// `t`: `m = min(⌊t·a⌋, 2^ex−1)`. Returns `(codes, ⟨v_abs, a⟩ = ⟨r,v⟩/ℓ)`.
fn quantize_ex(a: &[f64], ex_bits: u32, t: f64) -> (Vec<u8>, f64) {
    let max_code = (1u64 << ex_bits) - 1;
    let mut codes = vec![0u8; a.len()];
    let mut ipnorm = 0.0f64;
    for (i, &ai) in a.iter().enumerate() {
        let m = (((t * ai) + FLOOR_EPS) as u64).min(max_code);
        codes[i] = m as u8;
        ipnorm += (m as f64 + 0.5) * ai;
    }
    (codes, ipnorm)
}

/// Total-order f64 wrapper for the heap (critical values are finite by
/// construction: `a[i] > 0` is checked before every push).
#[derive(PartialEq, PartialOrd)]
struct NotNanF64(f64);
impl Eq for NotNanF64 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for NotNanF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---- Matrix helpers (self-contained; mirror rabitq.rs, tested here) --------

impl TurboRabitQuantizer {
    /// `R · v` (row-major matrix-vector product).
    fn matvec(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for (r, o) in out.iter_mut().enumerate() {
            *o = super::simd::dot_product_simd(&self.rotation[r * d..(r + 1) * d], v);
        }
        out
    }
}

/// Row-major `dim×dim` orthonormal matrix via modified Gram–Schmidt on seeded
/// Gaussian rows. Deterministic for `(dim, seed)`.
fn random_orthonormal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
        loop {
            for prev in &rows {
                let proj = dot(&v, prev);
                for (vi, &pi) in v.iter_mut().zip(prev) {
                    *vi -= proj * pi;
                }
            }
            let norm = dot(&v, &v).sqrt();
            if norm >= 1e-6 {
                let inv = 1.0 / norm;
                for vi in &mut v {
                    *vi *= inv;
                }
                break;
            }
            v = (0..dim).map(|_| gaussian(&mut rng)).collect();
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

    fn unit_vec(rng: &mut StdRng, d: usize) -> Vec<f32> {
        let mut v = rng_vec(rng, d);
        let n = dot(&v, &v).sqrt();
        for x in &mut v {
            *x /= n;
        }
        v
    }

    #[test]
    fn recommend_bits_tracks_centroid_dominance() {
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 128;

        // Cone-shaped corpus: a large shared offset + tiny residuals → high dominance → b4.
        // (This is the nomic-embedding failure mode in miniature.)
        let offset: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 3.0).collect();
        let cone: Vec<Vec<f32>> = (0..500)
            .map(|_| {
                offset
                    .iter()
                    .map(|&o| o + (rng.random::<f32>() - 0.5) * 0.1)
                    .collect()
            })
            .collect();
        assert_eq!(
            recommend_turborabit_bits(&cone),
            4,
            "cone corpus should want max bits"
        );

        // Centred, spread corpus (mean ≈ 0, residuals ≈ signal) → low dominance → b2.
        let spread: Vec<Vec<f32>> = (0..500).map(|_| rng_vec(&mut rng, dim)).collect();
        assert_eq!(
            recommend_turborabit_bits(&spread),
            2,
            "spread corpus needs few bits"
        );

        // Degenerate input must not panic and must return a sane default.
        assert_eq!(recommend_turborabit_bits(&[]), 3);
    }

    #[test]
    fn rotation_is_orthonormal() {
        let d = 48;
        let r = random_orthonormal(d, 7);
        for i in 0..d {
            let ri = &r[i * d..(i + 1) * d];
            assert!((dot(ri, ri) - 1.0).abs() < 1e-4, "row {i} not unit");
            for j in (i + 1)..d {
                let rj = &r[j * d..(j + 1) * d];
                assert!(dot(ri, rj).abs() < 1e-3, "rows {i},{j} not orthogonal");
            }
        }
    }

    /// The heap search must find the global optimum: no `t` on a fine grid over
    /// `(0, t_end]` may beat the returned `t*`'s cosine objective (beyond FP slop).
    #[test]
    fn rescale_search_is_optimal() {
        let objective = |a: &[f64], ex_bits: u32, t: f64| -> f64 {
            let (codes, _) = quantize_ex(a, ex_bits, t);
            let mut num = 0.0;
            let mut den = 0.0;
            for (&m, &ai) in codes.iter().zip(a) {
                num += (m as f64 + 0.5) * ai;
                den += (m as f64 + 0.5) * (m as f64 + 0.5);
            }
            num / den.sqrt()
        };

        let mut rng = StdRng::seed_from_u64(42);
        for &d in &[32usize, 96] {
            for ex_bits in 1..=3u32 {
                for trial in 0..8 {
                    let v = unit_vec(&mut rng, d);
                    let a: Vec<f64> = v.iter().map(|x| x.abs() as f64).collect();
                    let t_star = best_rescale_factor(&a, ex_bits);
                    let got = objective(&a, ex_bits, t_star);

                    let max_a = a.iter().cloned().fold(0.0f64, f64::max);
                    let t_end = (((1u64 << ex_bits) - 1) + N_ENUM as u64) as f64 / max_a;
                    let mut best_grid = 0.0f64;
                    for k in 1..=4000 {
                        let t = t_end * k as f64 / 4000.0;
                        best_grid = best_grid.max(objective(&a, ex_bits, t));
                    }
                    assert!(
                        got >= best_grid - 1e-9,
                        "d={d} ex={ex_bits} trial={trial}: heap search {got} \
                         beaten by grid {best_grid}"
                    );
                }
            }
        }
    }

    /// B = 1 must be classic RaBitQ: same rotation seed + same training data
    /// ⇒ the two estimators agree (they are the same algebra, folded differently).
    #[test]
    fn b1_matches_frozen_rabitq() {
        let mut rng = StdRng::seed_from_u64(9);
        let d = 64;
        let train: Vec<Vec<f32>> = (0..300).map(|_| rng_vec(&mut rng, d)).collect();
        let seed = 12345u64;
        let ours = TurboRabitQuantizer::fit_with_seed(&train, 1, seed);
        let frozen = super::super::rabitq::RaBitQuantizer::fit_with_seed(&train, seed);

        for _ in 0..50 {
            let x = rng_vec(&mut rng, d);
            let q = rng_vec(&mut rng, d);
            let a = ours.estimate_dist_sq(&ours.prepare_query(&q), &ours.encode(&x));
            let b = frozen.estimate_dist_sq(&frozen.prepare_query(&q), &frozen.encode(&x));
            let denom = b.abs().max(1e-3);
            assert!(
                ((a - b) / denom).abs() < 1e-3,
                "B=1 diverges from frozen RaBitQ: {a} vs {b}"
            );
        }
    }

    /// Unbiasedness is over the random rotation. Fix one (x, q) pair, average
    /// the estimate across many independently-seeded quantizers.
    #[test]
    fn estimator_is_approximately_unbiased() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(11);
        let train: Vec<Vec<f32>> = (0..200).map(|_| rng_vec(&mut rng, d)).collect();
        let x = rng_vec(&mut rng, d);
        let q = rng_vec(&mut rng, d);
        let truth: f32 = x.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum();

        for bits in [2usize, 4] {
            let n = 200;
            let mut acc = 0.0f64;
            for s in 0..n {
                let quant = TurboRabitQuantizer::fit_with_seed(&train, bits, 1000 + s as u64);
                let est = quant.estimate_dist_sq(&quant.prepare_query(&q), &quant.encode(&x));
                acc += est as f64;
            }
            let mean = (acc / n as f64) as f32;
            let rel = (mean - truth) / truth;
            assert!(
                rel.abs() < 0.05,
                "B={bits} estimator biased: mean {mean} vs truth {truth} (rel {rel})"
            );
        }
    }

    /// Held-out queries vs brute-force ground truth — NEVER self-retrieval.
    /// More bits ⇒ better recall, and B=4 must beat B=1 decisively.
    #[test]
    fn held_out_recall_improves_with_bits() {
        let d = 128;
        let (n, nq, k) = (600usize, 60usize, 10usize);
        let mut rng = StdRng::seed_from_u64(3);
        let base: Vec<Vec<f32>> = (0..n).map(|_| unit_vec(&mut rng, d)).collect();
        let queries: Vec<Vec<f32>> = (0..nq).map(|_| unit_vec(&mut rng, d)).collect();

        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut ds: Vec<(usize, f32)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i, x.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum()))
                    .collect();
                ds.sort_by(|a, b| a.1.total_cmp(&b.1));
                ds.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect();

        let recall_at = |bits: usize| -> f32 {
            let quant = TurboRabitQuantizer::fit(&base, bits);
            let codes: Vec<TurboRabitCode> = base.iter().map(|x| quant.encode(x)).collect();
            let mut hit = 0usize;
            for (q, gt) in queries.iter().zip(&truth) {
                let pq = quant.prepare_query(q);
                let mut est: Vec<(usize, f32)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, quant.estimate_dist_sq(&pq, c)))
                    .collect();
                est.sort_by(|a, b| a.1.total_cmp(&b.1));
                let top: std::collections::HashSet<usize> =
                    est.into_iter().take(k).map(|(i, _)| i).collect();
                hit += gt.iter().filter(|i| top.contains(i)).count();
            }
            hit as f32 / (nq * k) as f32
        };

        let r1 = recall_at(1);
        let r2 = recall_at(2);
        let r4 = recall_at(4);
        assert!(r2 > r1, "B=2 ({r2}) should beat B=1 ({r1})");
        assert!(r4 > r2, "B=4 ({r4}) should beat B=2 ({r2})");
        assert!(r4 > 0.70, "B=4 recall too low: {r4}");
    }

    /// Vector == centroid must not NaN/panic; the estimate degrades to ‖q−c‖².
    #[test]
    fn zero_residual_is_guarded() {
        let mut rng = StdRng::seed_from_u64(5);
        let d = 32;
        let train: Vec<Vec<f32>> = (0..100).map(|_| rng_vec(&mut rng, d)).collect();
        let quant = TurboRabitQuantizer::fit(&train, 3);
        let centroid: Vec<f32> = {
            let mut c = vec![0.0f32; d];
            for v in &train {
                for (ci, &x) in c.iter_mut().zip(v) {
                    *ci += x;
                }
            }
            c.iter().map(|x| x / train.len() as f32).collect()
        };
        let code = quant.encode(&centroid);
        let q = rng_vec(&mut rng, d);
        let est = quant.estimate_dist_sq(&quant.prepare_query(&q), &code);
        assert!(est.is_finite(), "degenerate encode produced {est}");
        let qn_sq: f32 = q
            .iter()
            .zip(&centroid)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            (est - qn_sq).abs() / qn_sq.max(1e-3) < 0.05,
            "degenerate estimate {est} should be ≈ ‖q−c‖² = {qn_sq}"
        );
    }
}
