//! TurboQuant — data-oblivious multi-bit vector quantization with an unbiased
//! inner-product estimator (Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant",
//! arXiv:2504.19874, 2025).
//!
//! Deliberately **separate from and independent of** [`super::rabitq`]. RaBitQ is
//! the stable 1-bit baseline; TurboQuant is the multi-bit, data-oblivious
//! alternative we measure against it. The few shared helpers (orthonormal
//! rotation, Gaussian sampling) are duplicated here on purpose so RaBitQ stays
//! frozen; each copy carries its own test so the two cannot silently diverge.
//!
//! # Construction (all data-oblivious — no k-means on the data)
//!
//! 1. Seeded random orthonormal rotation `Π` and Gaussian sketch matrix `S`,
//!    both `d×d`.
//! 2. For a **unit** vector `x`, `y = Π·x` is uniform on the sphere, so in high
//!    dimension each coordinate `yⱼ ~ N(0, 1/d)`. The MSE-optimal `(b-1)`-bit
//!    scalar codebook is therefore the Lloyd–Max quantizer of a Gaussian —
//!    **derived here** in closed form (conditional means over the normal), then
//!    scaled by `1/√d`. No training data touched.
//! 3. Quantize `y` per coordinate → `idx`. Reconstruct `x̃_mse = Πᵀ·ỹ` with
//!    `ỹⱼ = c_{idxⱼ}`, and take the residual `r = x − x̃_mse`.
//! 4. 1-bit Quantized-JL sketch of the residual: `qjl = sign(S·r)`, store `γ = ‖r‖`.
//! 5. Unbiased inner-product estimate against a query `q`:
//!    ```text
//!    ⟨q, x̃⟩ ≈ ⟨Πq, ỹ⟩  +  γ · (√(π/2)/d) · ⟨Sq, qjl⟩
//!    ```
//!    The MSE term gives low reconstruction error; the QJL term debiases the
//!    inner product (an MSE-optimal quantizer alone is biased for IP). Both terms
//!    are `O(d)` per candidate after two `O(d²)` query matvecs (`Πq`, `Sq`) —
//!    the same hot-path shape as RaBitQ.
//!
//! `total_bits = b`: the scalar code uses `b-1` bits/coord, the QJL residual 1.
//! `b = 1` degenerates to a pure 1-bit QJL sketch (no MSE term).

use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// Default seed for `Π` and `S`, so builds are reproducible by default.
const DEFAULT_SEED: u64 = 0x_5455_5242_4F51_5421; // "TURBOQ!" ish

/// Number of Rademacher-flip + FWHT rounds in the structured rotation. One round is the
/// Fast-TurboQuant minimum; the RaBitQ-Library `FhtKacRotator` uses 4 for Haar-like mixing
/// at negligible cost (each round is `O(P log P)` additions), and we match it.
const FHT_ROUNDS: usize = 4;

/// The padded dimension the structured rotation operates in: the next power of two
/// (FWHT butterflies need one). Codes and the rotated query live at this length; the
/// QJL sketch and residual stay at the original `dim`. One definition — the HNSW arena
/// sizes its nibble section with this same function.
pub const fn fht_padded_dim(dim: usize) -> usize {
    dim.next_power_of_two()
}

/// Data-oblivious multi-bit quantizer. Holds only dimension-derived state
/// (rotation flips, sketch, Gaussian codebook) — never anything fitted to the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizer {
    dim: usize,
    /// `dim.next_power_of_two()` — the length the rotation, codes, and rotated query
    /// live at. Fast-TurboQuant's zero-padding: energy spreads over more, smaller
    /// coordinates, which the paper reports *reduces* quantization MSE.
    padded_dim: usize,
    /// Bits per coordinate for the MSE scalar code (`= total_bits - 1`).
    mse_bits: usize,
    /// Rademacher sign diagonals for the structured rotation `Π = (H·D_k)^(FHT_ROUNDS)`
    /// (Fast-TurboQuant, arXiv:2606.21448): `FHT_ROUNDS · padded_dim/8` bytes, rounds
    /// concatenated, bit `i%8` of byte `i/8` within each round. Replaces the dense
    /// `dim×dim` matrix — `O(P log P)` additions instead of `O(d²)` multiplies, and no
    /// `d²·4`-byte matrix to store or stream.
    flips: Vec<u8>,
    /// Row-major `dim×dim` Gaussian QJL sketch `S` (i.i.d. `N(0,1)`). Deliberately NOT
    /// replaced by the structured transform: the QJL debias constant `√(π/2)/d` is
    /// derived for a Gaussian sketch, and the sketch runs over the (unpadded) residual.
    sketch: Vec<f32>,
    /// `2^mse_bits` reconstruction levels for `N(0, 1/padded_dim)` (empty if `mse_bits == 0`).
    levels: Vec<f32>,
}

/// TurboQuant code for one vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboCode {
    /// One MSE codebook index per PADDED coordinate (`< 2^mse_bits`; length
    /// [`TurboQuantizer::padded_dim`]). Empty if `mse_bits == 0`. One `u8` per
    /// coordinate here for clarity; the HNSW arena packs to nibbles.
    pub idx: Vec<u8>,
    /// Packed QJL sign bits of `S·r` (`ceil(dim/8)` bytes).
    pub qjl: Vec<u8>,
    /// `γ = ‖r‖` — the residual norm scaling the QJL term.
    pub gamma: f32,
}

/// A query rotated + sketched once, reused across many candidate estimates.
pub struct PreparedQuery {
    /// `Π·q`.
    pq: Vec<f32>,
    /// `S·q`.
    sq: Vec<f32>,
}

impl PreparedQuery {
    /// `Π·q` — the rotated query, borrowed (for callers folding the estimate into their own kernel).
    pub fn pq(&self) -> &[f32] {
        &self.pq
    }
    /// `S·q` — the sketched query, borrowed.
    pub fn sq(&self) -> &[f32] {
        &self.sq
    }
}

impl TurboQuantizer {
    /// Build a quantizer for `dim`-dimensional unit vectors with `total_bits`
    /// bits/coordinate (`b-1` scalar + 1 QJL), using the default seed.
    ///
    /// Data-oblivious: needs only `dim`, no training vectors.
    ///
    /// # Panics
    /// Panics if `dim == 0` or `total_bits == 0`.
    pub fn new(dim: usize, total_bits: usize) -> Self {
        Self::with_seed(dim, total_bits, DEFAULT_SEED)
    }

    /// Build with an explicit seed for `Π` and `S`.
    pub fn with_seed(dim: usize, total_bits: usize, seed: u64) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert!(total_bits > 0, "total_bits must be >= 1");
        let mse_bits = total_bits - 1;
        let padded_dim = fht_padded_dim(dim);

        // Independent seeds for the flips and the sketch so they are uncorrelated.
        let mut rng = StdRng::seed_from_u64(seed);
        let flips: Vec<u8> = (0..FHT_ROUNDS * padded_dim.div_ceil(8))
            .map(|_| rng.random::<u8>())
            .collect();
        let sketch = gaussian_matrix(dim, seed ^ 0x9E37_79B9_7F4A_7C15);

        // Codebook: Lloyd–Max levels for the unit Gaussian, scaled to N(0, 1/P) — the
        // rotation spreads a unit vector's energy over the PADDED coordinates.
        let levels = if mse_bits == 0 {
            Vec::new()
        } else {
            let unit = derive_gaussian_lloyd_max(1usize << mse_bits);
            let sigma = 1.0 / (padded_dim as f32).sqrt();
            unit.into_iter().map(|c| c * sigma).collect()
        };

        Self {
            dim,
            padded_dim,
            mse_bits,
            flips,
            sketch,
            levels,
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bits per coordinate of the MSE scalar code (`total_bits − 1`; 0 = pure QJL).
    pub fn mse_bits(&self) -> usize {
        self.mse_bits
    }

    /// The `2^mse_bits` Lloyd–Max reconstruction levels (empty if `mse_bits == 0`).
    /// Exposed for the packed arena walk, which dequantizes codes through this table
    /// in its own kernel instead of going through [`TurboCode`].
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }

    /// `√(π/2)/d` — the QJL debias scale. One definition, shared by
    /// [`Self::estimate_ip`] and the packed arena walk, so the two paths cannot
    /// disagree on the constant.
    pub fn qjl_scale(&self) -> f32 {
        (std::f32::consts::PI / 2.0).sqrt() / self.dim as f32
    }

    /// Padded (power-of-two) dimension — the length of `TurboCode::idx` and of the
    /// rotated query [`PreparedQuery::pq`].
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Number of QJL sign bytes per code.
    fn qjl_bytes(&self) -> usize {
        self.dim.div_ceil(8)
    }

    /// `Π·x` — the structured rotation: `FHT_ROUNDS` rounds of (Rademacher sign flip →
    /// normalized FWHT), applied to the zero-padded input, in place. Orthonormal by
    /// construction: each diagonal is its own inverse and the normalized Hadamard is its
    /// own inverse, so [`Self::rotate_inv`] just replays the rounds backwards.
    fn rotate(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.padded_dim);
        let round_bytes = self.padded_dim.div_ceil(8);
        for k in 0..FHT_ROUNDS {
            let flips = &self.flips[k * round_bytes..(k + 1) * round_bytes];
            for (i, v) in x.iter_mut().enumerate() {
                if (flips[i / 8] >> (i % 8)) & 1 == 1 {
                    *v = -*v;
                }
            }
            fwht_normalized(x);
        }
    }

    /// `Πᵀ·y` — exact inverse of [`Self::rotate`]: rounds in reverse, FWHT first
    /// (self-inverse), then the same sign diagonal.
    fn rotate_inv(&self, y: &mut [f32]) {
        debug_assert_eq!(y.len(), self.padded_dim);
        let round_bytes = self.padded_dim.div_ceil(8);
        for k in (0..FHT_ROUNDS).rev() {
            fwht_normalized(y);
            let flips = &self.flips[k * round_bytes..(k + 1) * round_bytes];
            for (i, v) in y.iter_mut().enumerate() {
                if (flips[i / 8] >> (i % 8)) & 1 == 1 {
                    *v = -*v;
                }
            }
        }
    }

    /// Zero-pad `x` to the padded dimension.
    fn pad(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.padded_dim];
        out[..x.len()].copy_from_slice(x);
        out
    }

    /// Encode a (ideally unit-norm) vector into a TurboQuant code.
    pub fn encode(&self, x: &[f32]) -> TurboCode {
        debug_assert_eq!(x.len(), self.dim);
        let d = self.dim;

        // MSE scalar code on the rotated padded vector y = Π·pad(x). The reconstruction
        // x̃_mse comes back through the exact inverse rotation; only its first `dim`
        // coordinates matter for the residual (the query is zero-padded too, so the
        // padded tail of x̃ never meets a nonzero query coordinate at estimate time).
        let (idx, x_mse) = if self.mse_bits == 0 {
            (Vec::new(), vec![0.0f32; d])
        } else {
            let mut y = self.pad(x);
            self.rotate(&mut y);
            let mut idx = vec![0u8; self.padded_dim];
            let mut y_hat = vec![0.0f32; self.padded_dim];
            for (j, &yj) in y.iter().enumerate() {
                let k = nearest_level(&self.levels, yj);
                idx[j] = k as u8;
                y_hat[j] = self.levels[k];
            }
            self.rotate_inv(&mut y_hat);
            y_hat.truncate(d);
            (idx, y_hat)
        };

        // Residual r = x − x̃_mse, and its 1-bit QJL sketch qjl = sign(S·r).
        let mut r = vec![0.0f32; d];
        let mut gamma_sq = 0.0f32;
        for ((ri, &xi), &mi) in r.iter_mut().zip(x).zip(&x_mse) {
            *ri = xi - mi;
            gamma_sq += (xi - mi) * (xi - mi);
        }
        let sr = matvec(&self.sketch, &r, d);
        let mut qjl = vec![0u8; self.qjl_bytes()];
        for (i, &v) in sr.iter().enumerate() {
            if v >= 0.0 {
                qjl[i / 8] |= 1 << (i % 8);
            }
        }

        TurboCode {
            idx,
            qjl,
            gamma: gamma_sq.sqrt(),
        }
    }

    /// Prepare a query once for repeated [`estimate_ip`](Self::estimate_ip) calls.
    /// `pq` is padded-length (rotated), `sq` is dim-length (sketched).
    pub fn prepare_query(&self, q: &[f32]) -> PreparedQuery {
        debug_assert_eq!(q.len(), self.dim);
        let mut pq = self.pad(q);
        self.rotate(&mut pq);
        PreparedQuery {
            pq,
            sq: matvec(&self.sketch, q, self.dim),
        }
    }

    /// Unbiased estimate of `⟨q, x⟩` for the encoded `x`.
    ///
    /// `⟨Πq, ỹ⟩ + γ · (√(π/2)/d) · ⟨Sq, qjl⟩`.
    pub fn estimate_ip(&self, query: &PreparedQuery, code: &TurboCode) -> f32 {
        // MSE term: Σⱼ (Πq)ⱼ · c_{idxⱼ}.
        let mut mse = 0.0f32;
        if self.mse_bits != 0 {
            for (&pqj, &k) in query.pq.iter().zip(&code.idx) {
                mse += pqj * self.levels[k as usize];
            }
        }

        // QJL term: Σᵣ (Sq)ᵣ · signᵣ, signᵣ = ±1 from the packed bit.
        let mut s = 0.0f32;
        for (r, &sqr) in query.sq.iter().enumerate() {
            let bit = (code.qjl[r / 8] >> (r % 8)) & 1;
            if bit == 1 {
                s += sqr;
            } else {
                s -= sqr;
            }
        }
        mse + code.gamma * self.qjl_scale() * s
    }

    /// Cosine distance estimate for unit-norm vectors: `1 − ⟨q, x⟩`.
    pub fn estimate_cosine_distance(&self, query: &PreparedQuery, code: &TurboCode) -> f32 {
        1.0 - self.estimate_ip(query, code)
    }
}

// ---- Codebook derivation: Lloyd–Max quantizer of the unit Gaussian --------

/// Derive the `n`-level MSE-optimal (Lloyd–Max) reconstruction levels for a
/// standard normal `N(0,1)`, in closed form.
///
/// Fixed-point iteration: decision boundaries are midpoints of adjacent levels,
/// and each level is the Gaussian conditional mean of its cell,
/// `E[X | t_{i-1} < X ≤ t_i] = (φ(t_{i-1}) − φ(t_i)) / (Φ(t_i) − Φ(t_{i-1}))`.
/// Both integrals are closed form, so no sampling is needed.
fn derive_gaussian_lloyd_max(n: usize) -> Vec<f32> {
    assert!(n >= 2 && n.is_power_of_two());
    // Initialise levels spread across roughly [-3, 3].
    let mut c: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f64 + 0.5) / n as f64)
        .collect();

    for _ in 0..200 {
        // Boundaries: t[0] = -inf, t[n] = +inf, interior = midpoints.
        let mut t = vec![f64::NEG_INFINITY; n + 1];
        t[n] = f64::INFINITY;
        for i in 1..n {
            t[i] = 0.5 * (c[i - 1] + c[i]);
        }
        let mut max_delta = 0.0f64;
        for i in 0..n {
            let (a, b) = (t[i], t[i + 1]);
            let mass = norm_cdf(b) - norm_cdf(a); // Φ(b) − Φ(a)
            let new_c = if mass > 1e-12 {
                (norm_pdf(a) - norm_pdf(b)) / mass // (φ(a) − φ(b)) / mass
            } else {
                c[i]
            };
            max_delta = max_delta.max((new_c - c[i]).abs());
            c[i] = new_c;
        }
        if max_delta < 1e-9 {
            break;
        }
    }
    c.into_iter().map(|v| v as f32).collect()
}

/// Index of the nearest level to `v` (levels are sorted ascending).
#[inline]
fn nearest_level(levels: &[f32], v: f32) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (k, &c) in levels.iter().enumerate() {
        let d = (v - c).abs();
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

// ---- Normal pdf/cdf (closed-form codebook math) ---------------------------

#[inline]
fn norm_pdf(x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[inline]
fn norm_cdf(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

// ---- Matrix helpers (self-contained; mirror rabitq.rs, each tested here) ---

/// `M · v` for row-major `d×d` `M`.
fn matvec(m: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    for (r, o) in out.iter_mut().enumerate() {
        *o = super::simd::dot_product_simd(&m[r * d..(r + 1) * d], v);
    }
    out
}

/// In-place normalized fast Walsh–Hadamard transform: `x ← H·x / √len`.
/// `len` must be a power of two. Self-inverse (H² = len·I, and the √len
/// normalization is applied once per call), which is what lets the inverse
/// rotation simply replay rounds backwards.
fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= scale;
    }
}

/// Row-major `dim×dim` matrix of i.i.d. `N(0,1)` entries, seeded.
fn gaussian_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim * dim).map(|_| gaussian(&mut rng)).collect()
}

/// Test-only reference dot (the shipped paths use `simd::dot_product_simd`).
#[cfg(test)]
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

    fn unit_vec(rng: &mut StdRng, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| gaussian(rng)).collect();
        let n = dot(&v, &v).sqrt();
        for x in &mut v {
            *x /= n;
        }
        v
    }

    /// The normalized FWHT is its own inverse — the property the inverse rotation
    /// (`rotate_inv` replaying rounds backwards) relies on.
    #[test]
    fn fwht_is_involutive() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut x: Vec<f32> = (0..256).map(|_| gaussian(&mut rng)).collect();
        let orig = x.clone();
        fwht_normalized(&mut x);
        assert!(
            x.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3),
            "transform must actually change the vector"
        );
        fwht_normalized(&mut x);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4, "H(Hx) != x: {a} vs {b}");
        }
    }

    /// The full structured rotation must be orthonormal: inner products (and hence
    /// norms) are preserved, and rotate_inv undoes rotate exactly — including at a
    /// NON-power-of-two dim, where the zero-padding path is live.
    #[test]
    fn rotation_is_orthonormal_and_invertible() {
        let mut rng = StdRng::seed_from_u64(8);
        for &d in &[64usize, 97] {
            let q = TurboQuantizer::with_seed(d, 3, 11);
            let a: Vec<f32> = (0..d).map(|_| gaussian(&mut rng)).collect();
            let b: Vec<f32> = (0..d).map(|_| gaussian(&mut rng)).collect();
            let ip = dot(&a, &b);

            let (mut ra, mut rb) = (q.pad(&a), q.pad(&b));
            q.rotate(&mut ra);
            q.rotate(&mut rb);
            let rip = dot(&ra, &rb);
            assert!(
                (rip - ip).abs() < 1e-3 * ip.abs().max(1.0),
                "d={d}: rotation not orthonormal: <Ra,Rb>={rip} vs <a,b>={ip}"
            );

            q.rotate_inv(&mut ra);
            for (i, (&got, &want)) in ra.iter().zip(a.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "d={d}: rotate_inv(rotate(a))[{i}] = {got}, want {want}"
                );
            }
            // The padded tail must come back to (near-)zero too.
            for (i, &got) in ra.iter().enumerate().skip(d) {
                assert!(got.abs() < 1e-4, "d={d}: padded tail [{i}] = {got}, want 0");
            }
        }
    }

    #[test]
    fn lloyd_max_reproduces_known_gaussian_levels() {
        // Textbook Max–Lloyd reconstruction levels for a unit Gaussian.
        let two = derive_gaussian_lloyd_max(2);
        assert!((two[0] + 0.7979).abs() < 1e-3, "got {two:?}");
        assert!((two[1] - 0.7979).abs() < 1e-3, "got {two:?}");

        let four = derive_gaussian_lloyd_max(4);
        let expected = [-1.5104, -0.4528, 0.4528, 1.5104];
        for (g, e) in four.iter().zip(expected) {
            assert!((g - e).abs() < 2e-3, "got {four:?}");
        }
    }

    #[test]
    fn estimator_is_approximately_unbiased() {
        // Unbiasedness is over the random choice of (Π, S). Fix one (q, x) pair,
        // average the estimate across many independent quantizers.
        let d = 64;
        let mut rng = StdRng::seed_from_u64(11);
        let x = unit_vec(&mut rng, d);
        let q = unit_vec(&mut rng, d);
        let truth = dot(&q, &x);

        let n = 400;
        let mut acc = 0.0f64;
        for s in 0..n {
            let quant = TurboQuantizer::with_seed(d, 2, 1000 + s as u64);
            let code = quant.encode(&x);
            let pq = quant.prepare_query(&q);
            acc += quant.estimate_ip(&pq, &code) as f64;
        }
        let mean = (acc / n as f64) as f32;
        assert!(
            (mean - truth).abs() < 0.02,
            "estimator biased: mean {mean} vs truth {truth}"
        );
    }

    #[test]
    fn held_out_recall_improves_with_bits() {
        // Held-out queries vs brute-force IP ground truth — NEVER self-retrieval.
        let d = 128;
        let (n, nq, k) = (600usize, 60usize, 10usize);
        let mut rng = StdRng::seed_from_u64(3);
        let base: Vec<Vec<f32>> = (0..n).map(|_| unit_vec(&mut rng, d)).collect();
        let queries: Vec<Vec<f32>> = (0..nq).map(|_| unit_vec(&mut rng, d)).collect();

        // Brute-force top-k by true inner product.
        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut ips: Vec<(usize, f32)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i, dot(q, x)))
                    .collect();
                ips.sort_by(|a, b| b.1.total_cmp(&a.1));
                ips.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect();

        let recall_at = |bits: usize| -> f32 {
            let quant = TurboQuantizer::new(d, bits);
            let codes: Vec<TurboCode> = base.iter().map(|x| quant.encode(x)).collect();
            let mut hit = 0usize;
            for (q, gt) in queries.iter().zip(&truth) {
                let pq = quant.prepare_query(q);
                let mut est: Vec<(usize, f32)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, quant.estimate_ip(&pq, c)))
                    .collect();
                est.sort_by(|a, b| b.1.total_cmp(&a.1));
                let top: std::collections::HashSet<usize> =
                    est.into_iter().take(k).map(|(i, _)| i).collect();
                hit += gt.iter().filter(|i| top.contains(i)).count();
            }
            hit as f32 / (nq * k) as f32
        };

        let r1 = recall_at(1); // pure QJL sign sketch
        let r2 = recall_at(2);
        let r4 = recall_at(4);
        // More bits ⇒ better recall, and 4-bit is genuinely useful on random unit data.
        assert!(r2 > r1, "b=2 ({r2}) should beat b=1 ({r1})");
        assert!(r4 > r2, "b=4 ({r4}) should beat b=2 ({r2})");
        assert!(r4 > 0.55, "b=4 recall too low: {r4}");
    }
}
