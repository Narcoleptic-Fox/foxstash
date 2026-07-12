//! SIMD-accelerated vector operations
//!
//! This module provides high-performance SIMD implementations of vector operations
//! for x86_64 (AVX2, SSE) and ARM (NEON) architectures. The implementations use
//! the `pulp` crate for portable SIMD abstraction with runtime CPU detection.
//!
//! # Performance
//!
//! SIMD implementations use AVX2, SSE, or NEON vector instructions depending on
//! platform and runtime CPU detection. Performance depends on:
//! - Vector length (longer vectors benefit more)
//! - CPU architecture and SIMD support
//! - Memory alignment and cache behavior
//!
//! # Architecture Support
//!
//! - **x86_64**: AVX2 (8x f32), SSE (4x f32), scalar fallback
//! - **ARM**: NEON (4x f32), scalar fallback
//! - **Other**: Scalar fallback
//!
//! # Usage
//!
//! ```
//! use foxstash_core::vector::simd::{dot_product_simd, cosine_similarity_simd};
//!
//! let a = vec![1.0; 384];
//! let b = vec![2.0; 384];
//!
//! let dot = dot_product_simd(&a, &b);
//! let similarity = cosine_similarity_simd(&a, &b);
//! ```

use pulp::Simd;

/// Computes dot product using SIMD acceleration.
///
/// This function automatically detects and uses the best available SIMD
/// instruction set (AVX2, SSE, NEON, or scalar fallback).
///
/// # Arguments
///
/// * `a` - First vector (must have same length as `b`)
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Returns the dot product as a scalar f32 value.
///
/// # Panics
///
/// Panics if vectors have different lengths (use checked version for safety).
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::simd::dot_product_simd;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = dot_product_simd(&a, &b);
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let simd = pulp::Arch::new();

    simd.dispatch(|| dot_product_simd_impl(simd, a, b))
}

/// Computes L2 (Euclidean) distance using SIMD acceleration.
///
/// Calculates: `sqrt(sum((a[i] - b[i])^2))`
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Returns the non-negative L2 distance.
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::simd::l2_distance_simd;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let distance = l2_distance_simd(&a, &b);
/// assert!((distance - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let simd = pulp::Arch::new();

    simd.dispatch(|| l2_distance_simd_impl(simd, a, b))
}

/// Squared L2 distance — the same kernel as [`l2_distance_simd`] without the final `sqrt`.
///
/// `sqrt` is monotonic, so squared L2 induces exactly the same *ordering* as L2. Nearest-
/// neighbour search only ever compares distances, so the square root is pure overhead in
/// the inner loop: a SIFT query at `ef_search=500` computes ~8,500 distances, and every
/// one of those was paying for a root nobody reads.
///
/// Use this for ranking; take the root only on the handful of results you return.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::simd::l2_squared_distance_simd;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// assert!((l2_squared_distance_simd(&a, &b) - 25.0).abs() < 1e-5); // 5^2
/// ```
#[inline]
pub fn l2_squared_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let simd = pulp::Arch::new();

    simd.dispatch(|| l2_squared_distance_simd_impl(simd, a, b))
}

/// Computes cosine similarity using SIMD acceleration.
///
/// Calculates: dot(a, b) / (||a|| * ||b||)
///
/// Returns a value in [-1, 1] where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Returns cosine similarity in range [-1, 1].
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::simd::cosine_similarity_simd;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
/// let similarity = cosine_similarity_simd(&a, &b);
/// assert!((similarity - 0.0).abs() < 1e-5);
/// ```
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    if a.is_empty() {
        return 1.0; // Convention: empty vectors are maximally similar
    }

    let simd = pulp::Arch::new();

    simd.dispatch(|| {
        let dot = dot_product_simd_impl(simd, a, b);
        let norm_a = magnitude_simd_impl(simd, a);
        let norm_b = magnitude_simd_impl(simd, b);

        // Handle zero vectors
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        // Compute similarity and clamp to [-1, 1] to handle numerical errors
        let similarity = dot / (norm_a * norm_b);
        similarity.clamp(-1.0, 1.0)
    })
}

/// Computes the L2 norm (magnitude) of a vector using SIMD acceleration.
///
/// Returns `sqrt(sum(v[i]^2))`.
#[inline]
pub fn norm_simd(v: &[f32]) -> f32 {
    let simd = pulp::Arch::new();
    simd.dispatch(Magnitude { vector: v })
}

/// Computes cosine distance with a precomputed norm for vector `b`.
///
/// This is the fused hot-path: a single `dispatch` call with two SIMD
/// accumulators (dot product + norm_a²) in one pass over the data.
/// The caller supplies `norm_b` (precomputed and cached per stored vector).
///
/// Returns `1.0 - dot(a,b) / (||a|| * norm_b)`, i.e. cosine distance in [0, 2].
#[inline]
pub fn cosine_distance_prenorm(a: &[f32], b: &[f32], norm_b: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    if norm_b == 0.0 {
        return 1.0;
    }

    let simd = pulp::Arch::new();
    simd.dispatch(FusedCosineDistance { a, b, norm_b })
}

/// Fused cosine distance: single SIMD pass computing dot(a,b) and ||a||² simultaneously.
struct FusedCosineDistance<'a> {
    a: &'a [f32],
    b: &'a [f32],
    norm_b: f32,
}

impl pulp::WithSimd for FusedCosineDistance<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let a = self.a;
        let b = self.b;
        let norm_b = self.norm_b;
        let (a_chunks, a_tail) = S::as_simd_f32s(a);
        let (b_chunks, b_tail) = S::as_simd_f32s(b);

        let mut dot_acc = simd.splat_f32s(0.0);
        let mut norm_a_acc = simd.splat_f32s(0.0);
        for (&a_vec, &b_vec) in a_chunks.iter().zip(b_chunks.iter()) {
            dot_acc = simd.mul_add_e_f32s(a_vec, b_vec, dot_acc);
            norm_a_acc = simd.mul_add_e_f32s(a_vec, a_vec, norm_a_acc);
        }

        let mut dot = simd.reduce_sum_f32s(dot_acc);
        let mut norm_a_sq = simd.reduce_sum_f32s(norm_a_acc);

        debug_assert_eq!(a_tail.len(), b_tail.len());
        for (&a_scalar, &b_scalar) in a_tail.iter().zip(b_tail.iter()) {
            dot += a_scalar * b_scalar;
            norm_a_sq += a_scalar * a_scalar;
        }

        let norm_a = norm_a_sq.sqrt();
        if norm_a == 0.0 {
            return 1.0;
        }

        let similarity = dot / (norm_a * norm_b);
        1.0 - similarity.clamp(-1.0, 1.0)
    }
}

/// Internal implementation of dot product with SIMD.
///
/// This function is generic over SIMD architecture and will use the best
/// available instruction set at runtime.
#[inline(always)]
fn dot_product_simd_impl(simd: pulp::Arch, a: &[f32], b: &[f32]) -> f32 {
    struct DotProduct<'a> {
        a: &'a [f32],
        b: &'a [f32],
    }

    impl pulp::WithSimd for DotProduct<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.a;
            let b = self.b;
            let (a_chunks, a_tail) = S::as_simd_f32s(a);
            let (b_chunks, b_tail) = S::as_simd_f32s(b);

            let mut sum = simd.splat_f32s(0.0);
            for (&a_vec, &b_vec) in a_chunks.iter().zip(b_chunks.iter()) {
                sum = simd.mul_add_e_f32s(a_vec, b_vec, sum);
            }

            let mut result = simd.reduce_sum_f32s(sum);
            debug_assert_eq!(a_tail.len(), b_tail.len());
            for (&a_scalar, &b_scalar) in a_tail.iter().zip(b_tail.iter()) {
                result += a_scalar * b_scalar;
            }

            result
        }
    }

    simd.dispatch(DotProduct { a, b })
}

/// Internal implementation of L2 distance with SIMD.
#[inline(always)]
fn l2_distance_simd_impl(simd: pulp::Arch, a: &[f32], b: &[f32]) -> f32 {
    struct L2Distance<'a> {
        a: &'a [f32],
        b: &'a [f32],
    }

    impl pulp::WithSimd for L2Distance<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.a;
            let b = self.b;
            let (a_chunks, a_tail) = S::as_simd_f32s(a);
            let (b_chunks, b_tail) = S::as_simd_f32s(b);

            let mut sum_squares = simd.splat_f32s(0.0);
            for (&a_vec, &b_vec) in a_chunks.iter().zip(b_chunks.iter()) {
                let diff = simd.sub_f32s(a_vec, b_vec);
                sum_squares = simd.mul_add_e_f32s(diff, diff, sum_squares);
            }

            let mut result = simd.reduce_sum_f32s(sum_squares);
            debug_assert_eq!(a_tail.len(), b_tail.len());
            for (&a_scalar, &b_scalar) in a_tail.iter().zip(b_tail.iter()) {
                let diff = a_scalar - b_scalar;
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    simd.dispatch(L2Distance { a, b })
}

#[inline(always)]
fn l2_squared_distance_simd_impl(simd: pulp::Arch, a: &[f32], b: &[f32]) -> f32 {
    struct L2Squared<'a> {
        a: &'a [f32],
        b: &'a [f32],
    }

    impl pulp::WithSimd for L2Squared<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.a;
            let b = self.b;
            let (a_chunks, a_tail) = S::as_simd_f32s(a);
            let (b_chunks, b_tail) = S::as_simd_f32s(b);

            let mut sum_squares = simd.splat_f32s(0.0);
            for (&a_vec, &b_vec) in a_chunks.iter().zip(b_chunks.iter()) {
                let diff = simd.sub_f32s(a_vec, b_vec);
                sum_squares = simd.mul_add_e_f32s(diff, diff, sum_squares);
            }

            let mut result = simd.reduce_sum_f32s(sum_squares);
            debug_assert_eq!(a_tail.len(), b_tail.len());
            for (&a_scalar, &b_scalar) in a_tail.iter().zip(b_tail.iter()) {
                let diff = a_scalar - b_scalar;
                result += diff * diff;
            }

            result // no sqrt: monotonic in L2, and callers only compare
        }
    }

    simd.dispatch(L2Squared { a, b })
}

/// Vector magnitude WithSimd impl — used by both `magnitude_simd_impl` and `norm_simd`.
struct Magnitude<'a> {
    vector: &'a [f32],
}

impl pulp::WithSimd for Magnitude<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let vector = self.vector;
        let (chunks, tail) = S::as_simd_f32s(vector);

        let mut sum_squares = simd.splat_f32s(0.0);
        for &vector_chunk in chunks {
            sum_squares = simd.mul_add_e_f32s(vector_chunk, vector_chunk, sum_squares);
        }

        let mut result = simd.reduce_sum_f32s(sum_squares);

        for &value in tail {
            result += value * value;
        }

        result.sqrt()
    }
}

/// Internal implementation of vector magnitude with SIMD.
#[inline(always)]
fn magnitude_simd_impl(simd: pulp::Arch, vector: &[f32]) -> f32 {
    simd.dispatch(Magnitude { vector })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::ops::{cosine_similarity, dot_product, l2_distance};

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_dot_product_simd_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = dot_product_simd(&a, &b);
        let expected = dot_product(&a, &b).unwrap();

        assert!((result - expected).abs() < EPSILON);
        assert!((result - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_simd_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let result = dot_product_simd(&a, &b);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_simd_negative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];

        let result = dot_product_simd(&a, &b);
        let expected = dot_product(&a, &b).unwrap();

        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_simd_various_sizes() {
        // Test different sizes to verify remainder handling
        for size in [
            1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 383, 384, 767, 768,
        ] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

            let simd_result = dot_product_simd(&a, &b);
            let scalar_result = dot_product(&a, &b).unwrap();

            // Use relative epsilon for large results
            let epsilon = if scalar_result.abs() > 1000.0 {
                scalar_result.abs() * 1e-5 // 0.001% relative error
            } else {
                EPSILON
            };

            assert!(
                (simd_result - scalar_result).abs() < epsilon,
                "Size {}: SIMD={}, Scalar={}",
                size,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_dot_product_simd_misaligned_subslice_regression() {
        let size = 257;
        let a_storage: Vec<f32> = (0..(size + 3))
            .map(|i| ((i as f32) - 90.0) * 0.03125)
            .collect();
        let b_storage: Vec<f32> = (0..(size + 4))
            .map(|i| ((size + 4 - i) as f32 - 120.0) * 0.0625)
            .collect();

        let a = &a_storage[1..(1 + size)];
        let b = &b_storage[2..(2 + size)];

        let simd_result = dot_product_simd(a, b);
        let scalar_result = dot_product(a, b).unwrap();
        assert!((simd_result - scalar_result).abs() < 1e-4);
    }

    #[test]
    fn test_l2_distance_simd_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let result = l2_distance_simd(&a, &b);
        let expected = l2_distance(&a, &b).unwrap();

        assert!((result - expected).abs() < EPSILON);
        assert!((result - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_simd_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let result = l2_distance_simd(&a, &b);
        assert!(result < EPSILON);
    }

    #[test]
    fn test_l2_distance_simd_various_sizes() {
        for size in [
            1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 383, 384, 767, 768,
        ] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

            let simd_result = l2_distance_simd(&a, &b);
            let scalar_result = l2_distance(&a, &b).unwrap();

            // Use relative epsilon for large results
            let epsilon = if scalar_result.abs() > 1000.0 {
                scalar_result.abs() * 1e-5 // 0.001% relative error
            } else {
                EPSILON
            };

            assert!(
                (simd_result - scalar_result).abs() < epsilon,
                "Size {}: SIMD={}, Scalar={}",
                size,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_l2_distance_simd_misaligned_subslice_regression() {
        let size = 257;
        let a_storage: Vec<f32> = (0..(size + 3))
            .map(|i| ((i as f32) - 30.0) * 0.125)
            .collect();
        let b_storage: Vec<f32> = (0..(size + 4))
            .map(|i| ((i as f32) - 170.0) * -0.09375)
            .collect();

        let a = &a_storage[1..(1 + size)];
        let b = &b_storage[2..(2 + size)];

        let simd_result = l2_distance_simd(a, b);
        let scalar_result = l2_distance(a, b).unwrap();
        assert!((simd_result - scalar_result).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_similarity_simd_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let result = cosine_similarity_simd(&a, &b);
        let expected = cosine_similarity(&a, &b).unwrap();

        assert!((result - expected).abs() < EPSILON);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_simd_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let result = cosine_similarity_simd(&a, &b);
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_simd_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];

        let result = cosine_similarity_simd(&a, &b);
        assert!((result - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_simd_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        let result = cosine_similarity_simd(&a, &b);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_simd_various_sizes() {
        for size in [
            1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 383, 384, 767, 768, 1023, 1024,
        ] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
            let b: Vec<f32> = (0..size)
                .map(|i| 1.0 - (i as f32) / (size as f32))
                .collect();

            let simd_result = cosine_similarity_simd(&a, &b);
            let scalar_result = cosine_similarity(&a, &b).unwrap();

            // Cosine similarity is always in [-1, 1], but may have more rounding for large vectors
            let epsilon = if size > 100 { 1e-4 } else { EPSILON };

            assert!(
                (simd_result - scalar_result).abs() < epsilon,
                "Size {}: SIMD={}, Scalar={}",
                size,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_cosine_similarity_simd_misaligned_subslice_regression() {
        let size = 257;
        let a_storage: Vec<f32> = (0..(size + 3))
            .map(|i| (((i as f32) % 17.0) - 8.0) * 0.37)
            .collect();
        let b_storage: Vec<f32> = (0..(size + 4))
            .map(|i| (((i as f32) % 19.0) - 9.0) * -0.29)
            .collect();

        let a = &a_storage[1..(1 + size)];
        let b = &b_storage[2..(2 + size)];

        let simd_result = cosine_similarity_simd(a, b);
        let scalar_result = cosine_similarity(a, b).unwrap();
        assert!((simd_result - scalar_result).abs() < 1e-4);
    }

    #[test]
    fn test_simd_numerical_stability() {
        // Test with large values
        let a = vec![1e6; 384];
        let b = vec![2e6; 384];

        let simd_result = cosine_similarity_simd(&a, &b);
        let scalar_result = cosine_similarity(&a, &b).unwrap();

        assert!((simd_result - scalar_result).abs() < EPSILON);
        assert!((-1.0..=1.0).contains(&simd_result));

        // Test with small values
        let a = vec![1e-6; 384];
        let b = vec![2e-6; 384];

        let simd_result = cosine_similarity_simd(&a, &b);
        let scalar_result = cosine_similarity(&a, &b).unwrap();

        assert!((simd_result - scalar_result).abs() < EPSILON);
        assert!((-1.0..=1.0).contains(&simd_result));
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match")]
    fn test_dot_product_simd_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let _ = dot_product_simd(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match")]
    fn test_l2_distance_simd_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let _ = l2_distance_simd(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match")]
    fn test_cosine_similarity_simd_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let _ = cosine_similarity_simd(&a, &b);
    }

    #[test]
    fn test_norm_simd() {
        let v = vec![3.0, 4.0];
        let norm = norm_simd(&v);
        assert!((norm - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_distance_prenorm_matches_old() {
        // Compare fused prenorm distance against the original cosine_similarity_simd path
        for size in [3, 4, 8, 16, 32, 64, 128, 384, 768] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
            let b: Vec<f32> = (0..size)
                .map(|i| 1.0 - (i as f32) / (size as f32))
                .collect();

            let old_dist = 1.0 - cosine_similarity_simd(&a, &b);
            let norm_b = norm_simd(&b);
            let new_dist = cosine_distance_prenorm(&a, &b, norm_b);

            let epsilon = if size > 100 { 1e-4 } else { EPSILON };
            assert!(
                (old_dist - new_dist).abs() < epsilon,
                "Size {}: old={}, new={}",
                size,
                old_dist,
                new_dist
            );
        }
    }

    #[test]
    fn test_cosine_distance_prenorm_misaligned_subslice_regression() {
        let size = 257;
        let a_storage: Vec<f32> = (0..(size + 3))
            .map(|i| (((i as f32) % 13.0) - 6.0) * 0.41)
            .collect();
        let b_storage: Vec<f32> = (0..(size + 4))
            .map(|i| (((i as f32) % 11.0) - 5.0) * -0.23)
            .collect();

        let a = &a_storage[1..(1 + size)];
        let b = &b_storage[2..(2 + size)];
        let norm_b = norm_simd(b);

        let simd_result = cosine_distance_prenorm(a, b, norm_b);
        let scalar_result = 1.0 - cosine_similarity(a, b).unwrap();
        assert!((simd_result - scalar_result).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_distance_prenorm_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let norm_b = norm_simd(&b);
        // Zero query should return distance 1.0
        assert!((cosine_distance_prenorm(&a, &b, norm_b) - 1.0).abs() < EPSILON);
        // Zero stored vector (norm_b=0) should return distance 1.0
        assert!((cosine_distance_prenorm(&b, &a, 0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_distance_prenorm_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let norm_a = norm_simd(&a);
        let dist = cosine_distance_prenorm(&a, &a, norm_a);
        assert!(
            dist.abs() < EPSILON,
            "Identical vectors should have distance ~0, got {}",
            dist
        );
    }

    #[test]
    fn test_cosine_distance_prenorm_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let norm_b = norm_simd(&b);
        let dist = cosine_distance_prenorm(&a, &b, norm_b);
        assert!(
            (dist - 2.0).abs() < EPSILON,
            "Opposite vectors should have distance ~2, got {}",
            dist
        );
    }
}
