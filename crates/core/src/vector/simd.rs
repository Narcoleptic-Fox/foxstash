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

/// Asymmetric squared-L2 between an `f32` query and a node's 8-bit SQ8 codes.
///
/// "Asymmetric" means the query is **not** quantized: each database value is dequantized on
/// the fly (`min[d] + code * scale[d]`) and compared against the exact query component, so
/// quantization error enters on one side only. Quantizing both sides is cheaper per call and
/// strictly less accurate.
///
/// `min` and `scale` are **per dimension**, and must be. A single shared scale would let a
/// near-constant dimension's full 0-255 code swing weigh as much as a high-variance
/// dimension's — the bug that cost `SQ8HNSWIndex` 28 points of recall (commit 1df91b6).
///
/// The `u8 -> f32` widening is the entire cost of this kernel, and it is why the portable
/// `pulp` path is not used here: `pulp` operates on `f32` slices and cannot express the
/// widening, so a scalar version of this loop runs at ~124 ns per call against the f32 SIMD
/// path's ~87 ns — turning a bandwidth *saving* into a compute *regression*. AVX2 does the
/// widening in one instruction (`cvtepu8_epi32`), which is the only reason SQ8 traversal can
/// pay for itself at all.
#[inline]
pub fn sq8_asymmetric_l2_simd(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    debug_assert_eq!(query.len(), codes.len());
    debug_assert_eq!(query.len(), min.len());
    debug_assert_eq!(query.len(), scale.len());

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 widens 16 codes/iter (vs AVX2's 8) — the u8->f32 widening and the FMA both
        // double width. On Zen 4 (7840HS) this is the most-executed distance path under SQ8.
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature-detected; slices are equal length (checked) and the loop reads
            // at most `i + 16 <= n`, with a scalar tail for the remainder.
            return unsafe { sq8_asymmetric_l2_avx512(query, codes, min, scale) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection; all four slices are the same
            // length (checked above) and the loop never reads past `n - n % 8`.
            return unsafe { sq8_asymmetric_l2_avx2(query, codes, min, scale) };
        }
    }
    sq8_asymmetric_l2_scalar(query, codes, min, scale)
}

/// AVX-512 sibling of [`sq8_asymmetric_l2_avx2`]: 16 dims/iter. Widens 16 `u8` codes from a
/// 128-bit load via `cvtepu8_epi32`, dequantizes `min + code*scale`, accumulates `(q-deq)²`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq8_asymmetric_l2_avx512(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let c16 = _mm_loadu_si128(codes.as_ptr().add(i) as *const __m128i);
        let cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c16));
        let s = _mm512_loadu_ps(scale.as_ptr().add(i));
        let m = _mm512_loadu_ps(min.as_ptr().add(i));
        let q = _mm512_loadu_ps(query.as_ptr().add(i));
        let deq = _mm512_fmadd_ps(cf, s, m);
        let d = _mm512_sub_ps(q, deq);
        acc = _mm512_fmadd_ps(d, d, acc);
        i += 16;
    }
    let mut total = _mm512_reduce_add_ps(acc);
    for j in i..n {
        let deq = min[j] + codes[j] as f32 * scale[j];
        let d = query[j] - deq;
        total += d * d;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sq8_asymmetric_l2_avx2(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        // Widen 8 u8 codes to 8 f32 lanes — one instruction, and the reason this kernel
        // exists rather than a portable one.
        let c8 = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
        let c32 = _mm256_cvtepu8_epi32(c8);
        let cf = _mm256_cvtepi32_ps(c32);

        let s = _mm256_loadu_ps(scale.as_ptr().add(i));
        let m = _mm256_loadu_ps(min.as_ptr().add(i));
        let q = _mm256_loadu_ps(query.as_ptr().add(i));

        // deq = min + code * scale;  d = q - deq;  acc += d * d
        let deq = _mm256_fmadd_ps(cf, s, m);
        let d = _mm256_sub_ps(q, deq);
        acc = _mm256_fmadd_ps(d, d, acc);

        i += 8;
    }

    // Horizontal sum of the 8 lanes.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    for j in i..n {
        let deq = min[j] + codes[j] as f32 * scale[j];
        let d = query[j] - deq;
        total += d * d;
    }
    total
}

fn sq8_asymmetric_l2_scalar(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..codes.len() {
        let deq = min[i] + codes[i] as f32 * scale[i];
        let d = query[i] - deq;
        acc += d * d;
    }
    acc
}

/// Asymmetric dot product between an `f32` query and a node's 8-bit SQ8 codes.
///
/// Same shape as [`sq8_asymmetric_l2_simd`] — the query stays `f32`, each database value is
/// dequantized on the fly (`min[d] + code * scale[d]`) — but accumulates a product instead of
/// a squared difference. This is the building block for SQ8's cosine distance: the codes are
/// a metric-agnostic reconstruction of the original values (unlike RaBitQ's estimator, which
/// is folded specifically around squared L2 — see [`rabitq_asymmetric_l2_simd`]), so cosine
/// under SQ8 is just this dot product composed with the query's and the stored vector's norms
/// — both already available for free at the call site (the query's once per search, the
/// stored vector's from the node's own block, already read this visit) — rather than a new
/// per-vector encoding.
#[inline]
pub fn sq8_asymmetric_dot_simd(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    debug_assert_eq!(query.len(), codes.len());
    debug_assert_eq!(query.len(), min.len());
    debug_assert_eq!(query.len(), scale.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature-detected; equal-length slices, loop reads at most `i + 16 <= n`.
            return unsafe { sq8_asymmetric_dot_avx512(query, codes, min, scale) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection; all four slices are the same
            // length (checked above) and the loop never reads past `n - n % 8`.
            return unsafe { sq8_asymmetric_dot_avx2(query, codes, min, scale) };
        }
    }
    sq8_asymmetric_dot_scalar(query, codes, min, scale)
}

/// AVX-512 sibling of [`sq8_asymmetric_dot_avx2`]: 16 dims/iter, `q * (min + code*scale)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq8_asymmetric_dot_avx512(
    query: &[f32],
    codes: &[u8],
    min: &[f32],
    scale: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let c16 = _mm_loadu_si128(codes.as_ptr().add(i) as *const __m128i);
        let cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c16));
        let s = _mm512_loadu_ps(scale.as_ptr().add(i));
        let m = _mm512_loadu_ps(min.as_ptr().add(i));
        let q = _mm512_loadu_ps(query.as_ptr().add(i));
        let deq = _mm512_fmadd_ps(cf, s, m);
        acc = _mm512_fmadd_ps(q, deq, acc);
        i += 16;
    }
    let mut total = _mm512_reduce_add_ps(acc);
    for j in i..n {
        let deq = min[j] + codes[j] as f32 * scale[j];
        total += query[j] * deq;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sq8_asymmetric_dot_avx2(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = codes.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let c8 = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
        let c32 = _mm256_cvtepu8_epi32(c8);
        let cf = _mm256_cvtepi32_ps(c32);

        let s = _mm256_loadu_ps(scale.as_ptr().add(i));
        let m = _mm256_loadu_ps(min.as_ptr().add(i));
        let q = _mm256_loadu_ps(query.as_ptr().add(i));

        // deq = min + code * scale;  acc += q * deq
        let deq = _mm256_fmadd_ps(cf, s, m);
        acc = _mm256_fmadd_ps(q, deq, acc);

        i += 8;
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    for j in i..n {
        let deq = min[j] + codes[j] as f32 * scale[j];
        total += query[j] * deq;
    }
    total
}

fn sq8_asymmetric_dot_scalar(query: &[f32], codes: &[u8], min: &[f32], scale: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..codes.len() {
        let deq = min[i] + codes[i] as f32 * scale[i];
        acc += query[i] * deq;
    }
    acc
}

/// Asymmetric RaBitQ estimate of squared L2 between a prepared query and a node's 1-bit code.
///
/// "Asymmetric" in the same sense as [`sq8_asymmetric_l2_simd`]: the query stays `f32`
/// (already rotated into RaBitQ space by the caller — see
/// [`crate::vector::rabitq::RaBitQuantizer::prepare_query`]) and only the *database* side is
/// compressed, to 1 bit/dim.
///
/// This is the folded estimator from `crate::vector::rabitq` (Gao & Long, RaBitQ, SIGMOD
/// 2024), not Hamming distance: `S = Σ (2·bit − 1) · rq[i]`, then
/// `‖o − q‖² ≈ dtc_sq + qn_sq − 2 · est_factor · S`. `dtc_sq` (the vector's squared distance
/// to the corpus centroid) and `est_factor` (`dtc_sq / L1` of its rotated residual) are
/// per-vector scalars computed once at index-build time by
/// [`RaBitQuantizer::encode`](crate::vector::rabitq::RaBitQuantizer::encode); `rq` and
/// `qn_sq` are computed once per *query*, not per candidate — recomputing the rotation
/// (an O(dim²) matvec) on every node visit would turn an O(dim) distance into the very
/// cost this storage mode exists to avoid.
///
/// `bits` is packed 1 bit/dim, `bits[i/8]` bit `i%8`, matching
/// [`RaBitCode::bits`](crate::vector::rabitq::RaBitCode::bits)'s convention exactly — this
/// kernel does not call into `rabitq.rs` itself, so any packing mismatch here would silently
/// disagree with the derivation it claims to implement.
#[inline]
pub fn rabitq_asymmetric_l2_simd(
    rq: &[f32],
    bits: &[u8],
    dtc_sq: f32,
    est_factor: f32,
    qn_sq: f32,
) -> f32 {
    debug_assert_eq!(bits.len(), rq.len().div_ceil(8));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection; the loop never reads past
            // `rq.len() - rq.len() % 8`, and `bits` has at least `rq.len().div_ceil(8)`
            // bytes (checked above), which covers every byte index the loop computes.
            let s = unsafe { rabitq_signed_sum_avx2(rq, bits) };
            return (dtc_sq + qn_sq - 2.0 * est_factor * s).max(0.0);
        }
    }
    let s = rabitq_signed_sum_scalar(rq, bits);
    (dtc_sq + qn_sq - 2.0 * est_factor * s).max(0.0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rabitq_signed_sum_avx2(rq: &[f32], bits: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let n = rq.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    // Bit weights in lane order: lane 0 <-> bit 0 (value 1) .. lane 7 <-> bit 7 (value 128),
    // matching `bits[i/8] & (1 << (i % 8))`.
    let bit_masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_ps(1.0);
    let neg_ones = _mm256_set1_ps(-1.0);

    while i + 8 <= n {
        // One input byte covers exactly 8 dims — the same width as an AVX2 f32 lane.
        let byte = bits[i / 8] as i32;
        let byte_bcast = _mm256_set1_epi32(byte);
        let anded = _mm256_and_si256(byte_bcast, bit_masks);
        let is_set = _mm256_cmpgt_epi32(anded, zero);
        let signs = _mm256_blendv_ps(neg_ones, ones, _mm256_castsi256_ps(is_set));

        let rq_vec = _mm256_loadu_ps(rq.as_ptr().add(i));
        acc = _mm256_fmadd_ps(rq_vec, signs, acc);

        i += 8;
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    for j in i..n {
        let bit = (bits[j / 8] >> (j % 8)) & 1;
        total += if bit == 1 { rq[j] } else { -rq[j] };
    }
    total
}

fn rabitq_signed_sum_scalar(rq: &[f32], bits: &[u8]) -> f32 {
    let mut s = 0.0f32;
    for (i, &rqi) in rq.iter().enumerate() {
        let bit = (bits[i / 8] >> (i % 8)) & 1;
        s += if bit == 1 { rqi } else { -rqi };
    }
    s
}

/// `S = Σ (2·bitᵢ − 1) · rqᵢ` — the raw signed sum underneath
/// [`rabitq_asymmetric_l2_simd`], exposed on its own.
///
/// The multi-bit storages need this per **bit-plane**: a B-bit TurboRabit code
/// decomposes as `uᵢ + c_B = ½·Σₖ 2ᵏ·(2·bitₖ(i) − 1)`, so its packed estimator is
/// `B` calls to this one proven kernel (`dsq = dtc² + qn² + ½·f_rescale·Σₖ 2ᵏ·Sₖ`)
/// rather than a new unsafe path per storage mode. TurboQuant's QJL term is the
/// same shape over the sketched query. Packing convention is identical to
/// [`rabitq_asymmetric_l2_simd`]: `bits[i/8]` bit `i%8`.
#[inline]
pub fn rabitq_signed_sum(rq: &[f32], bits: &[u8]) -> f32 {
    debug_assert!(bits.len() >= rq.len().div_ceil(8));
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: same contract as the call in `rabitq_asymmetric_l2_simd` — feature
            // detection guards the ISA, and `bits` covers every byte the loop indexes.
            return unsafe { rabitq_signed_sum_avx2(rq, bits) };
        }
    }
    rabitq_signed_sum_scalar(rq, bits)
}

/// `Σ levels[nibbleᵢ] · pqᵢ` — dot of a nibble-packed code sequence, dequantized
/// through a ≤8-entry `f32` LUT, against a full-precision query.
///
/// This is TurboQuant's packed MSE term: its Lloyd–Max levels are affine in the
/// code bits only up to 2 MSE bits (the symmetric 4-level codebook happens to be
/// exactly affine; the 8-level one is not), so the bit-plane trick above cannot
/// dequantize them — a real gather is needed. With ≤3 MSE bits the whole codebook
/// fits one AVX2 register and `vpermd` does the gather in-register.
///
/// Packing: nibble `i` lives in `nibbles[i/2]`, low nibble for even `i` (LSB-first,
/// mirroring the bit convention of [`rabitq_signed_sum`]). `levels` may have fewer
/// than 8 entries; it is zero-padded internally, and every code must index within
/// the original length.
#[inline]
pub fn nibble_lut_dot_simd(pq: &[f32], nibbles: &[u8], levels: &[f32]) -> f32 {
    debug_assert!(nibbles.len() >= pq.len().div_ceil(2));
    debug_assert!(levels.len() <= 8, "LUT kernel supports at most 3-bit codes");
    let mut lut = [0.0f32; 8];
    lut[..levels.len()].copy_from_slice(levels);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature-detected; the loop reads 4 bytes at `i/2` only while
            // `i + 8 <= pq.len()`, and `nibbles.len() >= ceil(len/2) >= i/2 + 4`.
            return unsafe { nibble_lut_dot_avx2(pq, nibbles, &lut) };
        }
    }
    nibble_lut_dot_scalar(pq, nibbles, &lut)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nibble_lut_dot_avx2(pq: &[f32], nibbles: &[u8], lut: &[f32; 8]) -> f32 {
    use std::arch::x86_64::*;

    let n = pq.len();
    let mut acc = _mm256_setzero_ps();
    let lut_vec = _mm256_loadu_ps(lut.as_ptr());
    // Lane j holds dim i+j, whose nibble is nibble j of the u32 at byte i/2
    // (i is always even — it steps by 8). Little-endian: nibble j = (w >> 4j) & 0xF.
    let shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let nib_mask = _mm256_set1_epi32(0xF);

    let mut i = 0;
    while i + 8 <= n {
        let w = unsafe { (nibbles.as_ptr().add(i / 2) as *const u32).read_unaligned() };
        let idx = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(w as i32), shifts),
            nib_mask,
        );
        // vpermd uses only the low 3 bits of each lane, so idx ∈ 0..16 cannot read
        // out of register — codes ≥ levels.len() would gather a padded 0.0, and the
        // encoder never emits them (codes are < 2^mse_bits ≤ 8 by construction).
        let deq = _mm256_permutevar8x32_ps(lut_vec, idx);
        let pq_vec = _mm256_loadu_ps(pq.as_ptr().add(i));
        acc = _mm256_fmadd_ps(pq_vec, deq, acc);
        i += 8;
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    for j in i..n {
        let nib = (nibbles[j / 2] >> (4 * (j % 2))) & 0xF;
        total += lut[(nib & 0x7) as usize] * pq[j];
    }
    total
}

fn nibble_lut_dot_scalar(pq: &[f32], nibbles: &[u8], lut: &[f32; 8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &pqi) in pq.iter().enumerate() {
        let nib = (nibbles[i / 2] >> (4 * (i % 2))) & 0xF;
        acc += lut[(nib & 0x7) as usize] * pqi;
    }
    acc
}

/// `Σ uᵢ · rqᵢ` — dot of nibble-packed unsigned codes (`uᵢ ∈ 0..16`) against a
/// full-precision query, converting the code value directly (no LUT).
///
/// This is TurboRabit's FUSED kernel: the Extended-RaBitQ estimate is *linear in
/// the code value* (`dsq = dtc² + qn² + f_rescale·(⟨u,rq⟩ + c_B·Σrq)`), so unlike
/// TurboQuant's Lloyd–Max levels there is nothing to gather — one `cvtepi32_ps` +
/// one FMA per lane replaces the previous B bit-plane passes (B×dim FMAs and B
/// horizontal reductions collapse to dim FMAs and one). Packing convention is
/// identical to [`nibble_lut_dot_simd`]: nibble `i` in `nibbles[i/2]`, low nibble
/// for even `i`.
#[inline]
pub fn nibble_uint_dot_simd(rq: &[f32], nibbles: &[u8]) -> f32 {
    debug_assert!(nibbles.len() >= rq.len().div_ceil(2));

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 processes 16 dims/iter (512-bit) vs AVX2's 8; on Zen 4 (7840HS) the wider
        // FMA halves the iteration count of this ALU-bound kernel. Feature-gated with an AVX2
        // fallback for older cores.
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature-detected; reads 8 bytes at `i/2` only while `i + 16 <= len`,
            // and `nibbles.len() >= ceil(len/2) >= i/2 + 8` (checked above).
            return unsafe { nibble_uint_dot_avx512(rq, nibbles) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature-detected; reads 4 bytes at `i/2` only while `i + 8 <= len`,
            // and `nibbles.len() >= ceil(len/2) >= i/2 + 4` (checked above).
            return unsafe { nibble_uint_dot_avx2(rq, nibbles) };
        }
    }
    nibble_uint_dot_scalar(rq, nibbles)
}

/// AVX-512 sibling of [`nibble_uint_dot_avx2`]: 16 dims per iteration. The 16 nibbles for
/// `dims i..i+16` occupy 8 bytes = two `u32` words `w0` (dims i..i+8) and `w1` (dims i+8..i+16);
/// lanes 0..7 broadcast `w0`, lanes 8..15 broadcast `w1`, each right-shifted by its lane's nibble
/// offset then masked. Only AVX512F is required (no DQ/BW).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn nibble_uint_dot_avx512(rq: &[f32], nibbles: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let n = rq.len();
    let mut acc = _mm512_setzero_ps();
    // lane 0..7: shifts 0,4,..,28 applied to w0; lane 8..15: same applied to w1.
    // _mm512_set_epi32 takes lane 15 first, lane 0 last.
    let shifts = _mm512_set_epi32(28, 24, 20, 16, 12, 8, 4, 0, 28, 24, 20, 16, 12, 8, 4, 0);
    let nib_mask = _mm512_set1_epi32(0xF);

    let mut i = 0;
    while i + 16 <= n {
        let w0 = unsafe { (nibbles.as_ptr().add(i / 2) as *const u32).read_unaligned() } as i32;
        let w1 = unsafe { (nibbles.as_ptr().add(i / 2 + 4) as *const u32).read_unaligned() } as i32;
        let words = _mm512_set_epi32(
            w1, w1, w1, w1, w1, w1, w1, w1, w0, w0, w0, w0, w0, w0, w0, w0,
        );
        let u = _mm512_and_si512(_mm512_srlv_epi32(words, shifts), nib_mask);
        let uf = _mm512_cvtepi32_ps(u);
        let rq_vec = _mm512_loadu_ps(rq.as_ptr().add(i));
        acc = _mm512_fmadd_ps(rq_vec, uf, acc);
        i += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc);
    for j in i..n {
        let nib = (nibbles[j / 2] >> (4 * (j % 2))) & 0xF;
        total += nib as f32 * rq[j];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nibble_uint_dot_avx2(rq: &[f32], nibbles: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let n = rq.len();
    let mut acc = _mm256_setzero_ps();
    // Lane j holds dim i+j = nibble j of the u32 at byte i/2 (i steps by 8, so even).
    let shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let nib_mask = _mm256_set1_epi32(0xF);

    let mut i = 0;
    while i + 8 <= n {
        let w = unsafe { (nibbles.as_ptr().add(i / 2) as *const u32).read_unaligned() };
        let u = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(w as i32), shifts),
            nib_mask,
        );
        let uf = _mm256_cvtepi32_ps(u);
        let rq_vec = _mm256_loadu_ps(rq.as_ptr().add(i));
        acc = _mm256_fmadd_ps(rq_vec, uf, acc);
        i += 8;
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    for j in i..n {
        let nib = (nibbles[j / 2] >> (4 * (j % 2))) & 0xF;
        total += nib as f32 * rq[j];
    }
    total
}

fn nibble_uint_dot_scalar(rq: &[f32], nibbles: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &rqi) in rq.iter().enumerate() {
        let nib = (nibbles[i / 2] >> (4 * (i % 2))) & 0xF;
        acc += nib as f32 * rqi;
    }
    acc
}

#[cfg(test)]
mod rabitq_asymmetric_tests {
    use super::*;

    /// The AVX2 path and the scalar path must agree. A dim that is NOT a multiple of the
    /// 8-lane width exercises the tail loop, and NOT a multiple of 8 bits also forces a
    /// partial final byte in `bits`.
    #[test]
    fn avx2_matches_scalar() {
        let dim: usize = 131; // not a multiple of 8: exercises both the SIMD tail and a partial byte
        let rq: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.29).cos() * 7.0).collect();
        let n_bytes = dim.div_ceil(8);
        let bits: Vec<u8> = (0..n_bytes).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let dtc_sq = 4.5f32;
        let est_factor = 0.83f32;
        let qn_sq = 6.1f32;

        let dispatched = rabitq_asymmetric_l2_simd(&rq, &bits, dtc_sq, est_factor, qn_sq);

        let s = rabitq_signed_sum_scalar(&rq, &bits);
        let scalar = (dtc_sq + qn_sq - 2.0 * est_factor * s).max(0.0);

        let rel = (dispatched - scalar).abs() / scalar.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "AVX2 kernel disagrees with scalar: {dispatched} vs {scalar} (rel {rel:.2e})"
        );
    }

    /// Cross-check against `crate::vector::rabitq`'s own (allocating, reference)
    /// `estimate_dist_sq` — the kernel above reimplements that math without going through
    /// `RaBitCode`/`PreparedQuery`, so this is the check that the reimplementation didn't
    /// silently drift from the derivation it claims to match.
    #[test]
    fn matches_rabitq_module_reference_estimator() {
        use crate::vector::rabitq::RaBitQuantizer;
        use rand::{rngs::StdRng, RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(21);
        let dim = 97; // also not a multiple of 8
        let train: Vec<Vec<f32>> = (0..300)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let q = RaBitQuantizer::fit(&train);

        let o: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        let code = q.encode(&o);
        let prepared = q.prepare_query(&query);
        let reference = q.estimate_dist_sq(&prepared, &code);

        let kernel = rabitq_asymmetric_l2_simd(
            prepared.rq(),
            &code.bits,
            code.dtc_sq,
            code.est_factor,
            prepared.qn_sq(),
        );

        let rel = (kernel - reference).abs() / reference.abs().max(1e-6);
        assert!(
            rel < 1e-4,
            "kernel disagrees with rabitq.rs reference estimator: {kernel} vs {reference} (rel {rel:.2e})"
        );
    }
}

#[cfg(test)]
mod nibble_lut_dot_tests {
    use super::*;

    /// AVX2 and scalar paths must agree — odd dim exercises the tail loop AND a
    /// half-used final nibble byte; nibble values span the full 3-bit LUT range.
    #[test]
    fn avx2_matches_scalar() {
        let dim: usize = 131;
        let pq: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.31).sin() * 5.0).collect();
        let nibbles: Vec<u8> = (0..dim.div_ceil(2))
            .map(|i| {
                let lo = (i * 3 + 1) % 8;
                let hi = (i * 5 + 2) % 8;
                (lo | (hi << 4)) as u8
            })
            .collect();
        let levels: Vec<f32> = vec![-2.15, -1.34, -0.76, -0.24, 0.24, 0.76, 1.34, 2.15];

        let dispatched = nibble_lut_dot_simd(&pq, &nibbles, &levels);

        let mut lut = [0.0f32; 8];
        lut.copy_from_slice(&levels);
        let scalar = nibble_lut_dot_scalar(&pq, &nibbles, &lut);

        let rel = (dispatched - scalar).abs() / scalar.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "nibble LUT kernel disagrees with scalar: {dispatched} vs {scalar} (rel {rel:.2e})"
        );
    }

    /// AVX2 and scalar paths of the uint kernel must agree — odd dim exercises the
    /// tail loop and the half-used final nibble byte; codes span the full 0..16 range.
    #[test]
    fn uint_avx2_matches_scalar() {
        let dim: usize = 131;
        let rq: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos() * 4.0).collect();
        let nibbles: Vec<u8> = (0..dim.div_ceil(2))
            .map(|i| {
                let lo = (i * 7 + 3) % 16;
                let hi = (i * 11 + 5) % 16;
                (lo | (hi << 4)) as u8
            })
            .collect();
        let dispatched = nibble_uint_dot_simd(&rq, &nibbles);
        let scalar = nibble_uint_dot_scalar(&rq, &nibbles);
        let rel = (dispatched - scalar).abs() / scalar.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "uint kernel disagrees with scalar: {dispatched} vs {scalar} (rel {rel:.2e})"
        );
    }

    /// The AVX-512 uint kernel (16 dims/iter) must match scalar directly, not just via dispatch.
    /// Dims span the 16-lane boundary — 16 (exact), 17/31 (1- and 15-wide tails), 768 (the real
    /// embedding dim), and 131 (8 full blocks + 3 tail) — with codes across the full 0..16 range.
    #[test]
    fn uint_avx512_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable on this host; skipping direct AVX-512 check");
                return;
            }
            for dim in [16usize, 17, 31, 32, 47, 128, 131, 768] {
                let rq: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos() * 4.0).collect();
                let nibbles: Vec<u8> = (0..dim.div_ceil(2))
                    .map(|i| (((i * 7 + 3) % 16) | (((i * 11 + 5) % 16) << 4)) as u8)
                    .collect();
                // SAFETY: avx512f feature-detected just above.
                let avx512 = unsafe { nibble_uint_dot_avx512(&rq, &nibbles) };
                let scalar = nibble_uint_dot_scalar(&rq, &nibbles);
                let rel = (avx512 - scalar).abs() / scalar.abs().max(1e-6);
                assert!(
                    rel < 1e-5,
                    "dim {dim}: AVX-512 {avx512} vs scalar {scalar} (rel {rel:.2e})"
                );
            }
        }
    }

    /// A short LUT (2 levels = 1 MSE bit) must zero-pad, not read garbage.
    #[test]
    fn short_lut_is_padded() {
        let pq = vec![1.0f32; 16];
        let nibbles = vec![0x10u8; 8]; // alternating codes 0, 1
        let levels = vec![-0.8f32, 0.8];
        let got = nibble_lut_dot_simd(&pq, &nibbles, &levels);
        assert!(
            (got - 0.0).abs() < 1e-5,
            "8×(-0.8) + 8×0.8 should cancel, got {got}"
        );
    }
}

#[cfg(test)]
mod sq8_asymmetric_tests {
    use super::*;

    /// The AVX2 path and the scalar path must agree. A SIMD kernel that silently disagrees
    /// with its fallback produces results that depend on which machine you ran on.
    #[test]
    fn avx2_matches_scalar() {
        let dim = 131; // deliberately not a multiple of 8, to exercise the tail
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 12.0).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let min: Vec<f32> = (0..dim).map(|i| -(i as f32) * 0.11).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i % 5) as f32 * 0.03).collect();

        let dispatched = sq8_asymmetric_l2_simd(&query, &codes, &min, &scale);
        let scalar = sq8_asymmetric_l2_scalar(&query, &codes, &min, &scale);

        let rel = (dispatched - scalar).abs() / scalar.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "AVX2 kernel disagrees with scalar: {dispatched} vs {scalar} (rel {rel:.2e})"
        );
    }

    /// Same check for the dot-product kernel (SQ8's cosine building block), at a dim that is
    /// NOT a multiple of the 8-lane width.
    #[test]
    fn dot_avx2_matches_scalar() {
        let dim = 131;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 12.0).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let min: Vec<f32> = (0..dim).map(|i| -(i as f32) * 0.11).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i % 5) as f32 * 0.03).collect();

        let dispatched = sq8_asymmetric_dot_simd(&query, &codes, &min, &scale);
        let scalar = sq8_asymmetric_dot_scalar(&query, &codes, &min, &scale);

        let rel = (dispatched - scalar).abs() / scalar.abs().max(1e-6);
        assert!(
            rel < 1e-5,
            "AVX2 dot kernel disagrees with scalar: {dispatched} vs {scalar} (rel {rel:.2e})"
        );
    }

    /// The AVX-512 SQ8 kernels (16 dims/iter) must match scalar directly, across the 16-lane
    /// boundary — 16 (exact), 17/31 (1- and 15-wide tails), 768 (the real embedding dim).
    #[test]
    fn avx512_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable; skipping direct SQ8 AVX-512 check");
                return;
            }
            for dim in [16usize, 17, 31, 32, 131, 384, 768] {
                let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 12.0).collect();
                let codes: Vec<u8> = (0..dim).map(|i| ((i * 7 + 3) % 256) as u8).collect();
                let min: Vec<f32> = (0..dim).map(|i| -(i as f32) * 0.11).collect();
                let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i % 5) as f32 * 0.03).collect();
                // SAFETY: avx512f feature-detected just above.
                let l2 = unsafe { sq8_asymmetric_l2_avx512(&query, &codes, &min, &scale) };
                let l2s = sq8_asymmetric_l2_scalar(&query, &codes, &min, &scale);
                let dot = unsafe { sq8_asymmetric_dot_avx512(&query, &codes, &min, &scale) };
                let dots = sq8_asymmetric_dot_scalar(&query, &codes, &min, &scale);
                assert!(
                    (l2 - l2s).abs() / l2s.abs().max(1e-6) < 1e-5,
                    "dim {dim}: L2 AVX-512 {l2} vs scalar {l2s}"
                );
                assert!(
                    (dot - dots).abs() / dots.abs().max(1e-6) < 1e-5,
                    "dim {dim}: dot AVX-512 {dot} vs scalar {dots}"
                );
            }
        }
    }
}
