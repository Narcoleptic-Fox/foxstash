//! SIMD-accelerated vector operations
//!
//! This module provides high-performance SIMD implementations of vector operations
//! for x86_64 (AVX2, SSE) and ARM (NEON) architectures. The implementations use
//! the `pulp` crate for portable SIMD abstraction with runtime CPU detection.
//!
//! # Performance
//!
//! SIMD implementations provide 3-4x speedup over scalar operations for typical
//! embedding dimensions (384, 768, 1024). The exact speedup depends on:
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
/// Calculates: sqrt(sum((a[i] - b[i])^2))
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
            let n = a.len();

            // SIMD lane count (4 for SSE/NEON, 8 for AVX2)
            let lane_count = std::mem::size_of::<S::f32s>() / std::mem::size_of::<f32>();

            // Number of full SIMD chunks
            let simd_end = n - n % lane_count;

            // Process SIMD chunks
            let mut sum = simd.f32s_splat(0.0);

            let mut i = 0;
            while i < simd_end {
                // Load vectors from a and b
                let a_vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&a[i..]));
                let b_vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&b[i..]));

                // Multiply and accumulate
                sum = simd.f32s_mul_add_e(a_vec, b_vec, sum);

                i += lane_count;
            }

            // Horizontal sum of SIMD accumulator
            let mut result = simd.f32s_reduce_sum(sum);

            // Handle remainder elements
            for i in simd_end..n {
                result += a[i] * b[i];
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
            let n = a.len();

            let lane_count = std::mem::size_of::<S::f32s>() / std::mem::size_of::<f32>();
            let simd_end = n - n % lane_count;

            let mut sum_squares = simd.f32s_splat(0.0);

            let mut i = 0;
            while i < simd_end {
                let a_vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&a[i..]));
                let b_vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&b[i..]));

                // Compute difference
                let diff = simd.f32s_sub(a_vec, b_vec);

                // Multiply and accumulate: diff^2
                sum_squares = simd.f32s_mul_add_e(diff, diff, sum_squares);

                i += lane_count;
            }

            let mut result = simd.f32s_reduce_sum(sum_squares);

            // Handle remainder
            for i in simd_end..n {
                let diff = a[i] - b[i];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    simd.dispatch(L2Distance { a, b })
}

/// Internal implementation of vector magnitude with SIMD.
#[inline(always)]
fn magnitude_simd_impl(simd: pulp::Arch, vector: &[f32]) -> f32 {
    struct Magnitude<'a> {
        vector: &'a [f32],
    }

    impl pulp::WithSimd for Magnitude<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let vector = self.vector;
            let n = vector.len();

            let lane_count = std::mem::size_of::<S::f32s>() / std::mem::size_of::<f32>();
            let simd_end = n - n % lane_count;

            let mut sum_squares = simd.f32s_splat(0.0);

            let mut i = 0;
            while i < simd_end {
                let vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&vector[i..]));

                // Multiply and accumulate: vec^2
                sum_squares = simd.f32s_mul_add_e(vec, vec, sum_squares);

                i += lane_count;
            }

            let mut result = simd.f32s_reduce_sum(sum_squares);

            // Handle remainder
            for i in simd_end..n {
                result += vector[i] * vector[i];
            }

            result.sqrt()
        }
    }

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
}
