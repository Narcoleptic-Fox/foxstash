//! Vector operations for RAG system
//!
//! This module provides high-performance vector operations for similarity search
//! and embedding manipulation. All functions are optimized for hot-path performance.

use crate::{RagError, Result};

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity measures the cosine of the angle between two vectors,
/// returning a value in the range [-1, 1] where:
/// - 1.0 indicates identical direction
/// - 0.0 indicates orthogonal vectors
/// - -1.0 indicates opposite direction
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Returns the cosine similarity score in the range [-1, 1].
///
/// # Errors
///
/// Returns `RagError::DimensionMismatch` if the vectors have different dimensions.
///
/// # Performance
///
/// This function is optimized for hot-path performance and should be inlined
/// in most cases. For large batches, consider pre-normalizing vectors and using
/// dot product directly.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::ops::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
/// let similarity = cosine_similarity(&a, &b).unwrap();
/// assert!((similarity - 0.0).abs() < 1e-6);
/// ```
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    if a.is_empty() {
        return Ok(1.0); // Convention: empty vectors are maximally similar
    }

    let dot = dot_product_unchecked(a, b);
    let norm_a = magnitude(a);
    let norm_b = magnitude(b);

    // Handle zero vectors
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    // Clamp to [-1, 1] to handle numerical errors
    let similarity = dot / (norm_a * norm_b);
    Ok(similarity.clamp(-1.0, 1.0))
}

/// Computes the Euclidean (L2) distance between two vectors.
///
/// The L2 distance is the straight-line distance between two points in
/// Euclidean space, calculated as: sqrt(sum((a\[i\] - b\[i\])^2))
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Returns the non-negative Euclidean distance.
///
/// # Errors
///
/// Returns `RagError::DimensionMismatch` if the vectors have different dimensions.
///
/// # Performance
///
/// This function uses optimized iterators and should be inlined for small vectors.
/// For distance-based sorting, consider using squared distance to avoid the sqrt.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::ops::l2_distance;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let distance = l2_distance(&a, &b).unwrap();
/// assert!((distance - 5.0).abs() < 1e-6);
/// ```
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let squared_sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    Ok(squared_sum.sqrt())
}

/// Computes the dot product of two vectors.
///
/// The dot product is the sum of element-wise products: sum(a\[i\] * b\[i\])
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Returns the dot product as a scalar value.
///
/// # Errors
///
/// Returns `RagError::DimensionMismatch` if the vectors have different dimensions.
///
/// # Performance
///
/// This is a critical hot-path function. The implementation uses optimized
/// iteration and will benefit from auto-vectorization on most platforms.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::ops::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let product = dot_product(&a, &b).unwrap();
/// assert!((product - 32.0).abs() < 1e-6);
/// ```
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(dot_product_unchecked(a, b))
}

/// Normalizes a vector to unit length (L2 norm = 1).
///
/// This operation modifies the vector in-place, scaling all components
/// so that the resulting vector has a magnitude of 1.0.
///
/// # Arguments
///
/// * `vector` - Mutable reference to the vector to normalize
///
/// # Behavior
///
/// - If the vector has zero magnitude, it remains unchanged
/// - Empty vectors remain unchanged
///
/// # Performance
///
/// This function performs two passes over the data (magnitude calculation
/// and scaling). For better performance when normalizing many vectors,
/// consider batching or using SIMD operations.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::ops::normalize;
///
/// let mut v = vec![3.0, 4.0];
/// normalize(&mut v);
/// assert!((v[0] - 0.6).abs() < 1e-6);
/// assert!((v[1] - 0.8).abs() < 1e-6);
/// ```
#[inline]
pub fn normalize(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    let norm = magnitude(vector);

    if norm == 0.0 {
        return; // Don't modify zero vectors
    }

    let inv_norm = 1.0 / norm;
    for x in vector.iter_mut() {
        *x *= inv_norm;
    }
}

/// Checks if two vectors are approximately equal within a tolerance.
///
/// This function performs element-wise comparison with the specified epsilon
/// value to account for floating-point precision issues.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
/// * `epsilon` - Maximum allowed difference per component
///
/// # Returns
///
/// Returns `true` if all corresponding elements differ by at most `epsilon`,
/// `false` otherwise. Returns `false` if dimensions don't match.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::ops::approx_equal;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0001, 2.0001, 3.0001];
/// assert!(approx_equal(&a, &b, 0.001));
/// assert!(!approx_equal(&a, &b, 0.00001));
/// ```
#[inline]
pub fn approx_equal(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= epsilon)
}

/// Internal helper: computes dot product without dimension checking.
///
/// # Safety
///
/// Caller must ensure vectors have the same length.
#[inline(always)]
fn dot_product_unchecked(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Internal helper: computes the L2 magnitude (Euclidean norm) of a vector.
///
/// Returns sqrt(sum(x\[i\]^2))
#[inline(always)]
fn magnitude(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        // Pre-normalized vectors
        let a = vec![0.6, 0.8];
        let b = vec![0.8, 0.6];
        let similarity = cosine_similarity(&a, &b).unwrap();
        let expected = 0.6 * 0.8 + 0.8 * 0.6; // = 0.96
        assert!((similarity - expected).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let distance = l2_distance(&a, &b).unwrap();
        assert!((distance - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_unit() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let distance = l2_distance(&a, &b).unwrap();
        assert!((distance - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_pythagorean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = l2_distance(&a, &b).unwrap();
        assert!((distance - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = l2_distance(&a, &b);
        assert!(matches!(result, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_l2_distance_negative_values() {
        let a = vec![-1.0, -1.0];
        let b = vec![1.0, 1.0];
        let distance = l2_distance(&a, &b).unwrap();
        let expected = (8.0_f32).sqrt(); // sqrt(4 + 4)
        assert!((distance - expected).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_high_dimension() {
        let a = vec![1.0; 384]; // Typical embedding dimension
        let b = vec![2.0; 384];
        let distance = l2_distance(&a, &b).unwrap();
        let expected = (384.0_f32).sqrt(); // sqrt(384 * 1.0)
        assert!((distance - expected).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_positive() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let product = dot_product(&a, &b).unwrap();
        assert!((product - 32.0).abs() < EPSILON); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let product = dot_product(&a, &b).unwrap();
        assert!((product - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_negative() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let product = dot_product(&a, &b).unwrap();
        assert!((product - (-5.0)).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = dot_product(&a, &b);
        assert!(matches!(result, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let product = dot_product(&a, &b).unwrap();
        assert!((product - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        normalize(&mut v);
        assert!((v[0] - 1.0).abs() < EPSILON);
        assert!((v[1] - 0.0).abs() < EPSILON);
        assert!((v[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_standard() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < EPSILON);
        assert!((v[1] - 0.8).abs() < EPSILON);

        // Verify it's actually unit length
        let magnitude = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        assert!((v[0] - 0.0).abs() < EPSILON);
        assert!((v[1] - 0.0).abs() < EPSILON);
        assert!((v[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_negative_values() {
        let mut v = vec![-3.0, -4.0];
        normalize(&mut v);
        assert!((v[0] - (-0.6)).abs() < EPSILON);
        assert!((v[1] - (-0.8)).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_empty() {
        let mut v: Vec<f32> = vec![];
        normalize(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn test_normalize_high_dimension() {
        let mut v = vec![1.0; 384];
        normalize(&mut v);

        // Each component should be 1/sqrt(384)
        let expected = 1.0 / (384.0_f32).sqrt();
        for &val in &v {
            assert!((val - expected).abs() < EPSILON);
        }

        // Verify unit length (use larger epsilon for accumulated error)
        let magnitude = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-5); // Relaxed epsilon for high dimensions
    }

    #[test]
    fn test_approx_equal_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(approx_equal(&a, &b, EPSILON));
    }

    #[test]
    fn test_approx_equal_within_epsilon() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert!(approx_equal(&a, &b, 0.001));
    }

    #[test]
    fn test_approx_equal_outside_epsilon() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert!(!approx_equal(&a, &b, 0.00001));
    }

    #[test]
    fn test_approx_equal_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(!approx_equal(&a, &b, EPSILON));
    }

    #[test]
    fn test_approx_equal_negative_values() {
        let a = vec![-1.0, -2.0];
        let b = vec![-1.0001, -2.0001];
        assert!(approx_equal(&a, &b, 0.001));
    }

    #[test]
    fn test_approx_equal_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(approx_equal(&a, &b, EPSILON));
    }

    #[test]
    fn test_approx_equal_zero_epsilon() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(approx_equal(&a, &b, 0.0));

        let c = vec![1.0001, 2.0];
        assert!(!approx_equal(&a, &c, 0.0));
    }

    // Integration tests
    #[test]
    fn test_normalized_vectors_cosine_similarity() {
        let mut a = vec![3.0, 4.0];
        let mut b = vec![5.0, 12.0];

        normalize(&mut a);
        normalize(&mut b);

        // For normalized vectors, cosine similarity equals dot product
        let similarity = cosine_similarity(&a, &b).unwrap();
        let dot = dot_product(&a, &b).unwrap();

        assert!((similarity - dot).abs() < EPSILON);
    }

    #[test]
    fn test_l2_distance_relationship_to_cosine() {
        // For unit vectors: L2^2 = 2(1 - cosine_similarity)
        let mut a = vec![1.0, 2.0, 3.0];
        let mut b = vec![4.0, 5.0, 6.0];

        normalize(&mut a);
        normalize(&mut b);

        let similarity = cosine_similarity(&a, &b).unwrap();
        let distance = l2_distance(&a, &b).unwrap();
        let expected_distance_squared = 2.0 * (1.0 - similarity);

        assert!((distance * distance - expected_distance_squared).abs() < EPSILON);
    }

    #[test]
    fn test_performance_typical_embedding() {
        // Test with realistic embedding dimensions (MiniLM-L6-v2: 384)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let b: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) / 384.0).collect();

        let _similarity = cosine_similarity(&a, &b).unwrap();
        let _distance = l2_distance(&a, &b).unwrap();
        let _dot = dot_product(&a, &b).unwrap();
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let a = vec![1e6, 2e6, 3e6];
        let b = vec![4e6, 5e6, 6e6];

        let similarity = cosine_similarity(&a, &b).unwrap();

        // Should still be in valid range despite large values
        assert!((-1.0..=1.0).contains(&similarity));
    }

    #[test]
    fn test_numerical_stability_small_values() {
        let a = vec![1e-6, 2e-6, 3e-6];
        let b = vec![4e-6, 5e-6, 6e-6];

        let similarity = cosine_similarity(&a, &b).unwrap();

        // Should still compute valid similarity
        assert!((-1.0..=1.0).contains(&similarity));
    }
}
