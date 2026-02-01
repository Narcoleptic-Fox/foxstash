//! Vector operations and utilities for RAG system
//!
//! This module provides high-performance vector operations essential for
//! similarity search and embedding manipulation in the RAG system.
//!
//! # Core Operations
//!
//! - **Similarity Metrics**: Cosine similarity for measuring vector similarity
//! - **Distance Metrics**: L2 (Euclidean) distance for spatial relationships
//! - **Vector Algebra**: Dot product, normalization, and comparison operations
//!
//! # Performance Characteristics
//!
//! All operations in this module are optimized for hot-path performance:
//! - Functions are marked with inline hints for small vectors
//! - Efficient iterator usage for auto-vectorization
//! - Minimal allocations and cache-friendly access patterns
//! - SIMD acceleration with runtime CPU detection (3-4x speedup)
//!
//! # SIMD Acceleration
//!
//! The module automatically uses SIMD instructions when available:
//! - **x86_64**: AVX2, SSE2, or scalar fallback
//! - **ARM**: NEON or scalar fallback
//! - Runtime detection ensures optimal performance on any CPU
//!
//! Use the `*_auto()` functions for automatic SIMD/scalar selection:
//! ```
//! use foxstash_core::vector::{cosine_similarity_auto, dot_product_auto};
//!
//! let a = vec![1.0; 384];
//! let b = vec![2.0; 384];
//!
//! // Automatically uses SIMD if available
//! let similarity = cosine_similarity_auto(&a, &b).unwrap();
//! let dot = dot_product_auto(&a, &b).unwrap();
//! ```
//!
//! # Usage
//!
//! ```
//! use foxstash_core::vector::ops::{cosine_similarity, normalize};
//!
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! normalize(&mut embedding);
//!
//! let query = vec![0.6, 0.8, 0.0];
//! let similarity = cosine_similarity(&embedding, &query).unwrap();
//! ```

pub mod ops;
pub mod quantize;
pub mod simd;

use crate::{RagError, Result};

// Re-export commonly used functions
pub use ops::{
    approx_equal,
    cosine_similarity,
    dot_product,
    l2_distance,
    normalize,
};

// Re-export SIMD functions
pub use simd::{
    cosine_similarity_simd,
    dot_product_simd,
    l2_distance_simd,
};

/// Automatically selects between SIMD and scalar cosine similarity.
///
/// This function uses runtime CPU detection to choose the fastest available
/// implementation. On x86_64 with AVX2 or ARM with NEON, it uses SIMD
/// acceleration for 3-4x speedup. Otherwise, it falls back to scalar operations.
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
/// # Errors
///
/// Returns `RagError::DimensionMismatch` if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::cosine_similarity_auto;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
/// let similarity = cosine_similarity_auto(&a, &b).unwrap();
/// assert!((similarity - 0.0).abs() < 1e-5);
/// ```
#[inline]
pub fn cosine_similarity_auto(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    // Always use SIMD - pulp handles runtime detection and fallback
    Ok(simd::cosine_similarity_simd(a, b))
}

/// Automatically selects between SIMD and scalar L2 distance.
///
/// This function uses runtime CPU detection to choose the fastest available
/// implementation.
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
/// # Errors
///
/// Returns `RagError::DimensionMismatch` if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::l2_distance_auto;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let distance = l2_distance_auto(&a, &b).unwrap();
/// assert!((distance - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn l2_distance_auto(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(simd::l2_distance_simd(a, b))
}

/// Automatically selects between SIMD and scalar dot product.
///
/// This function uses runtime CPU detection to choose the fastest available
/// implementation.
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
/// Returns `RagError::DimensionMismatch` if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use foxstash_core::vector::dot_product_auto;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let product = dot_product_auto(&a, &b).unwrap();
/// assert!((product - 32.0).abs() < 1e-5);
/// ```
#[inline]
pub fn dot_product_auto(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RagError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(simd::dot_product_simd(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_auto_functions_match_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        // Test cosine similarity
        let auto_sim = cosine_similarity_auto(&a, &b).unwrap();
        let scalar_sim = cosine_similarity(&a, &b).unwrap();
        assert!((auto_sim - scalar_sim).abs() < EPSILON);

        // Test L2 distance
        let auto_dist = l2_distance_auto(&a, &b).unwrap();
        let scalar_dist = l2_distance(&a, &b).unwrap();
        assert!((auto_dist - scalar_dist).abs() < EPSILON);

        // Test dot product
        let auto_dot = dot_product_auto(&a, &b).unwrap();
        let scalar_dot = dot_product(&a, &b).unwrap();
        assert!((auto_dot - scalar_dot).abs() < EPSILON);
    }

    #[test]
    fn test_auto_functions_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert!(matches!(
            cosine_similarity_auto(&a, &b),
            Err(RagError::DimensionMismatch { .. })
        ));

        assert!(matches!(
            l2_distance_auto(&a, &b),
            Err(RagError::DimensionMismatch { .. })
        ));

        assert!(matches!(
            dot_product_auto(&a, &b),
            Err(RagError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_auto_functions_typical_embeddings() {
        // Test with typical embedding sizes
        for size in [384, 768, 1024] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
            let b: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) / (size as f32)).collect();

            let auto_sim = cosine_similarity_auto(&a, &b).unwrap();
            let scalar_sim = cosine_similarity(&a, &b).unwrap();
            assert!(
                (auto_sim - scalar_sim).abs() < 1e-4,  // Relaxed for large vectors
                "Size {}: auto={}, scalar={}",
                size, auto_sim, scalar_sim
            );

            let auto_dist = l2_distance_auto(&a, &b).unwrap();
            let scalar_dist = l2_distance(&a, &b).unwrap();
            // Use relative epsilon for distance
            let epsilon = scalar_dist.abs() * 1e-5;
            assert!(
                (auto_dist - scalar_dist).abs() < epsilon,
                "Size {}: auto={}, scalar={}",
                size, auto_dist, scalar_dist
            );
        }
    }
}
