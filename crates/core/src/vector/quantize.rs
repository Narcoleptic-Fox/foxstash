//! Vector quantization for memory-efficient storage
//!
//! This module provides quantization methods to reduce memory footprint while
//! maintaining acceptable search quality:
//!
//! - **Scalar Quantization (SQ8)**: f32 → i8 (4x compression, ~95% recall)
//! - **Binary Quantization (BQ)**: f32 → bit (32x compression, ~85% recall)
//!
//! # Memory Comparison (1M vectors × 384 dims)
//!
//! | Format | Size | Compression |
//! |--------|------|-------------|
//! | f32    | 1.5 GB | 1x |
//! | int8   | 384 MB | 4x |
//! | binary | 48 MB  | 32x |
//!
//! # Usage
//!
//! ```
//! use foxstash_core::vector::quantize::{ScalarQuantizer, BinaryQuantizer, Quantizer};
//!
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//!
//! // Scalar quantization (4x compression)
//! let sq = ScalarQuantizer::fit(&[vector.clone()]);
//! let quantized = sq.quantize(&vector);
//! let reconstructed = sq.dequantize(&quantized);
//!
//! // Binary quantization (32x compression)
//! let bq = BinaryQuantizer::new(4);
//! let binary = bq.quantize(&vector);
//! ```

use pulp::Simd;
use serde::{Deserialize, Serialize};

/// Trait for vector quantization
pub trait Quantizer: Send + Sync {
    /// Quantized representation type
    type Quantized: Clone + Send + Sync;

    /// Quantize a vector
    fn quantize(&self, vector: &[f32]) -> Self::Quantized;

    /// Dequantize back to f32 (lossy)
    fn dequantize(&self, quantized: &Self::Quantized) -> Vec<f32>;

    /// Compute distance between quantized vectors (fast path)
    fn distance_quantized(&self, a: &Self::Quantized, b: &Self::Quantized) -> f32;

    /// Compute distance between f32 query and quantized vector (asymmetric)
    fn distance_asymmetric(&self, query: &[f32], quantized: &Self::Quantized) -> f32;
}

// ============================================================================
// Scalar Quantization (SQ8)
// ============================================================================

/// Scalar quantization parameters for a single dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizationParams {
    /// Minimum value in training data
    pub min: f32,
    /// Maximum value in training data
    pub max: f32,
    /// Scale factor: (max - min) / 255
    pub scale: f32,
}

impl ScalarQuantizationParams {
    /// Create parameters from min/max values
    pub fn new(min: f32, max: f32) -> Self {
        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        Self { min, max, scale }
    }

    /// Quantize a single value to u8
    #[inline]
    pub fn quantize_value(&self, value: f32) -> u8 {
        let normalized = (value - self.min) / self.scale;
        normalized.clamp(0.0, 255.0) as u8
    }

    /// Dequantize a u8 back to f32
    #[inline]
    pub fn dequantize_value(&self, quantized: u8) -> f32 {
        (quantized as f32) * self.scale + self.min
    }
}

/// Scalar quantization (SQ8): f32 → u8
///
/// Maps each dimension independently to [0, 255] range based on min/max values
/// from training data. Provides 4x memory reduction with minimal quality loss.
///
/// # Example
///
/// ```
/// use foxstash_core::vector::quantize::{ScalarQuantizer, Quantizer};
///
/// // Fit quantizer on training data
/// let training_vectors = vec![
///     vec![0.1, 0.5, 0.9],
///     vec![0.2, 0.4, 0.8],
///     vec![0.3, 0.6, 0.7],
/// ];
/// let quantizer = ScalarQuantizer::fit(&training_vectors);
///
/// // Quantize new vectors
/// let query = vec![0.15, 0.45, 0.85];
/// let quantized = quantizer.quantize(&query);
///
/// // Compute distance efficiently
/// let db_vec = quantizer.quantize(&training_vectors[0]);
/// let distance = quantizer.distance_quantized(&quantized, &db_vec);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    /// Per-dimension quantization parameters
    params: Vec<ScalarQuantizationParams>,
    /// Dimensionality
    dim: usize,
}

/// Quantized vector representation for SQ8
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizedVector {
    /// Quantized values (u8 per dimension)
    pub data: Vec<u8>,
}

impl ScalarQuantizer {
    /// Fit quantizer parameters from training vectors
    ///
    /// Computes min/max for each dimension across all training vectors.
    ///
    /// # Panics
    ///
    /// Panics if training vectors have inconsistent dimensions.
    pub fn fit(training_vectors: &[Vec<f32>]) -> Self {
        assert!(!training_vectors.is_empty(), "Need at least one training vector");

        let dim = training_vectors[0].len();
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for vector in training_vectors {
            assert_eq!(vector.len(), dim, "Inconsistent vector dimensions");
            for (i, &val) in vector.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        let params: Vec<_> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| ScalarQuantizationParams::new(min, max))
            .collect();

        Self { params, dim }
    }

    /// Create quantizer with known min/max bounds
    ///
    /// Use this when you know the expected value range (e.g., normalized embeddings).
    pub fn with_bounds(dim: usize, min: f32, max: f32) -> Self {
        let params = vec![ScalarQuantizationParams::new(min, max); dim];
        Self { params, dim }
    }

    /// Create quantizer for normalized vectors ([-1, 1] range)
    pub fn for_normalized(dim: usize) -> Self {
        Self::with_bounds(dim, -1.0, 1.0)
    }

    /// Get the dimensionality
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get quantization parameters for analysis
    pub fn params(&self) -> &[ScalarQuantizationParams] {
        &self.params
    }
}

impl Quantizer for ScalarQuantizer {
    type Quantized = ScalarQuantizedVector;

    fn quantize(&self, vector: &[f32]) -> Self::Quantized {
        debug_assert_eq!(vector.len(), self.dim);

        let data: Vec<u8> = vector
            .iter()
            .zip(self.params.iter())
            .map(|(&val, param)| param.quantize_value(val))
            .collect();

        ScalarQuantizedVector { data }
    }

    fn dequantize(&self, quantized: &Self::Quantized) -> Vec<f32> {
        quantized
            .data
            .iter()
            .zip(self.params.iter())
            .map(|(&val, param)| param.dequantize_value(val))
            .collect()
    }

    fn distance_quantized(&self, a: &Self::Quantized, b: &Self::Quantized) -> f32 {
        // L2 distance in quantized space (scaled)
        sq8_l2_distance_simd(&a.data, &b.data)
    }

    fn distance_asymmetric(&self, query: &[f32], quantized: &Self::Quantized) -> f32 {
        // Asymmetric distance: full precision query vs quantized database
        sq8_asymmetric_l2_distance_simd(query, &quantized.data, &self.params)
    }
}

// ============================================================================
// Binary Quantization (BQ)
// ============================================================================

/// Binary quantization: f32 → bit
///
/// Maps each dimension to a single bit based on sign (positive = 1, negative = 0).
/// Provides 32x memory reduction. Best used for initial filtering with reranking.
///
/// Distance is computed using Hamming distance (number of differing bits).
///
/// # Example
///
/// ```
/// use foxstash_core::vector::quantize::{BinaryQuantizer, Quantizer};
///
/// let quantizer = BinaryQuantizer::new(128);
///
/// let vec_a = vec![0.5; 128];  // All positive → all 1s
/// let vec_b = vec![-0.5; 128]; // All negative → all 0s
///
/// let qa = quantizer.quantize(&vec_a);
/// let qb = quantizer.quantize(&vec_b);
///
/// // Maximum Hamming distance (all bits differ)
/// let distance = quantizer.distance_quantized(&qa, &qb);
/// assert_eq!(distance, 128.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    dim: usize,
    /// Number of bytes needed to store dim bits
    byte_len: usize,
}

/// Quantized vector representation for binary quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizedVector {
    /// Packed bits (ceil(dim/8) bytes)
    pub data: Vec<u8>,
}

impl BinaryQuantizer {
    /// Create a binary quantizer for vectors of given dimension
    pub fn new(dim: usize) -> Self {
        let byte_len = (dim + 7) / 8; // Ceiling division
        Self { dim, byte_len }
    }

    /// Get the dimensionality
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the byte length of quantized vectors
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl Quantizer for BinaryQuantizer {
    type Quantized = BinaryQuantizedVector;

    fn quantize(&self, vector: &[f32]) -> Self::Quantized {
        debug_assert_eq!(vector.len(), self.dim);

        let mut data = vec![0u8; self.byte_len];

        for (i, &val) in vector.iter().enumerate() {
            if val >= 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        BinaryQuantizedVector { data }
    }

    fn dequantize(&self, quantized: &Self::Quantized) -> Vec<f32> {
        // Binary quantization is highly lossy - we can only recover sign
        let mut result = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            result[i] = if bit == 1 { 1.0 } else { -1.0 };
        }

        result
    }

    fn distance_quantized(&self, a: &Self::Quantized, b: &Self::Quantized) -> f32 {
        // Hamming distance (number of differing bits)
        hamming_distance_simd(&a.data, &b.data) as f32
    }

    fn distance_asymmetric(&self, query: &[f32], quantized: &Self::Quantized) -> f32 {
        // Asymmetric: count mismatches between query sign and quantized bits
        let mut mismatches = 0u32;

        for (i, &val) in query.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let quantized_bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            let query_bit = if val >= 0.0 { 1 } else { 0 };

            if quantized_bit != query_bit {
                mismatches += 1;
            }
        }

        mismatches as f32
    }
}

// ============================================================================
// SIMD-Accelerated Distance Functions
// ============================================================================

/// SIMD-accelerated L2 distance for SQ8 vectors
#[inline]
pub fn sq8_l2_distance_simd(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let simd = pulp::Arch::new();
    simd.dispatch(|| sq8_l2_distance_impl(simd, a, b))
}

/// Internal SQ8 L2 distance implementation
#[inline(always)]
fn sq8_l2_distance_impl(simd: pulp::Arch, a: &[u8], b: &[u8]) -> f32 {
    struct Sq8L2<'a> {
        a: &'a [u8],
        b: &'a [u8],
    }

    impl pulp::WithSimd for Sq8L2<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
            let mut sum_sq: u32 = 0;

            for (&av, &bv) in self.a.iter().zip(self.b.iter()) {
                let diff = (av as i32) - (bv as i32);
                sum_sq += (diff * diff) as u32;
            }

            (sum_sq as f32).sqrt()
        }
    }

    simd.dispatch(Sq8L2 { a, b })
}

/// SIMD-accelerated asymmetric L2 distance (f32 query vs SQ8 database)
#[inline]
pub fn sq8_asymmetric_l2_distance_simd(
    query: &[f32],
    quantized: &[u8],
    params: &[ScalarQuantizationParams],
) -> f32 {
    debug_assert_eq!(query.len(), quantized.len());
    debug_assert_eq!(query.len(), params.len());

    let simd = pulp::Arch::new();
    simd.dispatch(|| sq8_asymmetric_l2_impl(simd, query, quantized, params))
}

/// Internal asymmetric L2 implementation
#[inline(always)]
fn sq8_asymmetric_l2_impl(
    simd: pulp::Arch,
    query: &[f32],
    quantized: &[u8],
    params: &[ScalarQuantizationParams],
) -> f32 {
    struct AsymL2<'a> {
        query: &'a [f32],
        quantized: &'a [u8],
        params: &'a [ScalarQuantizationParams],
    }

    impl pulp::WithSimd for AsymL2<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
            let mut sum_sq: f32 = 0.0;

            for ((&q, &qv), param) in self.query.iter()
                .zip(self.quantized.iter())
                .zip(self.params.iter())
            {
                let dequantized = param.dequantize_value(qv);
                let diff = q - dequantized;
                sum_sq += diff * diff;
            }

            sum_sq.sqrt()
        }
    }

    simd.dispatch(AsymL2 { query, quantized, params })
}

/// SIMD-accelerated Hamming distance for binary vectors
#[inline]
pub fn hamming_distance_simd(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    let simd = pulp::Arch::new();
    simd.dispatch(|| hamming_distance_impl(simd, a, b))
}

/// Internal Hamming distance implementation
#[inline(always)]
fn hamming_distance_impl(simd: pulp::Arch, a: &[u8], b: &[u8]) -> u32 {
    struct Hamming<'a> {
        a: &'a [u8],
        b: &'a [u8],
    }

    impl pulp::WithSimd for Hamming<'_> {
        type Output = u32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
            // Process 8 bytes at a time using u64 for efficient popcount
            let mut distance = 0u32;

            // Process full u64 chunks
            let chunks = self.a.len() / 8;
            for i in 0..chunks {
                let offset = i * 8;
                let a_u64 = u64::from_le_bytes([
                    self.a[offset], self.a[offset + 1], self.a[offset + 2], self.a[offset + 3],
                    self.a[offset + 4], self.a[offset + 5], self.a[offset + 6], self.a[offset + 7],
                ]);
                let b_u64 = u64::from_le_bytes([
                    self.b[offset], self.b[offset + 1], self.b[offset + 2], self.b[offset + 3],
                    self.b[offset + 4], self.b[offset + 5], self.b[offset + 6], self.b[offset + 7],
                ]);
                distance += (a_u64 ^ b_u64).count_ones();
            }

            // Handle remainder bytes
            for i in (chunks * 8)..self.a.len() {
                distance += (self.a[i] ^ self.b[i]).count_ones();
            }

            distance
        }
    }

    simd.dispatch(Hamming { a, b })
}

/// Compute dot product between f32 query and binary quantized vector
///
/// This is useful for cosine similarity approximation with binary vectors.
/// Returns the number of positive query components that match positive bits
/// minus those that mismatch.
#[inline]
pub fn binary_dot_product(query: &[f32], quantized: &BinaryQuantizedVector, dim: usize) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..dim {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let bit = ((quantized.data[byte_idx] >> bit_idx) & 1) as f32;
        // Map bit: 0 → -1, 1 → +1
        let sign = bit * 2.0 - 1.0;
        sum += query[i] * sign;
    }

    sum
}

// ============================================================================
// Product Quantization (PQ) - For very large datasets
// ============================================================================

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizerConfig {
    /// Original vector dimension
    pub dim: usize,
    /// Number of subvectors
    pub num_subvectors: usize,
    /// Bits per subvector (number of centroids = 2^bits)
    pub bits_per_subvector: usize,
}

impl ProductQuantizerConfig {
    /// Create a default PQ config for given dimension
    ///
    /// Uses 8 subvectors with 8 bits (256 centroids) each by default.
    pub fn default_for_dim(dim: usize) -> Self {
        let num_subvectors = 8.min(dim);
        Self {
            dim,
            num_subvectors,
            bits_per_subvector: 8,
        }
    }

    /// Dimension of each subvector
    pub fn subvector_dim(&self) -> usize {
        self.dim / self.num_subvectors
    }

    /// Number of centroids per subvector
    pub fn num_centroids(&self) -> usize {
        1 << self.bits_per_subvector
    }

    /// Compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        self.num_subvectors * ((self.bits_per_subvector + 7) / 8)
    }
}

// Note: Full PQ implementation with k-means training would go here.
// For now, we provide the config structure for future expansion.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // ========================================================================
    // Scalar Quantization Tests
    // ========================================================================

    #[test]
    fn test_scalar_quantizer_fit() {
        let vectors = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.2, 0.3, 0.8],
            vec![0.1, 0.6, 0.9],
        ];

        let sq = ScalarQuantizer::fit(&vectors);
        assert_eq!(sq.dim(), 3);

        // Check that params capture min/max
        assert!((sq.params[0].min - 0.0).abs() < EPSILON);
        assert!((sq.params[0].max - 0.2).abs() < EPSILON);
        assert!((sq.params[2].min - 0.8).abs() < EPSILON);
        assert!((sq.params[2].max - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_scalar_quantizer_roundtrip() {
        let vectors = vec![
            vec![-1.0, 0.0, 1.0],
            vec![-0.5, 0.5, 0.5],
        ];

        let sq = ScalarQuantizer::fit(&vectors);
        let original = vec![-0.7, 0.3, 0.8];
        let quantized = sq.quantize(&original);
        let reconstructed = sq.dequantize(&quantized);

        // Should be close but not exact (quantization is lossy)
        for (o, r) in original.iter().zip(reconstructed.iter()) {
            assert!((o - r).abs() < 0.02, "orig={}, recon={}", o, r);
        }
    }

    #[test]
    fn test_scalar_quantizer_for_normalized() {
        let sq = ScalarQuantizer::for_normalized(384);
        assert_eq!(sq.dim(), 384);

        // Test with normalized vector
        let vector: Vec<f32> = (0..384).map(|i| (i as f32 / 192.0) - 1.0).collect();
        let quantized = sq.quantize(&vector);
        let reconstructed = sq.dequantize(&quantized);

        // Should be close
        let max_error: f32 = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(max_error < 0.01, "Max error: {}", max_error);
    }

    #[test]
    fn test_sq8_distance_quantized() {
        let sq = ScalarQuantizer::for_normalized(4);

        let a = vec![1.0, 0.0, -1.0, 0.5];
        let b = vec![1.0, 0.0, -1.0, 0.5];

        let qa = sq.quantize(&a);
        let qb = sq.quantize(&b);

        let dist = sq.distance_quantized(&qa, &qb);
        assert!(dist < 1.0, "Same vectors should have near-zero distance");
    }

    #[test]
    fn test_sq8_distance_different() {
        let sq = ScalarQuantizer::for_normalized(4);

        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];

        let qa = sq.quantize(&a);
        let qb = sq.quantize(&b);

        let dist = sq.distance_quantized(&qa, &qb);
        assert!(dist > 100.0, "Opposite vectors should have large distance");
    }

    // ========================================================================
    // Binary Quantization Tests
    // ========================================================================

    #[test]
    fn test_binary_quantizer_basic() {
        let bq = BinaryQuantizer::new(8);
        assert_eq!(bq.dim(), 8);
        assert_eq!(bq.byte_len(), 1);
    }

    #[test]
    fn test_binary_quantizer_byte_len() {
        // Test various dimensions
        assert_eq!(BinaryQuantizer::new(1).byte_len(), 1);
        assert_eq!(BinaryQuantizer::new(8).byte_len(), 1);
        assert_eq!(BinaryQuantizer::new(9).byte_len(), 2);
        assert_eq!(BinaryQuantizer::new(16).byte_len(), 2);
        assert_eq!(BinaryQuantizer::new(384).byte_len(), 48);
    }

    #[test]
    fn test_binary_quantizer_all_positive() {
        let bq = BinaryQuantizer::new(8);
        let vector = vec![0.5, 0.3, 0.1, 0.9, 0.2, 0.4, 0.6, 0.8];
        let quantized = bq.quantize(&vector);

        // All positive → all bits set
        assert_eq!(quantized.data[0], 0xFF);
    }

    #[test]
    fn test_binary_quantizer_all_negative() {
        let bq = BinaryQuantizer::new(8);
        let vector = vec![-0.5, -0.3, -0.1, -0.9, -0.2, -0.4, -0.6, -0.8];
        let quantized = bq.quantize(&vector);

        // All negative → no bits set
        assert_eq!(quantized.data[0], 0x00);
    }

    #[test]
    fn test_binary_quantizer_mixed() {
        let bq = BinaryQuantizer::new(8);
        let vector = vec![0.5, -0.3, 0.1, -0.9, 0.2, -0.4, 0.6, -0.8];
        // Positive at indices: 0, 2, 4, 6 → bits 0, 2, 4, 6 → 0b01010101
        let quantized = bq.quantize(&vector);
        assert_eq!(quantized.data[0], 0b01010101);
    }

    #[test]
    fn test_binary_hamming_distance() {
        let bq = BinaryQuantizer::new(8);

        let a = vec![1.0; 8];   // All positive
        let b = vec![-1.0; 8];  // All negative

        let qa = bq.quantize(&a);
        let qb = bq.quantize(&b);

        let dist = bq.distance_quantized(&qa, &qb);
        assert_eq!(dist, 8.0); // All 8 bits differ
    }

    #[test]
    fn test_binary_hamming_same() {
        let bq = BinaryQuantizer::new(16);

        let a = vec![0.5, -0.3, 0.1, -0.9, 0.2, -0.4, 0.6, -0.8,
                     0.5, -0.3, 0.1, -0.9, 0.2, -0.4, 0.6, -0.8];

        let qa = bq.quantize(&a);
        let qb = bq.quantize(&a);

        let dist = bq.distance_quantized(&qa, &qb);
        assert_eq!(dist, 0.0); // Same vector → zero distance
    }

    #[test]
    fn test_binary_dequantize() {
        let bq = BinaryQuantizer::new(4);
        let vector = vec![0.5, -0.3, 0.1, -0.9];
        let quantized = bq.quantize(&vector);
        let dequantized = bq.dequantize(&quantized);

        // Dequantize only recovers sign
        assert_eq!(dequantized, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_binary_large_dimension() {
        let bq = BinaryQuantizer::new(384);
        let vector: Vec<f32> = (0..384)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();

        let quantized = bq.quantize(&vector);
        assert_eq!(quantized.data.len(), 48);

        let dequantized = bq.dequantize(&quantized);
        for (i, &val) in dequantized.iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(val, expected);
        }
    }

    // ========================================================================
    // SIMD Function Tests
    // ========================================================================

    #[test]
    fn test_hamming_distance_simd_basic() {
        let a = vec![0b11110000u8, 0b10101010];
        let b = vec![0b00001111u8, 0b10101010];

        let dist = hamming_distance_simd(&a, &b);
        // First byte: all 8 bits differ
        // Second byte: 0 bits differ
        assert_eq!(dist, 8);
    }

    #[test]
    fn test_hamming_distance_simd_same() {
        let a = vec![0xFF, 0x00, 0xAB, 0xCD];
        let b = a.clone();

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_sq8_l2_distance_simd_basic() {
        let a = vec![0u8, 50, 100, 150, 200, 250];
        let b = vec![0u8, 50, 100, 150, 200, 250];

        let dist = sq8_l2_distance_simd(&a, &b);
        assert!(dist < EPSILON);
    }

    #[test]
    fn test_sq8_l2_distance_simd_different() {
        let a = vec![0u8, 0, 0, 0];
        let b = vec![255u8, 255, 255, 255];

        let dist = sq8_l2_distance_simd(&a, &b);
        // Expected: sqrt(255^2 * 4) = sqrt(260100) ≈ 510
        assert!((dist - 510.0).abs() < 1.0);
    }

    // ========================================================================
    // Product Quantization Config Tests
    // ========================================================================

    #[test]
    fn test_pq_config_defaults() {
        let config = ProductQuantizerConfig::default_for_dim(384);

        assert_eq!(config.dim, 384);
        assert_eq!(config.num_subvectors, 8);
        assert_eq!(config.bits_per_subvector, 8);
        assert_eq!(config.subvector_dim(), 48);
        assert_eq!(config.num_centroids(), 256);
        assert_eq!(config.compressed_size(), 8); // 8 bytes for 8 subvectors × 8 bits
    }

    // ========================================================================
    // Accuracy Tests
    // ========================================================================

    #[test]
    fn test_sq8_recall_approximation() {
        // Generate random vectors and test that SQ8 preserves ordering
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let dim = 128;
        let num_vectors = 100;

        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect())
            .collect();

        let sq = ScalarQuantizer::fit(&vectors);
        let quantized: Vec<_> = vectors.iter().map(|v| sq.quantize(v)).collect();

        // Pick a random query
        let query_idx = 42;
        let query = &vectors[query_idx];
        let query_q = &quantized[query_idx];

        // Find nearest neighbors using exact and quantized distance
        let mut exact_distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_idx)
            .map(|(i, v)| {
                let dist: f32 = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        let mut quantized_distances: Vec<(usize, f32)> = quantized
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_idx)
            .map(|(i, q)| (i, sq.distance_quantized(query_q, q)))
            .collect();

        exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        quantized_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check recall@10: how many of top-10 exact neighbors are in top-10 quantized
        let exact_top10: std::collections::HashSet<_> =
            exact_distances[..10].iter().map(|(i, _)| *i).collect();
        let quantized_top10: std::collections::HashSet<_> =
            quantized_distances[..10].iter().map(|(i, _)| *i).collect();

        let recall = exact_top10.intersection(&quantized_top10).count();
        // SQ8 should have at least 70% recall@10
        assert!(recall >= 7, "Recall@10: {}/10", recall);
    }

    #[test]
    fn test_binary_recall_approximation() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        let dim = 128;
        let num_vectors = 100;

        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect())
            .collect();

        let bq = BinaryQuantizer::new(dim);
        let quantized: Vec<_> = vectors.iter().map(|v| bq.quantize(v)).collect();

        let query_idx = 42;
        let query = &vectors[query_idx];
        let query_q = &quantized[query_idx];

        // Find nearest using exact cosine and quantized hamming
        let mut exact_distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_idx)
            .map(|(i, v)| {
                let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cosine = dot / (norm_q * norm_v);
                (i, 1.0 - cosine) // Convert to distance
            })
            .collect();

        let mut quantized_distances: Vec<(usize, f32)> = quantized
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_idx)
            .map(|(i, q)| (i, bq.distance_quantized(query_q, q)))
            .collect();

        exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        quantized_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Binary should have at least 50% recall@10 (it's more lossy)
        let exact_top10: std::collections::HashSet<_> =
            exact_distances[..10].iter().map(|(i, _)| *i).collect();
        let quantized_top10: std::collections::HashSet<_> =
            quantized_distances[..10].iter().map(|(i, _)| *i).collect();

        let recall = exact_top10.intersection(&quantized_top10).count();
        assert!(recall >= 5, "Binary recall@10: {}/10", recall);
    }
}
