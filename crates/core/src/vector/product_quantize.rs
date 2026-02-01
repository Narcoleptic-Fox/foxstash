//! Product Quantization (PQ) for extreme compression
//!
//! Product Quantization achieves high compression ratios by:
//! 1. Splitting vectors into M subvectors
//! 2. Learning K centroids for each subspace via k-means
//! 3. Encoding each subvector as its nearest centroid index
//!
//! # Compression Example (384-dim, M=8, K=256)
//!
//! - Original: 384 × 4 bytes = 1536 bytes
//! - PQ encoded: 8 bytes (one u8 per subvector)
//! - Compression: **192x**
//!
//! # Search Methods
//!
//! - **Symmetric Distance Computation (SDC)**: Both query and DB vectors quantized
//! - **Asymmetric Distance Computation (ADC)**: Full precision query, quantized DB
//!   - ADC is more accurate and only slightly slower
//!
//! # Example
//!
//! ```
//! use foxstash_core::vector::product_quantize::{ProductQuantizer, PQConfig};
//!
//! // Configure PQ: 8 subvectors, 256 centroids each
//! let config = PQConfig::new(128, 8, 8); // dim, M, bits
//!
//! // Train on sample vectors (need enough for clustering)
//! let training_data: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| (0..128).map(|j| ((i * j) % 100) as f32 / 100.0).collect())
//!     .collect();
//! let pq = ProductQuantizer::train(&training_data, config).unwrap();
//!
//! // Encode vectors
//! let vector: Vec<f32> = vec![0.5; 128];
//! let codes = pq.encode(&vector);
//!
//! // Fast distance computation
//! let query: Vec<f32> = vec![0.6; 128];
//! let dist = pq.asymmetric_distance(&query, &codes);
//! ```

use crate::{RagError, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Product Quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Original vector dimension
    pub dim: usize,
    /// Number of subvectors (M)
    pub num_subvectors: usize,
    /// Bits per subvector code (typically 8 for 256 centroids)
    pub bits_per_code: usize,
    /// K-means iterations during training
    pub kmeans_iterations: usize,
    /// K-means initialization samples
    pub kmeans_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl PQConfig {
    /// Create a new PQ configuration
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (must be divisible by num_subvectors)
    /// * `num_subvectors` - Number of subvectors (M), typically 8-64
    /// * `bits_per_code` - Bits per code (8 = 256 centroids, 4 = 16 centroids)
    pub fn new(dim: usize, num_subvectors: usize, bits_per_code: usize) -> Self {
        Self {
            dim,
            num_subvectors,
            bits_per_code,
            kmeans_iterations: 25,
            kmeans_samples: 10_000,
            seed: None,
        }
    }

    /// Set k-means iterations
    pub fn with_kmeans_iterations(mut self, iterations: usize) -> Self {
        self.kmeans_iterations = iterations;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Dimension of each subvector
    pub fn subvector_dim(&self) -> usize {
        self.dim / self.num_subvectors
    }

    /// Number of centroids per subvector
    pub fn num_centroids(&self) -> usize {
        1 << self.bits_per_code
    }

    /// Size of encoded vector in bytes
    pub fn code_size(&self) -> usize {
        // For 8 bits, 1 byte per subvector
        // For 4 bits, pack 2 codes per byte
        (self.num_subvectors * self.bits_per_code + 7) / 8
    }

    /// Compression ratio compared to f32
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * 4) as f32 / self.code_size() as f32
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.dim % self.num_subvectors != 0 {
            return Err(RagError::IndexError(format!(
                "Dimension {} must be divisible by num_subvectors {}",
                self.dim, self.num_subvectors
            )));
        }
        if self.bits_per_code > 8 {
            return Err(RagError::IndexError(
                "bits_per_code must be <= 8".to_string()
            ));
        }
        Ok(())
    }
}

// ============================================================================
// PQ Codes
// ============================================================================

/// PQ-encoded vector (just the centroid indices)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCode {
    /// Centroid indices for each subvector
    pub codes: Vec<u8>,
}

impl PQCode {
    /// Create from raw codes
    pub fn new(codes: Vec<u8>) -> Self {
        Self { codes }
    }

    /// Get code for subvector m
    #[inline]
    pub fn get(&self, m: usize) -> u8 {
        self.codes[m]
    }
}

// ============================================================================
// Product Quantizer
// ============================================================================

/// Product Quantizer with trained codebooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    config: PQConfig,
    /// Codebooks: [M][K][D/M] - M subspaces, K centroids each, D/M dimensions
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Train a product quantizer on sample vectors
    ///
    /// # Arguments
    /// * `training_data` - Sample vectors for k-means clustering
    /// * `config` - PQ configuration
    ///
    /// # Returns
    /// Trained ProductQuantizer
    pub fn train(training_data: &[Vec<f32>], config: PQConfig) -> Result<Self> {
        config.validate()?;

        if training_data.is_empty() {
            return Err(RagError::IndexError("Training data is empty".to_string()));
        }

        let dim = config.dim;
        let m = config.num_subvectors;
        let k = config.num_centroids();
        let sub_dim = config.subvector_dim();

        // Validate training data dimensions
        for (_i, v) in training_data.iter().enumerate() {
            if v.len() != dim {
                return Err(RagError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }

        let mut rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Sample training data if too large
        let samples: Vec<&Vec<f32>> = if training_data.len() > config.kmeans_samples {
            training_data
                .choose_multiple(&mut rng, config.kmeans_samples)
                .collect()
        } else {
            training_data.iter().collect()
        };

        // Train codebook for each subspace
        let mut codebooks = Vec::with_capacity(m);

        for subspace in 0..m {
            let start = subspace * sub_dim;
            let end = start + sub_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = samples
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Run k-means
            let centroids = kmeans(&subvectors, k, config.kmeans_iterations, &mut rng)?;
            codebooks.push(centroids);
        }

        Ok(Self { config, codebooks })
    }

    /// Create from pre-computed codebooks
    pub fn from_codebooks(config: PQConfig, codebooks: Vec<Vec<Vec<f32>>>) -> Result<Self> {
        config.validate()?;

        if codebooks.len() != config.num_subvectors {
            return Err(RagError::IndexError(format!(
                "Expected {} codebooks, got {}",
                config.num_subvectors,
                codebooks.len()
            )));
        }

        Ok(Self { config, codebooks })
    }

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> PQCode {
        debug_assert_eq!(vector.len(), self.config.dim);

        let m = self.config.num_subvectors;
        let sub_dim = self.config.subvector_dim();
        let mut codes = Vec::with_capacity(m);

        for subspace in 0..m {
            let start = subspace * sub_dim;
            let subvector = &vector[start..start + sub_dim];

            // Find nearest centroid
            let code = self.find_nearest_centroid(subspace, subvector);
            codes.push(code);
        }

        PQCode::new(codes)
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, code: &PQCode) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.config.dim);

        for (subspace, &centroid_idx) in code.codes.iter().enumerate() {
            let centroid = &self.codebooks[subspace][centroid_idx as usize];
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Compute asymmetric L2 distance (full precision query vs PQ codes)
    ///
    /// This is more accurate than symmetric distance and uses precomputed
    /// distance tables for efficiency.
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], code: &PQCode) -> f32 {
        let sub_dim = self.config.subvector_dim();
        let mut dist_sq = 0.0f32;

        for (subspace, &centroid_idx) in code.codes.iter().enumerate() {
            let start = subspace * sub_dim;
            let query_sub = &query[start..start + sub_dim];
            let centroid = &self.codebooks[subspace][centroid_idx as usize];

            // L2 distance for this subspace
            for i in 0..sub_dim {
                let diff = query_sub[i] - centroid[i];
                dist_sq += diff * diff;
            }
        }

        dist_sq.sqrt()
    }

    /// Precompute distance table for a query (for batch search)
    ///
    /// Returns [M][K] table where table[m][k] = distance from query subvector m to centroid k
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let m = self.config.num_subvectors;
        let k = self.config.num_centroids();
        let sub_dim = self.config.subvector_dim();

        let mut table = vec![vec![0.0f32; k]; m];

        for subspace in 0..m {
            let start = subspace * sub_dim;
            let query_sub = &query[start..start + sub_dim];

            for centroid_idx in 0..k {
                let centroid = &self.codebooks[subspace][centroid_idx];
                let mut dist_sq = 0.0f32;

                for i in 0..sub_dim {
                    let diff = query_sub[i] - centroid[i];
                    dist_sq += diff * diff;
                }

                table[subspace][centroid_idx] = dist_sq;
            }
        }

        table
    }

    /// Fast distance using precomputed table
    #[inline]
    pub fn distance_with_table(&self, table: &[Vec<f32>], code: &PQCode) -> f32 {
        let mut dist_sq = 0.0f32;

        for (subspace, &centroid_idx) in code.codes.iter().enumerate() {
            dist_sq += table[subspace][centroid_idx as usize];
        }

        dist_sq.sqrt()
    }

    /// Symmetric distance between two PQ codes (fastest, least accurate)
    pub fn symmetric_distance(&self, a: &PQCode, b: &PQCode) -> f32 {
        let mut dist_sq = 0.0f32;

        for (subspace, (&code_a, &code_b)) in a.codes.iter().zip(b.codes.iter()).enumerate() {
            let centroid_a = &self.codebooks[subspace][code_a as usize];
            let centroid_b = &self.codebooks[subspace][code_b as usize];

            for i in 0..centroid_a.len() {
                let diff = centroid_a[i] - centroid_b[i];
                dist_sq += diff * diff;
            }
        }

        dist_sq.sqrt()
    }

    /// Get configuration
    pub fn config(&self) -> &PQConfig {
        &self.config
    }

    /// Get codebooks for analysis/serialization
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Quantization error for a vector (L2 distance to reconstruction)
    pub fn quantization_error(&self, vector: &[f32]) -> f32 {
        let code = self.encode(vector);
        let reconstructed = self.decode(&code);

        let mut error_sq = 0.0f32;
        for (a, b) in vector.iter().zip(reconstructed.iter()) {
            let diff = a - b;
            error_sq += diff * diff;
        }

        error_sq.sqrt()
    }

    fn find_nearest_centroid(&self, subspace: usize, subvector: &[f32]) -> u8 {
        let codebook = &self.codebooks[subspace];
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (idx, centroid) in codebook.iter().enumerate() {
            let mut dist_sq = 0.0f32;
            for i in 0..subvector.len() {
                let diff = subvector[i] - centroid[i];
                dist_sq += diff * diff;
            }

            if dist_sq < best_dist {
                best_dist = dist_sq;
                best_idx = idx;
            }
        }

        best_idx as u8
    }
}

// ============================================================================
// K-Means Clustering
// ============================================================================

/// Simple k-means clustering for codebook training
fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    iterations: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<Vec<Vec<f32>>> {
    if data.is_empty() {
        return Err(RagError::IndexError("Empty data for k-means".to_string()));
    }

    let dim = data[0].len();
    let n = data.len();

    // Initialize centroids with k-means++
    let mut centroids = kmeans_plusplus_init(data, k, rng);

    // Iterative refinement
    for _ in 0..iterations {
        // Assign points to nearest centroid
        let mut assignments = vec![0usize; n];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let mut best_centroid = 0;
            let mut best_dist = f32::INFINITY;

            for (c, centroid) in centroids.iter().enumerate() {
                let dist = l2_dist_sq(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = c;
                }
            }

            assignments[i] = best_centroid;
            counts[best_centroid] += 1;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];

        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            for j in 0..dim {
                new_centroids[c][j] += point[j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    new_centroids[c][j] /= counts[c] as f32;
                }
            } else {
                // Empty cluster - reinitialize randomly
                new_centroids[c] = data.choose(rng).unwrap().clone();
            }
        }

        centroids = new_centroids;
    }

    Ok(centroids)
}

/// K-means++ initialization
fn kmeans_plusplus_init(
    data: &[Vec<f32>],
    k: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<Vec<f32>> {
    use rand::Rng;

    let n = data.len();
    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.push(data[first_idx].clone());

    // Choose remaining centroids with probability proportional to D(x)^2
    let mut distances = vec![f32::INFINITY; n];

    for _ in 1..k {
        // Update distances to nearest centroid
        for (i, point) in data.iter().enumerate() {
            let dist = l2_dist_sq(point, centroids.last().unwrap());
            distances[i] = distances[i].min(dist);
        }

        // Sample proportional to D^2
        let total: f32 = distances.iter().sum();
        if total == 0.0 {
            // All points are centroids, pick randomly
            let idx = rng.gen_range(0..n);
            centroids.push(data[idx].clone());
            continue;
        }

        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0f32;
        let mut chosen_idx = 0;

        for (i, &dist) in distances.iter().enumerate() {
            cumsum += dist;
            if cumsum >= threshold {
                chosen_idx = i;
                break;
            }
        }

        centroids.push(data[chosen_idx].clone());
    }

    centroids
}

/// Squared L2 distance
#[inline]
fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ============================================================================
// Optimized PQ Index (for HNSW integration)
// ============================================================================

/// Precomputed centroid distances for fast symmetric search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQDistanceCache {
    /// [M][K][K] pairwise centroid distances per subspace
    distances: Vec<Vec<Vec<f32>>>,
}

impl PQDistanceCache {
    /// Build distance cache from codebooks
    pub fn build(pq: &ProductQuantizer) -> Self {
        let m = pq.config.num_subvectors;
        let k = pq.config.num_centroids();

        let mut distances = Vec::with_capacity(m);

        for subspace in 0..m {
            let codebook = &pq.codebooks[subspace];
            let mut subspace_dists = vec![vec![0.0f32; k]; k];

            for i in 0..k {
                for j in i..k {
                    let dist = l2_dist_sq(&codebook[i], &codebook[j]).sqrt();
                    subspace_dists[i][j] = dist;
                    subspace_dists[j][i] = dist;
                }
            }

            distances.push(subspace_dists);
        }

        Self { distances }
    }

    /// Fast symmetric distance using precomputed centroid distances
    #[inline]
    pub fn distance(&self, a: &PQCode, b: &PQCode) -> f32 {
        let mut dist_sq = 0.0f32;

        for (subspace, (&code_a, &code_b)) in a.codes.iter().zip(b.codes.iter()).enumerate() {
            let d = self.distances[subspace][code_a as usize][code_b as usize];
            dist_sq += d * d;
        }

        dist_sq.sqrt()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_pq_config() {
        let config = PQConfig::new(384, 8, 8);
        assert_eq!(config.subvector_dim(), 48);
        assert_eq!(config.num_centroids(), 256);
        assert_eq!(config.code_size(), 8);
        assert!((config.compression_ratio() - 192.0).abs() < 0.1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pq_config_validation() {
        // Dimension not divisible by M
        let config = PQConfig::new(385, 8, 8);
        assert!(config.validate().is_err());

        // Valid config
        let config = PQConfig::new(384, 8, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pq_train_encode_decode() {
        let dim = 64;
        let m = 8;
        let config = PQConfig::new(dim, m, 8).with_seed(42).with_kmeans_iterations(10);

        let training_data = generate_random_vectors(500, dim, 42);
        let pq = ProductQuantizer::train(&training_data, config).unwrap();

        // Encode a vector
        let vector = generate_random_vectors(1, dim, 99)[0].clone();
        let code = pq.encode(&vector);

        assert_eq!(code.codes.len(), m);

        // Decode and check reconstruction error
        let reconstructed = pq.decode(&code);
        assert_eq!(reconstructed.len(), dim);

        // Error should be reasonable (not exact due to quantization)
        // Note: Random data has higher quantization error than real embeddings
        let error = pq.quantization_error(&vector);
        assert!(error < 4.0, "Quantization error too high: {}", error);
    }

    #[test]
    fn test_pq_asymmetric_distance() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(10);

        let training_data = generate_random_vectors(500, dim, 42);
        let pq = ProductQuantizer::train(&training_data, config).unwrap();

        let query = generate_random_vectors(1, dim, 100)[0].clone();
        let db_vec = generate_random_vectors(1, dim, 200)[0].clone();
        let db_code = pq.encode(&db_vec);

        // Asymmetric distance
        let adc_dist = pq.asymmetric_distance(&query, &db_code);

        // Compare to true distance
        let true_dist = l2_dist_sq(&query, &db_vec).sqrt();

        // ADC should be close to true distance
        let relative_error = (adc_dist - true_dist).abs() / true_dist.max(0.001);
        assert!(relative_error < 0.5, "ADC error too high: {}", relative_error);
    }

    #[test]
    fn test_pq_distance_table() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(10);

        let training_data = generate_random_vectors(500, dim, 42);
        let pq = ProductQuantizer::train(&training_data, config).unwrap();

        let query = generate_random_vectors(1, dim, 100)[0].clone();
        let db_vec = generate_random_vectors(1, dim, 200)[0].clone();
        let db_code = pq.encode(&db_vec);

        // Compute distance table
        let table = pq.compute_distance_table(&query);

        // Distance with table should match asymmetric distance
        let table_dist = pq.distance_with_table(&table, &db_code);
        let adc_dist = pq.asymmetric_distance(&query, &db_code);

        assert!((table_dist - adc_dist).abs() < 1e-5);
    }

    #[test]
    fn test_pq_symmetric_distance() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(10);

        let training_data = generate_random_vectors(500, dim, 42);
        let pq = ProductQuantizer::train(&training_data, config).unwrap();

        let vec_a = generate_random_vectors(1, dim, 100)[0].clone();
        let vec_b = generate_random_vectors(1, dim, 200)[0].clone();

        let code_a = pq.encode(&vec_a);
        let code_b = pq.encode(&vec_b);

        // Symmetric distance
        let sym_dist = pq.symmetric_distance(&code_a, &code_b);

        // Should be positive
        assert!(sym_dist >= 0.0);

        // Distance to self should be 0
        let self_dist = pq.symmetric_distance(&code_a, &code_a);
        assert!(self_dist < 1e-5);
    }

    #[test]
    fn test_pq_distance_cache() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(10);

        let training_data = generate_random_vectors(500, dim, 42);
        let pq = ProductQuantizer::train(&training_data, config).unwrap();

        let cache = PQDistanceCache::build(&pq);

        let vec_a = generate_random_vectors(1, dim, 100)[0].clone();
        let vec_b = generate_random_vectors(1, dim, 200)[0].clone();

        let code_a = pq.encode(&vec_a);
        let code_b = pq.encode(&vec_b);

        // Cached distance should match symmetric distance
        let cached_dist = cache.distance(&code_a, &code_b);
        let sym_dist = pq.symmetric_distance(&code_a, &code_b);

        assert!((cached_dist - sym_dist).abs() < 1e-5);
    }

    #[test]
    fn test_kmeans_basic() {
        let data = generate_random_vectors(100, 16, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let centroids = kmeans(&data, 4, 10, &mut rng).unwrap();

        assert_eq!(centroids.len(), 4);
        for centroid in &centroids {
            assert_eq!(centroid.len(), 16);
        }
    }

    #[test]
    fn test_pq_recall() {
        // Test that PQ preserves nearest neighbor relationships
        let dim = 128;
        let n = 500;
        let config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(15);

        let data = generate_random_vectors(n, dim, 42);
        let pq = ProductQuantizer::train(&data, config).unwrap();

        // Encode all vectors
        let codes: Vec<PQCode> = data.iter().map(|v| pq.encode(v)).collect();

        // Random queries
        let queries = generate_random_vectors(10, dim, 999);

        let mut total_recall = 0.0;
        let k = 10;

        for query in &queries {
            // True nearest neighbors
            let mut true_dists: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2_dist_sq(query, v).sqrt()))
                .collect();
            true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // PQ nearest neighbors (using ADC)
            let table = pq.compute_distance_table(query);
            let mut pq_dists: Vec<(usize, f32)> = codes
                .iter()
                .enumerate()
                .map(|(i, code)| (i, pq.distance_with_table(&table, code)))
                .collect();
            pq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Count overlap in top-k
            let true_topk: std::collections::HashSet<usize> =
                true_dists[..k].iter().map(|(i, _)| *i).collect();
            let pq_topk: std::collections::HashSet<usize> =
                pq_dists[..k].iter().map(|(i, _)| *i).collect();

            let overlap = true_topk.intersection(&pq_topk).count();
            total_recall += overlap as f32 / k as f32;
        }

        let avg_recall = total_recall / queries.len() as f32;
        println!("PQ Recall@{}: {:.2}%", k, avg_recall * 100.0);

        // PQ should achieve at least 50% recall@10
        assert!(avg_recall >= 0.5, "PQ recall too low: {:.2}%", avg_recall * 100.0);
    }

    #[test]
    fn test_pq_compression_ratio() {
        let config = PQConfig::new(384, 8, 8);
        // 384 * 4 bytes = 1536 bytes original
        // 8 bytes compressed
        // Ratio = 192x
        assert!((config.compression_ratio() - 192.0).abs() < 0.1);

        let config = PQConfig::new(768, 16, 8);
        // 768 * 4 bytes = 3072 bytes original
        // 16 bytes compressed
        // Ratio = 192x
        assert!((config.compression_ratio() - 192.0).abs() < 0.1);

        let config = PQConfig::new(384, 48, 8);
        // 384 * 4 bytes = 1536 bytes original
        // 48 bytes compressed
        // Ratio = 32x
        assert!((config.compression_ratio() - 32.0).abs() < 0.1);
    }
}
