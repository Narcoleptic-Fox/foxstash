//! HNSW Index with Product Quantization
//!
//! Combines HNSW's fast approximate search with PQ's extreme compression.
//!
//! # Compression
//!
//! For 384-dim vectors with M=8, K=256:
//! - Original: 1.5 GB per million vectors
//! - PQ compressed: 8 MB per million vectors (192x compression!)
//!
//! # Usage
//!
//! ```ignore
//! use foxstash_core::index::hnsw_pq::{PQHNSWIndex, PQHNSWConfig};
//! use foxstash_core::vector::product_quantize::PQConfig;
//! use foxstash_core::Document;
//!
//! // Configure PQ: 8 subvectors, 256 centroids each
//! let pq_config = PQConfig::new(384, 8, 8);
//!
//! // Train PQ on sample vectors
//! let training_data = load_sample_vectors();
//! let mut index = PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default())?;
//!
//! // Add documents
//! for doc in documents {
//!     index.add(doc)?;
//! }
//!
//! // Search (uses ADC for accurate results)
//! let results = index.search(&query, 10)?;
//! ```

use crate::vector::product_quantize::{PQCode, PQConfig, PQDistanceCache, ProductQuantizer};
use crate::{Document, RagError, Result, SearchResult};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for PQ HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQHNSWConfig {
    /// Number of bidirectional links per element (except layer 0)
    pub m: usize,
    /// Number of bidirectional links for layer 0
    pub m0: usize,
    /// Candidate list size during construction
    pub ef_construction: usize,
    /// Candidate list size during search
    pub ef_search: usize,
    /// Level normalization factor
    pub ml: f32,
    /// Whether to use distance cache for faster symmetric search
    pub use_distance_cache: bool,
    /// Store original vectors for reranking
    pub store_original: bool,
    /// Number of candidates to rerank with original vectors
    pub rerank_candidates: usize,
}

impl Default for PQHNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
            use_distance_cache: true,
            store_original: false,
            rerank_candidates: 0,
        }
    }
}

impl PQHNSWConfig {
    /// Enable reranking with original vectors for better accuracy
    ///
    /// Trades memory for accuracy: stores original vectors and reranks
    /// top candidates using exact distance.
    pub fn with_reranking(mut self, candidates: usize) -> Self {
        self.store_original = true;
        self.rerank_candidates = candidates;
        self
    }

    /// Set ef_search parameter
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

// ============================================================================
// PQ HNSW Node
// ============================================================================

/// Node in PQ HNSW graph
#[derive(Debug, Clone)]
struct PQNode {
    /// Document ID
    id: String,
    /// Document content
    content: String,
    /// PQ-encoded embedding
    code: PQCode,
    /// Original embedding (if store_original is true)
    original: Option<Vec<f32>>,
    /// Metadata
    metadata: Option<serde_json::Value>,
    /// Connections per layer
    connections: Vec<HashSet<usize>>,
}

// ============================================================================
// PQ HNSW Index
// ============================================================================

/// HNSW index with Product Quantization
///
/// Achieves extreme compression (up to 192x) while maintaining good search quality.
/// Uses Asymmetric Distance Computation (ADC) for accurate search with full-precision queries.
pub struct PQHNSWIndex {
    config: PQHNSWConfig,
    pq: ProductQuantizer,
    distance_cache: Option<PQDistanceCache>,
    nodes: Vec<PQNode>,
    entry_point: Option<usize>,
    max_layer: usize,
}

impl PQHNSWIndex {
    /// Train a new PQ HNSW index on sample vectors
    ///
    /// # Arguments
    /// * `pq_config` - Product quantization configuration
    /// * `training_data` - Sample vectors for codebook training
    /// * `config` - HNSW configuration
    pub fn train(
        pq_config: PQConfig,
        training_data: &[Vec<f32>],
        config: PQHNSWConfig,
    ) -> Result<Self> {
        let pq = ProductQuantizer::train(training_data, pq_config)?;

        let distance_cache = if config.use_distance_cache {
            Some(PQDistanceCache::build(&pq))
        } else {
            None
        };

        Ok(Self {
            config,
            pq,
            distance_cache,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        })
    }

    /// Create index from pre-trained quantizer
    pub fn from_quantizer(pq: ProductQuantizer, config: PQHNSWConfig) -> Self {
        let distance_cache = if config.use_distance_cache {
            Some(PQDistanceCache::build(&pq))
        } else {
            None
        };

        Self {
            config,
            pq,
            distance_cache,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Add a document to the index
    pub fn add(&mut self, document: Document) -> Result<()> {
        let dim = self.pq.config().dim;
        if document.embedding.len() != dim {
            return Err(RagError::DimensionMismatch {
                expected: dim,
                actual: document.embedding.len(),
            });
        }

        let node_id = self.nodes.len();
        let node_level = self.random_level();

        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(HashSet::new());
        }

        let code = self.pq.encode(&document.embedding);
        let original = if self.config.store_original {
            Some(document.embedding)
        } else {
            None
        };

        let node = PQNode {
            id: document.id,
            content: document.content,
            code,
            original,
            metadata: document.metadata,
            connections,
        };

        self.nodes.push(node);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        self.insert_node(node_id, node_level);

        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Search for k nearest neighbors using ADC
    ///
    /// Uses Asymmetric Distance Computation for best accuracy:
    /// full precision query vs PQ-compressed database vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let dim = self.pq.config().dim;
        if query.len() != dim {
            return Err(RagError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Precompute distance table for fast ADC
        let distance_table = self.pq.compute_distance_table(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Search upper layers
        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer_adc(&distance_table, &current_nearest, 1, layer);
        }

        // Search layer 0 with ef_search candidates
        let ef = self.config.ef_search.max(k);
        let candidates = if self.config.rerank_candidates > 0 {
            self.config.ef_search.max(self.config.rerank_candidates)
        } else {
            ef
        };
        current_nearest = self.search_layer_adc(&distance_table, &current_nearest, candidates, 0);

        // Optionally rerank with original vectors
        let results = if self.config.store_original && self.config.rerank_candidates > 0 {
            self.rerank_with_original(query, &current_nearest, k)
        } else {
            self.to_search_results_adc(query, &current_nearest, &distance_table, k)
        };

        Ok(results)
    }

    /// Search using symmetric distance (fastest, lower accuracy)
    pub fn search_symmetric(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let dim = self.pq.config().dim;
        if query.len() != dim {
            return Err(RagError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Quantize query for symmetric search
        let query_code = self.pq.encode(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer_symmetric(&query_code, &current_nearest, 1, layer);
        }

        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer_symmetric(&query_code, &current_nearest, ef, 0);

        let results = self.to_search_results_symmetric(&query_code, &current_nearest, k);

        Ok(results)
    }

    /// Get number of documents
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get the product quantizer
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.pq
    }

    /// Memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let code_size = self.pq.config().num_subvectors;
        let original_size = if self.config.store_original {
            self.pq.config().dim * 4
        } else {
            0
        };
        let overhead_per_node = 100;

        // Codebook size
        let codebook_size = self.pq.config().num_subvectors
            * self.pq.config().num_centroids()
            * self.pq.config().subvector_dim()
            * 4;

        // Distance cache size
        let cache_size = if self.distance_cache.is_some() {
            self.pq.config().num_subvectors
                * self.pq.config().num_centroids()
                * self.pq.config().num_centroids()
                * 4
        } else {
            0
        };

        self.nodes.len() * (code_size + original_size + overhead_per_node)
            + codebook_size
            + cache_size
    }

    /// Get compression ratio vs full f32 storage
    pub fn compression_ratio(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let full_size = self.nodes.len() * self.pq.config().dim * 4;
        let actual_size = self.memory_usage();

        full_size as f32 / actual_size as f32
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen();
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    fn insert_node(&mut self, node_id: usize, node_level: usize) {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];
        let node_code = self.nodes[node_id].code.clone();

        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer_symmetric(&node_code, &current_nearest, 1, layer);
        }

        for layer in (0..=node_level).rev() {
            current_nearest = self.search_layer_symmetric(
                &node_code,
                &current_nearest,
                self.config.ef_construction,
                layer,
            );

            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors(&current_nearest, &node_code, m);

            for &neighbor_id in &neighbors {
                self.nodes[node_id].connections[layer].insert(neighbor_id);

                if layer < self.nodes[neighbor_id].connections.len() {
                    self.nodes[neighbor_id].connections[layer].insert(node_id);

                    let neighbor_m = if layer == 0 {
                        self.config.m0
                    } else {
                        self.config.m
                    };
                    if self.nodes[neighbor_id].connections[layer].len() > neighbor_m {
                        let neighbor_code = self.nodes[neighbor_id].code.clone();
                        let neighbor_connections: Vec<usize> = self.nodes[neighbor_id].connections
                            [layer]
                            .iter()
                            .copied()
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_code,
                            neighbor_m,
                        );
                        self.nodes[neighbor_id].connections[layer] = pruned.into_iter().collect();
                    }
                }
            }
        }
    }

    fn search_layer_adc(
        &self,
        distance_table: &[Vec<f32>],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        for &ep in entry_points {
            let dist = self
                .pq
                .distance_with_table(distance_table, &self.nodes[ep].code);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            best.push((OrderedFloat(dist), ep));
            visited.insert(ep);
        }

        while let Some(Reverse((current_dist, current_id))) = candidates.pop() {
            if best.len() >= ef {
                if let Some(&(furthest_dist, _)) = best.peek() {
                    if current_dist > furthest_dist {
                        break;
                    }
                }
            }

            if layer < self.nodes[current_id].connections.len() {
                for &neighbor_id in &self.nodes[current_id].connections[layer] {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        let dist = self
                            .pq
                            .distance_with_table(distance_table, &self.nodes[neighbor_id].code);
                        let dist_ord = OrderedFloat(dist);

                        if best.len() < ef {
                            candidates.push(Reverse((dist_ord, neighbor_id)));
                            best.push((dist_ord, neighbor_id));
                        } else if let Some(&(furthest_dist, _)) = best.peek() {
                            if dist_ord < furthest_dist {
                                candidates.push(Reverse((dist_ord, neighbor_id)));
                                best.push((dist_ord, neighbor_id));
                                if best.len() > ef {
                                    best.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut results: Vec<(f32, usize)> = best
            .into_iter()
            .map(|(OrderedFloat(dist), id)| (dist, id))
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.into_iter().map(|(_, id)| id).collect()
    }

    fn search_layer_symmetric(
        &self,
        query_code: &PQCode,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        for &ep in entry_points {
            let dist = self.distance_symmetric(query_code, &self.nodes[ep].code);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            best.push((OrderedFloat(dist), ep));
            visited.insert(ep);
        }

        while let Some(Reverse((current_dist, current_id))) = candidates.pop() {
            if best.len() >= ef {
                if let Some(&(furthest_dist, _)) = best.peek() {
                    if current_dist > furthest_dist {
                        break;
                    }
                }
            }

            if layer < self.nodes[current_id].connections.len() {
                for &neighbor_id in &self.nodes[current_id].connections[layer] {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        let dist =
                            self.distance_symmetric(query_code, &self.nodes[neighbor_id].code);
                        let dist_ord = OrderedFloat(dist);

                        if best.len() < ef {
                            candidates.push(Reverse((dist_ord, neighbor_id)));
                            best.push((dist_ord, neighbor_id));
                        } else if let Some(&(furthest_dist, _)) = best.peek() {
                            if dist_ord < furthest_dist {
                                candidates.push(Reverse((dist_ord, neighbor_id)));
                                best.push((dist_ord, neighbor_id));
                                if best.len() > ef {
                                    best.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut results: Vec<(f32, usize)> = best
            .into_iter()
            .map(|(OrderedFloat(dist), id)| (dist, id))
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.into_iter().map(|(_, id)| id).collect()
    }

    fn select_neighbors(&self, candidates: &[usize], query_code: &PQCode, m: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance_symmetric(query_code, &self.nodes[id].code);
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    fn distance_symmetric(&self, a: &PQCode, b: &PQCode) -> f32 {
        if let Some(ref cache) = self.distance_cache {
            cache.distance(a, b)
        } else {
            self.pq.symmetric_distance(a, b)
        }
    }

    fn to_search_results_adc(
        &self,
        _query: &[f32],
        node_ids: &[usize],
        distance_table: &[Vec<f32>],
        k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = node_ids
            .iter()
            .map(|&id| {
                let node = &self.nodes[id];
                let dist = self.pq.distance_with_table(distance_table, &node.code);
                // Convert distance to similarity score
                let score = 1.0 / (1.0 + dist);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }

    fn to_search_results_symmetric(
        &self,
        query_code: &PQCode,
        node_ids: &[usize],
        k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = node_ids
            .iter()
            .map(|&id| {
                let node = &self.nodes[id];
                let dist = self.distance_symmetric(query_code, &node.code);
                let score = 1.0 / (1.0 + dist);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }

    fn rerank_with_original(
        &self,
        query: &[f32],
        node_ids: &[usize],
        k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = node_ids
            .iter()
            .filter_map(|&id| {
                let node = &self.nodes[id];
                node.original.as_ref().map(|original| {
                    let score = crate::vector::cosine_similarity(query, original).unwrap_or(0.0);
                    SearchResult {
                        id: node.id.clone(),
                        content: node.content.clone(),
                        score,
                        metadata: node.metadata.clone(),
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }
}

/// OrderedFloat for BinaryHeap
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                    .collect()
            })
            .collect()
    }

    fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding,
            metadata: None,
        }
    }

    #[test]
    fn test_pq_hnsw_config() {
        let config = PQHNSWConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 200);
        assert!(!config.store_original);
    }

    #[test]
    fn test_pq_hnsw_basic() {
        let dim = 64;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(500, dim, 42);

        let mut index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();
        assert!(index.is_empty());

        let doc = create_test_document("doc1", generate_random_vectors(1, dim, 100)[0].clone());
        index.add(doc).unwrap();

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_pq_hnsw_search() {
        let dim = 64;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(500, dim, 42);

        let mut index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();

        // Add documents
        for i in 0..100 {
            let doc = create_test_document(
                &format!("doc{}", i),
                generate_random_vectors(1, dim, i)[0].clone(),
            );
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = generate_random_vectors(1, dim, 999)[0].clone();
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by score descending
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_pq_hnsw_compression() {
        let dim = 384;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(500, dim, 42);

        let mut index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();

        // Use enough vectors to amortize fixed overhead (codebook + distance cache)
        for i in 0..5000 {
            let doc = create_test_document(
                &format!("doc{}", i),
                generate_random_vectors(1, dim, i)[0].clone(),
            );
            index.add(doc).unwrap();
        }

        let memory = index.memory_usage();
        let full_size = 5000 * dim * 4;

        println!("PQ HNSW memory: {} bytes", memory);
        println!("Full f32 would be: {} bytes", full_size);
        println!("Compression ratio: {:.1}x", index.compression_ratio());

        // PQ codes are 8 bytes vs 1536 bytes (192x for vectors alone)
        // But we have fixed overhead from codebook + distance cache + HNSW graph
        // For 5K vectors, should still achieve > 2x compression
        assert!(
            memory < full_size / 2,
            "Memory should be < 50% of full size, got {}%",
            (memory as f64 / full_size as f64 * 100.0) as u32
        );
    }

    #[test]
    fn test_pq_hnsw_with_reranking() {
        let dim = 64;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(500, dim, 42);

        let config = PQHNSWConfig::default().with_reranking(50);
        let mut index = PQHNSWIndex::train(pq_config, &training_data, config).unwrap();

        for i in 0..100 {
            let doc = create_test_document(
                &format!("doc{}", i),
                generate_random_vectors(1, dim, i)[0].clone(),
            );
            index.add(doc).unwrap();
        }

        let query = generate_random_vectors(1, dim, 999)[0].clone();
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // With reranking, scores should be cosine similarity values
        for result in &results {
            assert!(result.score >= -1.0 && result.score <= 1.0);
        }
    }

    #[test]
    fn test_pq_hnsw_dimension_mismatch() {
        let dim = 64;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(100, dim, 42);

        let mut index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();

        // Wrong dimension
        let doc = create_test_document("doc1", vec![0.5; 32]);
        assert!(index.add(doc).is_err());
    }

    #[test]
    fn test_pq_hnsw_symmetric_search() {
        let dim = 64;
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let training_data = generate_random_vectors(500, dim, 42);

        let mut index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();

        for i in 0..100 {
            let doc = create_test_document(
                &format!("doc{}", i),
                generate_random_vectors(1, dim, i)[0].clone(),
            );
            index.add(doc).unwrap();
        }

        let query = generate_random_vectors(1, dim, 999)[0].clone();

        // Symmetric should be faster but less accurate
        let symmetric_results = index.search_symmetric(&query, 10).unwrap();
        assert_eq!(symmetric_results.len(), 10);

        // ADC should generally give better results
        let adc_results = index.search(&query, 10).unwrap();
        assert_eq!(adc_results.len(), 10);
    }
}
