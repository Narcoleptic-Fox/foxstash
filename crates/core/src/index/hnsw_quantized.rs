//! Quantized HNSW index for memory-efficient similarity search
//!
//! This module provides HNSW variants that use quantized vectors for storage,
//! dramatically reducing memory footprint while maintaining good search quality.
//!
//! # Quantization Modes
//!
//! - **SQ8 (Scalar Quantization)**: 4x compression, near-exact recall
//! - **RaBitQ**: 32x compression, 1 bit/dim with an unbiased estimator — the
//!   recommended 32x mode
//! - **Binary**: 32x compression — **deprecated**, degenerate on non-negative data
//!
//! # Two-Phase Search (Recommended)
//!
//! A 1-bit code is a coarse ranker. Use it to retrieve a candidate pool, then
//! rerank that pool against full-precision vectors:
//!
//! ```
//! use foxstash_core::index::hnsw_quantized::{RaBitQHNSWIndex, QuantizedHNSWConfig};
//! use foxstash_core::Document;
//!
//! let dim = 8;
//! let training: Vec<Vec<f32>> = (0..32)
//!     .map(|i| (0..dim).map(|d| ((i + d) % 5) as f32).collect())
//!     .collect();
//!
//! // RaBitQ must be fitted (it needs a centroid), then store full precision to rerank.
//! let mut index = RaBitQHNSWIndex::fit(&training, QuantizedHNSWConfig::default());
//!
//! for (i, v) in training.iter().enumerate() {
//!     index.add_with_full_precision(Document {
//!         id: format!("doc{}", i),
//!         content: format!("Content {}", i),
//!         embedding: v.clone(),
//!         metadata: None,
//!     }).unwrap();
//! }
//!
//! // Two-phase: 1-bit filter (100 candidates) -> exact rerank (top 2)
//! let results = index.search_and_rerank(&training[0], 100, 2).unwrap();
//! assert!(results.len() <= 2);
//! ```
//!
//! # Memory Comparison (1M vectors × 384 dims)
//!
//! | Index Type | Memory | Recall@10 (SIFT10K, pool=100) |
//! |------------|--------|-------------------------------|
//! | Full f32   | 1.5 GB | 100% baseline                 |
//! | SQ8        | 384 MB | 100.0%                        |
//! | RaBitQ     | 48 MB  | 73.2%                         |
//! | Binary     | 48 MB  | 1.2% — deprecated, see below  |
//!
//! Recall figures are measured, not estimated: `cargo run --release -p foxstash-benches
//! --example quantizer_sift` (10k vectors, 128d, 1000 queries, top-10, rerank pool=100,
//! ground truth shipped with the dataset).
//!
//! Binary's 1.2% is not a typo. Its zero threshold sets every bit on non-negative data
//! (SIFT, and any ReLU-activated embedding), collapsing every code to all-ones. At the
//! same 32x compression RaBitQ centers the data and reaches 73.2%.

use crate::vector::quantize::{
    BinaryQuantizedVector, BinaryQuantizer, Quantizer, ScalarQuantizedVector, ScalarQuantizer,
};
use crate::vector::rabitq::{PreparedQuery, RaBitCode, RaBitQuantizer};
use crate::{Document, RagError, Result, SearchResult};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for quantized HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedHNSWConfig {
    /// Number of bidirectional links created for each element (except layer 0)
    pub m: usize,
    /// Number of bidirectional links created for each element in layer 0
    pub m0: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search
    pub ef_search: usize,
    /// Normalization factor for level generation
    pub ml: f32,
}

impl Default for QuantizedHNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
        }
    }
}

// ============================================================================
// SQ8 HNSW Index
// ============================================================================

/// Node in SQ8 HNSW graph
#[derive(Debug, Clone)]
struct SQ8Node {
    id: String,
    content: String,
    quantized: ScalarQuantizedVector,
    metadata: Option<serde_json::Value>,
    connections: Vec<HashSet<usize>>,
}

/// HNSW index with scalar quantization (SQ8)
///
/// Stores vectors as u8 (4x compression) while maintaining high recall.
/// Supports asymmetric search (full precision query vs quantized database).
///
/// # Example
///
/// ```
/// use foxstash_core::index::hnsw_quantized::{SQ8HNSWIndex, QuantizedHNSWConfig};
/// use foxstash_core::Document;
///
/// // Create index with normalized vector bounds
/// let mut index = SQ8HNSWIndex::for_normalized(384, QuantizedHNSWConfig::default());
///
/// // Add documents
/// let doc = Document {
///     id: "doc1".to_string(),
///     content: "Hello world".to_string(),
///     embedding: vec![0.1; 384],
///     metadata: None,
/// };
/// index.add(doc).unwrap();
///
/// // Search
/// let results = index.search(&vec![0.1; 384], 5).unwrap();
/// ```
pub struct SQ8HNSWIndex {
    embedding_dim: usize,
    config: QuantizedHNSWConfig,
    quantizer: ScalarQuantizer,
    nodes: Vec<SQ8Node>,
    entry_point: Option<usize>,
    max_layer: usize,
}

impl SQ8HNSWIndex {
    /// Create index with custom quantizer
    pub fn new(quantizer: ScalarQuantizer, config: QuantizedHNSWConfig) -> Self {
        let embedding_dim = quantizer.dim();
        Self {
            embedding_dim,
            config,
            quantizer,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Create index for normalized embeddings ([-1, 1] range)
    pub fn for_normalized(dim: usize, config: QuantizedHNSWConfig) -> Self {
        Self::new(ScalarQuantizer::for_normalized(dim), config)
    }

    /// Create index by fitting quantizer on training data
    pub fn fit(training_vectors: &[Vec<f32>], config: QuantizedHNSWConfig) -> Self {
        let quantizer = ScalarQuantizer::fit(training_vectors);
        Self::new(quantizer, config)
    }

    /// Add a document to the index
    pub fn add(&mut self, document: Document) -> Result<()> {
        if document.embedding.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
            });
        }

        if document.embedding.iter().any(|v| !v.is_finite()) {
            return Err(RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let node_id = self.nodes.len();
        let node_level = self.random_level();

        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(HashSet::new());
        }

        let quantized = self.quantizer.quantize(&document.embedding);

        let node = SQ8Node {
            id: document.id,
            content: document.content,
            quantized,
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

    /// Search for k nearest neighbors using asymmetric distance
    ///
    /// The query is kept at full precision while comparing against quantized vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Quantize query for symmetric search (faster but slightly lower quality)
        let query_quantized = self.quantizer.quantize(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&query_quantized, &current_nearest, 1, layer);
        }

        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer(&query_quantized, &current_nearest, ef, 0);

        // Compute final scores using asymmetric distance for better accuracy
        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                let node = &self.nodes[node_id];
                // Use asymmetric distance for final ranking
                let dist = self.quantizer.distance_asymmetric(query, &node.quantized);
                // Convert distance to similarity score (1 / (1 + dist))
                let score = 1.0 / (1.0 + dist);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(k);

        Ok(results)
    }

    /// Search using symmetric quantized distance (faster, slightly lower quality)
    pub fn search_symmetric(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let query_quantized = self.quantizer.quantize(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&query_quantized, &current_nearest, 1, layer);
        }

        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer(&query_quantized, &current_nearest, ef, 0);

        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                let node = &self.nodes[node_id];
                let dist = self
                    .quantizer
                    .distance_quantized(&query_quantized, &node.quantized);
                let score = 1.0 / (1.0 + dist);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(k);

        Ok(results)
    }

    /// Returns number of documents in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the quantizer for analysis
    pub fn quantizer(&self) -> &ScalarQuantizer {
        &self.quantizer
    }

    /// Memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let vec_size = self.embedding_dim; // u8 per dimension
        let overhead_per_node = 100; // Approximate: id, content, connections
        self.nodes.len() * (vec_size + overhead_per_node)
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let uniform: f32 = rng.random::<f32>().max(f32::EPSILON);
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    fn insert_node(&mut self, node_id: usize, node_level: usize) {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];
        let node_quantized = self.nodes[node_id].quantized.clone();

        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&node_quantized, &current_nearest, 1, layer);
        }

        for layer in (0..=node_level).rev() {
            current_nearest = self.search_layer(
                &node_quantized,
                &current_nearest,
                self.config.ef_construction,
                layer,
            );

            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors(&current_nearest, &node_quantized, m);

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
                        let neighbor_quantized = self.nodes[neighbor_id].quantized.clone();
                        let neighbor_connections: Vec<usize> = self.nodes[neighbor_id].connections
                            [layer]
                            .iter()
                            .copied()
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_quantized,
                            neighbor_m,
                        );
                        self.nodes[neighbor_id].connections[layer] = pruned.into_iter().collect();
                    }
                }
            }
        }
    }

    fn search_layer(
        &self,
        query: &ScalarQuantizedVector,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        for &ep in entry_points {
            let dist = self
                .quantizer
                .distance_quantized(query, &self.nodes[ep].quantized);
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
                            .quantizer
                            .distance_quantized(query, &self.nodes[neighbor_id].quantized);
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
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results.into_iter().map(|(_, id)| id).collect()
    }

    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &ScalarQuantizedVector,
        m: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self
                    .quantizer
                    .distance_quantized(query, &self.nodes[id].quantized);
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

impl crate::index::VectorIndex for SQ8HNSWIndex {
    fn add(&mut self, document: Document) -> Result<()> {
        self.add(document)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

// ============================================================================
// Binary HNSW Index
// ============================================================================

/// Node in Binary HNSW graph
#[derive(Debug, Clone)]
struct BinaryNode {
    id: String,
    content: String,
    quantized: BinaryQuantizedVector,
    /// Optional full precision vector for reranking
    full_precision: Option<Vec<f32>>,
    metadata: Option<serde_json::Value>,
    connections: Vec<HashSet<usize>>,
}

/// HNSW index with binary quantization
///
/// Stores vectors as packed bits (32x compression). Best used for initial
/// candidate retrieval followed by reranking with higher precision.
///
/// # ⚠️ Degenerate on non-negative embeddings
///
/// [`BinaryQuantizer`] thresholds each dimension at zero, so on data that is
/// wholly non-negative — SIFT descriptors, and the output of any ReLU-activated
/// model — *every* bit is set, all codes collapse to all-ones, every Hamming
/// distance is zero, and graph traversal degenerates to arbitrary order.
///
/// Measured on SIFT10K (10k vectors, 128d, top-10, pool=100): **1.2%** recall@10,
/// i.e. chance. Mean-centering the threshold lifts it to 50.1%, and RaBitQ at the
/// same 32x compression reaches 73.2%.
///
/// Use [`RaBitQHNSWIndex`] instead: identical footprint, centers the data, and a
/// strictly better estimator.
///
/// # Example
///
/// ```
/// use foxstash_core::index::hnsw_quantized::{BinaryHNSWIndex, QuantizedHNSWConfig};
/// use foxstash_core::Document;
///
/// # #[allow(deprecated)] {
/// // Create binary index
/// let mut index = BinaryHNSWIndex::new(384, QuantizedHNSWConfig::default());
///
/// // Add with full precision storage for reranking
/// let doc = Document {
///     id: "doc1".to_string(),
///     content: "Hello world".to_string(),
///     embedding: vec![0.1; 384],
///     metadata: None,
/// };
/// index.add_with_full_precision(doc).unwrap();
///
/// // Two-phase search: binary filter → full precision rerank
/// let results = index.search_and_rerank(&vec![0.1; 384], 100, 10).unwrap();
/// # }
/// ```
#[deprecated(
    since = "0.6.0",
    note = "zero-threshold binary codes degenerate on non-negative embeddings (1.2% recall@10 on SIFT10K). \
            Use RaBitQHNSWIndex: same 32x compression, centered, 73.2% on the same benchmark."
)]
pub struct BinaryHNSWIndex {
    embedding_dim: usize,
    config: QuantizedHNSWConfig,
    quantizer: BinaryQuantizer,
    nodes: Vec<BinaryNode>,
    entry_point: Option<usize>,
    max_layer: usize,
    /// Whether full precision vectors are stored
    store_full_precision: bool,
}

#[allow(deprecated)]
impl BinaryHNSWIndex {
    /// Create index (binary only, no full precision storage)
    pub fn new(dim: usize, config: QuantizedHNSWConfig) -> Self {
        Self {
            embedding_dim: dim,
            config,
            quantizer: BinaryQuantizer::new(dim),
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            store_full_precision: false,
        }
    }

    /// Create index with full precision storage for reranking
    pub fn with_full_precision(dim: usize, config: QuantizedHNSWConfig) -> Self {
        let mut index = Self::new(dim, config);
        index.store_full_precision = true;
        index
    }

    /// Add document (binary only)
    pub fn add(&mut self, document: Document) -> Result<()> {
        self.add_internal(document, false)
    }

    /// Add document with full precision storage for reranking
    pub fn add_with_full_precision(&mut self, document: Document) -> Result<()> {
        self.add_internal(document, true)
    }

    fn add_internal(&mut self, document: Document, store_full: bool) -> Result<()> {
        if document.embedding.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
            });
        }

        if document.embedding.iter().any(|v| !v.is_finite()) {
            return Err(RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let node_id = self.nodes.len();
        let node_level = self.random_level();

        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(HashSet::new());
        }

        let quantized = self.quantizer.quantize(&document.embedding);
        let full_precision = if store_full || self.store_full_precision {
            Some(document.embedding)
        } else {
            None
        };

        let node = BinaryNode {
            id: document.id,
            content: document.content,
            quantized,
            full_precision,
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

    /// Search using Hamming distance (fast, lower quality)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let query_quantized = self.quantizer.quantize(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&query_quantized, &current_nearest, 1, layer);
        }

        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer(&query_quantized, &current_nearest, ef, 0);

        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                let node = &self.nodes[node_id];
                let dist = self
                    .quantizer
                    .distance_quantized(&query_quantized, &node.quantized);
                // Convert Hamming distance to similarity (max distance = dim)
                let score = 1.0 - (dist / self.embedding_dim as f32);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(k);

        Ok(results)
    }

    /// Two-phase search: binary filter → full precision rerank
    ///
    /// First retrieves `candidates` using binary search, then reranks using
    /// full precision cosine similarity if available.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `candidates` - Number of candidates to retrieve in binary phase
    /// * `k` - Number of final results
    pub fn search_and_rerank(
        &self,
        query: &[f32],
        candidates: usize,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 1: Binary search
        let query_quantized = self.quantizer.quantize(query);
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&query_quantized, &current_nearest, 1, layer);
        }

        let ef = self.config.ef_search.max(candidates);
        current_nearest = self.search_layer(&query_quantized, &current_nearest, ef, 0);
        current_nearest.truncate(candidates);

        // Phase 2: Rerank with full precision (if available)
        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                let node = &self.nodes[node_id];

                let score = if let Some(ref full_vec) = node.full_precision {
                    // Full precision cosine similarity
                    crate::vector::cosine_similarity(query, full_vec).unwrap_or(0.0)
                } else {
                    // Fall back to binary similarity
                    let dist = self
                        .quantizer
                        .distance_quantized(&query_quantized, &node.quantized);
                    1.0 - (dist / self.embedding_dim as f32)
                };

                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(k);

        Ok(results)
    }

    /// Returns number of documents
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let binary_size = self.quantizer.byte_len();
        let full_size = if self.store_full_precision {
            self.embedding_dim * 4 // f32
        } else {
            0
        };
        let overhead_per_node = 100;
        self.nodes.len() * (binary_size + full_size + overhead_per_node)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let uniform: f32 = rng.random::<f32>().max(f32::EPSILON);
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    fn insert_node(&mut self, node_id: usize, node_level: usize) {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];
        let node_quantized = self.nodes[node_id].quantized.clone();

        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer(&node_quantized, &current_nearest, 1, layer);
        }

        for layer in (0..=node_level).rev() {
            current_nearest = self.search_layer(
                &node_quantized,
                &current_nearest,
                self.config.ef_construction,
                layer,
            );

            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors(&current_nearest, &node_quantized, m);

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
                        let neighbor_quantized = self.nodes[neighbor_id].quantized.clone();
                        let neighbor_connections: Vec<usize> = self.nodes[neighbor_id].connections
                            [layer]
                            .iter()
                            .copied()
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_quantized,
                            neighbor_m,
                        );
                        self.nodes[neighbor_id].connections[layer] = pruned.into_iter().collect();
                    }
                }
            }
        }
    }

    fn search_layer(
        &self,
        query: &BinaryQuantizedVector,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        for &ep in entry_points {
            let dist = self
                .quantizer
                .distance_quantized(query, &self.nodes[ep].quantized);
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
                            .quantizer
                            .distance_quantized(query, &self.nodes[neighbor_id].quantized);
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
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results.into_iter().map(|(_, id)| id).collect()
    }

    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &BinaryQuantizedVector,
        m: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self
                    .quantizer
                    .distance_quantized(query, &self.nodes[id].quantized);
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

#[allow(deprecated)]
impl crate::index::VectorIndex for BinaryHNSWIndex {
    fn add(&mut self, document: Document) -> Result<()> {
        self.add(document)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

// ============================================================================
// OrderedFloat Helper
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ============================================================================
// RaBitQ HNSW Index
// ============================================================================

/// Node in RaBitQ HNSW graph
#[derive(Debug, Clone)]
struct RaBitQNode {
    id: String,
    content: String,
    code: RaBitCode,
    /// Optional full precision vector for reranking
    full_precision: Option<Vec<f32>>,
    metadata: Option<serde_json::Value>,
    connections: Vec<HashSet<usize>>,
}

/// HNSW index with RaBitQ 1-bit quantization (32x compression).
///
/// Supersedes [`BinaryHNSWIndex`]: same 1 bit/dim footprint, but RaBitQ's
/// unbiased estimator is a far better first-stage ranker than a Hamming proxy,
/// and — critically — it centers the data, so it does not degenerate on
/// non-negative embeddings the way a zero-threshold binary code does.
///
/// The quantizer must be fitted on training data (it needs a centroid), so
/// there is no `new(dim)` constructor — use [`RaBitQHNSWIndex::fit`].
///
/// Measured on SIFT10K (10k vectors, 128d, top-10, pool=100, exact-L2 rerank):
/// RaBitQ **73.2%** recall@10 vs binary's **1.2%** (binary's zero threshold
/// collapses on this non-negative data; even mean-centered it reaches only 50.1%).
///
/// # Example
///
/// ```
/// use foxstash_core::index::hnsw_quantized::{RaBitQHNSWIndex, QuantizedHNSWConfig};
/// use foxstash_core::Document;
///
/// let dim = 8;
/// let training: Vec<Vec<f32>> = (0..64)
///     .map(|i| (0..dim).map(|d| ((i + d) % 7) as f32).collect())
///     .collect();
///
/// // Fit the quantizer, then store full-precision vectors so we can rerank.
/// let mut index = RaBitQHNSWIndex::fit(&training, QuantizedHNSWConfig::default());
///
/// for (i, v) in training.iter().enumerate() {
///     index.add_with_full_precision(Document {
///         id: format!("doc{}", i),
///         content: format!("Content {}", i),
///         embedding: v.clone(),
///         metadata: None,
///     }).unwrap();
/// }
///
/// // Two-phase: 1-bit graph traversal (100 candidates) -> exact rerank (top 5)
/// let results = index.search_and_rerank(&training[0], 100, 5).unwrap();
/// assert!(results.len() <= 5);
/// ```
pub struct RaBitQHNSWIndex {
    embedding_dim: usize,
    config: QuantizedHNSWConfig,
    quantizer: RaBitQuantizer,
    nodes: Vec<RaBitQNode>,
    entry_point: Option<usize>,
    max_layer: usize,
}

impl RaBitQHNSWIndex {
    /// Create index with an already-fitted quantizer
    pub fn new(quantizer: RaBitQuantizer, config: QuantizedHNSWConfig) -> Self {
        let embedding_dim = quantizer.dim();
        Self {
            embedding_dim,
            config,
            quantizer,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Create index by fitting the quantizer on training data.
    ///
    /// RaBitQ needs a centroid, so this (or [`RaBitQHNSWIndex::new`] with a
    /// pre-fitted quantizer) is the only way to construct the index. Training
    /// vectors should be representative of what will be indexed.
    pub fn fit(training_vectors: &[Vec<f32>], config: QuantizedHNSWConfig) -> Self {
        Self::new(RaBitQuantizer::fit(training_vectors), config)
    }

    /// Add a document, storing only its 1-bit code (32x compression).
    ///
    /// Without full-precision vectors, [`RaBitQHNSWIndex::search_and_rerank`] cannot
    /// rerank exactly; use [`RaBitQHNSWIndex::add_with_full_precision`] if you want that.
    pub fn add(&mut self, document: Document) -> Result<()> {
        self.add_inner(document, false)
    }

    /// Add a document, also retaining its full-precision vector for exact reranking.
    pub fn add_with_full_precision(&mut self, document: Document) -> Result<()> {
        self.add_inner(document, true)
    }

    fn add_inner(&mut self, document: Document, keep_full: bool) -> Result<()> {
        if document.embedding.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
            });
        }

        if document.embedding.iter().any(|v| !v.is_finite()) {
            return Err(RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let node_id = self.nodes.len();
        let node_level = self.random_level();

        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(HashSet::new());
        }

        let code = self.quantizer.encode(&document.embedding);
        // The inserting vector is its own query during graph construction.
        let prepared = self.quantizer.prepare_query(&document.embedding);

        self.nodes.push(RaBitQNode {
            id: document.id,
            content: document.content,
            code,
            full_precision: keep_full.then(|| document.embedding.clone()),
            metadata: document.metadata,
            connections,
        });

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        self.insert_node(node_id, node_level, &prepared);

        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Search using the RaBitQ estimate for graph traversal.
    ///
    /// For best recall prefer [`RaBitQHNSWIndex::search_and_rerank`] — a 1-bit code
    /// is a coarse ranker, and the whole point of it is to feed a rerank stage.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let scored = self.traverse(query, self.config.ef_search.max(k))?;

        let mut results: Vec<SearchResult> = scored
            .iter()
            .map(|&(dist, id)| {
                let node = &self.nodes[id];
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score: 1.0 / (1.0 + dist.max(0.0)),
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        results.truncate(k);
        Ok(results)
    }

    /// Two-phase search: 1-bit graph traversal for `pool` candidates, then rerank
    /// the pool by exact L2 against stored full-precision vectors and cut to `k`.
    ///
    /// Nodes added via [`RaBitQHNSWIndex::add`] have no full-precision vector and
    /// keep their estimated distance, so mixing the two `add` methods will rank
    /// estimated and exact distances against each other. Prefer using
    /// [`RaBitQHNSWIndex::add_with_full_precision`] for every document.
    pub fn search_and_rerank(
        &self,
        query: &[f32],
        pool: usize,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let candidates = self.traverse(query, pool.max(k))?;

        let mut reranked: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&(est, id)| {
                let exact = self.nodes[id]
                    .full_precision
                    .as_ref()
                    .map(|v| l2_sq(query, v))
                    .unwrap_or(est);
                (exact, id)
            })
            .collect();

        reranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        reranked.truncate(k);

        Ok(reranked
            .into_iter()
            .map(|(dist, id)| {
                let node = &self.nodes[id];
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score: 1.0 / (1.0 + dist.max(0.0)),
                    metadata: node.metadata.clone(),
                }
            })
            .collect())
    }

    /// Walk the graph and return up to `ef` (distance, node_id) pairs, nearest first.
    fn traverse(&self, query: &[f32], ef: usize) -> Result<Vec<(f32, usize)>> {
        if query.len() != self.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let prepared = self.quantizer.prepare_query(query);
        let mut current = vec![self.entry_point.unwrap()];

        for layer in (1..=self.max_layer).rev() {
            current = self.search_layer(&prepared, &current, 1, layer);
        }
        let ids = self.search_layer(&prepared, &current, ef, 0);

        Ok(ids
            .into_iter()
            .map(|id| {
                (
                    self.quantizer
                        .estimate_dist_sq(&prepared, &self.nodes[id].code),
                    id,
                )
            })
            .collect())
    }

    /// Returns number of documents in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the quantizer for analysis
    pub fn quantizer(&self) -> &RaBitQuantizer {
        &self.quantizer
    }

    /// Memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let code_size = self.embedding_dim.div_ceil(8); // 1 bit per dimension
        let full_size = if self.nodes[0].full_precision.is_some() {
            self.embedding_dim * 4
        } else {
            0
        };
        let overhead_per_node = 100; // Approximate: id, content, connections
        self.nodes.len() * (code_size + full_size + overhead_per_node)
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let uniform: f32 = rng.random::<f32>().max(f32::EPSILON);
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    fn insert_node(&mut self, node_id: usize, node_level: usize, prepared: &PreparedQuery) {
        let mut current = vec![self.entry_point.unwrap()];

        for layer in (node_level + 1..=self.max_layer).rev() {
            current = self.search_layer(prepared, &current, 1, layer);
        }

        for layer in (0..=node_level).rev() {
            current = self.search_layer(prepared, &current, self.config.ef_construction, layer);

            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors(&current, prepared, m);

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
                        // Reprune the neighbor from its own perspective: its stored code is
                        // all we have, so decode-free pruning uses the neighbor as the query.
                        let neighbor_prepared = self.nodes[neighbor_id]
                            .full_precision
                            .as_ref()
                            .map(|v| self.quantizer.prepare_query(v));
                        let connections: Vec<usize> = self.nodes[neighbor_id].connections[layer]
                            .iter()
                            .copied()
                            .collect();

                        let pruned = match neighbor_prepared.as_ref() {
                            Some(np) => self.select_neighbors(&connections, np, neighbor_m),
                            // No full-precision vector to re-prepare from; keep the closest
                            // by the inserting node's view rather than dropping arbitrarily.
                            None => self.select_neighbors(&connections, prepared, neighbor_m),
                        };
                        self.nodes[neighbor_id].connections[layer] = pruned.into_iter().collect();
                    }
                }
            }
        }
    }

    fn search_layer(
        &self,
        query: &PreparedQuery,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        for &ep in entry_points {
            let dist = self.quantizer.estimate_dist_sq(query, &self.nodes[ep].code);
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
                            .quantizer
                            .estimate_dist_sq(query, &self.nodes[neighbor_id].code);
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
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results.into_iter().map(|(_, id)| id).collect()
    }

    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &PreparedQuery,
        m: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                (
                    self.quantizer.estimate_dist_sq(query, &self.nodes[id].code),
                    id,
                )
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

impl crate::index::VectorIndex for RaBitQHNSWIndex {
    fn add(&mut self, document: Document) -> Result<()> {
        // Retain full precision by default: the estimator is a coarse first stage and
        // callers reaching through the trait cannot opt into reranking otherwise.
        self.add_with_full_precision(document)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_and_rerank(query, self.config.ef_search.max(k), k)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding,
            metadata: None,
        }
    }

    fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim)
            .map(|_| rand::RngExt::random_range(&mut rng, -1.0..1.0))
            .collect()
    }

    // ========================================================================
    // SQ8 HNSW Tests
    // ========================================================================

    #[test]
    fn test_sq8_hnsw_basic() {
        let index = SQ8HNSWIndex::for_normalized(128, QuantizedHNSWConfig::default());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_sq8_hnsw_add_single() {
        let mut index = SQ8HNSWIndex::for_normalized(3, QuantizedHNSWConfig::default());
        let doc = create_test_document("doc1", vec![0.5, -0.3, 0.8]);
        assert!(index.add(doc).is_ok());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_sq8_hnsw_search() {
        let mut index = SQ8HNSWIndex::for_normalized(128, QuantizedHNSWConfig::default());

        for i in 0..100 {
            let embedding = generate_random_vector(128, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        let query = generate_random_vector(128, 999);
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
        // Results should be sorted by score descending
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_sq8_hnsw_dimension_mismatch() {
        let mut index = SQ8HNSWIndex::for_normalized(128, QuantizedHNSWConfig::default());
        let doc = create_test_document("doc1", vec![0.5; 64]); // Wrong dimension
        assert!(index.add(doc).is_err());
    }

    #[test]
    fn test_sq8_memory_savings() {
        let dim = 384usize;
        let num_docs = 1000usize;

        let mut index = SQ8HNSWIndex::for_normalized(dim, QuantizedHNSWConfig::default());

        for i in 0..num_docs {
            let embedding = generate_random_vector(dim, i as u64);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        let memory = index.memory_usage();
        let full_precision = num_docs * dim * 4; // f32 size

        // SQ8 should use ~1/4 the memory for vectors
        assert!(
            memory < full_precision,
            "SQ8 memory: {}, full: {}",
            memory,
            full_precision
        );
    }

    // ========================================================================
    // Binary HNSW Tests
    // ========================================================================

    #[test]
    fn test_binary_hnsw_basic() {
        let index = BinaryHNSWIndex::new(128, QuantizedHNSWConfig::default());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_binary_hnsw_search() {
        let mut index = BinaryHNSWIndex::new(128, QuantizedHNSWConfig::default());

        for i in 0..100 {
            let embedding = generate_random_vector(128, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        let query = generate_random_vector(128, 999);
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_binary_hnsw_search_and_rerank() {
        let mut index = BinaryHNSWIndex::with_full_precision(128, QuantizedHNSWConfig::default());

        for i in 0..100 {
            let embedding = generate_random_vector(128, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add_with_full_precision(doc).unwrap();
        }

        let query = generate_random_vector(128, 999);

        // Two-phase search should give better results than binary-only
        let results = index.search_and_rerank(&query, 50, 10).unwrap();

        assert_eq!(results.len(), 10);
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_binary_memory_savings() {
        let dim = 384usize;
        let num_docs = 1000usize;

        let mut index = BinaryHNSWIndex::new(dim, QuantizedHNSWConfig::default());

        for i in 0..num_docs {
            let embedding = generate_random_vector(dim, i as u64);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        let memory = index.memory_usage();
        let full_precision = num_docs * dim * 4; // f32 size

        // Binary should use ~1/32 the memory for vectors
        assert!(
            memory < full_precision / 10,
            "Binary memory: {}, full: {}",
            memory,
            full_precision
        );
    }

    // ========================================================================
    // Recall Comparison Tests
    // ========================================================================

    #[test]
    fn test_recall_comparison() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let dim = 128;
        let num_docs = 500;
        let k = 10;

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..num_docs)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::RngExt::random_range(&mut rng, -1.0..1.0))
                    .collect()
            })
            .collect();

        // Build indices
        let mut sq8_index = SQ8HNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());
        let mut binary_index =
            BinaryHNSWIndex::with_full_precision(dim, QuantizedHNSWConfig::default());

        for (i, vec) in vectors.iter().enumerate() {
            let doc = create_test_document(&format!("doc{}", i), vec.clone());
            sq8_index.add(doc.clone()).unwrap();
            binary_index.add_with_full_precision(doc).unwrap();
        }

        // Test with random queries
        let query = generate_random_vector(dim, 9999);

        // Compute ground truth (brute force)
        let mut ground_truth: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let similarity = crate::vector::cosine_similarity(&query, v).unwrap();
                (i, similarity)
            })
            .collect();
        ground_truth.sort_by(|a, b| b.1.total_cmp(&a.1));
        let ground_truth_top_k: std::collections::HashSet<_> =
            ground_truth[..k].iter().map(|(i, _)| *i).collect();

        // Test SQ8 recall
        let sq8_results = sq8_index.search(&query, k).unwrap();
        let sq8_ids: std::collections::HashSet<_> = sq8_results
            .iter()
            .map(|r| r.id.strip_prefix("doc").unwrap().parse::<usize>().unwrap())
            .collect();
        let sq8_recall = ground_truth_top_k.intersection(&sq8_ids).count();

        // Test Binary recall (with reranking)
        let binary_results = binary_index.search_and_rerank(&query, 50, k).unwrap();
        let binary_ids: std::collections::HashSet<_> = binary_results
            .iter()
            .map(|r| r.id.strip_prefix("doc").unwrap().parse::<usize>().unwrap())
            .collect();
        let binary_recall = ground_truth_top_k.intersection(&binary_ids).count();

        println!("SQ8 recall@{}: {}/{}", k, sq8_recall, k);
        println!("Binary+rerank recall@{}: {}/{}", k, binary_recall, k);

        // Note: Recall can vary significantly based on data distribution.
        // With random vectors in high dimensions, recall tends to be lower.
        // We're testing that the indices work, not exact recall guarantees.
        // SQ8 should have at least 40% recall (conservative for random data)
        assert!(sq8_recall >= 4, "SQ8 recall too low: {}/{}", sq8_recall, k);
        // Binary with reranking should have at least 30% recall
        assert!(
            binary_recall >= 3,
            "Binary recall too low: {}/{}",
            binary_recall,
            k
        );
    }

    #[test]
    fn test_sq8_add_nan_embedding_rejected() {
        let mut index = SQ8HNSWIndex::for_normalized(8, QuantizedHNSWConfig::default());
        let doc =
            create_test_document("nan_doc", vec![0.1, 0.2, f32::NAN, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let result = index.add(doc);
        assert!(result.is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_binary_add_nan_embedding_rejected() {
        let mut index = BinaryHNSWIndex::new(8, QuantizedHNSWConfig::default());
        let doc =
            create_test_document("nan_doc", vec![0.1, 0.2, 0.3, f32::NAN, 0.5, 0.6, 0.7, 0.8]);
        let result = index.add(doc);
        assert!(result.is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_binary_add_with_full_precision_nan_embedding_rejected() {
        let mut index = BinaryHNSWIndex::with_full_precision(8, QuantizedHNSWConfig::default());
        let doc = create_test_document("nan_doc", vec![f32::NAN; 8]);
        let result = index.add_with_full_precision(doc);
        assert!(result.is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_quantized_search_with_nan_query_does_not_panic() {
        let dim = 8;

        let mut sq8 = SQ8HNSWIndex::for_normalized(dim, QuantizedHNSWConfig::default());
        sq8.add(create_test_document("sq8_doc", vec![0.1; dim]))
            .unwrap();

        let mut binary = BinaryHNSWIndex::new(dim, QuantizedHNSWConfig::default());
        binary
            .add(create_test_document("bin_doc", vec![0.2; dim]))
            .unwrap();

        let query = vec![f32::NAN; dim];

        let sq8_outcome = std::panic::catch_unwind(|| sq8.search(&query, 1));
        assert!(
            sq8_outcome.is_ok(),
            "SQ8 search panicked when query contains NaN"
        );

        let binary_outcome = std::panic::catch_unwind(|| binary.search(&query, 1));
        assert!(
            binary_outcome.is_ok(),
            "Binary search panicked when query contains NaN"
        );
    }

    // ========================================================================
    // RaBitQ HNSW Tests
    // ========================================================================

    /// SIFT-shaped fixture: wholly non-negative, values across the full 0..255 range,
    /// with cluster structure *and* real per-dimension variance within each cluster.
    ///
    /// Both properties matter. Non-negativity is what makes a zero-threshold binary
    /// code degenerate (every bit sets). The per-dimension variance is what makes the
    /// data separable *at all* — a fixture whose intra-cluster spread is tiny relative
    /// to its mean is unseparable by any 1-bit code, and would fail this test for
    /// reasons that have nothing to do with the bug under test.
    fn sift_like(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let clusters = 8;
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| (next() % 256) as f32).collect())
            .collect();

        (0..count)
            .map(|i| {
                let c = &centers[i % clusters];
                c.iter()
                    .map(|&x| {
                        let noise = (next() % 81) as f32 - 40.0; // +/- 40
                        (x + noise).clamp(0.0, 255.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Regression test for the bug this index exists to fix.
    ///
    /// `BinaryQuantizer` thresholds at zero, so on wholly non-negative data every bit
    /// sets, all codes collapse to all-ones, and retrieval degenerates to arbitrary
    /// order. RaBitQ centers on a fitted centroid, so it must still find the exact
    /// vector it was given as a query.
    #[test]
    fn rabitq_retrieves_exactly_on_nonnegative_data() {
        let dim = 32;
        let vectors = sift_like(128, dim);
        assert!(
            vectors.iter().flatten().all(|&x| x >= 0.0),
            "fixture must be non-negative to exercise the bug"
        );

        let mut index = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            index
                .add_with_full_precision(create_test_document(&format!("doc{i}"), v.clone()))
                .unwrap();
        }
        assert_eq!(index.len(), vectors.len());

        // Query with vectors that are already in the index: the top hit must be itself.
        let mut hits = 0;
        for (i, v) in vectors.iter().enumerate() {
            let results = index.search_and_rerank(v, 64, 1).unwrap();
            if results.first().map(|r| r.id.as_str()) == Some(format!("doc{i}").as_str()) {
                hits += 1;
            }
        }

        // Exact-match self-retrieval through a 1-bit first stage + exact rerank should
        // be essentially perfect. Binary scores near zero here.
        assert!(
            hits >= 122, // >= 95% of 128
            "RaBitQ self-retrieval on non-negative data: {hits}/128 (expected >= 122). \
             A collapse toward 0 means the codes are degenerate."
        );
    }

    /// Pins the bug that RaBitQHNSWIndex exists to fix, on the same fixture.
    ///
    /// If this ever starts passing at RaBitQ-like rates, someone has fixed
    /// BinaryQuantizer's threshold and the deprecation can be revisited.
    #[test]
    #[allow(deprecated)]
    fn binary_degenerates_on_nonnegative_data() {
        let dim = 32;
        let vectors = sift_like(128, dim);

        let mut binary = BinaryHNSWIndex::with_full_precision(dim, QuantizedHNSWConfig::default());
        let mut rabitq = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            let id = format!("doc{i}");
            binary
                .add_with_full_precision(create_test_document(&id, v.clone()))
                .unwrap();
            rabitq
                .add_with_full_precision(create_test_document(&id, v.clone()))
                .unwrap();
        }

        // Every code is all-ones, so the binary quantizer cannot distinguish any two
        // vectors: quantize the whole corpus and it collapses to a single distinct code.
        let bq = BinaryQuantizer::new(dim);
        let distinct: HashSet<Vec<u8>> = vectors
            .iter()
            .map(|v| bq.quantize(v).data.clone())
            .collect();
        assert_eq!(
            distinct.len(),
            1,
            "expected all binary codes to collapse to one; got {} distinct",
            distinct.len()
        );

        // And that collapse shows up as retrieval quality, through the real index.
        let score = |hits: usize| hits as f32 / vectors.len() as f32;
        let count_self_hits = |f: &dyn Fn(&[f32], usize) -> Vec<String>| {
            vectors
                .iter()
                .enumerate()
                .filter(|(i, v)| f(v, 1).first() == Some(&format!("doc{i}")))
                .count()
        };

        let bin_hits = count_self_hits(&|v, k| {
            binary
                .search(v, k)
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect()
        });
        let rb_hits = count_self_hits(&|v, k| {
            rabitq
                .search_and_rerank(v, 64, k)
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect()
        });

        assert!(
            score(rb_hits) > score(bin_hits) + 0.5,
            "RaBitQ should dominate degenerate binary by a wide margin on non-negative data; \
             got RaBitQ {:.0}% vs Binary {:.0}%",
            score(rb_hits) * 100.0,
            score(bin_hits) * 100.0
        );
    }

    #[test]
    fn rabitq_rejects_dimension_mismatch() {
        let vectors = sift_like(32, 16);
        let mut index = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());
        index
            .add_with_full_precision(create_test_document("a", vectors[0].clone()))
            .unwrap();

        assert!(index
            .add(create_test_document("bad", vec![1.0; 8]))
            .is_err());
        assert!(index.search(&[1.0; 8], 1).is_err());
    }

    #[test]
    fn rabitq_empty_index_returns_no_results() {
        let vectors = sift_like(16, 8);
        let index = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());

        assert!(index.is_empty());
        assert!(index.search(&vectors[0], 5).unwrap().is_empty());
        assert!(index
            .search_and_rerank(&vectors[0], 10, 5)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn rabitq_rejects_non_finite_embeddings() {
        let vectors = sift_like(16, 8);
        let mut index = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());

        let mut bad = vectors[0].clone();
        bad[0] = f32::NAN;
        assert!(index.add(create_test_document("nan", bad)).is_err());

        let mut inf = vectors[0].clone();
        inf[0] = f32::INFINITY;
        assert!(index.add(create_test_document("inf", inf)).is_err());
    }

    #[test]
    fn rabitq_clear_resets_index() {
        let vectors = sift_like(32, 8);
        let mut index = RaBitQHNSWIndex::fit(&vectors, QuantizedHNSWConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            index
                .add_with_full_precision(create_test_document(&format!("d{i}"), v.clone()))
                .unwrap();
        }
        assert_eq!(index.len(), 32);

        index.clear();
        assert!(index.is_empty());
        assert!(index.search(&vectors[0], 1).unwrap().is_empty());
    }
}
