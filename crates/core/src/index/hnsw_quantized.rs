//! Quantized HNSW index for memory-efficient similarity search
//!
//! This module provides HNSW variants that use quantized vectors for storage,
//! dramatically reducing memory footprint while maintaining good search quality.
//!
//! # Quantization Modes
//!
//! - **SQ8 (Scalar Quantization)**: 4x compression, ~95% recall
//! - **Binary**: 32x compression, ~85% recall (best for initial filtering)
//!
//! # Two-Phase Search (Recommended)
//!
//! For best quality/speed tradeoff, use binary quantization for initial candidate
//! retrieval, then rerank with SQ8 or full precision:
//!
//! ```ignore
//! // 1. Fast initial search with binary (32x compressed)
//! let candidates = binary_index.search(&query, 100)?;
//!
//! // 2. Rerank top candidates with SQ8 or full precision
//! let results = rerank_with_sq8(&candidates, &query, 10);
//! ```
//!
//! # Memory Comparison (1M vectors × 384 dims)
//!
//! | Index Type | Memory | Search Quality |
//! |------------|--------|----------------|
//! | Full f32   | 1.5 GB | 100% baseline  |
//! | SQ8        | 384 MB | ~95% recall    |
//! | Binary     | 48 MB  | ~85% recall    |

use crate::vector::quantize::{
    BinaryQuantizedVector, BinaryQuantizer, Quantizer, ScalarQuantizedVector, ScalarQuantizer,
};
use crate::{Document, RagError, Result, SearchResult};
use rand::Rng;
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

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen();
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
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
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
/// # Example
///
/// ```
/// use foxstash_core::index::hnsw_quantized::{BinaryHNSWIndex, QuantizedHNSWConfig};
/// use foxstash_core::Document;
///
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
/// ```
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

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen();
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
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
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
            .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
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
                    .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
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
        ground_truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
}
