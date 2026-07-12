//! Quantized HNSW index for memory-efficient similarity search
//!
//! This module provides [`RaBitQHNSWIndex`], an HNSW variant that uses RaBitQ 1-bit
//! quantized vectors for storage (32x compression, 1 bit/dim with an unbiased
//! estimator), dramatically reducing memory footprint while maintaining good search
//! quality.
//!
//! For 4x (SQ8) scalar-quantized storage, use the main
//! [`HNSWIndex`](super::hnsw::HNSWIndex) with
//! [`Storage::SQ8`](super::hnsw::Storage::SQ8) instead — it dominates a dedicated SQ8
//! HNSW variant on every measured axis (recall, QPS, and build time).
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
//! | Index Type | Memory | Traversal reads          | Rerank            |
//! |------------|--------|---------------------------|--------------------|
//! | Full f32              | 1.5 GB | full vector per node visit | n/a (already exact) |
//! | SQ8 (main `HNSWIndex`) | 384 MB | 1-byte/dim code per visit | `rerank_candidates` against retained f32 |
//! | RaBitQ (this module)  | 48 MB  | 1-bit/dim code per visit  | `search_and_rerank` against retained f32 |
//!
//! Both traverse the graph on their compressed code only — the retained full-precision
//! vector, if any, is never touched during traversal, only during the rerank pass over
//! the final candidate pool. Recall depends on the rerank pool size and is not asserted
//! here; measure it with `cargo run --release -p foxstash-benches --example
//! quantizer_sift` before relying on a number, per this repo's history of stale
//! benchmark claims (see `benchmarks/RESULTS.md`).

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
    ///
    /// This bounds the coarse (quantized-code) traversal, matching
    /// [`HNSWConfig`](super::hnsw::HNSWConfig)'s default so the two index families are
    /// tuned comparably rather than one silently exploring a smaller candidate set.
    pub ef_search: usize,
    /// Candidate pool size for the rerank stage (`search_and_rerank`), on index variants
    /// that support one.
    ///
    /// This is a separate knob from `ef_search`: `ef_search` bounds how far the *coarse*
    /// traversal looks, `rerank_candidates` bounds how many of those coarse hits get
    /// rescored against full-precision vectors. A larger pool costs more exact-distance
    /// computations at rerank time — each one touches a full-precision vector, spending
    /// back some of the DRAM traffic the compressed traversal saved — but a pool smaller
    /// than the true top-k's rank under the coarse metric will silently drop correct
    /// results before rerank ever sees them.
    pub rerank_candidates: usize,
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
            ef_search: 100,
            rerank_candidates: 100,
            ml: 1.0 / (m as f32).ln(),
        }
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
/// RaBitQ's unbiased estimator centers the data before thresholding, so unlike a
/// zero-threshold binary code it does not degenerate on wholly non-negative
/// embeddings (SIFT descriptors, and the output of any ReLU-activated model), where
/// a zero threshold sets every bit and collapses every code to the same value.
///
/// The quantizer must be fitted on training data (it needs a centroid), so
/// there is no `new(dim)` constructor — use [`RaBitQHNSWIndex::fit`].
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
        self.search_and_rerank(query, self.config.rerank_candidates.max(k), k)
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

    #[test]
    fn quantized_hnsw_config_defaults_match_full_precision_ef_search() {
        let config = QuantizedHNSWConfig::default();
        // Regression guard: this used to be 50, silently exploring half the candidate
        // set that HNSWConfig's default (100) does.
        assert_eq!(config.ef_search, 100);
        assert!(config.rerank_candidates > 0);
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

    /// Regression test for the class of bug RaBitQ's centering avoids: a zero-threshold
    /// binary code would set every bit on wholly non-negative data, collapsing all
    /// codes to the same value and degenerating retrieval to arbitrary order. RaBitQ
    /// centers on a fitted centroid, so it must still find the exact vector it was
    /// given as a query.
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
        // be essentially perfect.
        assert!(
            hits >= 122, // >= 95% of 128
            "RaBitQ self-retrieval on non-negative data: {hits}/128 (expected >= 122). \
             A collapse toward 0 means the codes are degenerate."
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
