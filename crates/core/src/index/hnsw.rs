//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! Based on the paper "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018).
//!
//! The HNSW algorithm creates a multi-layer graph structure where:
//! - Layer 0 contains all elements
//! - Higher layers contain exponentially fewer elements
//! - Each element has connections to its nearest neighbors at each layer
//! - Search starts at the top layer and zooms in to find nearest neighbors

use crate::{Document, Result, SearchResult};
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::{max, Reverse};
use std::collections::{BinaryHeap, HashSet};

/// Cross-platform cache line prefetch (read hint).
/// x86_64: _mm_prefetch T0, aarch64: PLDL1KEEP, other: no-op.
#[inline(always)]
unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Use inline asm instead of std::arch::aarch64::_prefetch which is unstable
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch multiple cache lines starting at `ptr` (64 bytes per line).
#[inline(always)]
unsafe fn prefetch_embedding(ptr: *const u8, cache_lines: usize) {
    for i in 0..cache_lines {
        prefetch_read(ptr.add(i * 64));
    }
}

/// Wrapper for f32 that implements Ord for use in BinaryHeap
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

/// Packed bitset for visited node tracking.
///
/// 1 bit per node = 12.5 KB for 100K nodes (fits L1 cache).
/// Replaces the previous generation-based Vec<u64> (800 KB, spills to L2/L3).
struct BitsetVisited {
    bits: Vec<u64>,
}

impl BitsetVisited {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
        }
    }

    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        debug_assert!(
            (node >> 6) < self.bits.len(),
            "BitsetVisited::is_visited: node {} out of bounds (capacity {})",
            node,
            self.bits.len() * 64
        );
        let word = unsafe { *self.bits.get_unchecked(node >> 6) };
        word & (1u64 << (node & 63)) != 0
    }

    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        debug_assert!(
            (node >> 6) < self.bits.len(),
            "BitsetVisited::mark_visited: node {} out of bounds (capacity {})",
            node,
            self.bits.len() * 64
        );
        unsafe {
            *self.bits.get_unchecked_mut(node >> 6) |= 1u64 << (node & 63);
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.bits.fill(0);
    }
}

/// Reusable search context to avoid allocations during search
///
/// Provides ~2-3x speedup over allocating new structures each query.
/// Uses packed bitset for cache-friendly visited tracking (~12.5 KB for 100K nodes).
pub struct SearchContext {
    /// Packed bitset for visited tracking (fits L1 cache)
    visited: BitsetVisited,
    /// Number of nodes this context supports
    capacity: usize,
    /// Reusable min-heap for candidates
    candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>>,
    /// Reusable max-heap for best results
    best: BinaryHeap<(OrderedFloat, usize)>,
}

impl SearchContext {
    /// Create a new search context for an index with `n` nodes
    pub fn new(n: usize) -> Self {
        Self {
            visited: BitsetVisited::new(n),
            capacity: n,
            candidates: BinaryHeap::with_capacity(256),
            best: BinaryHeap::with_capacity(256),
        }
    }

    /// Reset for a new search — clears bitset (~12.5 KB memset for 100K nodes)
    #[inline]
    fn reset(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.best.clear();
    }

    /// Check if node was visited in current search
    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        self.visited.is_visited(node)
    }

    /// Mark node as visited
    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        self.visited.mark_visited(node);
    }
}

/// Strategy for building the HNSW index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    /// Sequential insertion - slower but guarantees high recall (97%+)
    /// Best for: production search, accuracy-critical applications
    #[default]
    Sequential,
    /// Parallel insertion using layer-copying approach
    /// Fast build (6-10x faster), good recall at small scale (<50k)
    /// May have lower recall at larger scales (needs more work)
    Parallel,
    /// Automatically choose based on dataset size
    /// Uses Parallel for <50k vectors (where it works well)
    /// Uses Sequential for larger datasets (reliability over speed)
    Auto,
}

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Number of bidirectional links created for each element (except layer 0)
    /// Typical value: 16-32. Higher values increase recall but use more memory.
    pub m: usize,

    /// Number of bidirectional links created for each element in layer 0
    /// Typical value: 2 * m (64 for m=32)
    pub m0: usize,

    /// Size of the dynamic candidate list during construction
    /// Typical value: 100-200. Higher values improve quality but slow down construction.
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search
    /// Typical value: 50-100. Higher values improve recall but slow down search.
    pub ef_search: usize,

    /// Normalization factor for level generation
    /// Typical value: 1.0 / ln(m) ≈ 0.29 for m=32
    pub ml: f32,

    /// Use the heuristic neighbor selection algorithm (Algorithm 4 from paper)
    /// When true, selects diverse neighbors that aren't "behind" already-selected ones.
    /// This improves graph connectivity and recall at slight construction cost.
    pub use_heuristic: bool,

    /// When using heuristic, also consider neighbors of candidates (extend_candidates)
    /// This can find better neighbors but increases construction time.
    pub extend_candidates: bool,

    /// When pruning, keep some pruned connections for better connectivity
    /// Only applies when use_heuristic is true.
    pub keep_pruned_connections: bool,

    /// Build strategy: Sequential (high recall), Parallel (faster), or Auto
    pub build_strategy: BuildStrategy,

    /// Random seed for reproducible builds (None = random)
    pub seed: Option<u64>,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 32; // Match instant-distance for good recall
        Self {
            m,
            m0: m * 2,
            ef_construction: 100,
            ef_search: 100,
            ml: 1.0 / (m as f32).ln(),
            use_heuristic: true,
            extend_candidates: false,
            keep_pruned_connections: true,
            build_strategy: BuildStrategy::default(),
            seed: None,
        }
    }
}

impl HNSWConfig {
    /// Use simple nearest-neighbor selection (faster construction, lower recall)
    pub fn with_simple_selection(mut self) -> Self {
        self.use_heuristic = false;
        self
    }

    /// Enable extended candidate search (better quality, slower construction)
    pub fn with_extended_candidates(mut self) -> Self {
        self.extend_candidates = true;
        self
    }

    /// Set ef_search parameter
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set build strategy
    /// - Sequential: slower but guarantees high recall (97%+)
    /// - Parallel: faster using instant-distance's layer-copying approach
    /// - Auto: Sequential for <50k vectors, Parallel for larger
    pub fn with_build_strategy(mut self, strategy: BuildStrategy) -> Self {
        self.build_strategy = strategy;
        self
    }

    /// Set random seed for reproducible builds
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set ef_construction parameter
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set M parameter (connections per node)
    pub fn with_m(mut self, m: usize) -> Self {
        assert!(m <= 127, "m must be <= 127 (m0 = 2*m must fit in u8)");
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f32).ln();
        self
    }
}

/// HNSW index for efficient similarity search
///
/// Uses Struct-of-Arrays (SoA) layout for better cache locality:
/// - Hot path: embeddings stored contiguously for SIMD-friendly access
/// - Cold path: document metadata stored separately
pub struct HNSWIndex {
    /// Dimensionality of embeddings
    embedding_dim: usize,
    /// Configuration parameters
    config: HNSWConfig,

    // === HOT PATH (accessed during every distance computation) ===
    /// All embeddings stored contiguously: embeddings[i * dim .. (i+1) * dim]
    embeddings: Vec<f32>,
    /// Precomputed L2 norms for each vector (avoids recomputing per distance call)
    norms: Vec<f32>,

    // === GRAPH STRUCTURE ===
    /// Connections for each node at each layer: connections[node_id][layer] -> neighbors
    /// Uses Vec<u32> instead of HashSet for cache-friendly traversal (4-5x faster search)
    connections: Vec<Vec<Vec<u32>>>,

    /// Flat layer 0 connection cache: `node_id * m0 + i` → neighbor_id.
    /// Eliminates triple-indirection for the hot search path (layer 0 = 99% of search time).
    connections_l0: Vec<u32>,
    /// Actual neighbor count per node at layer 0
    connections_l0_count: Vec<u8>,

    // === COLD PATH (only accessed when returning results) ===
    /// Document IDs
    ids: Vec<String>,
    /// Document content
    contents: Vec<String>,
    /// Document metadata
    metadata: Vec<Option<serde_json::Value>>,

    /// Entry point (node index with highest layer)
    entry_point: Option<usize>,
    /// Maximum layer in the index
    max_layer: usize,
}

impl HNSWIndex {
    /// Creates a new HNSW index with custom configuration
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embedding vectors
    /// * `config` - HNSW configuration parameters
    pub fn new(embedding_dim: usize, config: HNSWConfig) -> Self {
        assert!(
            config.m <= 127,
            "m must be <= 127 (m0 = 2*m must fit in u8), got m={}",
            config.m
        );
        Self {
            embedding_dim,
            config,
            embeddings: Vec::new(),
            norms: Vec::new(),
            connections: Vec::new(),
            connections_l0: Vec::new(),
            connections_l0_count: Vec::new(),
            ids: Vec::new(),
            contents: Vec::new(),
            metadata: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Creates a new HNSW index with default configuration
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embedding vectors
    pub fn with_defaults(embedding_dim: usize) -> Self {
        Self::new(embedding_dim, HNSWConfig::default())
    }

    /// Build an HNSW index from embeddings using the configured strategy
    ///
    /// This is the recommended way to create an index from bulk embeddings.
    /// The build strategy is controlled by `config.build_strategy`:
    /// - `Sequential`: Slower but guarantees high recall (97%+)
    /// - `Parallel`: Faster using instant-distance's layer-copying approach
    /// - `Auto`: Sequential for <50k vectors, Parallel for larger
    ///
    /// # Arguments
    /// * `embeddings` - Vector of embedding vectors (all must have same dimension)
    /// * `config` - HNSW configuration parameters
    ///
    /// # Returns
    /// A new HNSWIndex built from the embeddings
    ///
    /// # Example
    /// ```
    /// use foxstash_core::index::{HNSWIndex, HNSWConfig, BuildStrategy};
    ///
    /// // Create a small set of 4-dimensional embeddings
    /// let embeddings = vec![
    ///     vec![1.0, 0.0, 0.0, 0.0],
    ///     vec![0.0, 1.0, 0.0, 0.0],
    ///     vec![0.0, 0.0, 1.0, 0.0],
    ///     vec![0.0, 0.0, 0.0, 1.0],
    ///     vec![0.5, 0.5, 0.0, 0.0],
    /// ];
    ///
    /// let config = HNSWConfig::default()
    ///     .with_build_strategy(BuildStrategy::Parallel)
    ///     .with_seed(42);
    /// let index = HNSWIndex::build(embeddings, config);
    /// assert_eq!(index.len(), 5);
    /// ```
    pub fn build(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        if embeddings.is_empty() {
            return Self::new(0, config);
        }

        let expected_dim = embeddings[0].len();
        for (i, embedding) in embeddings.iter().enumerate() {
            assert!(
                embedding.len() == expected_dim,
                "All embeddings must have the same dimension: expected {}, got {} at index {}",
                expected_dim,
                embedding.len(),
                i
            );
        }

        let n = embeddings.len();
        let strategy = match config.build_strategy {
            BuildStrategy::Auto => {
                // Parallel works well for <50k, Sequential for larger
                if n < 50_000 {
                    BuildStrategy::Parallel
                } else {
                    BuildStrategy::Sequential
                }
            }
            other => other,
        };

        match strategy {
            BuildStrategy::Sequential => Self::build_sequential(embeddings, config),
            BuildStrategy::Parallel | BuildStrategy::Auto => {
                Self::build_parallel(embeddings, config)
            }
        }
    }

    /// Build using sequential insertion (high recall, slower)
    fn build_sequential(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        let embedding_dim = embeddings[0].len();
        let n = embeddings.len();

        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);
        let ml = config.ml;

        // Pre-generate all node levels (clamp to EPSILON to prevent ln(0) = -inf)
        let levels: Vec<usize> = (0..n)
            .map(|_| {
                let r: f32 = rng.gen::<f32>().max(f32::EPSILON);
                (-r.ln() * ml).floor() as usize
            })
            .collect();

        let _max_level = *levels.iter().max().unwrap_or(&0);

        // Sort by level descending
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| levels[b].cmp(&levels[a]));

        // Create index
        let mut index = Self::new(embedding_dim, config);

        // Pre-allocate
        index.embeddings.reserve(n * embedding_dim);
        index.norms.reserve(n);
        index.connections.reserve(n);
        index.ids.reserve(n);
        index.contents.reserve(n);
        index.metadata.reserve(n);

        // Add nodes in sorted order
        for &i in &sorted_indices {
            let level = levels[i];
            let node_id = index.len();

            // Create connections (using Vec<u32> for cache-friendly traversal)
            let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                node_connections.push(Vec::new());
            }

            // Add to storage
            let norm = crate::vector::simd::norm_simd(&embeddings[i]);
            index.embeddings.extend_from_slice(&embeddings[i]);
            index.norms.push(norm);
            index.connections.push(node_connections);
            index.ids.push(i.to_string());
            index.contents.push(String::new());
            index.metadata.push(None);

            if index.entry_point.is_none() {
                index.entry_point = Some(node_id);
                index.max_layer = level;
                continue;
            }

            index.insert_node(node_id, level);

            if level > index.max_layer {
                index.max_layer = level;
                index.entry_point = Some(node_id);
            }
        }

        index.build_l0_cache();
        index
    }

    /// Returns the number of nodes in the index
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if the index is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Get embedding slice for a node (hot path - inline for performance)
    #[inline]
    fn get_embedding(&self, node_id: usize) -> &[f32] {
        let start = node_id * self.embedding_dim;
        let end = start + self.embedding_dim;
        &self.embeddings[start..end]
    }

    /// Get layer 0 neighbors from the flat cache (single indirection).
    #[inline(always)]
    fn get_neighbors_l0(&self, node_id: usize) -> &[u32] {
        let m0 = self.config.m0;
        let start = node_id * m0;
        let count = self.connections_l0_count[node_id] as usize;
        &self.connections_l0[start..start + count]
    }

    /// Build flat layer 0 cache from the nested `connections` structure.
    /// Called once after index construction for fast search-time access.
    fn build_l0_cache(&mut self) {
        let n = self.len();
        let m0 = self.config.m0;
        debug_assert!(m0 <= 255, "m0 exceeds u8 capacity for connections_l0_count");
        self.connections_l0 = vec![0u32; n * m0];
        self.connections_l0_count = vec![0u8; n];

        for i in 0..n {
            if !self.connections[i].is_empty() {
                let neighbors = &self.connections[i][0];
                let count = neighbors.len().min(m0);
                let start = i * m0;
                self.connections_l0[start..start + count].copy_from_slice(&neighbors[..count]);
                self.connections_l0_count[i] = count as u8;
            }
        }
    }

    /// Extend flat L0 cache to accommodate a new node (for incremental add).
    fn extend_l0_cache_for_new_node(&mut self) {
        let m0 = self.config.m0;
        self.connections_l0
            .resize(self.connections_l0.len() + m0, 0);
        self.connections_l0_count.push(0);
    }

    /// Adds a document to the index
    ///
    /// # Arguments
    /// * `document` - Document with embedding to add
    ///
    /// # Errors
    /// Returns error if embedding dimension doesn't match index dimension
    pub fn add(&mut self, document: Document) -> Result<()> {
        if document.embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
            });
        }

        if document.embedding.iter().any(|v| !v.is_finite()) {
            return Err(crate::RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".into(),
            ));
        }

        let node_id = self.len();
        let node_level = self.random_level();

        // Create connections for each layer (Vec<u32> for cache-friendly traversal)
        let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            node_connections.push(Vec::new());
        }

        // Add to SoA storage
        let norm = crate::vector::simd::norm_simd(&document.embedding);
        self.embeddings.extend_from_slice(&document.embedding);
        self.norms.push(norm);
        self.connections.push(node_connections);
        self.ids.push(document.id);
        self.contents.push(document.content);
        self.metadata.push(document.metadata);

        // Extend flat L0 cache for the new node
        self.extend_l0_cache_for_new_node();

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        let affected = self.insert_node(node_id, node_level);

        // Update only the nodes whose L0 connections changed (O(m0) per node).
        self.sync_l0_cache_for_nodes(&affected);

        // Update entry point if this node has more layers
        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Add just an embedding vector (faster than add() for bulk operations)
    ///
    /// This method is optimized for benchmarks and bulk data loading where
    /// you only need to store embeddings without document content.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this vector
    /// * `embedding` - The embedding vector to add
    ///
    /// # Errors
    /// Returns error if embedding dimension doesn't match index dimension
    pub fn add_embedding(&mut self, id: String, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }

        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(crate::RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".into(),
            ));
        }

        let node_id = self.len();
        let node_level = self.random_level();

        // Create connections for each layer (Vec<u32> for cache-friendly traversal)
        let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            node_connections.push(Vec::new());
        }

        // Add to SoA storage
        let norm = crate::vector::simd::norm_simd(&embedding);
        self.embeddings.extend_from_slice(&embedding);
        self.norms.push(norm);
        self.connections.push(node_connections);
        self.ids.push(id);
        self.contents.push(String::new());
        self.metadata.push(None);

        // Extend flat L0 cache for the new node
        self.extend_l0_cache_for_new_node();

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        let affected = self.insert_node(node_id, node_level);

        // Update only the nodes whose L0 connections changed (O(m0) per node).
        self.sync_l0_cache_for_nodes(&affected);

        // Update entry point if this node has more layers
        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Searches for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `k` - Number of results to return
    ///
    /// # Errors
    /// Returns error if query dimension doesn't match index dimension
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut ctx = SearchContext::new(self.len());
        self.search_with_context(query, k, &mut ctx)
    }

    /// Searches for k nearest neighbors for multiple queries in parallel
    ///
    /// # Arguments
    /// * `queries` - Slice of query embedding vectors
    /// * `k` - Number of results to return per query
    ///
    /// # Errors
    /// Returns error if any query dimension doesn't match index dimension
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        queries
            .par_iter()
            .map(|query| self.search(query, k))
            .collect()
    }

    /// Creates a reusable search context for faster repeated searches
    ///
    /// Use with `search_with_context` for ~2-3x speedup when doing many queries
    pub fn create_search_context(&self) -> SearchContext {
        SearchContext::new(self.len())
    }

    /// Fast search using a reusable context (avoids allocations)
    ///
    /// ~2-3x faster than `search()` when doing many queries.
    /// Create context once with `create_search_context()`, reuse for all queries.
    pub fn search_with_context(
        &self,
        query: &[f32],
        k: usize,
        ctx: &mut SearchContext,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure search context is large enough for current index size.
        if ctx.capacity < self.len() {
            *ctx = SearchContext::new(self.len());
        }

        // Precompute query norm once for fused distance
        let query_norm = crate::vector::simd::norm_simd(query);

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Search from top layer to layer 1
        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(query, &current_nearest, 1, layer, ctx, query_norm);
        }

        // Search layer 0 with ef_search candidates
        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer(query, &current_nearest, ef, 0, ctx, query_norm);

        // Convert to SearchResults (cold path — distances already computed above)
        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                // Use fused distance for final scoring too
                let dist = self.distance_to_node(query, node_id, query_norm);
                let score = 1.0 - dist;
                SearchResult {
                    id: self.ids[node_id].clone(),
                    content: self.contents[node_id].clone(),
                    score,
                    metadata: self.metadata[node_id].clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(k);

        Ok(results)
    }

    /// Fast batch search using reusable contexts
    pub fn search_batch_fast(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;
        use std::cell::RefCell;

        // Thread-local search contexts
        thread_local! {
            static CTX: RefCell<Option<SearchContext>> = const { RefCell::new(None) };
        }

        queries
            .par_iter()
            .map(|query| {
                CTX.with(|ctx| {
                    let mut ctx_ref = ctx.borrow_mut();
                    if ctx_ref.is_none() || ctx_ref.as_ref().unwrap().capacity < self.len() {
                        *ctx_ref = Some(SearchContext::new(self.len()));
                    }
                    self.search_with_context(query, k, ctx_ref.as_mut().unwrap())
                })
            })
            .collect()
    }

    /// Clears all documents from the index
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.norms.clear();
        self.connections.clear();
        self.connections_l0.clear();
        self.connections_l0_count.clear();
        self.ids.clear();
        self.contents.clear();
        self.metadata.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get all documents in the index
    ///
    /// Returns a vector containing clones of all documents in the index.
    /// Useful for serialization and persistence.
    ///
    /// # Returns
    /// * `Vec<Document>` - Vector of all documents
    pub fn get_all_documents(&self) -> Vec<Document> {
        (0..self.len())
            .map(|i| Document {
                id: self.ids[i].clone(),
                content: self.contents[i].clone(),
                embedding: self.get_embedding(i).to_vec(),
                metadata: self.metadata[i].clone(),
            })
            .collect()
    }

    /// Get the HNSW configuration
    ///
    /// # Returns
    /// * `&HNSWConfig` - Reference to the configuration
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    /// Get the entry point node index
    ///
    /// # Returns
    /// * `Option<usize>` - Entry point node index, or None if index is empty
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Get the maximum layer in the index
    ///
    /// # Returns
    /// * `usize` - Maximum layer
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Get the embedding dimension
    ///
    /// # Returns
    /// * `usize` - Embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Generates a random level for a new node using exponential decay.
    ///
    /// Clamps the uniform sample to `[EPSILON, 1)` to prevent `ln(0.0) = -inf`.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen::<f32>().max(f32::EPSILON);
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    /// Inserts a node into the graph structure.
    ///
    /// Returns the IDs of all nodes whose layer-0 connections were modified
    /// (the new node itself plus all layer-0 neighbors it linked to). The caller
    /// can pass this list to `sync_l0_cache_for_nodes` instead of doing a full
    /// `build_l0_cache` rebuild.
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to insert
    /// * `node_level` - Maximum layer of the node
    fn insert_node(&mut self, node_id: usize, node_level: usize) -> Vec<usize> {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Get embedding once (hot path optimization)
        let node_embedding = self.get_embedding(node_id).to_vec();
        let query_norm = crate::vector::simd::norm_simd(&node_embedding);
        let mut ctx = SearchContext::new(self.len());

        // Search for nearest neighbors from top to target layer + 1
        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer(
                &node_embedding,
                &current_nearest,
                1,
                layer,
                &mut ctx,
                query_norm,
            );
        }

        // Collect L0-modified nodes for incremental cache update.
        // The new node itself is always modified at layer 0.
        let mut l0_modified: Vec<usize> = vec![node_id];

        // Insert into layers from top to bottom
        for layer in (0..=node_level).rev() {
            current_nearest = self.search_layer(
                &node_embedding,
                &current_nearest,
                self.config.ef_construction,
                layer,
                &mut ctx,
                query_norm,
            );

            // Determine M for this layer
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            // Select M nearest neighbors
            let neighbors = self.select_neighbors(&current_nearest, &node_embedding, m, layer);

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                // Add link from new node to neighbor (avoid duplicates)
                let neighbor_u32 = neighbor_id as u32;
                if !self.connections[node_id][layer].contains(&neighbor_u32) {
                    self.connections[node_id][layer].push(neighbor_u32);
                }

                // Only add bidirectional link if neighbor exists at this layer
                if layer < self.connections[neighbor_id].len() {
                    // Add link from neighbor to new node
                    let node_u32 = node_id as u32;
                    if !self.connections[neighbor_id][layer].contains(&node_u32) {
                        self.connections[neighbor_id][layer].push(node_u32);
                    }

                    // Prune neighbor's connections if needed
                    let neighbor_m = if layer == 0 {
                        self.config.m0
                    } else {
                        self.config.m
                    };

                    if self.connections[neighbor_id][layer].len() > neighbor_m {
                        let neighbor_embedding = self.get_embedding(neighbor_id).to_vec();
                        let neighbor_connections: Vec<usize> = self.connections[neighbor_id][layer]
                            .iter()
                            .map(|&x| x as usize)
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_embedding,
                            neighbor_m,
                            layer,
                        );

                        self.connections[neighbor_id][layer] =
                            pruned.into_iter().map(|x| x as u32).collect();
                    }

                    // Track layer-0 nodes modified by bidirectional linking.
                    if layer == 0 {
                        l0_modified.push(neighbor_id);
                    }
                }
            }
        }

        l0_modified
    }

    /// Updates the flat L0 cache for a specific subset of nodes.
    ///
    /// Called after incremental insertions to avoid the O(n) full rebuild
    /// that `build_l0_cache` performs. Each node update is O(m0).
    fn sync_l0_cache_for_nodes(&mut self, node_ids: &[usize]) {
        let m0 = self.config.m0;
        // Ensure cache is large enough for all nodes.
        let needed = self.connections.len() * m0;
        if self.connections_l0.len() < needed {
            self.connections_l0.resize(needed, 0u32);
            self.connections_l0_count
                .resize(self.connections.len(), 0u8);
        }
        for &node_id in node_ids {
            if node_id < self.connections.len() && !self.connections[node_id].is_empty() {
                let neighbors = &self.connections[node_id][0];
                let count = neighbors.len().min(m0);
                let start = node_id * m0;
                self.connections_l0[start..start + count].copy_from_slice(&neighbors[..count]);
                for j in count..m0 {
                    self.connections_l0[start + j] = 0;
                }
                self.connections_l0_count[node_id] = count as u8;
            }
        }
    }

    /// Searches a specific layer for nearest neighbors.
    ///
    /// Uses reusable context, fused SIMD distance, prefetching, and batch heap updates.
    #[inline]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        ctx: &mut SearchContext,
        query_norm: f32,
    ) -> Vec<usize> {
        ctx.reset();

        // Initialize with entry points
        for &ep in entry_points {
            let dist = self.distance_to_node(query, ep, query_norm);
            ctx.candidates.push(Reverse((OrderedFloat(dist), ep)));
            ctx.best.push((OrderedFloat(dist), ep));
            ctx.mark_visited(ep);
        }

        while let Some(Reverse((current_dist, current_id))) = ctx.candidates.pop() {
            // If current is farther than the ef-th nearest, we're done
            if ctx.best.len() >= ef {
                if let Some(&(furthest_dist, _)) = ctx.best.peek() {
                    if current_dist > furthest_dist {
                        break;
                    }
                }
            }

            // Get neighbors: use flat L0 cache for layer 0, nested structure for upper layers
            let neighbors_l0_slice;
            let neighbors: &[u32] = if layer == 0 && !self.connections_l0.is_empty() {
                neighbors_l0_slice = self.get_neighbors_l0(current_id);
                neighbors_l0_slice
            } else if layer < self.connections[current_id].len() {
                &self.connections[current_id][layer]
            } else {
                &[]
            };

            if !neighbors.is_empty() {
                let n_neighbors = neighbors.len();

                // 128d * 4 bytes = 512 bytes = 8 cache lines; prefetch first 3 (192 bytes)
                const PREFETCH_AHEAD: usize = 2;
                const CACHE_LINES_PER_EMBEDDING: usize = 3;

                // Phase 1: Compute distances for all unvisited neighbors into stack buffer.
                // This separates compute from heap ops for better ILP and fewer branch
                // mispredictions.
                let mut batch_buf: [(f32, usize); 64] = [(0.0, 0); 64];
                let mut batch_count = 0usize;

                for (i, &neighbor_u32) in neighbors.iter().enumerate() {
                    let neighbor_id = neighbor_u32 as usize;

                    // Prefetch embeddings 2 neighbors ahead (multiple cache lines)
                    unsafe {
                        let lookahead = i + PREFETCH_AHEAD;
                        if lookahead < n_neighbors {
                            let ahead_id = neighbors[lookahead] as usize;
                            let ahead_ptr = self
                                .embeddings
                                .as_ptr()
                                .wrapping_add(ahead_id * self.embedding_dim)
                                as *const u8;
                            prefetch_embedding(ahead_ptr, CACHE_LINES_PER_EMBEDDING);

                            // Prefetch the visited bitset word for the lookahead neighbor
                            let bitset_ptr =
                                ctx.visited.bits.as_ptr().wrapping_add(ahead_id >> 6) as *const u8;
                            prefetch_read(bitset_ptr);

                            // Prefetch the norm for the lookahead neighbor
                            let norm_ptr = self.norms.as_ptr().wrapping_add(ahead_id) as *const u8;
                            prefetch_read(norm_ptr);
                        }
                    }

                    if !ctx.is_visited(neighbor_id) {
                        ctx.mark_visited(neighbor_id);
                        let dist = self.distance_to_node(query, neighbor_id, query_norm);
                        if batch_count < batch_buf.len() {
                            batch_buf[batch_count] = (dist, neighbor_id);
                            batch_count += 1;
                        }
                    }
                }

                // Phase 2: Batch heap updates from the computed distances.
                for &(dist, neighbor_id) in &batch_buf[..batch_count] {
                    let dist_ord = OrderedFloat(dist);

                    if ctx.best.len() < ef {
                        ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                        ctx.best.push((dist_ord, neighbor_id));
                    } else if let Some(&(furthest_dist, _)) = ctx.best.peek() {
                        if dist_ord < furthest_dist {
                            ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                            ctx.best.push((dist_ord, neighbor_id));

                            if ctx.best.len() > ef {
                                ctx.best.pop();
                            }
                        }
                    }
                }
            }
        }

        // Extract and return node IDs sorted by distance
        let mut results: Vec<(f32, usize)> = ctx
            .best
            .drain()
            .map(|(OrderedFloat(dist), id)| (dist, id))
            .collect();

        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results.into_iter().map(|(_, id)| id).collect()
    }

    /// Selects M best neighbors using a heuristic
    ///
    /// When `use_heuristic` is enabled (default), uses Algorithm 4 from the HNSW paper
    /// which ensures diversity by only selecting candidates that are closer to the query
    /// than to any already-selected neighbor. This prevents selecting neighbors that are
    /// "behind" other neighbors, improving graph connectivity.
    ///
    /// # Arguments
    /// * `candidates` - Candidate neighbor IDs
    /// * `query` - Query point
    /// * `m` - Number of neighbors to select
    /// * `layer` - Current layer
    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &[f32],
        m: usize,
        layer: usize,
    ) -> Vec<usize> {
        if !self.config.use_heuristic {
            // Simple heuristic: select M closest neighbors
            return self.select_neighbors_simple(candidates, query, m);
        }

        // Algorithm 4 from the HNSW paper: SELECT-NEIGHBORS-HEURISTIC
        // This ensures diversity by checking if each candidate is closer to query
        // than to any already-selected neighbor.

        // Optionally extend candidates with their neighbors
        let mut working_candidates: Vec<usize> = candidates.to_vec();
        if self.config.extend_candidates {
            let mut seen: HashSet<usize> = candidates.iter().copied().collect();
            for &candidate in candidates {
                if layer < self.connections[candidate].len() {
                    for &neighbor_u32 in &self.connections[candidate][layer] {
                        let neighbor = neighbor_u32 as usize;
                        if seen.insert(neighbor) {
                            working_candidates.push(neighbor);
                        }
                    }
                }
            }
        }

        // Score and sort candidates by distance to query
        let mut scored: Vec<(f32, usize)> = working_candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, self.get_embedding(id));
                (dist, id)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Select neighbors using the heuristic
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut pruned: Vec<(f32, usize)> = Vec::new();

        for (dist_to_query, candidate_id) in scored {
            if selected.len() >= m {
                break;
            }

            // Check if this candidate is closer to query than to any selected neighbor
            let candidate_embedding = self.get_embedding(candidate_id);
            let mut is_good = true;

            for &selected_id in &selected {
                let selected_embedding = self.get_embedding(selected_id);
                let dist_to_selected = self.distance(candidate_embedding, selected_embedding);

                // If candidate is closer to a selected neighbor than to query,
                // it's "behind" that neighbor and we should skip it
                if dist_to_selected < dist_to_query {
                    is_good = false;
                    pruned.push((dist_to_query, candidate_id));
                    break;
                }
            }

            if is_good {
                selected.push(candidate_id);
            }
        }

        // Optionally add back some pruned connections if we didn't get enough
        if self.config.keep_pruned_connections && selected.len() < m {
            for (_, pruned_id) in pruned {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&pruned_id) {
                    selected.push(pruned_id);
                }
            }
        }

        selected
    }

    /// Simple neighbor selection: just pick M closest
    #[inline]
    fn select_neighbors_simple(&self, candidates: &[usize], query: &[f32], m: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, self.get_embedding(id));
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Computes distance between two vectors
    /// Uses 1 - cosine_similarity for distance metric (SIMD accelerated)
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - crate::vector::simd::cosine_similarity_simd(a, b)
    }

    /// Fused distance from query to a stored node using precomputed norms.
    /// Single SIMD dispatch, single pass over data.
    #[inline]
    fn distance_to_node(&self, query: &[f32], node_id: usize, query_norm: f32) -> f32 {
        let embedding = self.get_embedding(node_id);
        let norm_b = self.norms[node_id];
        // Reuse the fused path but with query_norm precomputed too
        // We compute: 1 - dot(q, e) / (||q|| * ||e||)
        if query_norm == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        crate::vector::simd::cosine_distance_prenorm(query, embedding, norm_b)
    }

    /// Build an HNSW index from embeddings using parallel construction
    ///
    /// Uses instant-distance's layer-copying approach for safe parallelization:
    /// - All connections stored in zero layer (M*2 neighbors per node)
    /// - Upper layers are read-only snapshots (M neighbors, copied after each batch)
    /// - Process batches top-to-bottom: top batch sequential, rest parallel
    ///
    /// # Arguments
    /// * `embeddings` - Vector of embedding vectors (all must have same dimension)
    /// * `config` - HNSW configuration parameters
    ///
    /// # Returns
    /// A new HNSWIndex built from the embeddings
    pub fn build_parallel(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        assert!(!embeddings.is_empty(), "Cannot build from empty embeddings");
        let embedding_dim = embeddings[0].len();
        let n = embeddings.len();

        if n == 1 {
            return Self::build_single(embeddings, config);
        }

        let ml = config.ml;
        let ef_construction = config.ef_construction;
        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);

        // Calculate batch sizes (how many nodes per layer batch)
        // This determines insertion order, not graph structure
        let mut sizes = Vec::new();
        let mut num = n;
        loop {
            let next = (num as f32 * ml) as usize;
            if next < M_MAX {
                break;
            }
            sizes.push((num - next, num));
            num = next;
        }
        sizes.push((num, num));
        sizes.reverse();
        let num_batches = sizes.len();
        let top = LayerId(num_batches - 1);

        // Shuffle points randomly for insertion order
        assert!(n < u32::MAX as usize);
        let mut shuffled: Vec<(u32, usize)> = (0..n).map(|i| (rng.gen::<u32>(), i)).collect();
        shuffled.sort_unstable_by_key(|&(r, _)| r);

        // Reorder embeddings according to shuffle
        let points: Vec<Vec<f32>> = shuffled
            .iter()
            .map(|&(_, idx)| embeddings[idx].clone())
            .collect();

        // Build ranges for each batch
        let mut ranges = Vec::with_capacity(num_batches);
        for (i, (size, cumulative)) in sizes.into_iter().enumerate() {
            let start = cumulative - size;
            let batch_id = LayerId(num_batches - i - 1);
            // Skip first point (it's the entry point, inserted implicitly)
            ranges.push((batch_id, max(start, 1)..cumulative));
        }

        // Zero layer: all nodes, M*2 connections each (the LIVE data)
        let zero: Vec<RwLock<ZeroNode>> =
            (0..n).map(|_| RwLock::new(ZeroNode::default())).collect();

        // Upper layers: snapshots copied after each batch (READ-ONLY during search)
        let mut layers: Vec<Vec<UpperNode>> = vec![Vec::new(); top.0];

        // Search pool for thread-local state reuse
        let pool = SearchPool::new(n);

        // Process batches from top to bottom
        for (batch, range) in ranges {
            let end = range.end;

            if batch.0 == top.0 {
                // Top batch: insert sequentially (forms the backbone)
                for i in range {
                    Self::par_insert(
                        PointId(i as u32),
                        batch,
                        &zero,
                        &layers,
                        &points,
                        &pool,
                        ef_construction,
                        top,
                    );
                }
            } else {
                // Lower batches: insert in parallel (safe because upper layers are snapshots)
                range.into_par_iter().for_each(|i| {
                    Self::par_insert(
                        PointId(i as u32),
                        batch,
                        &zero,
                        &layers,
                        &points,
                        &pool,
                        ef_construction,
                        top,
                    );
                });
            }

            // After each batch, snapshot zero layer to create upper layer
            // layers[batch-1] = snapshot of zero[0..end] truncated to M neighbors
            if !batch.is_zero() {
                zero[..end]
                    .par_iter()
                    .map(|z| UpperNode::from_zero(&z.read()))
                    .collect_into_vec(&mut layers[batch.0 - 1]);
            }
        }

        // Convert to final index format
        Self::convert_parallel_to_index(zero, layers, points, shuffled, embedding_dim, config, top)
    }

    /// Insert a single node during parallel construction
    /// Always updates zero layer; searches use upper layer snapshots + zero layer
    #[allow(clippy::too_many_arguments)]
    fn par_insert(
        new: PointId,
        target_layer: LayerId, // The batch/layer this node belongs to
        zero: &[RwLock<ZeroNode>],
        layers: &[Vec<UpperNode>],
        points: &[Vec<f32>],
        pool: &SearchPool,
        ef_construction: usize,
        top: LayerId,
    ) {
        let mut search = pool.pop();
        search.visited.reserve(points.len());

        let point = &points[new.as_usize()];
        search.reset();

        // Start search from entry point (always node 0)
        search.push(PointId(0), point, points);

        // Descend through layers from top to bottom
        for cur in top.descend() {
            // Use ef=1 for greedy descent ABOVE target layer
            // Use ef_construction at target layer and below
            search.ef = if cur.0 <= target_layer.0 {
                ef_construction
            } else {
                1
            };

            if cur.0 > target_layer.0 {
                // Above target layer: search upper layer snapshot, then cull
                if cur.0 <= layers.len() && !layers[cur.0 - 1].is_empty() {
                    search.search_upper(point, &layers[cur.0 - 1], points, M_MAX);
                    search.cull();
                }
                // If snapshot doesn't exist, just continue descent
            } else {
                // At or below target layer: search zero layer and BREAK
                search.search_zero(point, zero, points, M0_MAX);
                break; // Key fix: don't keep searching!
            }
        }

        // Get best candidates from search
        let found = search.select_simple();

        // Add connections: new node → neighbors (in zero layer)
        {
            let mut node = zero[new.as_usize()].write();
            for (i, candidate) in found.iter().take(M0_MAX).enumerate() {
                node.nearest[i] = candidate.pid;
            }
        }

        // Add reverse connections: neighbors → new node (bidirectional)
        for candidate in found.iter().take(M0_MAX) {
            Self::add_reverse_connection(zero, points, new, candidate.pid);
        }

        pool.push(search);
    }

    /// Add reverse connection from neighbor to new node, maintaining SORTED order by distance
    /// This is critical: UpperNode::from_zero takes the first M entries, so they must be the M closest
    fn add_reverse_connection(
        zero: &[RwLock<ZeroNode>],
        points: &[Vec<f32>],
        new: PointId,
        neighbor: PointId,
    ) {
        let mut node = zero[neighbor.as_usize()].write();
        let neighbor_point = &points[neighbor.as_usize()];
        let new_dist = Self::parallel_distance(neighbor_point, &points[new.as_usize()]);

        let count = node.count();

        // Binary search for insertion position (sorted by distance, ascending)
        let pos = {
            let mut left = 0;
            let mut right = count;
            while left < right {
                let mid = (left + right) / 2;
                let mid_dist =
                    Self::parallel_distance(neighbor_point, &points[node.nearest[mid].as_usize()]);
                if mid_dist < new_dist {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            left
        };

        // If position is beyond capacity, this node is worse than all current neighbors
        if pos >= M0_MAX {
            return;
        }

        // Shift elements right to make room, dropping the last if at capacity
        let shift_end = count.min(M0_MAX - 1);
        for i in (pos..shift_end).rev() {
            node.nearest[i + 1] = node.nearest[i];
        }

        // Insert at sorted position
        node.nearest[pos] = new;
    }

    /// Convert parallel construction data to final HNSWIndex format
    fn convert_parallel_to_index(
        zero: Vec<RwLock<ZeroNode>>,
        layers: Vec<Vec<UpperNode>>,
        points: Vec<Vec<f32>>,
        shuffled: Vec<(u32, usize)>,
        embedding_dim: usize,
        config: HNSWConfig,
        top: LayerId,
    ) -> Self {
        let n = points.len();
        let num_layers = top.0 + 1; // Total layers including layer 0
        let zero_final: Vec<ZeroNode> = zero.into_iter().map(|n| n.into_inner()).collect();

        // Build connections from fixed-array format
        // Uses Vec<u32> for cache-friendly traversal (4-5x faster than HashSet)
        let mut connections: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(num_layers);

            // Layer 0: from zero layer (M0 connections)
            let layer0: Vec<u32> = zero_final[i].iter().map(|p| p.as_usize() as u32).collect();
            node_connections.push(layer0);

            // Upper layers: ONLY add connections where node actually existed in snapshot
            // Nodes should NOT have fake connections at layers they don't belong to
            // (This matches instant-distance's behavior where late nodes simply don't exist at upper layers)
            for layer in &layers {
                if i < layer.len() {
                    // Node exists in this snapshot - use its actual connections
                    let layer_conns: Vec<u32> =
                        layer[i].iter().map(|p| p.as_usize() as u32).collect();
                    node_connections.push(layer_conns);
                }
                // else: node doesn't exist at this layer - don't add fake connections!
            }

            connections.push(node_connections);
        }

        // Compute norms for all points
        let norms: Vec<f32> = points
            .iter()
            .map(|p| crate::vector::simd::norm_simd(p))
            .collect();

        // Flatten embeddings to SoA
        let flat_embeddings: Vec<f32> = points.into_iter().flatten().collect();

        // Create ID mapping (shuffled index → original index)
        let ids: Vec<String> = shuffled.iter().map(|&(_, orig)| orig.to_string()).collect();

        let mut index = Self {
            embedding_dim,
            config,
            embeddings: flat_embeddings,
            norms,
            connections,
            connections_l0: Vec::new(),
            connections_l0_count: Vec::new(),
            ids,
            contents: vec![String::new(); n],
            metadata: vec![None; n],
            entry_point: Some(0),
            max_layer: top.0,
        };
        index.build_l0_cache();
        index
    }

    /// Build single-node index (trivial case)
    fn build_single(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        let embedding_dim = embeddings[0].len();
        let norms = vec![crate::vector::simd::norm_simd(&embeddings[0])];
        let m0 = config.m0;
        Self {
            embedding_dim,
            config,
            embeddings: embeddings.into_iter().flatten().collect(),
            norms,
            connections: vec![vec![Vec::new()]],
            connections_l0: vec![0u32; m0],
            connections_l0_count: vec![0u8],
            ids: vec!["0".to_string()],
            contents: vec![String::new()],
            metadata: vec![None],
            entry_point: Some(0),
            max_layer: 0,
        }
    }

    /// Distance function for parallel construction (SIMD accelerated)
    #[inline]
    fn parallel_distance(a: &[f32], b: &[f32]) -> f32 {
        1.0 - crate::vector::simd::cosine_similarity_simd(a, b)
    }
}

// ============================================================================
// INSTANT-DISTANCE STYLE PARALLEL CONSTRUCTION
// Uses fixed-size arrays and layer-copying for safe parallelization
// ============================================================================

/// Maximum connections per node in layer 0 (M * 2)
const M0_MAX: usize = 64;
/// Maximum connections per node in upper layers (M)
const M_MAX: usize = 32;
/// Invalid point ID marker
const INVALID: u32 = u32::MAX;

/// Point ID wrapper (u32 for memory efficiency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PointId(u32);

impl PointId {
    fn as_usize(self) -> usize {
        self.0 as usize
    }

    fn is_valid(self) -> bool {
        self.0 != INVALID
    }
}

/// Layer 0 node with M*2 fixed connections
#[derive(Clone)]
struct ZeroNode {
    /// Fixed array of neighbor IDs (INVALID = empty slot)
    nearest: [PointId; M0_MAX],
}

impl Default for ZeroNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M0_MAX],
        }
    }
}

impl ZeroNode {
    /// Count of valid neighbors
    fn count(&self) -> usize {
        self.nearest.iter().take_while(|p| p.is_valid()).count()
    }

    /// Iterate over valid neighbors
    fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Upper layer node with M fixed connections
#[derive(Clone)]
struct UpperNode {
    nearest: [PointId; M_MAX],
}

impl Default for UpperNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M_MAX],
        }
    }
}

impl UpperNode {
    /// Create from ZeroNode, truncating to M neighbors
    fn from_zero(zero: &ZeroNode) -> Self {
        let mut node = Self::default();
        for (i, &pid) in zero.nearest.iter().take(M_MAX).enumerate() {
            node.nearest[i] = pid;
        }
        node
    }

    fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Visited bitmap with generation counter (O(1) clear)
struct Visited {
    store: Vec<u8>,
    generation: u8,
}

impl Visited {
    fn new(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    fn clear(&mut self) {
        if self.generation == 255 {
            self.store.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    fn insert(&mut self, pid: PointId) -> bool {
        let idx = pid.as_usize();
        if self.store[idx] == self.generation {
            false
        } else {
            self.store[idx] = self.generation;
            true
        }
    }

    fn reserve(&mut self, capacity: usize) {
        if self.store.len() < capacity {
            self.store.resize(capacity, 0);
        }
    }
}

/// Candidate for search (distance + point ID)
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    pid: PointId,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by distance, then by pid for stability
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.pid.cmp(&other.pid))
    }
}

/// Search state for parallel construction
struct Search {
    /// Candidates to explore (min-heap by distance)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Best results found (sorted by distance)
    nearest: Vec<Candidate>,
    /// Visited nodes
    visited: Visited,
    /// Current ef value
    ef: usize,
}

impl Search {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            visited: Visited::new(capacity),
            ef: 1,
        }
    }

    fn reset(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.visited.clear();
    }

    fn push(&mut self, pid: PointId, point: &[f32], points: &[Vec<f32>]) {
        let distance = HNSWIndex::parallel_distance(point, &points[pid.as_usize()]);
        let candidate = Candidate { distance, pid };
        self.candidates.push(Reverse(candidate));
        self.nearest.push(candidate);
        self.visited.insert(pid);
    }

    /// After searching a layer, prepare for the next layer down
    fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in &self.nearest {
            self.candidates.push(Reverse(candidate));
        }
        self.visited.clear();
        for c in &self.nearest {
            self.visited.insert(c.pid);
        }
    }

    /// Search within a layer (generic over layer type)
    fn search_zero(
        &mut self,
        point: &[f32],
        layer: &[RwLock<ZeroNode>],
        points: &[Vec<f32>],
        num: usize,
    ) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Explore neighbors
            let node = layer[candidate.pid.as_usize()].read();
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance =
                        HNSWIndex::parallel_distance(point, &points[neighbor_pid.as_usize()]);
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    // Add to candidates if potentially useful
                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));

                        // Insert into nearest (sorted)
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    fn search_upper(
        &mut self,
        point: &[f32],
        layer: &[UpperNode],
        points: &[Vec<f32>],
        num: usize,
    ) {
        if layer.is_empty() {
            return;
        }

        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Safety: skip if candidate is beyond current layer snapshot
            if candidate.pid.as_usize() >= layer.len() {
                continue;
            }

            let node = &layer[candidate.pid.as_usize()];
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance =
                        HNSWIndex::parallel_distance(point, &points[neighbor_pid.as_usize()]);
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    /// Get best candidates (sorted by distance)
    fn select_simple(&self) -> &[Candidate] {
        &self.nearest
    }
}

/// Pool of search states for thread-local reuse
struct SearchPool {
    pool: Mutex<Vec<Search>>,
    capacity: usize,
}

impl SearchPool {
    fn new(capacity: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            capacity,
        }
    }

    fn pop(&self) -> Search {
        self.pool
            .lock()
            .pop()
            .unwrap_or_else(|| Search::new(self.capacity))
    }

    fn push(&self, mut search: Search) {
        search.reset();
        self.pool.lock().push(search);
    }
}

/// Layer ID wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerId(usize);

impl LayerId {
    fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Iterate from this layer down to 0
    fn descend(self) -> impl Iterator<Item = LayerId> {
        (0..=self.0).rev().map(LayerId)
    }
}

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
        (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.m, 32); // Changed from 16 to match instant-distance
        assert_eq!(config.m0, 64); // m * 2
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 100);
        assert!((config.ml - (1.0 / 32_f32.ln())).abs() < 0.01);
        assert!(config.use_heuristic); // Heuristic enabled by default
        assert!(!config.extend_candidates);
        assert!(config.keep_pruned_connections);
    }

    #[test]
    fn test_hnsw_config_builders() {
        let config = HNSWConfig::default()
            .with_m(32)
            .with_ef_search(100)
            .with_ef_construction(400)
            .with_simple_selection()
            .with_extended_candidates();

        assert_eq!(config.m, 32);
        assert_eq!(config.m0, 64);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.ef_construction, 400);
        assert!(!config.use_heuristic);
        assert!(config.extend_candidates);
    }

    #[test]
    fn test_hnsw_new() {
        let index = HNSWIndex::with_defaults(128);
        assert_eq!(index.embedding_dim, 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_single_document() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);

        assert!(index.add(doc).is_ok());
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0]); // Wrong dimension

        assert!(index.add(doc).is_err());
    }

    #[test]
    fn rejects_inf_embedding() {
        let mut index = HNSWIndex::new(3, HNSWConfig::default());
        let doc = Document {
            id: "inf".to_string(),
            content: "test".to_string(),
            embedding: vec![f32::INFINITY, 0.0, 0.0],
            metadata: None,
        };
        assert!(index.add(doc).is_err());

        let doc_neg = Document {
            id: "neg_inf".to_string(),
            content: "test".to_string(),
            embedding: vec![0.0, f32::NEG_INFINITY, 0.0],
            metadata: None,
        };
        assert!(index.add(doc_neg).is_err());
    }

    #[test]
    fn search_context_resizes_after_add() {
        let mut index = HNSWIndex::new(3, HNSWConfig::default());
        index.add(Document {
            id: "a".into(), content: "a".into(),
            embedding: vec![1.0, 0.0, 0.0], metadata: None,
        }).unwrap();

        let mut ctx = index.create_search_context();

        for i in 0..10 {
            index.add(Document {
                id: format!("doc-{i}"), content: format!("content-{i}"),
                embedding: vec![(i as f32) * 0.1, 1.0 - (i as f32) * 0.1, 0.0],
                metadata: None,
            }).unwrap();
        }

        let results = index.search_with_context(&[1.0, 0.0, 0.0], 5, &mut ctx).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_empty_index() {
        let index = HNSWIndex::with_defaults(3);
        let query = vec![1.0, 0.0, 0.0];

        let results = index.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_document() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_multiple_documents() {
        let mut index = HNSWIndex::with_defaults(3);

        // Add documents with different embeddings
        let docs = vec![
            create_test_document("doc1", vec![1.0, 0.0, 0.0]),
            create_test_document("doc2", vec![0.0, 1.0, 0.0]),
            create_test_document("doc3", vec![0.0, 0.0, 1.0]),
            create_test_document("doc4", vec![1.0, 1.0, 0.0]),
        ];

        for doc in docs {
            index.add(doc).unwrap();
        }

        // Query closest to doc1
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_search_exact_match() {
        let mut index = HNSWIndex::with_defaults(3);

        let embedding = vec![0.5, 0.5, 0.7072];
        let doc = create_test_document("doc1", embedding.clone());
        index.add(doc).unwrap();

        let results = index.search(&embedding, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clear() {
        let mut index = HNSWIndex::with_defaults(3);

        for i in 0..5 {
            let doc = create_test_document(&format!("doc{}", i), vec![i as f32, 0.0, 0.0]);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 5);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_random_dataset_100_vectors() {
        let dim = 128;
        let mut index = HNSWIndex::with_defaults(dim);

        // Add 100 random vectors
        for i in 0..100 {
            let embedding = generate_random_vector(dim, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search with a random query
        let query = generate_random_vector(dim, 9999);
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by score (descending)
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_random_dataset_1000_vectors() {
        let dim = 64;
        let mut index = HNSWIndex::with_defaults(dim);

        // Add 1000 random vectors
        for i in 0..1000 {
            let embedding = generate_random_vector(dim, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 1000);

        // Perform multiple searches
        for seed in [111, 222, 333, 444, 555] {
            let query = generate_random_vector(dim, seed);
            let results = index.search(&query, 20).unwrap();

            assert_eq!(results.len(), 20);

            // Verify ordering
            for i in 0..results.len() - 1 {
                assert!(results[i].score >= results[i + 1].score);
            }

            // All scores should be between -1 and 1
            for result in &results {
                assert!(result.score >= -1.0 && result.score <= 1.0);
            }
        }
    }

    #[test]
    fn test_recall_with_known_neighbors() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        // Create a query vector
        let query = generate_random_vector(dim, 0);

        // Create 100 vectors with varying similarity to query
        for i in 0..100 {
            let mut embedding = generate_random_vector(dim, i + 1);

            // First 10 vectors are more similar to query
            if i < 10 {
                for j in 0..dim {
                    embedding[j] = query[j] * 0.9 + embedding[j] * 0.1;
                }
            }

            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        // Search for top 10
        let results = index.search(&query, 10).unwrap();

        // Count how many of the actual top 10 were found
        let mut recall_count = 0;
        for result in &results {
            let doc_num: usize = result.id.strip_prefix("doc").unwrap().parse().unwrap();
            if doc_num < 10 {
                recall_count += 1;
            }
        }

        // HNSW should find most of the true nearest neighbors
        // Expect at least 70% recall
        assert!(recall_count >= 7, "Recall too low: {}/10", recall_count);
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        let query = vec![1.0, 0.0]; // Wrong dimension
        assert!(index.search(&query, 1).is_err());
    }

    #[test]
    fn test_metadata_preservation() {
        let mut index = HNSWIndex::with_defaults(3);

        let mut doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        doc.metadata = Some(serde_json::json!({"category": "test", "priority": 5}));

        index.add(doc).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.is_some());

        let metadata = results[0].metadata.as_ref().unwrap();
        assert_eq!(metadata["category"], "test");
        assert_eq!(metadata["priority"], 5);
    }

    #[test]
    fn test_search_with_nan_query_does_not_panic() {
        let mut index = HNSWIndex::with_defaults(3);
        index
            .add(create_test_document("doc1", vec![1.0, 0.0, 0.0]))
            .unwrap();
        index
            .add(create_test_document("doc2", vec![0.0, 1.0, 0.0]))
            .unwrap();

        let query = vec![f32::NAN, 0.0, 0.0];
        let outcome = std::panic::catch_unwind(|| index.search(&query, 2));

        assert!(outcome.is_ok(), "search panicked when query contains NaN");
    }

    #[test]
    #[should_panic(expected = "All embeddings must have the same dimension")]
    fn test_build_rejects_mismatched_dimensions() {
        let _ = HNSWIndex::build(
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0]],
            HNSWConfig::default(),
        );
    }

    #[test]
    fn test_add_rejects_nan_embedding() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("nan_doc", vec![1.0, f32::NAN, 0.0]);
        let result = index.add(doc);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_add_embedding_rejects_nan() {
        let mut index = HNSWIndex::with_defaults(3);
        let result = index.add_embedding("nan_vec".into(), vec![f32::NAN, 0.0, 0.0]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_add_rejects_all_nan_embedding() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("all_nan", vec![f32::NAN, f32::NAN, f32::NAN]);
        assert!(index.add(doc).is_err());
    }

    #[test]
    fn test_zero_vector_accepted_and_searchable() {
        let mut index = HNSWIndex::with_defaults(3);

        // Zero vectors are valid (they just have zero norm)
        let doc_zero = create_test_document("zero", vec![0.0, 0.0, 0.0]);
        assert!(index.add(doc_zero).is_ok());

        let doc_normal = create_test_document("normal", vec![1.0, 0.0, 0.0]);
        assert!(index.add(doc_normal).is_ok());

        // Search should not panic with zero vectors in the index
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);

        // The normal vector should rank higher than the zero vector
        assert_eq!(results[0].id, "normal");
    }

    #[test]
    fn test_zero_vector_query_does_not_panic() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        // Zero query should not panic (distance_to_node handles zero norm)
        let query = vec![0.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_random_level_never_panics() {
        // Exercise random_level many times to verify ln(0) is impossible
        let index = HNSWIndex::with_defaults(3);
        for _ in 0..10_000 {
            let _level = index.random_level();
        }
    }
}
