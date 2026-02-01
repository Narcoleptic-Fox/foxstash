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
use rand::Rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Number of bidirectional links created for each element (except layer 0)
    /// Typical value: 16. Higher values increase recall but use more memory.
    pub m: usize,

    /// Number of bidirectional links created for each element in layer 0
    /// Typical value: 2 * m (32 for m=16)
    pub m0: usize,

    /// Size of the dynamic candidate list during construction
    /// Typical value: 200. Higher values improve quality but slow down construction.
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search
    /// Typical value: 50. Higher values improve recall but slow down search.
    pub ef_search: usize,

    /// Normalization factor for level generation
    /// Typical value: 1.0 / ln(m) ≈ 0.36 for m=16
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
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
            use_heuristic: true, // Use improved heuristic by default
            extend_candidates: false,
            keep_pruned_connections: true,
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

    /// Set ef_construction parameter
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set M parameter (connections per node)
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f32).ln();
        self
    }
}

/// Internal node in the HNSW graph
#[derive(Debug, Clone)]
struct Node {
    /// Document ID
    id: String,
    /// Document content
    content: String,
    /// Embedding vector
    embedding: Vec<f32>,
    /// Metadata
    metadata: Option<serde_json::Value>,
    /// Connections at each layer: layer -> set of neighbor IDs
    connections: Vec<HashSet<usize>>,
}

/// HNSW index for efficient similarity search
pub struct HNSWIndex {
    /// Dimensionality of embeddings
    embedding_dim: usize,
    /// Configuration parameters
    config: HNSWConfig,
    /// All nodes in the index
    nodes: Vec<Node>,
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
        Self {
            embedding_dim,
            config,
            nodes: Vec::new(),
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

        let node_id = self.nodes.len();
        let node_level = self.random_level();

        // Create node with empty connections for each layer
        let mut connections = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            connections.push(HashSet::new());
        }

        let node = Node {
            id: document.id,
            content: document.content,
            embedding: document.embedding,
            metadata: document.metadata,
            connections,
        };

        self.nodes.push(node);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        self.insert_node(node_id, node_level);

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
        if query.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Search from top layer to layer 1
        for layer in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer(query, &current_nearest, 1, layer);
        }

        // Search layer 0 with ef_search candidates
        let ef = self.config.ef_search.max(k);
        current_nearest = self.search_layer(query, &current_nearest, ef, 0);

        // Convert to SearchResults and return top k
        let mut results: Vec<SearchResult> = current_nearest
            .iter()
            .map(|&node_id| {
                let node = &self.nodes[node_id];
                let score = crate::vector::cosine_similarity(query, &node.embedding).unwrap_or(0.0);
                SearchResult {
                    id: node.id.clone(),
                    content: node.content.clone(),
                    score,
                    metadata: node.metadata.clone(),
                }
            })
            .collect();

        // Sort by score descending and take top k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Returns the number of documents in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clears all documents from the index
    pub fn clear(&mut self) {
        self.nodes.clear();
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
        self.nodes
            .iter()
            .map(|node| Document {
                id: node.id.clone(),
                content: node.content.clone(),
                embedding: node.embedding.clone(),
                metadata: node.metadata.clone(),
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

    /// Generates a random level for a new node using exponential decay
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen();
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    /// Inserts a node into the graph structure
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to insert
    /// * `node_level` - Maximum layer of the node
    fn insert_node(&mut self, node_id: usize, node_level: usize) {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Search for nearest neighbors from top to target layer + 1
        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer(
                &self.nodes[node_id].embedding.clone(),
                &current_nearest,
                1,
                layer,
            );
        }

        // Insert into layers from top to bottom
        for layer in (0..=node_level).rev() {
            let query = self.nodes[node_id].embedding.clone();
            current_nearest =
                self.search_layer(&query, &current_nearest, self.config.ef_construction, layer);

            // Determine M for this layer
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            // Select M nearest neighbors
            let neighbors = self.select_neighbors(&current_nearest, &query, m, layer);

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                // Add link from new node to neighbor
                self.nodes[node_id].connections[layer].insert(neighbor_id);

                // Only add bidirectional link if neighbor exists at this layer
                // (neighbor must have connections.len() > layer)
                if layer < self.nodes[neighbor_id].connections.len() {
                    // Add link from neighbor to new node
                    self.nodes[neighbor_id].connections[layer].insert(node_id);

                    // Prune neighbor's connections if needed
                    let neighbor_m = if layer == 0 {
                        self.config.m0
                    } else {
                        self.config.m
                    };

                    if self.nodes[neighbor_id].connections[layer].len() > neighbor_m {
                        let neighbor_embedding = self.nodes[neighbor_id].embedding.clone();
                        let neighbor_connections: Vec<usize> = self.nodes[neighbor_id].connections
                            [layer]
                            .iter()
                            .copied()
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_embedding,
                            neighbor_m,
                            layer,
                        );

                        self.nodes[neighbor_id].connections[layer] = pruned.into_iter().collect();
                    }
                }
            }
        }
    }

    /// Searches a specific layer for nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query embedding
    /// * `entry_points` - Starting points for search
    /// * `ef` - Number of nearest neighbors to find
    /// * `layer` - Layer to search
    ///
    /// # Returns
    /// Vector of node IDs sorted by distance (closest first)
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap by distance
        let mut best = BinaryHeap::new(); // Max-heap by distance

        // Initialize with entry points
        for &ep in entry_points {
            let dist = self.distance(query, &self.nodes[ep].embedding);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            best.push((OrderedFloat(dist), ep));
            visited.insert(ep);
        }

        while let Some(Reverse((current_dist, current_id))) = candidates.pop() {
            // If current is farther than the ef-th nearest, we're done
            if best.len() >= ef {
                if let Some(&(furthest_dist, _)) = best.peek() {
                    if current_dist > furthest_dist {
                        break;
                    }
                }
            }

            // Check all neighbors at this layer
            if layer < self.nodes[current_id].connections.len() {
                for &neighbor_id in &self.nodes[current_id].connections[layer] {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);

                        let dist = self.distance(query, &self.nodes[neighbor_id].embedding);
                        let dist_ord = OrderedFloat(dist);

                        if best.len() < ef {
                            candidates.push(Reverse((dist_ord, neighbor_id)));
                            best.push((dist_ord, neighbor_id));
                        } else if let Some(&(furthest_dist, _)) = best.peek() {
                            if dist_ord < furthest_dist {
                                candidates.push(Reverse((dist_ord, neighbor_id)));
                                best.push((dist_ord, neighbor_id));

                                // Keep only ef elements
                                if best.len() > ef {
                                    best.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract and return node IDs sorted by distance
        let mut results: Vec<(f32, usize)> = best
            .into_iter()
            .map(|(OrderedFloat(dist), id)| (dist, id))
            .collect();

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
                if layer < self.nodes[candidate].connections.len() {
                    for &neighbor in &self.nodes[candidate].connections[layer] {
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
                let dist = self.distance(query, &self.nodes[id].embedding);
                (dist, id)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Select neighbors using the heuristic
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut pruned: Vec<(f32, usize)> = Vec::new();

        for (dist_to_query, candidate_id) in scored {
            if selected.len() >= m {
                break;
            }

            // Check if this candidate is closer to query than to any selected neighbor
            let candidate_embedding = &self.nodes[candidate_id].embedding;
            let mut is_good = true;

            for &selected_id in &selected {
                let selected_embedding = &self.nodes[selected_id].embedding;
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
    fn select_neighbors_simple(&self, candidates: &[usize], query: &[f32], m: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, &self.nodes[id].embedding);
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Computes distance between two vectors
    /// Uses 1 - cosine_similarity for distance metric
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - crate::vector::cosine_similarity(a, b).unwrap_or(0.0)
    }
}

/// Wrapper for f32 that implements Ord for use in BinaryHeap
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
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
        assert!((config.ml - 0.36).abs() < 0.01);
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

        let embedding = vec![0.5, 0.5, 0.7071];
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
}
