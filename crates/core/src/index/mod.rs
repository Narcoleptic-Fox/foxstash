//! Vector index implementations
//!
//! This module provides multiple index types for vector similarity search:
//!
//! - [`FlatIndex`]: Brute-force search (100% accurate, O(n) search)
//! - [`HNSWIndex`]: Approximate nearest neighbors (fast, full precision)
//! - [`SQ8HNSWIndex`]: HNSW with scalar quantization (4x memory reduction)
//! - [`RaBitQHNSWIndex`]: HNSW with RaBitQ 1-bit quantization (32x memory reduction)
//! - [`BinaryHNSWIndex`]: **deprecated** — degenerate on non-negative embeddings
//!
//! # Memory Comparison (1M vectors × 384 dims)
//!
//! | Index | Memory | Recall@10* | Use Case |
//! |-------|--------|------------|----------|
//! | HNSW (f32) | 1.5 GB | 100% | Default choice |
//! | SQ8 HNSW | 384 MB | 100.0% | Memory constrained |
//! | RaBitQ HNSW | 48 MB | 73.2% | Massive datasets |
//! | Binary HNSW | 48 MB | 1.2% | Deprecated — use RaBitQ |
//!
//! *Measured on SIFT10K with a two-phase search (1-bit filter, pool=100, exact rerank):
//! `cargo run --release -p foxstash-benches --example quantizer_sift`. Binary's zero
//! threshold collapses on non-negative data; RaBitQ centers and does not.
//!
//! # Streaming Operations
//!
//! For large datasets, use the streaming module for memory-efficient batch ingestion:
//!
//! ```
//! use foxstash_core::index::streaming::{BatchBuilder, BatchConfig};
//! use foxstash_core::index::HNSWIndex;
//! use foxstash_core::Document;
//!
//! let mut index = HNSWIndex::with_defaults(4);
//!
//! let config = BatchConfig::default()
//!     .with_batch_size(1000)
//!     .with_progress(|p| println!("Progress: {}/{}", p.completed, p.total.unwrap_or(0)));
//!
//! let documents = vec![
//!     Document { id: "a".into(), content: "alpha".into(), embedding: vec![1.0, 0.0, 0.0, 0.0], metadata: None },
//!     Document { id: "b".into(), content: "beta".into(),  embedding: vec![0.0, 1.0, 0.0, 0.0], metadata: None },
//!     Document { id: "c".into(), content: "gamma".into(), embedding: vec![0.0, 0.0, 1.0, 0.0], metadata: None },
//! ];
//!
//! let mut builder = BatchBuilder::new(&mut index, config);
//! for doc in documents {
//!     builder.add(doc).unwrap();
//! }
//! let result = builder.finish();
//! assert_eq!(result.documents_indexed, 3);
//! ```

pub mod flat;
pub mod hnsw;
pub mod hnsw_pq;
pub mod hnsw_quantized;
pub mod streaming;

pub use flat::FlatIndex;
pub use hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
pub use hnsw_pq::{PQHNSWConfig, PQHNSWIndex};
#[allow(deprecated)]
pub use hnsw_quantized::{BinaryHNSWIndex, QuantizedHNSWConfig, RaBitQHNSWIndex, SQ8HNSWIndex};
pub use streaming::{
    BatchBuilder, BatchConfig, BatchIndex, BatchProgress, BatchResult, FilteredSearchBuilder,
    PaginationConfig, SearchPage, SearchResultIterator,
};

use crate::{Document, Result, SearchResult};

/// Trait for vector similarity indexes.
///
/// Provides a common interface across all index implementations (HNSW, Flat,
/// SQ8, Binary, PQ). Object-safe — works with `Box<dyn VectorIndex>`.
///
/// Construction is excluded because each index type has different configuration
/// requirements.
pub trait VectorIndex {
    /// Add a document to the index.
    fn add(&mut self, document: Document) -> Result<()>;

    /// Search for the k nearest neighbors to the query vector.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Return the number of documents in the index.
    fn len(&self) -> usize;

    /// Return true if the index contains no documents.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all documents from the index.
    fn clear(&mut self);

    /// Return the expected embedding dimension.
    fn embedding_dim(&self) -> usize;
}

/// Extension trait for indexes that retain original embeddings.
///
/// Only HNSW and Flat indexes can return full documents; quantized variants
/// discard original vectors during encoding.
pub trait VectorIndexSnapshot: VectorIndex {
    /// Return clones of all documents stored in the index.
    fn get_all_documents(&self) -> Vec<Document>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.into(),
            content: format!("content-{id}"),
            embedding,
            metadata: None,
        }
    }

    #[test]
    fn vector_index_object_safety_hnsw() {
        let mut index: Box<dyn VectorIndex> = Box::new(HNSWIndex::with_defaults(3));

        assert!(index.is_empty());
        assert_eq!(index.embedding_dim(), 3);

        index.add(make_doc("a", vec![1.0, 0.0, 0.0])).unwrap();
        index.add(make_doc("b", vec![0.0, 1.0, 0.0])).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "a");

        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn vector_index_snapshot_flat() {
        let mut index: Box<dyn VectorIndexSnapshot> = Box::new(FlatIndex::new(3));

        index.add(make_doc("x", vec![0.5, 0.5, 0.0])).unwrap();
        index.add(make_doc("y", vec![0.0, 0.5, 0.5])).unwrap();

        let docs = index.get_all_documents();
        assert_eq!(docs.len(), 2);

        let ids: std::collections::HashSet<_> = docs.iter().map(|d| d.id.as_str()).collect();
        assert!(ids.contains("x"));
        assert!(ids.contains("y"));
    }

    #[test]
    fn vector_index_snapshot_hnsw() {
        let mut index: Box<dyn VectorIndexSnapshot> = Box::new(HNSWIndex::with_defaults(3));

        index.add(make_doc("p", vec![1.0, 0.0, 0.0])).unwrap();
        let docs = index.get_all_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "p");
    }

    #[test]
    fn vector_index_sq8() {
        let mut index: Box<dyn VectorIndex> = Box::new(SQ8HNSWIndex::for_normalized(
            4,
            QuantizedHNSWConfig::default(),
        ));

        index.add(make_doc("q", vec![0.5, -0.3, 0.8, 0.1])).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.embedding_dim(), 4);

        let results = index.search(&[0.5, -0.3, 0.8, 0.1], 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn vector_index_binary() {
        let mut index: Box<dyn VectorIndex> =
            Box::new(BinaryHNSWIndex::new(4, QuantizedHNSWConfig::default()));

        index.add(make_doc("r", vec![0.1, 0.2, 0.3, 0.4])).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.embedding_dim(), 4);
    }

    #[test]
    fn batch_builder_via_blanket_impl() {
        let mut index = HNSWIndex::with_defaults(3);
        let config = BatchConfig::default().with_batch_size(10);
        let mut builder = BatchBuilder::new(&mut index, config);

        builder.add(make_doc("d1", vec![1.0, 0.0, 0.0])).unwrap();
        builder.add(make_doc("d2", vec![0.0, 1.0, 0.0])).unwrap();

        let result = builder.finish();
        assert_eq!(result.documents_indexed, 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn batch_builder_flat_via_blanket_impl() {
        let mut index = FlatIndex::new(3);
        let config = BatchConfig::default();
        let mut builder = BatchBuilder::new(&mut index, config);

        builder.add(make_doc("f1", vec![1.0, 0.0, 0.0])).unwrap();
        let result = builder.finish();
        assert_eq!(result.documents_indexed, 1);
    }
}
