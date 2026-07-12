//! Vector index implementations
//!
//! - [`HNSWIndex`]: approximate nearest neighbours. **The one you want.** Quantization is a
//!   [`Storage`] mode on this index, not a separate type — see below.
//! - [`FlatIndex`]: brute force. Exact, O(n) per query. Use it as a control, and for tiny
//!   corpora where an approximate index cannot pay for itself.
//! - [`RaBitQHNSWIndex`]: 1-bit quantization, 32x compression. Still a separate type; being
//!   folded into [`Storage`].
//!
//! # Quantization is a storage mode, not an index type
//!
//! ```
//! # use foxstash_core::index::{HNSWConfig, HNSWIndex, Storage, DistanceMetric};
//! let config = HNSWConfig {
//!     storage: Storage::SQ8,      // 8-bit codes inline in the node arena
//!     rerank_candidates: 100,     // rescore the top pool with exact f32 distances
//!     metric: DistanceMetric::L2,
//!     ..Default::default()
//! };
//! ```
//!
//! The graph is still *built* with exact f32 distances; only the traversal reads compressed
//! codes. That combination is what makes it fast **and** accurate: on SIFT1M it is 1.20x
//! hnswlib at 99.5% recall@10, because the hot node block shrinks 784 → 400 bytes and HNSW
//! search is memory-latency bound. `rerank_candidates: 0` drops the f32 vectors entirely
//! (0.73x hnswlib's memory) at a recall ceiling near 98.9%.
//!
//! A standalone `SQ8HNSWIndex` used to exist. It was deleted: the storage mode beat it on
//! recall, throughput *and* build time at every `ef` (see `benchmarks/RESULTS.md`), and it was
//! a metric footgun — hardcoded L2 with no `metric` field, while [`HNSWConfig`] defaults to
//! cosine, so swapping index types silently changed the question being asked.
//!
//! A plain zero-threshold binary quantizer is not offered: on non-negative data (SIFT, and
//! most embedding models) every bit is set and the code carries no information — it measured
//! 1.2% recall@10. RaBitQ centers each vector before thresholding, which is the whole
//! difference. See `crate::vector::quantize` for the comparison.
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
pub use hnsw::{
    BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, MemoryBreakdown, Searcher, Storage,
};
pub use hnsw_pq::{PQHNSWConfig, PQHNSWIndex};
pub use hnsw_quantized::{QuantizedHNSWConfig, RaBitQHNSWIndex};
pub use streaming::{
    BatchBuilder, BatchConfig, BatchIndex, BatchProgress, BatchResult, FilteredSearchBuilder,
    PaginationConfig, SearchPage, SearchResultIterator,
};

use crate::{Document, Result, SearchResult};

/// Trait for vector similarity indexes.
///
/// Provides a common interface across all index implementations (HNSW, Flat,
/// SQ8, RaBitQ, PQ). Object-safe — works with `Box<dyn VectorIndex>`.
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
    fn vector_index_sq8_storage_is_object_safe() {
        // SQ8 is a storage mode now, not an index type — but it must still work through
        // `Box<dyn VectorIndex>`, which is what this asserts. (The standalone `SQ8HNSWIndex`
        // this test used to construct was deleted: the storage mode beat it on recall,
        // throughput and build time at every ef.)
        //
        // Quantized storage must be trained before it can encode anything — SQ8 needs the
        // corpus's per-dimension min/scale. `train()` is therefore part of the object-safe
        // path, and this test exists partly to keep it that way.
        let mut sq8 = HNSWIndex::new(
            4,
            HNSWConfig {
                storage: Storage::SQ8,
                rerank_candidates: 100,
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        );
        let sample = vec![
            vec![0.5, -0.3, 0.8, 0.1],
            vec![-0.2, 0.9, -0.5, 0.4],
            vec![0.1, 0.1, 0.2, -0.9],
        ];
        sq8.train(&sample).unwrap();

        let mut index: Box<dyn VectorIndex> = Box::new(sq8);
        index.add(make_doc("q", vec![0.5, -0.3, 0.8, 0.1])).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.embedding_dim(), 4);

        let results = index.search(&[0.5, -0.3, 0.8, 0.1], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "q");
    }

    #[test]
    fn quantized_storage_rejects_add_before_train_instead_of_panicking() {
        // This used to be an index-out-of-bounds panic on the very first document: the SQ8
        // codebook was only ever fitted inside `build`/`build_parallel`, so `new()` + `add()`
        // indexed into an empty `q_scale`. It was public, documented, reachable from
        // foxstash-db (every insert on an SQ8 collection crashed) — and every SQ8 test went
        // through `build_parallel`, so nothing could fail on it.
        let mut index: Box<dyn VectorIndex> = Box::new(HNSWIndex::new(
            4,
            HNSWConfig {
                storage: Storage::SQ8,
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        ));

        let err = index
            .add(make_doc("q", vec![0.5, -0.3, 0.8, 0.1]))
            .expect_err("untrained SQ8 must reject add(), not panic");

        assert!(
            matches!(err, crate::RagError::NotTrained(_)),
            "expected NotTrained, got {err:?}"
        );
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
