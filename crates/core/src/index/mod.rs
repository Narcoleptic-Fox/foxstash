//! Vector index implementations
//!
//! - [`HNSWIndex`]: approximate nearest neighbours. **The one you want.** Quantization is a
//!   [`Storage`] mode on this index, not a separate type — see below.
//! - [`FlatIndex`]: brute force. Exact, O(n) per query. Use it as a control, and for tiny
//!   corpora where an approximate index cannot pay for itself.
//!
//! There is exactly ONE approximate index type. There used to be four. `SQ8HNSWIndex`,
//! `RaBitQHNSWIndex` and `PQHNSWIndex` were all deleted — see "The standalone index types, and
//! why they are gone" below.
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
//! # The standalone index types, and why they are gone
//!
//! Three of them existed. All three are deleted. They shared one shape: a "fat node"
//! (`Vec<HashSet<usize>>` adjacency plus a `String` id and `String` content **per node**), no
//! rayon so builds were sequential, and **no `metric` field at all** — hardcoded L2, while
//! [`HNSWConfig`] defaults to *cosine*. Swapping index type to save memory silently changed the
//! question being asked.
//!
//! * **`SQ8HNSWIndex`** — superseded by [`Storage::SQ8`], which beat it on recall, throughput
//!   *and* build time at every `ef`.
//! * **`RaBitQHNSWIndex`** — superseded by [`Storage::RaBitQ`]. Same capability, minus the
//!   pathology. (Its one unique trick, a per-query rerank pool, survives as
//!   [`HNSWIndex::set_rerank_candidates`].)
//! * **`PQHNSWIndex`** — *not* superseded. Deleted because it was **dominated**. Its selling
//!   point was 192x compression of the vector payload, and it could not convert that into a
//!   usable index. Measured on GIST (960-d, 100k, L2 — PQ's best case, since it is L2-only):
//!
//! ```text
//!                                   MB   recall@10     QPS
//!   PQHNSWIndex, no rerank          18      23.07%    1293
//!   PQHNSWIndex, rerank 100        402      62.27%     790   <- ceiling
//!   PQHNSWIndex, rerank 400        402      60.97%     446   <- gets WORSE
//!   Storage::RaBitQ + rerank       440      97.97%    1970
//!   Storage::SQ8, no rerank        139      98.40%     760
//! ```
//!
//!   The ~62% is a **ceiling, not a knob**: the graph is traversed on PQ codes, so the candidate
//!   pool handed to the rescoring stage does not *contain* the true neighbours — and you cannot
//!   rerank your way to items you never retrieved. Widening the pool made recall fall. Worse, the
//!   compression evaporates precisely when it becomes useful: reaching even 62% requires
//!   retaining the f32 vectors (402 MB), at which point [`Storage::RaBitQ`] costs 440 MB and
//!   delivers 98%.
//!
//!   Note what the docs said before anyone re-measured: **"~55% recall@10"**. That figure was
//!   produced with `rerank_candidates` at its default of **0** — the accuracy stage switched off.
//!   The true no-rerank number is 23%. A bad number produced by a *disabled feature* makes the
//!   feature look inherently bad, and then nobody re-measures it. This library has now been bitten
//!   by that four separate times; see `benchmarks/RESULTS.md`.
//!
//! The [`ProductQuantizer`](crate::vector::product_quantize::ProductQuantizer) primitive is kept.
//! It is a perfectly good quantizer. It is just not a viable way to traverse a graph.
//!
//! A plain zero-threshold binary quantizer is not offered: on non-negative data (SIFT, and
//! most embedding models) every bit is set and the code carries no information — it measured
//! 1.2% recall@10. RaBitQ centers each vector before thresholding, which is the whole
//! difference. See `crate::vector::quantize` for the comparison.
//!
//! # Streaming Operations
//!
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

pub use flat::FlatIndex;
pub use hnsw::{
    BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, MemoryBreakdown, Searcher, Storage,
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

}
