//! Vector index implementations
//!
//! This module provides multiple index types for vector similarity search:
//!
//! - [`FlatIndex`]: Brute-force search (100% accurate, O(n) search)
//! - [`HNSWIndex`]: Approximate nearest neighbors (fast, full precision)
//! - [`SQ8HNSWIndex`]: HNSW with scalar quantization (4x memory reduction)
//! - [`BinaryHNSWIndex`]: HNSW with binary quantization (32x memory reduction)
//!
//! # Memory Comparison (1M vectors × 384 dims)
//!
//! | Index | Memory | Recall | Use Case |
//! |-------|--------|--------|----------|
//! | HNSW (f32) | 1.5 GB | ~95% | Default choice |
//! | SQ8 HNSW | 384 MB | ~90% | Memory constrained |
//! | Binary HNSW | 48 MB | ~80%* | Massive datasets |
//!
//! *Binary recall improves significantly with two-phase search (filter + rerank).
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
pub use hnsw::{BuildStrategy, HNSWConfig, HNSWIndex};
pub use hnsw_pq::{PQHNSWConfig, PQHNSWIndex};
pub use hnsw_quantized::{BinaryHNSWIndex, QuantizedHNSWConfig, SQ8HNSWIndex};
pub use streaming::{
    BatchBuilder, BatchConfig, BatchIndex, BatchProgress, BatchResult, FilteredSearchBuilder,
    PaginationConfig, SearchPage, SearchResultIterator,
};
