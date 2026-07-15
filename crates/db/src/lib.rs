//! foxstash-db — Database layer for foxstash
//!
//! Provides collections, metadata filtering, WAL-backed persistence,
//! recovery, and concurrent access on top of foxstash-core's vector engine.
//!
//! # Architecture
//!
//! ```text
//! foxstash-core (HNSWIndex, IncrementalStorage)
//!       |
//! foxstash-db  (Collection, VectorStore, Filter, Recovery)
//!       |
//! consumer     (foxloom, CLI tools, etc.)
//! ```
//!
//! # Platform Support
//!
//! The pure-algorithm modules ([`hybrid`], [`inverted_index`], [`tokenizer`],
//! [`text_index`]) compile on all targets including `wasm32`. The filesystem-
//! dependent modules (`collection`, `store`, `recovery`, `filter`, `id_map`)
//! are only available on non-WASM targets.

// ── Pure algorithm modules — available on all targets including wasm32 ──

pub mod hybrid;
pub mod inverted_index;
pub mod text_index;
pub mod tokenizer;

// ── Filesystem-dependent modules — desktop/server only ──

#[cfg(not(target_arch = "wasm32"))]
pub mod collection;
#[cfg(not(target_arch = "wasm32"))]
pub mod filter;
#[cfg(not(target_arch = "wasm32"))]
pub mod id_map;
#[cfg(not(target_arch = "wasm32"))]
pub mod recovery;
#[cfg(not(target_arch = "wasm32"))]
pub mod store;

// ── Desktop-only imports and types ──

#[cfg(not(target_arch = "wasm32"))]
use foxstash_core::index::{HNSWConfig, Storage};
#[cfg(not(target_arch = "wasm32"))]
use crate::inverted_index::BM25Config;
use foxstash_core::storage::IncrementalConfig;
#[cfg(not(target_arch = "wasm32"))]
use thiserror::Error;

/// Errors from the database layer.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Error)]
pub enum DbError {
    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("collection already exists: {0}")]
    CollectionExists(String),

    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("core error: {0}")]
    Core(#[from] foxstash_core::RagError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("recovery error: {0}")]
    Recovery(String),

    #[error("validation error: {0}")]
    Validation(String),

    /// `Collection` ingests documents one at a time. Quantized storage needs a codebook
    /// (`SQ8`'s per-dimension min/scale, or RaBitQ's rotation) fitted on a corpus sample
    /// *before* the first vector is encoded — `HNSWIndex::build`/`build_parallel` do that,
    /// `HNSWIndex::new` does not. A `Collection` backed by an untrained quantized index
    /// panics on its first insert, so this is rejected up front instead.
    #[error(
        "collection storage must be Storage::F32: {storage:?} requires a codebook trained on \
         a corpus sample before any vector can be encoded, but a collection ingests documents \
         incrementally and has no such sample at construction time. For quantized storage, \
         build the index once via foxstash_core::index::HNSWIndex::build_parallel over the \
         full corpus instead of through Collection::insert."
    )]
    UnsupportedIncrementalStorage { storage: Storage },
}

#[cfg(not(target_arch = "wasm32"))]
pub type Result<T> = std::result::Result<T, DbError>;

/// Configuration for opening a [`VectorStore`].
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct DbConfig {
    /// HNSW index configuration.
    pub hnsw: HNSWConfig,
    /// Incremental storage configuration.
    pub storage: IncrementalConfig,
    /// BM25 scoring parameters for the keyword half of hybrid search.
    ///
    /// `BM25Config` was public, and `InvertedIndex::with_config` was public, and there was no path
    /// between them: every construction site in `Collection` and `recovery` called
    /// `InvertedIndex::new()`, so `k1` and `b` were unreachable from any public API. A knob you
    /// cannot turn is not a knob. This field is the path.
    pub bm25: BM25Config,
    /// Embedding dimensionality shared by every collection in this store.
    ///
    /// All documents inserted into any collection must have embeddings of
    /// exactly this length. If you need collections with different dimensions
    /// (e.g. different embedding models), open separate `VectorStore` instances
    /// pointing to different directories.
    pub embedding_dim: usize,
    /// Whether to auto-checkpoint after threshold mutations.
    pub auto_checkpoint: bool,
    /// Default hybrid search configuration.
    pub hybrid: HybridConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for DbConfig {
    fn default() -> Self {
        Self {
            hnsw: HNSWConfig::default(),
            storage: IncrementalConfig::default(),
            bm25: BM25Config::default(),
            embedding_dim: 384,
            auto_checkpoint: true,
            hybrid: HybridConfig::default(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DbConfig {
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        assert!(dim > 0, "embedding_dim must be greater than zero");
        self.embedding_dim = dim;
        self
    }

    pub fn with_hnsw(mut self, hnsw: HNSWConfig) -> Self {
        self.hnsw = hnsw;
        self
    }

    pub fn with_storage(mut self, storage: IncrementalConfig) -> Self {
        self.storage = storage;
        self
    }

    pub fn with_auto_checkpoint(mut self, auto: bool) -> Self {
        self.auto_checkpoint = auto;
        self
    }

    pub fn with_hybrid(mut self, hybrid: HybridConfig) -> Self {
        self.hybrid = hybrid;
        self
    }
}

// ── Re-exports: always available ──

pub use hybrid::{HybridConfig, MergeStrategy};
pub use text_index::TextIndex;

// ── Re-exports: desktop only ──

#[cfg(not(target_arch = "wasm32"))]
pub use collection::Collection;
#[cfg(not(target_arch = "wasm32"))]
pub use filter::Filter;
#[cfg(not(target_arch = "wasm32"))]
pub use foxstash_core::{Document, SearchResult};
#[cfg(not(target_arch = "wasm32"))]
pub use store::VectorStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "embedding_dim must be greater than zero")]
    fn config_rejects_zero_embedding_dim() {
        DbConfig::default().with_embedding_dim(0);
    }
}
