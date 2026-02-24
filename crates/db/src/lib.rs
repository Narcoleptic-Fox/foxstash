//! foxstash-db — Database layer for foxstash
//!
//! Provides collections, metadata filtering, WAL-backed persistence,
//! recovery, and concurrent access on top of foxstash-core's vector engine.
//!
//! # Architecture
//!
//! ```text
//! foxstash-core (HNSWIndex, IncrementalStorage)
//!       ↓
//! foxstash-db  (Collection, VectorStore, Filter, Recovery)
//!       ↓
//! consumer     (foxloom, CLI tools, etc.)
//! ```

#![cfg(not(target_arch = "wasm32"))]

pub mod collection;
pub mod filter;
pub mod hybrid;
pub mod id_map;
pub mod inverted_index;
pub mod recovery;
pub mod store;
pub mod text_index;
pub mod tokenizer;

use foxstash_core::index::HNSWConfig;
use foxstash_core::storage::IncrementalConfig;
use thiserror::Error;

/// Errors from the database layer.
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
}

pub type Result<T> = std::result::Result<T, DbError>;

/// Configuration for opening a [`VectorStore`].
#[derive(Debug, Clone)]
pub struct DbConfig {
    /// HNSW index configuration.
    pub hnsw: HNSWConfig,
    /// Incremental storage configuration.
    pub storage: IncrementalConfig,
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

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            hnsw: HNSWConfig::default(),
            storage: IncrementalConfig::default(),
            embedding_dim: 384,
            auto_checkpoint: true,
            hybrid: HybridConfig::default(),
        }
    }
}

impl DbConfig {
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
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

// Re-export key types consumers need.
pub use collection::Collection;
pub use filter::Filter;
pub use foxstash_core::{Document, SearchResult};
pub use hybrid::{HybridConfig, MergeStrategy};
pub use store::VectorStore;
pub use text_index::TextIndex;
