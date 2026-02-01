//! Foxstash - Core library
//!
//! High-performance vector search and embedding generation for local-first AI.

pub mod vector;
pub mod index;
pub mod embedding;
pub mod storage;

use thiserror::Error;

/// Result type for RAG operations
pub type Result<T> = std::result::Result<T, RagError>;

/// Error types for RAG operations
#[derive(Debug, Error)]
pub enum RagError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("Compression error: {0}")]
    CompressionError(#[from] storage::compression::CompressionError),
}

/// Document with embedding
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Search result with similarity score
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Configuration for RAG system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagConfig {
    pub embedding_dim: usize,
    pub max_documents: usize,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum IndexType {
    Flat,
    HNSW,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384, // MiniLM-L6-v2
            max_documents: 10_000,
            index_type: IndexType::HNSW,
        }
    }
}

// Re-export commonly used items
pub use vector::{cosine_similarity, dot_product, l2_distance, normalize};
