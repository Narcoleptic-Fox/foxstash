//! Embedding generation module
//!
//! This module provides text-to-vector embedding generation capabilities.
//! It currently supports ONNX Runtime-based models (e.g., MiniLM-L6-v2).
//!
//! # Features
//!
//! - **ONNX Runtime**: Enable with `onnx` feature flag for production embeddings
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # {
//! use foxstash_core::embedding::OnnxEmbedder;
//!
//! let embedder = OnnxEmbedder::new(
//!     "models/model.onnx",
//!     "models/tokenizer.json"
//! )?;
//!
//! // Single embedding
//! let embedding = embedder.embed("Hello, world!")?;
//! assert_eq!(embedding.len(), 384);
//!
//! // Batch embedding
//! let embeddings = embedder.embed_batch(&["First text", "Second text"])?;
//! assert_eq!(embeddings.len(), 2);
//! # Ok::<(), foxstash_core::RagError>(())
//! # }
//! ```

#[cfg(feature = "onnx")]
mod onnx;

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedder;

// Placeholder when ONNX feature is not enabled
#[cfg(not(feature = "onnx"))]
#[doc(hidden)]
pub struct OnnxEmbedder;

#[cfg(not(feature = "onnx"))]
impl OnnxEmbedder {
    /// This is a placeholder when the `onnx` feature is not enabled.
    ///
    /// To use ONNX embeddings, enable the `onnx` feature in your Cargo.toml:
    ///
    /// ```toml
    /// [dependencies]
    /// foxstash-core = { version = "*", features = ["onnx"] }
    /// ```
    #[allow(dead_code)]
    pub fn new<P: AsRef<std::path::Path>>(
        _model_path: P,
        _tokenizer_path: P,
    ) -> crate::Result<Self> {
        Err(crate::RagError::EmbeddingError(
            "ONNX feature is not enabled. Enable it in Cargo.toml with features = [\"onnx\"]"
                .to_string(),
        ))
    }
}
