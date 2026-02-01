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

pub mod flat;
pub mod hnsw;
pub mod hnsw_quantized;

pub use flat::FlatIndex;
pub use hnsw::{HNSWConfig, HNSWIndex};
pub use hnsw_quantized::{BinaryHNSWIndex, QuantizedHNSWConfig, SQ8HNSWIndex};
