//! Storage layer with compression and persistence
//!
//! This module provides efficient storage capabilities for the Foxstash system,
//! including multiple compression codecs and persistence mechanisms.

pub mod compression;

#[cfg(not(target_arch = "wasm32"))]
pub mod file;

pub use compression::{Codec, compress, decompress, compress_with, best_codec, CompressionStats};

#[cfg(not(target_arch = "wasm32"))]
pub use file::FileStorage;

// Legacy placeholder (will be removed in future)
pub struct MemoryStorage;
