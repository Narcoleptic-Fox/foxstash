//! Storage layer with compression and persistence
//!
//! This module provides efficient storage capabilities for the Foxstash system,
//! including multiple compression codecs and persistence mechanisms.
//!
//! # Storage Types
//!
//! - [`FileStorage`]: Simple file-based persistence with compression
//! - [`IncrementalStorage`]: WAL-based incremental persistence for efficient updates
//!
//! # Incremental Storage
//!
//! For large indexes with frequent updates, use `IncrementalStorage` to avoid
//! rewriting the entire index on each change:
//!
//! ```ignore
//! use foxstash_core::storage::incremental::{IncrementalStorage, IncrementalConfig};
//!
//! let config = IncrementalConfig::default()
//!     .with_checkpoint_threshold(10_000);
//!
//! let mut storage = IncrementalStorage::new("/tmp/index", config)?;
//!
//! // Log changes to WAL (fast append-only)
//! storage.log_add(&document)?;
//!
//! // Periodic checkpoint (full snapshot)
//! if storage.needs_checkpoint() {
//!     storage.checkpoint(&index, metadata)?;
//! }
//! ```

pub mod compression;

#[cfg(not(target_arch = "wasm32"))]
pub mod file;

#[cfg(not(target_arch = "wasm32"))]
pub mod incremental;

pub use compression::{Codec, compress, decompress, compress_with, best_codec, CompressionStats};

#[cfg(not(target_arch = "wasm32"))]
pub use file::FileStorage;

#[cfg(not(target_arch = "wasm32"))]
pub use incremental::{
    IncrementalStorage, IncrementalConfig, IndexMetadata, StorageStats,
    WalOperation, WalEntry, CheckpointMeta, Manifest, RecoveryHelper,
};

// Legacy placeholder (will be removed in future)
pub struct MemoryStorage;
