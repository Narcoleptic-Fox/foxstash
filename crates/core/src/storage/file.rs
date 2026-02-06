//! File-based storage for native platforms
//!
//! Provides persistent storage with compression, atomic writes,
//! and metadata management.
//!
//! # Features
//!
//! - **Atomic Writes**: Write to temporary files and rename to prevent corruption
//! - **Compression**: Configurable compression codecs for space efficiency
//! - **Metadata Tracking**: Store creation time, update time, and compression stats
//! - **Type Safety**: Separate methods for documents and indices
//!
//! # Examples
//!
//! ```no_run
//! use foxstash_core::storage::file::{FileStorage};
//! use foxstash_core::storage::compression::Codec;
//! use foxstash_core::Document;
//!
//! # fn main() -> foxstash_core::Result<()> {
//! // Create storage with default codec (None)
//! let storage = FileStorage::new("/tmp/rag_storage")?;
//!
//! // Or with compression
//! let storage = FileStorage::with_codec("/tmp/rag_storage", Codec::Gzip)?;
//!
//! // Save a document
//! let doc = Document {
//!     id: "doc1".to_string(),
//!     content: "Hello world".to_string(),
//!     embedding: vec![0.1; 384],
//!     metadata: None,
//! };
//! let stats = storage.save_document("doc1", &doc)?;
//! println!("Compression ratio: {:.2}", stats.ratio);
//!
//! // Load it back
//! let loaded = storage.load_document("doc1")?;
//! assert_eq!(loaded.id, "doc1");
//!
//! // List all stored items
//! let items = storage.list()?;
//! println!("Stored items: {:?}", items);
//! # Ok(())
//! # }
//! ```

#![cfg(not(target_arch = "wasm32"))]

use crate::storage::compression::{self, Codec, CompressionStats};
use crate::{Document, RagError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const STORAGE_VERSION: u32 = 1;
const DATA_EXTENSION: &str = "data";
const META_EXTENSION: &str = "meta";
const TMP_EXTENSION: &str = "tmp";

/// Metadata for stored items
///
/// Contains information about the stored item including version,
/// timestamps, and compression statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    /// Storage format version
    pub version: u32,
    /// Unix timestamp when item was created
    pub created_at: u64,
    /// Unix timestamp when item was last updated
    pub updated_at: u64,
    /// Type of stored item ("document", "flat_index", "hnsw_index")
    pub item_type: String,
    /// Compression codec used
    pub compression: Codec,
    /// Original size before compression (bytes)
    pub original_size: usize,
    /// Compressed size after compression (bytes)
    pub compressed_size: usize,
}

impl StorageMetadata {
    /// Create new metadata
    fn new(
        item_type: String,
        compression: Codec,
        original_size: usize,
        compressed_size: usize,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            version: STORAGE_VERSION,
            created_at: now,
            updated_at: now,
            item_type,
            compression,
            original_size,
            compressed_size,
        }
    }

    /// Update the updated_at timestamp
    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// File-based storage manager
///
/// Manages persistent storage of documents and indices on the filesystem.
/// Uses atomic writes to prevent corruption and supports configurable compression.
///
/// # Directory Structure
///
/// ```text
/// base_path/
/// ├── doc1.data       # Serialized and compressed document
/// ├── doc1.meta       # Metadata for document
/// ├── index1.data     # Serialized and compressed index
/// └── index1.meta     # Metadata for index
/// ```
#[derive(Debug)]
pub struct FileStorage {
    base_path: PathBuf,
    codec: Codec,
}

impl FileStorage {
    /// Create new file storage at the specified path
    ///
    /// Creates the directory if it doesn't exist. Uses no compression by default.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory path for storage
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - New FileStorage instance
    ///
    /// # Errors
    ///
    /// Returns error if directory creation fails or path is invalid.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// let storage = FileStorage::new("/tmp/my_storage").unwrap();
    /// ```
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_codec(base_path, Codec::None)
    }

    /// Create file storage with specific compression codec
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory path for storage
    /// * `codec` - Compression codec to use
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - New FileStorage instance
    ///
    /// # Errors
    ///
    /// Returns error if directory creation fails or path is invalid.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # use foxstash_core::storage::compression::Codec;
    /// let storage = FileStorage::with_codec("/tmp/my_storage", Codec::Gzip).unwrap();
    /// ```
    pub fn with_codec(base_path: impl AsRef<Path>, codec: Codec) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path).map_err(|e| {
                RagError::StorageError(format!("Failed to create storage directory: {}", e))
            })?;
        }

        // Verify it's a directory
        if !base_path.is_dir() {
            return Err(RagError::StorageError(format!(
                "Storage path is not a directory: {}",
                base_path.display()
            )));
        }

        Ok(Self { base_path, codec })
    }

    /// Save document with compression
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the document
    /// * `document` - Document to save
    ///
    /// # Returns
    ///
    /// * `Result<CompressionStats>` - Compression statistics
    ///
    /// # Errors
    ///
    /// Returns error if serialization or writing fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # use foxstash_core::Document;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test".to_string(),
    ///     embedding: vec![0.1; 384],
    ///     metadata: None,
    /// };
    /// let stats = storage.save_document("doc1", &doc)?;
    /// println!("Saved with ratio: {:.2}", stats.ratio);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_document(&self, id: &str, document: &Document) -> Result<CompressionStats> {
        // Use JSON serialization for documents because they contain serde_json::Value metadata
        let serialized = serde_json::to_vec(document)
            .map_err(|e| RagError::StorageError(format!("JSON serialization failed: {}", e)))?;

        // Compress the data
        let (compressed, stats) = compression::compress_with(&serialized, self.codec)
            .map_err(|e| RagError::StorageError(format!("Compression failed: {}", e)))?;

        // Create or update metadata
        let metadata = if self.exists(id) {
            let mut meta = self.get_metadata(id)?;
            meta.touch();
            meta.original_size = stats.original_size;
            meta.compressed_size = stats.compressed_size;
            meta.compression = stats.codec;
            meta
        } else {
            StorageMetadata::new(
                "document".to_string(),
                stats.codec,
                stats.original_size,
                stats.compressed_size,
            )
        };

        // Save data file atomically
        let data_path = self.item_path(id);
        self.write_atomic(&data_path, &compressed)?;

        // Save metadata file atomically
        let meta_path = self.metadata_path(id);
        let meta_bytes = bincode::serialize(&metadata)?;
        self.write_atomic(&meta_path, &meta_bytes)?;

        Ok(stats)
    }

    /// Load document
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the document
    ///
    /// # Returns
    ///
    /// * `Result<Document>` - Loaded document
    ///
    /// # Errors
    ///
    /// Returns error if document doesn't exist or deserialization fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// let doc = storage.load_document("doc1")?;
    /// println!("Loaded: {}", doc.id);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_document(&self, id: &str) -> Result<Document> {
        // Check if item exists
        if !self.exists(id) {
            return Err(RagError::StorageError(format!(
                "Document not found: {}",
                id
            )));
        }

        // Load metadata
        let metadata = self.get_metadata(id)?;

        // Check version compatibility
        if metadata.version != STORAGE_VERSION {
            return Err(RagError::StorageError(format!(
                "Incompatible storage version: expected {}, got {}",
                STORAGE_VERSION, metadata.version
            )));
        }

        // Load data file
        let data_path = self.item_path(id);
        let mut file = File::open(&data_path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        // Verify size matches metadata
        if compressed.len() != metadata.compressed_size {
            return Err(RagError::StorageError(format!(
                "Data corruption detected: size mismatch for {}",
                id
            )));
        }

        // Decompress (codec detected automatically from header)
        let decompressed = compression::decompress(&compressed)
            .map_err(|e| RagError::StorageError(format!("Decompression failed: {}", e)))?;

        // Deserialize using JSON
        let document: Document = serde_json::from_slice(&decompressed)
            .map_err(|e| RagError::StorageError(format!("JSON deserialization failed: {}", e)))?;

        Ok(document)
    }

    /// Save FlatIndex
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the index
    /// * `index` - FlatIndex to save
    ///
    /// # Returns
    ///
    /// * `Result<CompressionStats>` - Compression statistics
    ///
    /// # Errors
    ///
    /// Returns error if serialization or writing fails.
    pub fn save_flat_index(
        &self,
        name: &str,
        index: &FlatIndexWrapper,
    ) -> Result<CompressionStats> {
        self.save_with_metadata(name, index, "flat_index")
    }

    /// Load FlatIndex
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the index
    ///
    /// # Returns
    ///
    /// * `Result<FlatIndex>` - Loaded index
    ///
    /// # Errors
    ///
    /// Returns error if index doesn't exist or deserialization fails.
    pub fn load_flat_index(&self, name: &str) -> Result<FlatIndexWrapper> {
        self.load_with_metadata(name)
    }

    /// Save HNSWIndex
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the index
    /// * `index` - HNSWIndex to save
    ///
    /// # Returns
    ///
    /// * `Result<CompressionStats>` - Compression statistics
    ///
    /// # Errors
    ///
    /// Returns error if serialization or writing fails.
    pub fn save_hnsw_index(
        &self,
        name: &str,
        index: &HNSWIndexWrapper,
    ) -> Result<CompressionStats> {
        self.save_with_metadata(name, index, "hnsw_index")
    }

    /// Load HNSWIndex
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the index
    ///
    /// # Returns
    ///
    /// * `Result<HNSWIndex>` - Loaded index
    ///
    /// # Errors
    ///
    /// Returns error if index doesn't exist or deserialization fails.
    pub fn load_hnsw_index(&self, name: &str) -> Result<HNSWIndexWrapper> {
        self.load_with_metadata(name)
    }

    /// Delete item from storage
    ///
    /// Removes both the data and metadata files.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the item to delete
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if successful
    ///
    /// # Errors
    ///
    /// Returns error if deletion fails. Does not error if item doesn't exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// storage.delete("doc1")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&self, name: &str) -> Result<()> {
        let data_path = self.item_path(name);
        let meta_path = self.metadata_path(name);

        // Delete data file if it exists
        if data_path.exists() {
            fs::remove_file(&data_path).map_err(|e| {
                RagError::StorageError(format!("Failed to delete data file: {}", e))
            })?;
        }

        // Delete metadata file if it exists
        if meta_path.exists() {
            fs::remove_file(&meta_path).map_err(|e| {
                RagError::StorageError(format!("Failed to delete metadata file: {}", e))
            })?;
        }

        Ok(())
    }

    /// List all items in storage
    ///
    /// Returns names of all stored items (without extensions).
    ///
    /// # Returns
    ///
    /// * `Result<Vec<String>>` - List of item names
    ///
    /// # Errors
    ///
    /// Returns error if directory reading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// let items = storage.list()?;
    /// for item in items {
    ///     println!("Found: {}", item);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn list(&self) -> Result<Vec<String>> {
        let entries = fs::read_dir(&self.base_path).map_err(|e| {
            RagError::StorageError(format!("Failed to read storage directory: {}", e))
        })?;

        let mut names = std::collections::HashSet::new();

        for entry in entries {
            let entry = entry.map_err(|e| {
                RagError::StorageError(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == DATA_EXTENSION || ext == META_EXTENSION {
                        if let Some(stem) = path.file_stem() {
                            if let Some(name) = stem.to_str() {
                                names.insert(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        let mut result: Vec<String> = names.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get metadata for an item
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the item
    ///
    /// # Returns
    ///
    /// * `Result<StorageMetadata>` - Item metadata
    ///
    /// # Errors
    ///
    /// Returns error if metadata doesn't exist or can't be read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// let meta = storage.get_metadata("doc1")?;
    /// println!("Type: {}, Size: {} bytes", meta.item_type, meta.compressed_size);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_metadata(&self, name: &str) -> Result<StorageMetadata> {
        let meta_path = self.metadata_path(name);

        if !meta_path.exists() {
            return Err(RagError::StorageError(format!(
                "Metadata not found for item: {}",
                name
            )));
        }

        let mut file = File::open(&meta_path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        let metadata: StorageMetadata = bincode::deserialize(&contents)?;
        Ok(metadata)
    }

    /// Get total storage size in bytes
    ///
    /// Calculates the sum of all data and metadata files.
    ///
    /// # Returns
    ///
    /// * `Result<u64>` - Total size in bytes
    ///
    /// # Errors
    ///
    /// Returns error if directory reading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// let size = storage.total_size()?;
    /// println!("Storage uses {} bytes", size);
    /// # Ok(())
    /// # }
    /// ```
    pub fn total_size(&self) -> Result<u64> {
        let entries = fs::read_dir(&self.base_path).map_err(|e| {
            RagError::StorageError(format!("Failed to read storage directory: {}", e))
        })?;

        let mut total = 0u64;

        for entry in entries {
            let entry = entry.map_err(|e| {
                RagError::StorageError(format!("Failed to read directory entry: {}", e))
            })?;

            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total += metadata.len();
            }
        }

        Ok(total)
    }

    /// Clear all storage
    ///
    /// Removes all data and metadata files from storage.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if successful
    ///
    /// # Errors
    ///
    /// Returns error if file deletion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// storage.clear()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear(&self) -> Result<()> {
        let entries = fs::read_dir(&self.base_path).map_err(|e| {
            RagError::StorageError(format!("Failed to read storage directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                RagError::StorageError(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();
            if path.is_file() {
                fs::remove_file(&path)
                    .map_err(|e| RagError::StorageError(format!("Failed to delete file: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Check if item exists in storage
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the item
    ///
    /// # Returns
    ///
    /// * `bool` - true if item exists
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use foxstash_core::storage::file::FileStorage;
    /// # fn main() -> foxstash_core::Result<()> {
    /// let storage = FileStorage::new("/tmp/storage")?;
    /// if storage.exists("doc1") {
    ///     println!("Document exists!");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn exists(&self, name: &str) -> bool {
        self.item_path(name).exists() && self.metadata_path(name).exists()
    }

    // Internal helper methods

    /// Get path for item data file
    fn item_path(&self, name: &str) -> PathBuf {
        self.base_path.join(format!("{}.{}", name, DATA_EXTENSION))
    }

    /// Get path for item metadata file
    fn metadata_path(&self, name: &str) -> PathBuf {
        self.base_path.join(format!("{}.{}", name, META_EXTENSION))
    }

    /// Atomic write: write to temp file, then rename
    ///
    /// This ensures that even if the process crashes during write,
    /// the original file is not corrupted.
    ///
    /// # Arguments
    ///
    /// * `path` - Target file path
    /// * `data` - Data to write
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if successful
    ///
    /// # Errors
    ///
    /// Returns error if write or rename fails.
    fn write_atomic(&self, path: &Path, data: &[u8]) -> Result<()> {
        // Create temp file path
        let tmp_path = path.with_extension(TMP_EXTENSION);

        // Write to temp file
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(data)?;
            file.sync_all()?; // Ensure data is flushed to disk
        }

        // Atomically rename temp to final
        fs::rename(&tmp_path, path).map_err(|e| {
            // Try to clean up temp file if rename fails
            let _ = fs::remove_file(&tmp_path);
            RagError::IoError(e)
        })?;

        Ok(())
    }

    /// Save item with metadata
    ///
    /// Generic method for saving any serializable item with metadata tracking.
    fn save_with_metadata<T: Serialize>(
        &self,
        name: &str,
        item: &T,
        item_type: &str,
    ) -> Result<CompressionStats> {
        // Serialize the item
        let serialized = serde_json::to_vec(item)
            .map_err(|e| RagError::StorageError(format!("JSON serialization failed: {}", e)))?;

        // Compress the data
        let (compressed, stats) = compression::compress_with(&serialized, self.codec)
            .map_err(|e| RagError::StorageError(format!("Compression failed: {}", e)))?;

        // Create or update metadata
        let metadata = if self.exists(name) {
            let mut meta = self.get_metadata(name)?;
            meta.touch();
            meta.original_size = stats.original_size;
            meta.compressed_size = stats.compressed_size;
            meta.compression = stats.codec;
            meta
        } else {
            StorageMetadata::new(
                item_type.to_string(),
                stats.codec,
                stats.original_size,
                stats.compressed_size,
            )
        };

        // Save data file atomically
        let data_path = self.item_path(name);
        self.write_atomic(&data_path, &compressed)?;

        // Save metadata file atomically
        let meta_path = self.metadata_path(name);
        let meta_bytes = bincode::serialize(&metadata)?;
        self.write_atomic(&meta_path, &meta_bytes)?;

        Ok(stats)
    }

    /// Load item with metadata check
    ///
    /// Generic method for loading any deserializable item with metadata verification.
    fn load_with_metadata<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T> {
        // Check if item exists
        if !self.exists(name) {
            return Err(RagError::StorageError(format!("Item not found: {}", name)));
        }

        // Load metadata
        let metadata = self.get_metadata(name)?;

        // Check version compatibility
        if metadata.version != STORAGE_VERSION {
            return Err(RagError::StorageError(format!(
                "Incompatible storage version: expected {}, got {}",
                STORAGE_VERSION, metadata.version
            )));
        }

        // Load data file
        let data_path = self.item_path(name);
        let mut file = File::open(&data_path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        // Verify size matches metadata
        if compressed.len() != metadata.compressed_size {
            return Err(RagError::StorageError(format!(
                "Data corruption detected: size mismatch for {}",
                name
            )));
        }

        // Decompress (codec detected automatically from header)
        let decompressed = compression::decompress(&compressed)
            .map_err(|e| RagError::StorageError(format!("Decompression failed: {}", e)))?;

        // Deserialize
        let item: T = serde_json::from_slice::<T>(&decompressed)
            .map_err(|e| RagError::StorageError(format!("JSON deserialization failed: {}", e)))?;

        Ok(item)
    }
}

/// Wrapper for FlatIndex to enable serialization
///
/// Since FlatIndex uses HashMap internally, we need to ensure it's serializable.
/// This wrapper provides serialization support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndexWrapper {
    pub embedding_dim: usize,
    pub documents: Vec<Document>,
}

impl FlatIndexWrapper {
    /// Create wrapper from FlatIndex
    pub fn from_index(index: &crate::index::FlatIndex) -> Self {
        Self {
            embedding_dim: index.embedding_dim(),
            documents: index.get_all_documents(),
        }
    }

    /// Convert wrapper to FlatIndex
    pub fn to_index(&self) -> Result<crate::index::FlatIndex> {
        let mut index = crate::index::FlatIndex::new(self.embedding_dim);
        index.add_batch(self.documents.clone())?;
        Ok(index)
    }
}

/// Wrapper for HNSWIndex to enable serialization
///
/// HNSWIndex has complex internal structures, so we serialize it as a flat list
/// of documents and rebuild the index on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndexWrapper {
    pub embedding_dim: usize,
    pub documents: Vec<Document>,
    pub config: HNSWConfigWrapper,
}

/// Serializable wrapper for HNSWConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfigWrapper {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f32,
    #[serde(default = "default_use_heuristic")]
    pub use_heuristic: bool,
    #[serde(default)]
    pub extend_candidates: bool,
    #[serde(default = "default_keep_pruned")]
    pub keep_pruned_connections: bool,
}

fn default_use_heuristic() -> bool {
    true
}
fn default_keep_pruned() -> bool {
    true
}

impl From<&crate::index::HNSWConfig> for HNSWConfigWrapper {
    fn from(config: &crate::index::HNSWConfig) -> Self {
        Self {
            m: config.m,
            m0: config.m0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            ml: config.ml,
            use_heuristic: config.use_heuristic,
            extend_candidates: config.extend_candidates,
            keep_pruned_connections: config.keep_pruned_connections,
        }
    }
}

impl From<HNSWConfigWrapper> for crate::index::HNSWConfig {
    fn from(wrapper: HNSWConfigWrapper) -> Self {
        Self {
            m: wrapper.m,
            m0: wrapper.m0,
            ef_construction: wrapper.ef_construction,
            ef_search: wrapper.ef_search,
            ml: wrapper.ml,
            use_heuristic: wrapper.use_heuristic,
            extend_candidates: wrapper.extend_candidates,
            keep_pruned_connections: wrapper.keep_pruned_connections,
            build_strategy: crate::index::BuildStrategy::default(),
            seed: None,
        }
    }
}

impl HNSWIndexWrapper {
    /// Create wrapper from HNSWIndex
    pub fn from_index(index: &crate::index::HNSWIndex) -> Self {
        Self {
            embedding_dim: index.embedding_dim(),
            documents: index.get_all_documents(),
            config: HNSWConfigWrapper::from(index.config()),
        }
    }

    /// Convert wrapper to HNSWIndex
    pub fn to_index(&self) -> Result<crate::index::HNSWIndex> {
        let config: crate::index::HNSWConfig = self.config.clone().into();
        let mut index = crate::index::HNSWIndex::new(self.embedding_dim, config);
        for doc in &self.documents {
            index.add(doc.clone())?;
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_document(id: &str) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Test content for {}", id),
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            metadata: Some(serde_json::json!({"test": true})),
        }
    }

    fn create_test_flat_index() -> crate::index::FlatIndex {
        let mut index = crate::index::FlatIndex::new(5);
        index.add(create_test_document("doc1")).unwrap();
        index.add(create_test_document("doc2")).unwrap();
        index
    }

    fn create_test_hnsw_index() -> crate::index::HNSWIndex {
        let mut index = crate::index::HNSWIndex::with_defaults(5);
        index.add(create_test_document("doc1")).unwrap();
        index.add(create_test_document("doc2")).unwrap();
        index
    }

    #[test]
    fn test_new_storage() {
        let dir = tempdir().unwrap();
        let _storage = FileStorage::new(dir.path()).unwrap();
        assert!(dir.path().exists());
        assert!(dir.path().is_dir());
    }

    #[test]
    fn test_new_storage_with_codec() {
        let dir = tempdir().unwrap();
        let _storage = FileStorage::with_codec(dir.path(), Codec::Gzip).unwrap();
        assert!(dir.path().exists());
    }

    #[test]
    fn test_invalid_storage_path() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("file.txt");
        std::fs::write(&file_path, b"test").unwrap();

        let result = FileStorage::new(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_document_save_load() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let doc = create_test_document("doc1");
        let stats = storage.save_document("doc1", &doc).unwrap();

        assert!(stats.original_size > 0);
        assert_eq!(stats.codec, Codec::None);

        let loaded = storage.load_document("doc1").unwrap();
        assert_eq!(loaded.id, doc.id);
        assert_eq!(loaded.content, doc.content);
        assert_eq!(loaded.embedding, doc.embedding);
    }

    #[test]
    fn test_document_not_found() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let result = storage.load_document("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_index_persistence() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let index = create_test_flat_index();
        let wrapper = FlatIndexWrapper::from_index(&index);

        let stats = storage.save_flat_index("index1", &wrapper).unwrap();
        assert!(stats.original_size > 0);

        let loaded_wrapper = storage.load_flat_index("index1").unwrap();
        let loaded_index = loaded_wrapper.to_index().unwrap();

        assert_eq!(loaded_index.len(), index.len());
        assert_eq!(loaded_index.embedding_dim(), index.embedding_dim());
    }

    #[test]
    fn test_hnsw_index_persistence() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let index = create_test_hnsw_index();
        let wrapper = HNSWIndexWrapper::from_index(&index);

        let stats = storage.save_hnsw_index("index1", &wrapper).unwrap();
        assert!(stats.original_size > 0);

        let loaded_wrapper = storage.load_hnsw_index("index1").unwrap();
        let loaded_index = loaded_wrapper.to_index().unwrap();

        assert_eq!(loaded_index.len(), index.len());
        assert_eq!(loaded_index.embedding_dim(), index.embedding_dim());
    }

    #[test]
    fn test_atomic_write() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let path = dir.path().join("test.data");
        let data = b"test data";

        storage.write_atomic(&path, data).unwrap();

        assert!(path.exists());
        let read_data = std::fs::read(&path).unwrap();
        assert_eq!(read_data, data);

        // Verify no temp files left behind
        let tmp_path = dir.path().join("test.tmp");
        assert!(!tmp_path.exists());
    }

    #[test]
    fn test_metadata() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let doc = create_test_document("doc1");
        storage.save_document("doc1", &doc).unwrap();

        let metadata = storage.get_metadata("doc1").unwrap();
        assert_eq!(metadata.version, STORAGE_VERSION);
        assert_eq!(metadata.item_type, "document");
        assert!(metadata.created_at > 0);
        assert_eq!(metadata.created_at, metadata.updated_at);
        assert_eq!(metadata.compression, Codec::None);
        assert!(metadata.original_size > 0);
    }

    #[test]
    fn test_metadata_update() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let doc = create_test_document("doc1");
        storage.save_document("doc1", &doc).unwrap();

        let meta1 = storage.get_metadata("doc1").unwrap();

        // Wait a bit to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Save again
        storage.save_document("doc1", &doc).unwrap();

        let meta2 = storage.get_metadata("doc1").unwrap();
        assert_eq!(meta2.created_at, meta1.created_at);
        assert!(meta2.updated_at >= meta1.updated_at);
    }

    #[test]
    fn test_list_storage() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        assert_eq!(storage.list().unwrap().len(), 0);

        storage
            .save_document("doc1", &create_test_document("doc1"))
            .unwrap();
        storage
            .save_document("doc2", &create_test_document("doc2"))
            .unwrap();
        storage
            .save_document("doc3", &create_test_document("doc3"))
            .unwrap();

        let items = storage.list().unwrap();
        assert_eq!(items.len(), 3);
        assert!(items.contains(&"doc1".to_string()));
        assert!(items.contains(&"doc2".to_string()));
        assert!(items.contains(&"doc3".to_string()));
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let doc = create_test_document("doc1");
        storage.save_document("doc1", &doc).unwrap();

        assert!(storage.exists("doc1"));
        assert_eq!(storage.list().unwrap().len(), 1);

        storage.delete("doc1").unwrap();

        assert!(!storage.exists("doc1"));
        assert_eq!(storage.list().unwrap().len(), 0);
    }

    #[test]
    fn test_delete_nonexistent() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        // Should not error when deleting non-existent item
        let result = storage.delete("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        storage
            .save_document("doc1", &create_test_document("doc1"))
            .unwrap();
        storage
            .save_document("doc2", &create_test_document("doc2"))
            .unwrap();
        storage
            .save_document("doc3", &create_test_document("doc3"))
            .unwrap();

        assert_eq!(storage.list().unwrap().len(), 3);

        storage.clear().unwrap();

        assert_eq!(storage.list().unwrap().len(), 0);
    }

    #[test]
    fn test_storage_size() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        assert_eq!(storage.total_size().unwrap(), 0);

        storage
            .save_document("doc1", &create_test_document("doc1"))
            .unwrap();

        let size = storage.total_size().unwrap();
        assert!(size > 0);

        storage
            .save_document("doc2", &create_test_document("doc2"))
            .unwrap();

        let size2 = storage.total_size().unwrap();
        assert!(size2 > size);
    }

    #[test]
    fn test_compression_codecs() {
        let dir = tempdir().unwrap();

        // Test with different codecs
        #[allow(unused_mut)]
        let mut codecs = vec![Codec::None, Codec::Gzip];

        #[cfg(feature = "zstd")]
        codecs.push(Codec::Zstd);

        #[cfg(feature = "lz4")]
        codecs.push(Codec::Lz4);

        for codec in codecs {
            let storage = FileStorage::with_codec(dir.path(), codec).unwrap();
            let doc = create_test_document("doc1");

            let stats = storage.save_document("test", &doc).unwrap();
            assert!(stats.original_size > 0);

            let loaded = storage.load_document("test").unwrap();
            assert_eq!(loaded.id, doc.id);
            assert_eq!(loaded.content, doc.content);

            storage.delete("test").unwrap();
        }
    }

    #[test]
    fn test_exists() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        assert!(!storage.exists("doc1"));

        storage
            .save_document("doc1", &create_test_document("doc1"))
            .unwrap();

        assert!(storage.exists("doc1"));
        assert!(!storage.exists("doc2"));
    }

    #[test]
    fn test_flat_index_wrapper_roundtrip() {
        let index = create_test_flat_index();
        let wrapper = FlatIndexWrapper::from_index(&index);
        let restored = wrapper.to_index().unwrap();

        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.embedding_dim(), index.embedding_dim());

        // Test search works
        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = restored.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hnsw_index_wrapper_roundtrip() {
        let index = create_test_hnsw_index();
        let wrapper = HNSWIndexWrapper::from_index(&index);
        let restored = wrapper.to_index().unwrap();

        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.embedding_dim(), index.embedding_dim());

        // Test search works
        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = restored.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_concurrent_writes() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        // Write same document multiple times to test atomicity
        let doc = create_test_document("doc1");

        for _ in 0..10 {
            storage.save_document("doc1", &doc).unwrap();
            let loaded = storage.load_document("doc1").unwrap();
            assert_eq!(loaded.id, doc.id);
        }
    }

    #[test]
    fn test_large_document() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        // Create a large document
        let mut large_doc = create_test_document("large");
        large_doc.embedding = vec![0.5; 10000];
        large_doc.content = "x".repeat(100000);

        let stats = storage.save_document("large", &large_doc).unwrap();
        assert!(stats.original_size > 100000);

        let loaded = storage.load_document("large").unwrap();
        assert_eq!(loaded.id, large_doc.id);
        assert_eq!(loaded.embedding.len(), 10000);
        assert_eq!(loaded.content.len(), 100000);
    }
}
