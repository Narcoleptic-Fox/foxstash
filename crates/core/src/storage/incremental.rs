//! Incremental persistence with Write-Ahead Log (WAL)
//!
//! This module provides efficient incremental persistence for vector indexes:
//!
//! - **Write-Ahead Log (WAL)**: Append-only log for fast writes
//! - **Checkpointing**: Periodic full snapshots for fast recovery
//! - **Recovery**: Replay WAL from last checkpoint
//! - **Compaction**: Merge WAL into checkpoint to reclaim space
//!
//! # Architecture
//!
//! ```text
//! storage/
//! ├── checkpoint_00001.bin   # Full index snapshot
//! ├── checkpoint_00001.meta  # Checkpoint metadata
//! ├── wal_00001.log          # WAL entries since checkpoint
//! └── manifest.json          # Current state pointer
//! ```
//!
//! # Example
//!
//! ```ignore
//! use foxstash_core::storage::incremental::{IncrementalStorage, IncrementalConfig};
//! use foxstash_core::index::HNSWIndex;
//! use foxstash_core::Document;
//!
//! // Create incremental storage
//! let config = IncrementalConfig::default()
//!     .with_wal_sync_interval(100)    // Sync WAL every 100 ops
//!     .with_checkpoint_threshold(10000); // Checkpoint every 10K ops
//!
//! let mut storage = IncrementalStorage::new("/tmp/index_storage", config)?;
//!
//! // Load or create index
//! let mut index = storage.load_or_create::<HNSWIndex>(384)?;
//!
//! // Add documents - automatically logged to WAL
//! for doc in documents {
//!     storage.log_add(&doc)?;
//!     index.add(doc)?;
//! }
//!
//! // Explicit checkpoint (or automatic based on threshold)
//! storage.checkpoint(&index)?;
//!
//! // Recovery after crash - replays WAL automatically
//! let index = storage.recover::<HNSWIndex>()?;
//! ```

#![cfg(not(target_arch = "wasm32"))]

use crate::storage::compression::{self, Codec};
use crate::{Document, RagError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for incremental storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Number of WAL operations before triggering automatic checkpoint
    pub checkpoint_threshold: usize,
    /// Number of WAL operations before syncing to disk
    pub wal_sync_interval: usize,
    /// Maximum WAL file size in bytes before forcing checkpoint
    pub max_wal_size: usize,
    /// Compression codec for checkpoints
    pub checkpoint_codec: Codec,
    /// Whether to fsync after each WAL write (slower but safer)
    pub sync_on_write: bool,
    /// Keep old checkpoints for rollback (0 = delete immediately)
    pub keep_checkpoints: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            checkpoint_threshold: 10_000,
            wal_sync_interval: 100,
            max_wal_size: 100 * 1024 * 1024, // 100 MB
            checkpoint_codec: Codec::Gzip,
            sync_on_write: false,
            keep_checkpoints: 2,
        }
    }
}

impl IncrementalConfig {
    /// Set checkpoint threshold
    pub fn with_checkpoint_threshold(mut self, threshold: usize) -> Self {
        self.checkpoint_threshold = threshold;
        self
    }

    /// Set WAL sync interval
    pub fn with_wal_sync_interval(mut self, interval: usize) -> Self {
        self.wal_sync_interval = interval;
        self
    }

    /// Set maximum WAL size
    pub fn with_max_wal_size(mut self, size: usize) -> Self {
        self.max_wal_size = size;
        self
    }

    /// Set checkpoint compression codec
    pub fn with_checkpoint_codec(mut self, codec: Codec) -> Self {
        self.checkpoint_codec = codec;
        self
    }

    /// Enable sync on every write (safer but slower)
    pub fn with_sync_on_write(mut self, sync: bool) -> Self {
        self.sync_on_write = sync;
        self
    }

    /// Set number of old checkpoints to keep
    pub fn with_keep_checkpoints(mut self, count: usize) -> Self {
        self.keep_checkpoints = count;
        self
    }
}

// ============================================================================
// WAL Entry Types
// ============================================================================

/// Operations that can be logged to WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Add a document
    Add(Document),
    /// Remove a document by ID
    Remove(String),
    /// Clear all documents
    Clear,
    /// Marker for checkpoint completion
    Checkpoint { checkpoint_id: u64 },
}

/// A single WAL entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonically increasing sequence number
    pub seq: u64,
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// The operation
    pub operation: WalOperation,
    /// CRC32 checksum for integrity
    pub checksum: u32,
}

impl WalEntry {
    /// Create a new WAL entry
    fn new(seq: u64, operation: WalOperation) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let mut entry = Self {
            seq,
            timestamp,
            operation,
            checksum: 0,
        };

        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32 checksum
    fn compute_checksum(&self) -> u32 {
        let data = bincode::serialize(&(&self.seq, &self.timestamp, &self.operation)).unwrap();
        crc32fast::hash(&data)
    }

    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        let expected = {
            let mut copy = self.clone();
            copy.checksum = 0;
            copy.compute_checksum()
        };
        self.checksum == expected
    }
}

// ============================================================================
// Manifest
// ============================================================================

/// Manifest tracking current storage state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Current checkpoint ID
    pub current_checkpoint: Option<u64>,
    /// Current WAL sequence number
    pub wal_seq: u64,
    /// Number of operations since last checkpoint
    pub ops_since_checkpoint: usize,
    /// Total documents in index
    pub total_documents: usize,
    /// Index embedding dimension
    pub embedding_dim: usize,
    /// Index type ("hnsw", "flat", "sq8_hnsw", "binary_hnsw")
    pub index_type: String,
    /// Last modified timestamp
    pub last_modified: u64,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            current_checkpoint: None,
            wal_seq: 0,
            ops_since_checkpoint: 0,
            total_documents: 0,
            embedding_dim: 0,
            index_type: String::new(),
            last_modified: 0,
        }
    }
}

// ============================================================================
// Checkpoint Metadata
// ============================================================================

/// Metadata for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Checkpoint ID
    pub id: u64,
    /// WAL sequence at checkpoint time
    pub wal_seq: u64,
    /// Number of documents in checkpoint
    pub document_count: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Index type
    pub index_type: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Uncompressed size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression codec used
    pub codec: Codec,
}

// ============================================================================
// WAL Writer
// ============================================================================

/// Write-Ahead Log writer
struct WalWriter {
    file: BufWriter<File>,
    #[allow(dead_code)]
    path: PathBuf, // Kept for future: WAL rotation, recovery logging
    current_size: usize,
    sync_on_write: bool,
}

impl WalWriter {
    /// Open or create WAL file
    fn open(path: &Path, sync_on_write: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| RagError::StorageError(format!("Failed to open WAL: {}", e)))?;

        let current_size = file.metadata()
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            current_size,
            sync_on_write,
        })
    }

    /// Append entry to WAL
    fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let data = bincode::serialize(entry)?;
        let len = data.len() as u32;

        // Write length prefix + data
        self.file.write_all(&len.to_le_bytes())
            .map_err(|e| RagError::StorageError(format!("WAL write failed: {}", e)))?;
        self.file.write_all(&data)
            .map_err(|e| RagError::StorageError(format!("WAL write failed: {}", e)))?;

        self.current_size += 4 + data.len();

        if self.sync_on_write {
            self.sync()?;
        }

        Ok(())
    }

    /// Sync WAL to disk
    fn sync(&mut self) -> Result<()> {
        self.file.flush()
            .map_err(|e| RagError::StorageError(format!("WAL sync failed: {}", e)))?;
        self.file.get_ref().sync_all()
            .map_err(|e| RagError::StorageError(format!("WAL sync failed: {}", e)))?;
        Ok(())
    }

    /// Get current WAL size
    fn size(&self) -> usize {
        self.current_size
    }
}

// ============================================================================
// WAL Reader
// ============================================================================

/// Read entries from WAL
struct WalReader {
    file: BufReader<File>,
}

impl WalReader {
    /// Open WAL for reading
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| RagError::StorageError(format!("Failed to open WAL: {}", e)))?;
        Ok(Self {
            file: BufReader::new(file),
        })
    }

    /// Read all entries from a specific sequence number
    fn read_from(&mut self, from_seq: u64) -> Result<Vec<WalEntry>> {
        let mut entries = Vec::new();
        let mut len_buf = [0u8; 4];

        loop {
            // Read length prefix
            match self.file.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(RagError::StorageError(format!("WAL read failed: {}", e))),
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            self.file.read_exact(&mut data)
                .map_err(|e| RagError::StorageError(format!("WAL read failed: {}", e)))?;

            let entry: WalEntry = bincode::deserialize(&data)?;

            // Verify integrity
            if !entry.verify() {
                return Err(RagError::StorageError(format!(
                    "WAL entry {} failed integrity check",
                    entry.seq
                )));
            }

            // Only include entries after the requested sequence
            if entry.seq > from_seq {
                entries.push(entry);
            }
        }

        Ok(entries)
    }
}

// ============================================================================
// Incremental Storage
// ============================================================================

/// Incremental storage manager with WAL and checkpointing
///
/// Provides efficient incremental persistence with fast recovery.
pub struct IncrementalStorage {
    base_path: PathBuf,
    config: IncrementalConfig,
    manifest: Manifest,
    wal_writer: Option<WalWriter>,
    ops_since_sync: usize,
}

impl IncrementalStorage {
    /// Create or open incremental storage
    pub fn new<P: AsRef<Path>>(base_path: P, config: IncrementalConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create directory if needed
        fs::create_dir_all(&base_path)
            .map_err(|e| RagError::StorageError(format!("Failed to create storage dir: {}", e)))?;

        // Load or create manifest
        let manifest_path = base_path.join("manifest.json");
        let manifest = if manifest_path.exists() {
            let data = fs::read_to_string(&manifest_path)
                .map_err(|e| RagError::StorageError(format!("Failed to read manifest: {}", e)))?;
            serde_json::from_str(&data)
                .map_err(|e| RagError::StorageError(format!("Failed to parse manifest: {}", e)))?
        } else {
            Manifest::default()
        };

        // Open WAL writer
        let wal_path = base_path.join(format!("wal_{:05}.log", manifest.current_checkpoint.unwrap_or(0)));
        let wal_writer = WalWriter::open(&wal_path, config.sync_on_write)?;

        Ok(Self {
            base_path,
            config,
            manifest,
            wal_writer: Some(wal_writer),
            ops_since_sync: 0,
        })
    }

    /// Log an add operation to WAL
    pub fn log_add(&mut self, doc: &Document) -> Result<()> {
        self.log_operation(WalOperation::Add(doc.clone()))
    }

    /// Log a remove operation to WAL
    pub fn log_remove(&mut self, id: &str) -> Result<()> {
        self.log_operation(WalOperation::Remove(id.to_string()))
    }

    /// Log a clear operation to WAL
    pub fn log_clear(&mut self) -> Result<()> {
        self.log_operation(WalOperation::Clear)
    }

    /// Log an operation to WAL
    fn log_operation(&mut self, operation: WalOperation) -> Result<()> {
        self.manifest.wal_seq += 1;
        self.manifest.ops_since_checkpoint += 1;

        let entry = WalEntry::new(self.manifest.wal_seq, operation);

        if let Some(ref mut writer) = self.wal_writer {
            writer.append(&entry)?;
            self.ops_since_sync += 1;

            // Periodic sync
            if self.ops_since_sync >= self.config.wal_sync_interval {
                writer.sync()?;
                self.ops_since_sync = 0;
            }
        }

        // Update manifest
        self.manifest.last_modified = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(())
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.manifest.ops_since_checkpoint >= self.config.checkpoint_threshold
            || self.wal_writer.as_ref().map(|w| w.size()).unwrap_or(0) >= self.config.max_wal_size
    }

    /// Create a checkpoint from serializable index data
    pub fn checkpoint<T: Serialize>(&mut self, index: &T, meta: IndexMetadata) -> Result<CheckpointMeta> {
        // Sync WAL first
        if let Some(ref mut writer) = self.wal_writer {
            writer.sync()?;
        }

        let checkpoint_id = self.manifest.current_checkpoint.map(|c| c + 1).unwrap_or(1);

        // Serialize index
        let data = bincode::serialize(index)?;
        let original_size = data.len();

        // Compress
        let (compressed, _stats) = compression::compress_with(&data, self.config.checkpoint_codec)?;
        let compressed_size = compressed.len();

        // Write checkpoint file
        let checkpoint_path = self.base_path.join(format!("checkpoint_{:05}.bin", checkpoint_id));
        fs::write(&checkpoint_path, &compressed)
            .map_err(|e| RagError::StorageError(format!("Failed to write checkpoint: {}", e)))?;

        // Create checkpoint metadata
        let checkpoint_meta = CheckpointMeta {
            id: checkpoint_id,
            wal_seq: self.manifest.wal_seq,
            document_count: meta.document_count,
            embedding_dim: meta.embedding_dim,
            index_type: meta.index_type.clone(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            original_size,
            compressed_size,
            codec: self.config.checkpoint_codec,
        };

        // Write checkpoint metadata
        let meta_path = self.base_path.join(format!("checkpoint_{:05}.meta", checkpoint_id));
        let meta_json = serde_json::to_string_pretty(&checkpoint_meta)
            .map_err(|e| RagError::StorageError(format!("Failed to serialize meta: {}", e)))?;
        fs::write(&meta_path, &meta_json)
            .map_err(|e| RagError::StorageError(format!("Failed to write checkpoint meta: {}", e)))?;

        // Log checkpoint marker to WAL
        self.manifest.wal_seq += 1;
        let entry = WalEntry::new(
            self.manifest.wal_seq,
            WalOperation::Checkpoint { checkpoint_id },
        );
        if let Some(ref mut writer) = self.wal_writer {
            writer.append(&entry)?;
            writer.sync()?;
        }

        // Update manifest
        self.manifest.current_checkpoint = Some(checkpoint_id);
        self.manifest.ops_since_checkpoint = 0;
        self.manifest.total_documents = meta.document_count;
        self.manifest.embedding_dim = meta.embedding_dim;
        self.manifest.index_type = meta.index_type;
        self.save_manifest()?;

        // Rotate WAL
        self.rotate_wal(checkpoint_id)?;

        // Clean old checkpoints
        self.cleanup_old_checkpoints(checkpoint_id)?;

        Ok(checkpoint_meta)
    }

    /// Load checkpoint and return deserialized data
    pub fn load_checkpoint<T: for<'de> Deserialize<'de>>(&self) -> Result<Option<(T, CheckpointMeta)>> {
        let checkpoint_id = match self.manifest.current_checkpoint {
            Some(id) => id,
            None => return Ok(None),
        };

        // Load metadata
        let meta_path = self.base_path.join(format!("checkpoint_{:05}.meta", checkpoint_id));
        let meta_json = fs::read_to_string(&meta_path)
            .map_err(|e| RagError::StorageError(format!("Failed to read checkpoint meta: {}", e)))?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_json)
            .map_err(|e| RagError::StorageError(format!("Failed to parse checkpoint meta: {}", e)))?;

        // Load and decompress checkpoint
        let checkpoint_path = self.base_path.join(format!("checkpoint_{:05}.bin", checkpoint_id));
        let compressed = fs::read(&checkpoint_path)
            .map_err(|e| RagError::StorageError(format!("Failed to read checkpoint: {}", e)))?;
        let data = compression::decompress(&compressed)?;

        // Deserialize
        let index: T = bincode::deserialize(&data)?;

        Ok(Some((index, meta)))
    }

    /// Get WAL entries since last checkpoint
    pub fn get_wal_entries(&self) -> Result<Vec<WalEntry>> {
        let checkpoint_seq = if let Some(cp_id) = self.manifest.current_checkpoint {
            // Find the checkpoint marker seq
            let meta_path = self.base_path.join(format!("checkpoint_{:05}.meta", cp_id));
            if meta_path.exists() {
                let meta_json = fs::read_to_string(&meta_path)
                    .map_err(|e| RagError::StorageError(format!("Failed to read meta: {}", e)))?;
                let meta: CheckpointMeta = serde_json::from_str(&meta_json)
                    .map_err(|e| RagError::StorageError(format!("Failed to parse meta: {}", e)))?;
                meta.wal_seq
            } else {
                0
            }
        } else {
            0
        };

        let wal_path = self.base_path.join(format!(
            "wal_{:05}.log",
            self.manifest.current_checkpoint.unwrap_or(0)
        ));

        if !wal_path.exists() {
            return Ok(Vec::new());
        }

        let mut reader = WalReader::open(&wal_path)?;
        reader.read_from(checkpoint_seq)
    }

    /// Get current manifest state
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        let wal_size = self.wal_writer.as_ref().map(|w| w.size()).unwrap_or(0);

        let checkpoint_size = self.manifest.current_checkpoint
            .map(|id| {
                let path = self.base_path.join(format!("checkpoint_{:05}.bin", id));
                fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0)
            })
            .unwrap_or(0);

        StorageStats {
            checkpoint_id: self.manifest.current_checkpoint,
            wal_size,
            checkpoint_size,
            total_size: wal_size + checkpoint_size,
            ops_since_checkpoint: self.manifest.ops_since_checkpoint,
            total_documents: self.manifest.total_documents,
        }
    }

    /// Force sync WAL to disk
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.wal_writer {
            writer.sync()?;
        }
        self.save_manifest()?;
        Ok(())
    }

    fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.base_path.join("manifest.json");
        let json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| RagError::StorageError(format!("Failed to serialize manifest: {}", e)))?;
        fs::write(&manifest_path, &json)
            .map_err(|e| RagError::StorageError(format!("Failed to write manifest: {}", e)))?;
        Ok(())
    }

    fn rotate_wal(&mut self, checkpoint_id: u64) -> Result<()> {
        // Close current WAL
        if let Some(ref mut writer) = self.wal_writer {
            writer.sync()?;
        }

        // Delete old WAL if exists
        let old_wal = self.base_path.join(format!(
            "wal_{:05}.log",
            checkpoint_id.saturating_sub(1)
        ));
        if old_wal.exists() {
            let _ = fs::remove_file(&old_wal);
        }

        // Open new WAL
        let new_wal_path = self.base_path.join(format!("wal_{:05}.log", checkpoint_id));
        self.wal_writer = Some(WalWriter::open(&new_wal_path, self.config.sync_on_write)?);

        Ok(())
    }

    fn cleanup_old_checkpoints(&self, current_id: u64) -> Result<()> {
        if self.config.keep_checkpoints == 0 {
            return Ok(());
        }

        let cutoff = current_id.saturating_sub(self.config.keep_checkpoints as u64);

        for entry in fs::read_dir(&self.base_path)
            .map_err(|e| RagError::StorageError(format!("Failed to read dir: {}", e)))?
        {
            let entry = entry.map_err(|e| RagError::StorageError(format!("Dir entry error: {}", e)))?;
            let name = entry.file_name().to_string_lossy().to_string();

            if name.starts_with("checkpoint_") {
                // Extract checkpoint ID
                if let Some(id_str) = name.strip_prefix("checkpoint_").and_then(|s| s.split('.').next()) {
                    if let Ok(id) = id_str.parse::<u64>() {
                        if id < cutoff {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Metadata about the index for checkpointing
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    pub document_count: usize,
    pub embedding_dim: usize,
    pub index_type: String,
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Current checkpoint ID
    pub checkpoint_id: Option<u64>,
    /// WAL file size in bytes
    pub wal_size: usize,
    /// Checkpoint file size in bytes
    pub checkpoint_size: usize,
    /// Total storage size
    pub total_size: usize,
    /// Operations since last checkpoint
    pub ops_since_checkpoint: usize,
    /// Total documents in index
    pub total_documents: usize,
}

// ============================================================================
// Recovery Helper
// ============================================================================

/// Helper for recovering an index from storage
pub struct RecoveryHelper<'a> {
    storage: &'a IncrementalStorage,
}

impl<'a> RecoveryHelper<'a> {
    pub fn new(storage: &'a IncrementalStorage) -> Self {
        Self { storage }
    }

    /// Replay WAL entries on an index
    pub fn replay_wal<F>(&self, mut apply_op: F) -> Result<usize>
    where
        F: FnMut(&WalOperation) -> Result<()>,
    {
        let entries = self.storage.get_wal_entries()?;
        let count = entries.len();

        for entry in entries {
            match &entry.operation {
                WalOperation::Checkpoint { .. } => {
                    // Skip checkpoint markers
                    continue;
                }
                op => apply_op(op)?,
            }
        }

        Ok(count)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_document(id: &str, dim: usize) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding: vec![0.1; dim],
            metadata: None,
        }
    }

    #[test]
    fn test_config_builder() {
        let config = IncrementalConfig::default()
            .with_checkpoint_threshold(5000)
            .with_wal_sync_interval(50)
            .with_max_wal_size(50 * 1024 * 1024)
            .with_sync_on_write(true)
            .with_keep_checkpoints(3);

        assert_eq!(config.checkpoint_threshold, 5000);
        assert_eq!(config.wal_sync_interval, 50);
        assert_eq!(config.max_wal_size, 50 * 1024 * 1024);
        assert!(config.sync_on_write);
        assert_eq!(config.keep_checkpoints, 3);
    }

    #[test]
    fn test_wal_entry_integrity() {
        let entry = WalEntry::new(1, WalOperation::Add(create_test_document("doc1", 128)));
        assert!(entry.verify());

        // Tamper with entry
        let mut tampered = entry.clone();
        tampered.seq = 999;
        assert!(!tampered.verify());
    }

    #[test]
    fn test_storage_creation() {
        let dir = TempDir::new().unwrap();
        let storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        assert!(storage.manifest().current_checkpoint.is_none());
        assert_eq!(storage.manifest().wal_seq, 0);
    }

    #[test]
    fn test_wal_logging() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        // Log some operations
        storage.log_add(&create_test_document("doc1", 128)).unwrap();
        storage.log_add(&create_test_document("doc2", 128)).unwrap();
        storage.log_remove("doc1").unwrap();

        assert_eq!(storage.manifest().wal_seq, 3);
        assert_eq!(storage.manifest().ops_since_checkpoint, 3);

        // Force sync
        storage.sync().unwrap();

        // Read back WAL
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 3);

        match &entries[0].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc1"),
            _ => panic!("Expected Add operation"),
        }

        match &entries[2].operation {
            WalOperation::Remove(id) => assert_eq!(id, "doc1"),
            _ => panic!("Expected Remove operation"),
        }
    }

    #[test]
    fn test_checkpoint_and_recovery() {
        let dir = TempDir::new().unwrap();

        // Create storage and log some operations
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_checkpoint_threshold(100),
        ).unwrap();

        // Simulate index data (use String for serialization)
        let test_data: Vec<String> = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        for id in &test_data {
            storage.log_add(&create_test_document(id, 128)).unwrap();
        }

        // Create checkpoint
        let meta = storage.checkpoint(
            &test_data,
            IndexMetadata {
                document_count: 3,
                embedding_dim: 128,
                index_type: "test".to_string(),
            },
        ).unwrap();

        assert_eq!(meta.id, 1);
        assert_eq!(meta.document_count, 3);

        // Log more operations after checkpoint
        storage.log_add(&create_test_document("doc4", 128)).unwrap();
        storage.sync().unwrap();

        // Verify we can load checkpoint
        let (loaded_data, loaded_meta): (Vec<String>, CheckpointMeta) =
            storage.load_checkpoint().unwrap().unwrap();
        assert_eq!(loaded_data, test_data);
        assert_eq!(loaded_meta.id, 1);

        // Verify WAL has the post-checkpoint entry
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc4"),
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_needs_checkpoint() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_checkpoint_threshold(5),
        ).unwrap();

        for i in 0..4 {
            storage.log_add(&create_test_document(&format!("doc{}", i), 128)).unwrap();
        }
        assert!(!storage.needs_checkpoint());

        storage.log_add(&create_test_document("doc5", 128)).unwrap();
        assert!(storage.needs_checkpoint());
    }

    #[test]
    fn test_storage_stats() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        for i in 0..10 {
            storage.log_add(&create_test_document(&format!("doc{}", i), 128)).unwrap();
        }
        storage.sync().unwrap();

        let stats = storage.stats();
        assert!(stats.wal_size > 0);
        assert_eq!(stats.ops_since_checkpoint, 10);
        assert!(stats.checkpoint_id.is_none());
    }

    #[test]
    fn test_recovery_helper() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        // Log operations
        storage.log_add(&create_test_document("doc1", 128)).unwrap();
        storage.log_add(&create_test_document("doc2", 128)).unwrap();
        storage.log_remove("doc1").unwrap();
        storage.sync().unwrap();

        // Use recovery helper
        let helper = RecoveryHelper::new(&storage);
        let mut adds = 0;
        let mut removes = 0;

        helper.replay_wal(|op| {
            match op {
                WalOperation::Add(_) => adds += 1,
                WalOperation::Remove(_) => removes += 1,
                _ => {}
            }
            Ok(())
        }).unwrap();

        assert_eq!(adds, 2);
        assert_eq!(removes, 1);
    }

    #[test]
    fn test_persistence_across_reopens() {
        let dir = TempDir::new().unwrap();

        // First session
        {
            let mut storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
            storage.log_add(&create_test_document("doc1", 128)).unwrap();
            storage.log_add(&create_test_document("doc2", 128)).unwrap();
            storage.sync().unwrap();
        }

        // Reopen
        {
            let storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
            assert_eq!(storage.manifest().wal_seq, 2);

            let entries = storage.get_wal_entries().unwrap();
            assert_eq!(entries.len(), 2);
        }
    }
}
