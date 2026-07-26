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
//! ```no_run
//! use foxstash_core::storage::incremental::{
//!     IncrementalStorage, IncrementalConfig, IndexMetadata, RecoveryHelper,
//! };
//! use foxstash_core::storage::incremental::WalOperation;
//! use foxstash_core::index::HNSWIndex;
//! use foxstash_core::Document;
//!
//! fn main() -> Result<(), foxstash_core::RagError> {
//!     // Create incremental storage
//!     let config = IncrementalConfig::default()
//!         .with_wal_sync_interval(100)
//!         .with_checkpoint_threshold(10_000);
//!
//!     let mut storage = IncrementalStorage::new("/tmp/index_storage", config)?;
//!     let mut index = HNSWIndex::with_defaults(128);
//!
//!     // Add documents -- log to WAL, then apply to index
//!     let doc = Document {
//!         id: "doc1".into(),
//!         content: "Hello".into(),
//!         embedding: vec![0.1; 128],
//!         metadata: None,
//!     };
//!     storage.log_add(&doc)?;
//!     index.add(doc)?;
//!
//!     // Checkpoint when threshold is reached
//!     if storage.needs_checkpoint() {
//!         let meta = IndexMetadata {
//!             document_count: index.len(),
//!             embedding_dim: 128,
//!             index_type: "hnsw".into(),
//!         };
//!         storage.checkpoint(&index.get_all_documents(), meta)?;
//!     }
//!
//!     // Recovery: load last checkpoint, then replay WAL
//!     let helper = RecoveryHelper::new(&storage);
//!     helper.replay_wal(|op| {
//!         match op {
//!             WalOperation::Add(doc) => { index.add(doc.clone())?; }
//!             WalOperation::Clear => { index.clear(); }
//!             _ => {}
//!         }
//!         Ok(())
//!     })?;
//!     Ok(())
//! }
//! ```

#![cfg(not(target_arch = "wasm32"))]

use crate::storage::compression::{self, Codec};
use crate::{Document, RagError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static ATOMIC_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

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
        let data = serde_json::to_vec(&(&self.seq, &self.timestamp, &self.operation)).unwrap();
        crc32fast::hash(&data)
    }

    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        // compute_checksum() serializes (seq, timestamp, operation) — the checksum
        // field is already excluded from the hash input, so no clone needed.
        self.checksum == self.compute_checksum()
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
    /// On-disk layout version for the checkpoint payload.
    ///
    /// Unlike an index snapshot — a same-version *cache* that can always be
    /// rebuilt — a checkpoint is the durable copy of the user's documents. It
    /// cannot be regenerated from anything, so a shape change that
    /// deserializes into plausible-but-wrong data is unrecoverable.
    ///
    /// Checkpoints written before this field existed have no `format_version`
    /// key at all; `serde(default)` gives them 0, which
    /// [`CheckpointMeta::check_compatible`] reports as *older than versioning*
    /// rather than as corruption.
    #[serde(default)]
    pub format_version: u32,
    /// Crate version that wrote this checkpoint. Recorded for diagnostics, and
    /// deliberately **not** part of the compatibility test: the payload is
    /// self-describing JSON, so an unchanged layout stays readable across
    /// releases. Only [`Self::format_version`] gates loading.
    #[serde(default)]
    pub crate_version: String,
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

/// Current checkpoint layout version.
///
/// Bump this whenever the serialized shape of a checkpoint payload changes in a
/// way an older or newer reader would misinterpret. Adding an optional field
/// with a `serde` default does not qualify — JSON tolerates that. Removing a
/// field, changing a type, or changing what a field *means* does.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

impl CheckpointMeta {
    /// Reject a checkpoint this build cannot read correctly.
    ///
    /// The payload is JSON, so it is self-describing and survives most additive
    /// change on its own — this guards the cases JSON cannot catch, where the
    /// bytes still parse but no longer mean what they did.
    pub fn check_compatible(&self) -> Result<()> {
        if self.format_version == CHECKPOINT_FORMAT_VERSION {
            return Ok(());
        }
        if self.format_version == 0 {
            // Readable, and migrated in place on load — see `load_checkpoint`.
            //
            // v0 means "written before the meta carried a version". The version was
            // added to the METADATA, not to the payload, so a v0 payload has the
            // same shape a v1 reader expects. And the payload is self-describing
            // JSON: if an older build wrote a different `Document` shape,
            // deserialization fails loudly rather than misreading. So the honest
            // move is to attempt it and let the parse be the check — refusing
            // outright would have made every existing collection unopenable to buy
            // safety JSON already provides.
            return Ok(());
        }
        Err(RagError::StorageError(format!(
            "checkpoint {} is format v{}, this build reads v{} (foxstash {}, written by {}). \
             A checkpoint is durable user data, not a rebuildable cache — refusing rather \
             than risking a silent misread.",
            self.id,
            self.format_version,
            CHECKPOINT_FORMAT_VERSION,
            env!("CARGO_PKG_VERSION"),
            if self.crate_version.is_empty() {
                "unknown"
            } else {
                &self.crate_version
            },
        )))
    }
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

        let current_size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);

        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            current_size,
            sync_on_write,
        })
    }

    /// Append entry to WAL
    fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let data = serde_json::to_vec(entry)
            .map_err(|e| RagError::StorageError(format!("WAL serialize failed: {}", e)))?;
        let len = data.len() as u32;

        // Write length prefix + data
        self.file
            .write_all(&len.to_le_bytes())
            .map_err(|e| RagError::StorageError(format!("WAL write failed: {}", e)))?;
        self.file
            .write_all(&data)
            .map_err(|e| RagError::StorageError(format!("WAL write failed: {}", e)))?;

        self.current_size += 4 + data.len();

        if self.sync_on_write {
            self.sync()?;
        }

        Ok(())
    }

    /// Sync WAL to disk
    fn sync(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|e| RagError::StorageError(format!("WAL sync failed: {}", e)))?;
        self.file
            .get_ref()
            .sync_all()
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
            match self.file.read_exact(&mut data) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(RagError::StorageError(format!("WAL read failed: {}", e))),
            }

            let entry: WalEntry = serde_json::from_slice(&data)
                .map_err(|e| RagError::StorageError(format!("WAL deserialize failed: {}", e)))?;

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
        let wal_path = base_path.join(format!(
            "wal_{:05}.log",
            manifest.current_checkpoint.unwrap_or(0)
        ));
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
    pub fn checkpoint<T: Serialize>(
        &mut self,
        index: &T,
        meta: IndexMetadata,
    ) -> Result<CheckpointMeta> {
        // Sync WAL first
        if let Some(ref mut writer) = self.wal_writer {
            writer.sync()?;
        }

        let checkpoint_id = self.manifest.current_checkpoint.map(|c| c + 1).unwrap_or(1);

        // Serialize index
        let data = serde_json::to_vec(index)
            .map_err(|e| RagError::StorageError(format!("Checkpoint serialize failed: {}", e)))?;
        let original_size = data.len();

        // Compress
        let (compressed, _stats) = compression::compress_with(&data, self.config.checkpoint_codec)?;
        let compressed_size = compressed.len();

        // Write checkpoint file
        let checkpoint_path = self
            .base_path
            .join(format!("checkpoint_{:05}.bin", checkpoint_id));
        Self::write_atomic_file(&checkpoint_path, &compressed)
            .map_err(|e| RagError::StorageError(format!("Failed to write checkpoint: {}", e)))?;

        // Create checkpoint metadata
        let checkpoint_meta = CheckpointMeta {
            format_version: CHECKPOINT_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
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
        let meta_path = self
            .base_path
            .join(format!("checkpoint_{:05}.meta", checkpoint_id));
        let meta_json = serde_json::to_string_pretty(&checkpoint_meta)
            .map_err(|e| RagError::StorageError(format!("Failed to serialize meta: {}", e)))?;
        Self::write_atomic_file(&meta_path, meta_json.as_bytes()).map_err(|e| {
            RagError::StorageError(format!("Failed to write checkpoint meta: {}", e))
        })?;

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
    pub fn load_checkpoint<T: for<'de> Deserialize<'de>>(
        &self,
    ) -> Result<Option<(T, CheckpointMeta)>> {
        let checkpoint_id = match self.manifest.current_checkpoint {
            Some(id) => id,
            None => return Ok(None),
        };

        // Load metadata
        let meta_path = self
            .base_path
            .join(format!("checkpoint_{:05}.meta", checkpoint_id));
        let meta_json = fs::read_to_string(&meta_path).map_err(|e| {
            RagError::StorageError(format!("Failed to read checkpoint meta: {}", e))
        })?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_json).map_err(|e| {
            RagError::StorageError(format!("Failed to parse checkpoint meta: {}", e))
        })?;
        // Check before reading the payload: an incompatible checkpoint should
        // cost nothing and never be partially applied.
        meta.check_compatible()?;

        // Load and decompress checkpoint
        let checkpoint_path = self
            .base_path
            .join(format!("checkpoint_{:05}.bin", checkpoint_id));
        let compressed = fs::read(&checkpoint_path)
            .map_err(|e| RagError::StorageError(format!("Failed to read checkpoint: {}", e)))?;
        let data = compression::decompress(&compressed)?;

        // Deserialize. For a v0 checkpoint this parse IS the compatibility check:
        // JSON is self-describing, so a payload written against a different
        // `Document` shape fails here instead of being misread.
        let index: T = serde_json::from_slice(&data).map_err(|e| {
            if meta.format_version == 0 {
                RagError::StorageError(format!(
                    "checkpoint {} predates checkpoint versioning and does not match this \
                     build's document shape ({}). The data is not corrupt — it was written \
                     by an incompatible version. Export it with the version that wrote it. \
                     Underlying error: {e}",
                    meta.id,
                    env!("CARGO_PKG_VERSION"),
                ))
            } else {
                RagError::StorageError(format!("Checkpoint deserialize failed: {}", e))
            }
        })?;

        // Migrate a legacy checkpoint's metadata in place, now that its payload has
        // been proven readable. Only the meta sidecar is rewritten; the payload is
        // untouched because nothing about it needed to change. Best-effort: failing
        // to stamp it costs a re-check next open, not correctness.
        let mut meta = meta;
        if meta.format_version == 0 {
            meta.format_version = CHECKPOINT_FORMAT_VERSION;
            meta.crate_version = env!("CARGO_PKG_VERSION").to_string();
            if let Ok(json) = serde_json::to_string_pretty(&meta) {
                let _ = Self::write_atomic_file(&meta_path, json.as_bytes());
            }
        }

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

        let checkpoint_size = self
            .manifest
            .current_checkpoint
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
        Self::write_atomic_file(&manifest_path, json.as_bytes())
            .map_err(|e| RagError::StorageError(format!("Failed to write manifest: {}", e)))?;
        Ok(())
    }

    fn atomic_tmp_path(path: &Path) -> PathBuf {
        let file_name = path.file_name().and_then(|f| f.to_str()).unwrap_or("file");
        let counter = ATOMIC_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
        path.with_file_name(format!(
            "{}.{}.{}.tmp",
            file_name,
            std::process::id(),
            counter
        ))
    }

    fn write_atomic_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
        let tmp_path = Self::atomic_tmp_path(path);
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(data)?;
            file.sync_all()?;
        }
        fs::rename(&tmp_path, path).inspect_err(|_| {
            let _ = fs::remove_file(&tmp_path);
        })?;
        Ok(())
    }

    fn rotate_wal(&mut self, checkpoint_id: u64) -> Result<()> {
        // Close current WAL
        if let Some(ref mut writer) = self.wal_writer {
            writer.sync()?;
        }

        // Open new WAL FIRST (before deleting old).
        // This ensures we always have a valid WAL even if deletion fails.
        let new_wal_path = self.base_path.join(format!("wal_{:05}.log", checkpoint_id));
        let new_writer = WalWriter::open(&new_wal_path, self.config.sync_on_write)?;
        self.wal_writer = Some(new_writer);

        // Now safe to delete old WAL.
        let old_wal = self
            .base_path
            .join(format!("wal_{:05}.log", checkpoint_id.saturating_sub(1)));
        if old_wal.exists() && old_wal != new_wal_path {
            let _ = fs::remove_file(&old_wal);
        }

        Ok(())
    }

    fn cleanup_old_checkpoints(&self, current_id: u64) -> Result<()> {
        let cutoff = if self.config.keep_checkpoints == 0 {
            current_id.saturating_sub(1)
        } else {
            current_id.saturating_sub(self.config.keep_checkpoints as u64)
        };

        for entry in fs::read_dir(&self.base_path)
            .map_err(|e| RagError::StorageError(format!("Failed to read dir: {}", e)))?
        {
            let entry =
                entry.map_err(|e| RagError::StorageError(format!("Dir entry error: {}", e)))?;
            let name = entry.file_name().to_string_lossy().to_string();

            if name.starts_with("checkpoint_") {
                // Extract checkpoint ID
                if let Some(id_str) = name
                    .strip_prefix("checkpoint_")
                    .and_then(|s| s.split('.').next())
                {
                    if let Ok(id) = id_str.parse::<u64>() {
                        if id <= cutoff {
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

    /// A checkpoint written by this build round-trips; one claiming a different
    /// format version is refused **before** its payload is read.
    ///
    /// The rejection matters more than the round-trip. A checkpoint is the only
    /// copy of the user's documents — unlike an index snapshot, nothing can
    /// regenerate it — so a shape change that parses into plausible-but-wrong
    /// data is unrecoverable. This asserts the guard fires rather than assuming it.
    #[test]
    fn checkpoint_meta_is_versioned_and_incompatible_versions_are_refused() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        let docs = vec![create_test_document("a", 4), create_test_document("b", 4)];
        let meta = storage
            .checkpoint(
                &docs,
                IndexMetadata {
                    document_count: docs.len(),
                    embedding_dim: 4,
                    index_type: "test".into(),
                },
            )
            .unwrap();

        assert_eq!(meta.format_version, CHECKPOINT_FORMAT_VERSION);
        assert_eq!(meta.crate_version, env!("CARGO_PKG_VERSION"));

        // Round-trips at the current version.
        let (loaded, _) = storage
            .load_checkpoint::<Vec<Document>>()
            .unwrap()
            .expect("checkpoint present");
        assert_eq!(loaded.len(), 2);

        // A future version must be refused, not read.
        let meta_path = dir.path().join(format!("checkpoint_{:05}.meta", meta.id));
        let bumped = std::fs::read_to_string(&meta_path).unwrap().replace(
            &format!("\"format_version\": {CHECKPOINT_FORMAT_VERSION}"),
            &format!("\"format_version\": {}", CHECKPOINT_FORMAT_VERSION + 1),
        );
        std::fs::write(&meta_path, bumped).unwrap();
        let err = storage
            .load_checkpoint::<Vec<Document>>()
            .expect_err("a newer format must be refused");
        assert!(
            format!("{err}").contains("format v"),
            "error should name the version mismatch: {err}"
        );

        // A pre-versioning checkpoint (no field at all) must LOAD and be migrated
        // in place — this is what every existing on-disk collection looks like, and
        // refusing them would make the version stamp a data-loss event.
        let meta_json = std::fs::read_to_string(&meta_path).unwrap();
        let legacy = meta_json
            .lines()
            .filter(|l| !l.contains("format_version") && !l.contains("crate_version"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&meta_path, legacy).unwrap();

        let (loaded, migrated) = storage
            .load_checkpoint::<Vec<Document>>()
            .expect("a legacy checkpoint must load, not be refused")
            .expect("checkpoint present");
        assert_eq!(loaded.len(), 2, "legacy payload must come back intact");
        assert_eq!(
            migrated.format_version, CHECKPOINT_FORMAT_VERSION,
            "the returned meta should report the migrated version"
        );

        // The migration is persisted, so it happens once rather than every open.
        let on_disk: CheckpointMeta =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        assert_eq!(
            on_disk.format_version, CHECKPOINT_FORMAT_VERSION,
            "the meta sidecar should have been stamped on disk"
        );
        assert_eq!(on_disk.crate_version, env!("CARGO_PKG_VERSION"));
    }

    /// A legacy checkpoint whose payload does NOT match this build's document
    /// shape must fail with a message that says so, not be silently misread.
    ///
    /// This is what makes accepting v0 safe: the JSON parse is the compatibility
    /// check, so "old" and "incompatible" stay distinguishable.
    #[test]
    fn a_legacy_checkpoint_with_an_incompatible_payload_fails_clearly() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let docs = vec![create_test_document("a", 4)];
        let meta = storage
            .checkpoint(
                &docs,
                IndexMetadata {
                    document_count: 1,
                    embedding_dim: 4,
                    index_type: "test".into(),
                },
            )
            .unwrap();

        // Strip the version (making it legacy) and read it back as a type the
        // payload cannot possibly satisfy.
        let meta_path = dir.path().join(format!("checkpoint_{:05}.meta", meta.id));
        let legacy = std::fs::read_to_string(&meta_path)
            .unwrap()
            .lines()
            .filter(|l| !l.contains("format_version"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&meta_path, legacy).unwrap();

        let err = storage
            .load_checkpoint::<Vec<u64>>()
            .expect_err("an incompatible legacy payload must fail");
        assert!(
            format!("{err}").contains("predates checkpoint versioning"),
            "the error should name the legacy case rather than a bare parse error: {err}"
        );
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
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

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
        )
        .unwrap();

        // Simulate index data (use String for serialization)
        let test_data: Vec<String> =
            vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        for id in &test_data {
            storage.log_add(&create_test_document(id, 128)).unwrap();
        }

        // Create checkpoint
        let meta = storage
            .checkpoint(
                &test_data,
                IndexMetadata {
                    document_count: 3,
                    embedding_dim: 128,
                    index_type: "test".to_string(),
                },
            )
            .unwrap();

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
        )
        .unwrap();

        for i in 0..4 {
            storage
                .log_add(&create_test_document(&format!("doc{}", i), 128))
                .unwrap();
        }
        assert!(!storage.needs_checkpoint());

        storage.log_add(&create_test_document("doc5", 128)).unwrap();
        assert!(storage.needs_checkpoint());
    }

    #[test]
    fn test_storage_stats() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        for i in 0..10 {
            storage
                .log_add(&create_test_document(&format!("doc{}", i), 128))
                .unwrap();
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
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        // Log operations
        storage.log_add(&create_test_document("doc1", 128)).unwrap();
        storage.log_add(&create_test_document("doc2", 128)).unwrap();
        storage.log_remove("doc1").unwrap();
        storage.sync().unwrap();

        // Use recovery helper
        let helper = RecoveryHelper::new(&storage);
        let mut adds = 0;
        let mut removes = 0;

        helper
            .replay_wal(|op| {
                match op {
                    WalOperation::Add(_) => adds += 1,
                    WalOperation::Remove(_) => removes += 1,
                    _ => {}
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(adds, 2);
        assert_eq!(removes, 1);
    }

    #[test]
    fn test_persistence_across_reopens() {
        let dir = TempDir::new().unwrap();

        // First session
        {
            let mut storage =
                IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
            storage.log_add(&create_test_document("doc1", 128)).unwrap();
            storage.log_add(&create_test_document("doc2", 128)).unwrap();
            storage.sync().unwrap();
        }

        // Reopen
        {
            let storage =
                IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
            assert_eq!(storage.manifest().wal_seq, 2);

            let entries = storage.get_wal_entries().unwrap();
            assert_eq!(entries.len(), 2);
        }
    }

    #[test]
    fn test_keep_checkpoints_zero_prunes_old_checkpoints() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_keep_checkpoints(0),
        )
        .unwrap();

        let data = vec!["doc".to_string()];
        for checkpoint_no in 0..3 {
            storage
                .checkpoint(
                    &data,
                    IndexMetadata {
                        document_count: 1,
                        embedding_dim: 128,
                        index_type: format!("test_{}", checkpoint_no),
                    },
                )
                .unwrap();
        }

        let checkpoint_bins: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("checkpoint_") && name.ends_with(".bin"))
            .collect();

        assert_eq!(
            checkpoint_bins.len(),
            1,
            "keep_checkpoints=0 should keep only current checkpoint"
        );
    }

    #[test]
    fn test_keep_checkpoints_exact_retention_count() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_keep_checkpoints(2),
        )
        .unwrap();

        let data = vec!["doc".to_string()];
        for checkpoint_no in 0..5 {
            storage
                .checkpoint(
                    &data,
                    IndexMetadata {
                        document_count: 1,
                        embedding_dim: 128,
                        index_type: format!("test_{}", checkpoint_no),
                    },
                )
                .unwrap();
        }

        let checkpoint_bins: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("checkpoint_") && name.ends_with(".bin"))
            .collect();

        assert_eq!(
            checkpoint_bins.len(),
            2,
            "retention should keep exactly keep_checkpoints checkpoint files"
        );
    }

    #[test]
    fn test_wal_roundtrip_with_metadata() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        let doc = Document {
            id: "meta-doc".to_string(),
            content: "has metadata".to_string(),
            embedding: vec![0.1; 4],
            metadata: Some(serde_json::json!({
                "scope": "workspace",
                "tags": ["rust", "ai"],
                "priority": 5
            })),
        };

        storage.log_add(&doc).unwrap();
        storage.sync().unwrap();

        // Read back and verify metadata survived the roundtrip.
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            WalOperation::Add(recovered) => {
                assert_eq!(recovered.id, "meta-doc");
                let meta = recovered.metadata.as_ref().unwrap();
                assert_eq!(meta["scope"], "workspace");
                assert_eq!(meta["priority"], 5);
                assert_eq!(meta["tags"][0], "rust");
            }
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_checkpoint_roundtrip_with_metadata() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        let docs = vec![
            Document {
                id: "d1".to_string(),
                content: "first".to_string(),
                embedding: vec![1.0, 0.0],
                metadata: Some(serde_json::json!({"lang": "rust"})),
            },
            Document {
                id: "d2".to_string(),
                content: "second".to_string(),
                embedding: vec![0.0, 1.0],
                metadata: None,
            },
        ];

        storage
            .checkpoint(
                &docs,
                IndexMetadata {
                    document_count: 2,
                    embedding_dim: 2,
                    index_type: "hnsw".to_string(),
                },
            )
            .unwrap();

        let (loaded, meta): (Vec<Document>, CheckpointMeta) =
            storage.load_checkpoint().unwrap().unwrap();
        assert_eq!(meta.document_count, 2);
        assert_eq!(loaded.len(), 2);

        assert_eq!(loaded[0].id, "d1");
        assert_eq!(loaded[0].metadata.as_ref().unwrap()["lang"], "rust");

        assert_eq!(loaded[1].id, "d2");
        assert!(loaded[1].metadata.is_none());
    }

    #[test]
    fn test_recovery_ignores_truncated_tail_entry() {
        let dir = TempDir::new().unwrap();
        {
            let mut storage =
                IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
            storage.log_add(&create_test_document("doc1", 4)).unwrap();
            storage.sync().unwrap();
        }

        let wal_path = dir.path().join("wal_00000.log");
        let torn_entry = WalEntry::new(2, WalOperation::Add(create_test_document("doc2", 4)));
        let torn_payload = serde_json::to_vec(&torn_entry).unwrap();
        let torn_len = torn_payload.len() as u32;

        let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        file.write_all(&torn_len.to_le_bytes()).unwrap();
        file.write_all(&torn_payload[..torn_payload.len() / 2])
            .unwrap();
        file.sync_all().unwrap();

        let storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc1"),
            _ => panic!("expected Add operation"),
        }
    }

    #[test]
    fn test_atomic_writes_leave_no_tmp_files_after_repeated_checkpoints() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();

        for i in 0..8 {
            let payload = vec![format!("doc-{i}")];
            storage
                .checkpoint(
                    &payload,
                    IndexMetadata {
                        document_count: 1,
                        embedding_dim: 128,
                        index_type: "hnsw".to_string(),
                    },
                )
                .unwrap();
        }

        let has_tmp = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .any(|name| name.ends_with(".tmp"));
        assert!(!has_tmp);
    }

    // ========================================================================
    // Discriminating tests for `IncrementalConfig` fields flagged VACUOUS in the public-option
    // audit: previously covered only by `test_config_builder`, which checks the field is
    // *stored*, never that it's *honored*. NOT COMPILED — the team lead will compile and
    // sabotage-verify these directly.
    // ========================================================================

    /// `checkpoint_codec` must select which codec compresses the checkpoint file, not just
    /// round-trip through the builder. `compression::create_header` writes the codec's wire id
    /// as the FIRST byte of the compressed output: `Codec::None` -> 0, `Codec::Gzip` -> 1 (see
    /// `crate::storage::compression`, `Codec::id`/`Codec::from_id`). Two checkpoints of
    /// identical data under the two codecs must therefore differ in that first byte.
    ///
    /// Sabotage this catches: hardcode `checkpoint()` to always call
    /// `compression::compress_with(&data, Codec::Gzip)` regardless of
    /// `self.config.checkpoint_codec` — the `Codec::None` config below would still produce a
    /// file starting with byte 1 (Gzip's id), not 0.
    #[test]
    fn checkpoint_codec_controls_the_compression_used() {
        let data: Vec<String> = vec!["doc1".into(), "doc2".into(), "doc3".into()];
        let meta = || IndexMetadata {
            document_count: 3,
            embedding_dim: 128,
            index_type: "test".to_string(),
        };

        let dir_none = TempDir::new().unwrap();
        let mut storage_none = IncrementalStorage::new(
            dir_none.path(),
            IncrementalConfig::default().with_checkpoint_codec(Codec::None),
        )
        .unwrap();
        storage_none.checkpoint(&data, meta()).unwrap();
        let none_bytes = fs::read(dir_none.path().join("checkpoint_00001.bin")).unwrap();

        let dir_gzip = TempDir::new().unwrap();
        let mut storage_gzip = IncrementalStorage::new(
            dir_gzip.path(),
            IncrementalConfig::default().with_checkpoint_codec(Codec::Gzip),
        )
        .unwrap();
        storage_gzip.checkpoint(&data, meta()).unwrap();
        let gzip_bytes = fs::read(dir_gzip.path().join("checkpoint_00001.bin")).unwrap();

        assert_eq!(
            none_bytes[0], 0,
            "Codec::None checkpoint should have header byte 0, got {}",
            none_bytes[0]
        );
        assert_eq!(
            gzip_bytes[0], 1,
            "Codec::Gzip checkpoint should have header byte 1, got {} — checkpoint_codec is \
             being ignored",
            gzip_bytes[0]
        );
    }

    /// `keep_checkpoints` must bound how many old checkpoint files survive pruning after each
    /// new checkpoint. Six checkpoints are taken back-to-back; `keep_checkpoints: N` should leave
    /// exactly `N` `checkpoint_*.bin` files afterwards.
    ///
    /// Sabotage this catches: skip `cleanup_old_checkpoints` entirely, or hardcode its cutoff —
    /// both configs below would then leave the same file count (6, if pruning never runs, or
    /// some other fixed number) regardless of `keep_checkpoints`.
    #[test]
    fn keep_checkpoints_bounds_surviving_checkpoint_files() {
        let count_checkpoint_files = |dir: &std::path::Path| -> usize {
            fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with("checkpoint_") && name.ends_with(".bin")
                })
                .count()
        };

        let run_with = |keep_checkpoints: usize| -> usize {
            let dir = TempDir::new().unwrap();
            let mut storage = IncrementalStorage::new(
                dir.path(),
                IncrementalConfig::default().with_keep_checkpoints(keep_checkpoints),
            )
            .unwrap();
            for i in 0..6 {
                storage
                    .checkpoint(
                        &vec![format!("doc{i}")],
                        IndexMetadata {
                            document_count: 1,
                            embedding_dim: 4,
                            index_type: "test".to_string(),
                        },
                    )
                    .unwrap();
            }
            count_checkpoint_files(dir.path())
        };

        let kept_2 = run_with(2);
        let kept_5 = run_with(5);

        assert_eq!(
            kept_2, 2,
            "keep_checkpoints: 2 should leave exactly 2 checkpoint files after 6 checkpoints, \
             found {kept_2}"
        );
        assert_eq!(
            kept_5, 5,
            "keep_checkpoints: 5 should leave exactly 5 checkpoint files after 6 checkpoints, \
             found {kept_5} — keep_checkpoints is being ignored"
        );
    }

    /// `wal_sync_interval` must control how many WAL ops accumulate in the in-process
    /// `BufWriter` before an fsync flushes them to disk. `WalWriter::append` only calls `sync()`
    /// (which calls `flush()`) once `ops_since_sync >= config.wal_sync_interval`; below that,
    /// entries sit in the `BufWriter`'s buffer and are invisible to an independent read of the
    /// WAL file. A fresh storage's WAL file is always named `wal_00000.log` (see
    /// `IncrementalStorage::new`).
    ///
    /// Sabotage this catches: hardcode the sync trigger to some fixed op count (or never sync
    /// except via `sync_on_write`/an explicit `.sync()`) instead of reading
    /// `self.config.wal_sync_interval` — the `wal_sync_interval: 2` config below would then also
    /// leave the WAL file empty after 2 ops, like the `wal_sync_interval: 1000` config does.
    #[test]
    fn wal_sync_interval_controls_when_the_wal_hits_disk() {
        let dir_frequent = TempDir::new().unwrap();
        let mut storage_frequent = IncrementalStorage::new(
            dir_frequent.path(),
            IncrementalConfig::default().with_wal_sync_interval(2),
        )
        .unwrap();
        storage_frequent
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        storage_frequent
            .log_add(&create_test_document("doc2", 4))
            .unwrap();
        // 2 ops with wal_sync_interval=2: the sync trigger has fired inside the 2nd `log_add`.
        let frequent_len = fs::metadata(dir_frequent.path().join("wal_00000.log"))
            .map(|m| m.len())
            .unwrap_or(0);

        let dir_rare = TempDir::new().unwrap();
        let mut storage_rare = IncrementalStorage::new(
            dir_rare.path(),
            IncrementalConfig::default().with_wal_sync_interval(1000),
        )
        .unwrap();
        storage_rare
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        storage_rare
            .log_add(&create_test_document("doc2", 4))
            .unwrap();
        // Same 2 ops, but wal_sync_interval=1000: nothing has crossed the threshold, so the
        // BufWriter has not flushed and the on-disk file should still be empty.
        let rare_len = fs::metadata(dir_rare.path().join("wal_00000.log"))
            .map(|m| m.len())
            .unwrap_or(0);

        assert_eq!(
            rare_len, 0,
            "wal_sync_interval: 1000 after only 2 ops should leave the WAL file un-flushed \
             (0 bytes on disk), found {rare_len} bytes"
        );
        assert!(
            frequent_len > 0,
            "wal_sync_interval: 2 after 2 ops should have triggered a flush, but the WAL file \
             on disk is still empty — wal_sync_interval is being ignored"
        );
    }

    /// `max_wal_size` must feed `needs_checkpoint()`'s size-based trigger, independent of the
    /// op-count trigger (`checkpoint_threshold`, held enormous here so it can never fire).
    ///
    /// Sabotage this catches: drop the `|| ... >= self.config.max_wal_size` half of
    /// `needs_checkpoint`'s condition (or hardcode the size threshold) — the tiny-max-size
    /// config below would then also report `false` after logging a document, the same as the
    /// enormous-max-size config.
    #[test]
    fn max_wal_size_feeds_needs_checkpoint() {
        let dir_tiny = TempDir::new().unwrap();
        let mut storage_tiny = IncrementalStorage::new(
            dir_tiny.path(),
            IncrementalConfig::default()
                .with_checkpoint_threshold(1_000_000)
                .with_max_wal_size(1),
        )
        .unwrap();
        storage_tiny
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        assert!(
            storage_tiny.needs_checkpoint(),
            "max_wal_size: 1 should make needs_checkpoint() true after a single WAL entry, but \
             it reported false — max_wal_size is being ignored"
        );

        let dir_huge = TempDir::new().unwrap();
        let mut storage_huge = IncrementalStorage::new(
            dir_huge.path(),
            IncrementalConfig::default()
                .with_checkpoint_threshold(1_000_000)
                .with_max_wal_size(1024 * 1024 * 1024),
        )
        .unwrap();
        storage_huge
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        assert!(
            !storage_huge.needs_checkpoint(),
            "max_wal_size: 1 GB should leave needs_checkpoint() false after a single tiny WAL \
             entry, but it reported true"
        );
    }

    /// `sync_on_write` must force a sync (flush + fsync) after EVERY write, independent of
    /// `wal_sync_interval` — held enormous here so the interval-based trigger can never fire on
    /// its own, isolating `sync_on_write`'s effect.
    ///
    /// Sabotage this catches: drop the `if self.sync_on_write { self.sync()?; }` call in
    /// `WalWriter::append` (or hardcode it to `false`) — the `sync_on_write: true` config below
    /// would then also leave the WAL file empty after a single write.
    #[test]
    fn sync_on_write_forces_a_sync_on_every_write() {
        let dir_on = TempDir::new().unwrap();
        let mut storage_on = IncrementalStorage::new(
            dir_on.path(),
            IncrementalConfig::default()
                .with_wal_sync_interval(1000)
                .with_sync_on_write(true),
        )
        .unwrap();
        storage_on
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        let on_len = fs::metadata(dir_on.path().join("wal_00000.log"))
            .map(|m| m.len())
            .unwrap_or(0);

        let dir_off = TempDir::new().unwrap();
        let mut storage_off = IncrementalStorage::new(
            dir_off.path(),
            IncrementalConfig::default()
                .with_wal_sync_interval(1000)
                .with_sync_on_write(false),
        )
        .unwrap();
        storage_off
            .log_add(&create_test_document("doc1", 4))
            .unwrap();
        let off_len = fs::metadata(dir_off.path().join("wal_00000.log"))
            .map(|m| m.len())
            .unwrap_or(0);

        assert_eq!(
            off_len, 0,
            "sync_on_write: false with wal_sync_interval: 1000 should leave the WAL file \
             un-flushed after one write, found {off_len} bytes"
        );
        assert!(
            on_len > 0,
            "sync_on_write: true should flush after every write regardless of \
             wal_sync_interval, but the WAL file on disk is still empty — sync_on_write is \
             being ignored"
        );
    }
}
