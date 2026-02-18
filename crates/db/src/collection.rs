//! Per-collection handle with concurrent read/write access.
//!
//! Thread safety:
//! - `inner` (index + id_map): `RwLock` — reads don't block each other.
//! - `storage`: separate `Mutex` — WAL writes don't block searches.

use crate::filter::Filter;
use crate::id_map::IdMap;
use crate::recovery;
use crate::{DbConfig, DbError, Result};
use foxstash_core::index::HNSWIndex;
use foxstash_core::storage::incremental::{IncrementalStorage, IndexMetadata};
use foxstash_core::{Document, SearchResult};
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use std::path::Path;
use tracing::debug;

/// Mutable state behind the read-write lock.
struct CollectionInner {
    index: HNSWIndex,
    id_map: IdMap,
    /// Flat document store for get-by-id and checkpoint serialization.
    /// Positions align with the id_map, but tombstoned entries may be stale.
    documents: Vec<Document>,
}

/// A named collection of documents with vector search.
pub struct Collection {
    name: String,
    config: DbConfig,
    inner: RwLock<CollectionInner>,
    storage: Mutex<IncrementalStorage>,
}

impl Collection {
    /// Open or create a collection at the given path.
    pub fn open(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        let storage = IncrementalStorage::new(path, config.storage.clone())
            .map_err(DbError::Core)?;

        let state = recovery::recover(&storage, &config)?;

        debug!(
            name,
            live = state.id_map.live_count(),
            tombstoned = state.id_map.tombstone_count(),
            "collection opened"
        );

        Ok(Self {
            name: name.to_string(),
            config,
            inner: RwLock::new(CollectionInner {
                index: state.index,
                id_map: state.id_map,
                documents: state.documents,
            }),
            storage: Mutex::new(storage),
        })
    }

    /// Create a fresh (empty) collection at the given path.
    pub fn create(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        let storage = IncrementalStorage::new(path, config.storage.clone())
            .map_err(DbError::Core)?;

        Ok(Self {
            name: name.to_string(),
            inner: RwLock::new(CollectionInner {
                index: HNSWIndex::new(config.embedding_dim, config.hnsw.clone()),
                id_map: IdMap::new(),
                documents: Vec::new(),
            }),
            config,
            storage: Mutex::new(storage),
        })
    }

    /// Insert a document. If the ID already exists, the old version is tombstoned.
    pub fn insert(
        &self,
        id: String,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dim {
            return Err(DbError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: embedding.len(),
            });
        }

        let doc = Document {
            id: id.clone(),
            content,
            embedding,
            metadata,
        };

        // WAL first (crash-safe).
        {
            let mut storage = self.storage.lock();
            storage.log_add(&doc).map_err(DbError::Core)?;
        }

        // Then mutate in-memory state.
        {
            let mut inner = self.inner.write();
            // Tombstone previous version if re-inserting same ID.
            if inner.id_map.get(&id).is_some() {
                inner.id_map.remove(&id);
            }
            inner.index.add(doc.clone()).map_err(DbError::Core)?;
            inner.id_map.insert(id);
            inner.documents.push(doc);
        }

        self.maybe_auto_checkpoint()?;
        Ok(())
    }

    /// Soft-delete a document by ID. Returns `true` if the document existed.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let existed = {
            let mut inner = self.inner.write();
            inner.id_map.remove(id)
        };

        if existed {
            let mut storage = self.storage.lock();
            storage.log_remove(id).map_err(DbError::Core)?;
        }

        Ok(existed)
    }

    /// Search for the `k` nearest neighbors, optionally filtered by metadata.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.embedding_dim {
            return Err(DbError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: query.len(),
            });
        }

        let inner = self.inner.read();

        if inner.index.is_empty() {
            return Ok(Vec::new());
        }

        match filter {
            None => self.search_unfiltered(&inner, query, k),
            Some(f) => self.search_filtered(&inner, query, k, f),
        }
    }

    /// Get a document by ID.
    pub fn get(&self, id: &str) -> Result<Option<Document>> {
        let inner = self.inner.read();
        let pos = match inner.id_map.get(id) {
            Some(p) => p,
            None => return Ok(None),
        };
        // Walk the documents vec to find the one at this position.
        // Since documents are append-only and positions are sequential,
        // the document at `pos` is `inner.documents[pos]`.
        Ok(inner.documents.get(pos).cloned())
    }

    /// Flush WAL to disk and save manifest.
    pub fn flush(&self) -> Result<()> {
        let mut storage = self.storage.lock();
        storage.sync().map_err(DbError::Core)?;
        Ok(())
    }

    /// Compact: rebuild index from live documents only, checkpoint, reclaim tombstones.
    pub fn compact(&self) -> Result<()> {
        let live_docs = {
            let inner = self.inner.read();
            self.collect_live_documents(&inner)
        };

        let doc_count = live_docs.len();

        // Rebuild index + id_map from scratch.
        let mut new_index = HNSWIndex::new(self.config.embedding_dim, self.config.hnsw.clone());
        let mut new_id_map = IdMap::new();

        for doc in &live_docs {
            new_index.add(doc.clone()).map_err(DbError::Core)?;
            new_id_map.insert(doc.id.clone());
        }

        // Checkpoint the compacted state.
        {
            let mut storage = self.storage.lock();
            storage
                .checkpoint(
                    &live_docs,
                    IndexMetadata {
                        document_count: doc_count,
                        embedding_dim: self.config.embedding_dim,
                        index_type: "hnsw".into(),
                    },
                )
                .map_err(DbError::Core)?;
        }

        // Swap in the new state.
        {
            let mut inner = self.inner.write();
            inner.index = new_index;
            inner.id_map = new_id_map;
            inner.documents = live_docs;
        }

        debug!(name = %self.name, doc_count, "compaction complete");
        Ok(())
    }

    /// Number of live (non-tombstoned) documents.
    pub fn len(&self) -> usize {
        self.inner.read().id_map.live_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// List all live (non-tombstoned) document IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.inner
            .read()
            .id_map
            .live_ids()
            .map(String::from)
            .collect()
    }

    /// Check whether a live (non-tombstoned) document with this ID exists.
    pub fn contains(&self, id: &str) -> bool {
        self.inner.read().id_map.get(id).is_some()
    }

    // ── private helpers ─────────────────────────────────────────────

    /// Unfiltered search: query HNSW, exclude tombstones.
    fn search_unfiltered(
        &self,
        inner: &CollectionInner,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Over-fetch to compensate for tombstones.
        let fetch = if inner.id_map.tombstone_count() > 0 {
            k + inner.id_map.tombstone_count()
        } else {
            k
        };
        let raw = inner.index.search(query, fetch).map_err(DbError::Core)?;

        let results: Vec<SearchResult> = raw
            .into_iter()
            .filter(|r| !inner.id_map.is_tombstoned(&r.id))
            .take(k)
            .collect();

        Ok(results)
    }

    /// Filtered search: progressive over-fetch (2x → 4x → 8x).
    fn search_filtered(
        &self,
        inner: &CollectionInner,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        for multiplier in [2, 4, 8] {
            let fetch = k * multiplier;
            let raw = inner.index.search(query, fetch).map_err(DbError::Core)?;

            let results: Vec<SearchResult> = raw
                .into_iter()
                .filter(|r| {
                    !inner.id_map.is_tombstoned(&r.id)
                        && filter.matches(r.metadata.as_ref())
                })
                .take(k)
                .collect();

            if results.len() >= k {
                return Ok(results);
            }
        }

        // Final attempt with everything available.
        let raw = inner
            .index
            .search(query, inner.index.len())
            .map_err(DbError::Core)?;
        let results: Vec<SearchResult> = raw
            .into_iter()
            .filter(|r| {
                !inner.id_map.is_tombstoned(&r.id) && filter.matches(r.metadata.as_ref())
            })
            .take(k)
            .collect();

        Ok(results)
    }

    /// Collect all live (non-tombstoned) documents, deduplicating by ID.
    /// Reverse-iterates so the latest version of a re-inserted ID wins.
    fn collect_live_documents(&self, inner: &CollectionInner) -> Vec<Document> {
        let mut seen = std::collections::HashSet::new();
        let mut live: Vec<Document> = inner
            .documents
            .iter()
            .rev()
            .filter(|doc| !inner.id_map.is_tombstoned(&doc.id) && seen.insert(doc.id.clone()))
            .cloned()
            .collect();
        live.reverse();
        live
    }

    fn maybe_auto_checkpoint(&self) -> Result<()> {
        if !self.config.auto_checkpoint {
            return Ok(());
        }

        let needs = {
            let storage = self.storage.lock();
            storage.needs_checkpoint()
        };

        if needs {
            self.compact()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foxstash_core::storage::IncrementalConfig;
    use serde_json::json;
    use tempfile::TempDir;

    fn cfg(dim: usize) -> DbConfig {
        DbConfig {
            embedding_dim: dim,
            auto_checkpoint: false,
            storage: IncrementalConfig::default(),
            ..Default::default()
        }
    }

    #[test]
    fn create_and_insert() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(4)).unwrap();

        col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn insert_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(4)).unwrap();

        let result = col.insert("a".into(), "hi".into(), vec![1.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn delete_and_search() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();

        assert!(col.delete("a").unwrap());
        assert_eq!(col.len(), 1);

        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn delete_nonexistent() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        assert!(!col.delete("nope").unwrap());
    }

    #[test]
    fn get_by_id() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "alpha".into(),
            vec![1.0, 0.0, 0.0],
            Some(json!({"k": "v"})),
        )
        .unwrap();

        let doc = col.get("a").unwrap().unwrap();
        assert_eq!(doc.id, "a");
        assert_eq!(doc.content, "alpha");
        assert_eq!(doc.metadata.unwrap()["k"], "v");

        assert!(col.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn filtered_search() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "alpha".into(),
            vec![1.0, 0.0, 0.0],
            Some(json!({"scope": "workspace"})),
        )
        .unwrap();
        col.insert(
            "b".into(),
            "beta".into(),
            vec![0.9, 0.1, 0.0],
            Some(json!({"scope": "session"})),
        )
        .unwrap();
        col.insert(
            "c".into(),
            "gamma".into(),
            vec![0.8, 0.2, 0.0],
            Some(json!({"scope": "workspace"})),
        )
        .unwrap();

        let filter = Filter::eq("scope", "workspace");
        let results = col.search(&[1.0, 0.0, 0.0], 10, Some(&filter)).unwrap();

        assert!(results.iter().all(|r| r.id != "b"));
        assert!(results.len() >= 2);
    }

    #[test]
    fn compact_removes_tombstones() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();

        col.delete("b").unwrap();
        assert_eq!(col.len(), 2);

        col.compact().unwrap();
        assert_eq!(col.len(), 2);

        // After compaction, search should still work.
        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!results.iter().any(|r| r.id == "b"));
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = TempDir::new().unwrap();

        // Session 1: insert and flush.
        {
            let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
            col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
                .unwrap();
            col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
                .unwrap();
            col.flush().unwrap();
        }

        // Session 2: reopen and verify.
        {
            let col = Collection::open("test", dir.path(), cfg(3)).unwrap();
            assert_eq!(col.len(), 2);
            assert!(col.get("a").unwrap().is_some());
            assert!(col.get("b").unwrap().is_some());
        }
    }

    #[test]
    fn empty_collection() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        assert!(col.is_empty());

        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn compact_deduplicates_reinserted_ids() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        // Insert "x", then re-insert "x" with new content.
        col.insert("x".into(), "old".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("x".into(), "new".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        assert_eq!(col.len(), 1);

        col.compact().unwrap();
        assert_eq!(col.len(), 1);

        let doc = col.get("x").unwrap().unwrap();
        assert_eq!(doc.content, "new");
    }

    #[test]
    fn list_ids_and_contains() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();

        assert!(col.contains("a"));
        assert!(col.contains("b"));
        assert!(col.contains("c"));
        assert!(!col.contains("d"));

        let mut ids = col.list_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b", "c"]);

        // Delete one, verify updated.
        col.delete("b").unwrap();
        assert!(!col.contains("b"));

        let mut ids = col.list_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "c"]);
    }
}
