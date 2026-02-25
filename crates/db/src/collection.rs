//! Per-collection handle with concurrent read/write access.
//!
//! Thread safety:
//! - `inner` (index + id_map): `RwLock` — reads don't block each other.
//! - `storage`: separate `Mutex` — WAL writes don't block searches.

use crate::filter::Filter;
use crate::hybrid::{self, HybridConfig};
use crate::id_map::IdMap;
use crate::inverted_index::InvertedIndex;
use crate::recovery;
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
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
    text_index: InvertedIndex,
    tokenizer: SimpleTokenizer,
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
        let storage =
            IncrementalStorage::new(path, config.storage.clone()).map_err(DbError::Core)?;

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
                text_index: state.text_index,
                tokenizer: SimpleTokenizer::new(),
            }),
            storage: Mutex::new(storage),
        })
    }

    /// Create a fresh (empty) collection at the given path.
    pub fn create(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        let storage =
            IncrementalStorage::new(path, config.storage.clone()).map_err(DbError::Core)?;

        Ok(Self {
            name: name.to_string(),
            inner: RwLock::new(CollectionInner {
                index: HNSWIndex::new(config.embedding_dim, config.hnsw.clone()),
                id_map: IdMap::new(),
                documents: Vec::new(),
                text_index: InvertedIndex::new(),
                tokenizer: SimpleTokenizer::new(),
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
            // Clean up old inverted index postings before tombstoning.
            // id_map.get() returns None after remove(), so capture old_pos first.
            if let Some(old_pos) = inner.id_map.get(&id) {
                inner.text_index.remove(old_pos);
            }
            // Tombstone previous version if re-inserting same ID (no-op if absent).
            inner.id_map.remove(&id);
            inner.index.add(doc.clone()).map_err(DbError::Core)?;
            let pos = inner.id_map.insert(id);
            let tokens = inner.tokenizer.tokenize(&doc.content);
            inner.text_index.add(pos, &tokens);
            inner.documents.push(doc);
        }

        self.maybe_auto_checkpoint()?;
        Ok(())
    }

    /// Soft-delete a document by ID. Returns `true` if the document existed.
    pub fn delete(&self, id: &str) -> Result<bool> {
        // Hold write lock for the entire operation to prevent TOCTOU races.
        let mut inner = self.inner.write();
        if !inner.id_map.is_live(id) {
            return Ok(false);
        }

        // WAL first (crash-safe).
        {
            let mut storage = self.storage.lock();
            storage.log_remove(id).map_err(DbError::Core)?;
        }

        // Remove inverted index postings before tombstoning.
        if let Some(pos) = inner.id_map.get(id) {
            inner.text_index.remove(pos);
        }
        // Then apply tombstone in-memory.
        inner.id_map.remove(id);
        Ok(true)
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

        if k == 0 {
            return Ok(Vec::new());
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
        // Hold write lock for the entire operation to prevent concurrent mutations
        // from being silently dropped during the swap.
        let mut inner = self.inner.write();

        let live_docs = self.collect_live_documents(&inner);
        let doc_count = live_docs.len();

        let mut new_index = HNSWIndex::new(self.config.embedding_dim, self.config.hnsw.clone());
        let mut new_id_map = IdMap::new();
        let mut new_text_index = InvertedIndex::new();

        for doc in &live_docs {
            new_index.add(doc.clone()).map_err(DbError::Core)?;
            let pos = new_id_map.insert(doc.id.clone());
            let tokens = inner.tokenizer.tokenize(&doc.content);
            new_text_index.add(pos, &tokens);
        }

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

        inner.index = new_index;
        inner.id_map = new_id_map;
        inner.documents = live_docs;
        inner.text_index = new_text_index;

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

    /// Search by text content using BM25 scoring.
    pub fn search_text(
        &self,
        query: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let inner = self.inner.read();

        let tokens = inner.tokenizer.tokenize(query);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Over-fetch to compensate for tombstones + filter, clamped to index size.
        let fetch = if filter.is_some() {
            (k.saturating_mul(4)).min(inner.text_index.len())
        } else {
            (k + inner.id_map.tombstone_count()).min(inner.text_index.len())
        };
        let raw = inner.text_index.search(&tokens, fetch);

        let results: Vec<SearchResult> = raw
            .into_iter()
            .filter_map(|(pos, score)| {
                let id = inner.id_map.id_at(pos)?;
                if !inner.id_map.is_live(id) {
                    return None;
                }
                if let Some(f) = filter {
                    let doc = inner.documents.get(pos)?;
                    if !f.matches(doc.metadata.as_ref()) {
                        return None;
                    }
                }
                let doc = inner.documents.get(pos)?;
                Some(SearchResult {
                    id: doc.id.clone(),
                    content: doc.content.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                })
            })
            .take(k)
            .collect();

        Ok(results)
    }

    /// Hybrid search: combine vector similarity and BM25 keyword scores.
    pub fn search_hybrid(
        &self,
        query: &[f32],
        query_text: &str,
        k: usize,
        filter: Option<&Filter>,
        config: Option<&HybridConfig>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.embedding_dim {
            return Err(DbError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: query.len(),
            });
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        let default_config = HybridConfig::default();
        let config = config.unwrap_or(&default_config);

        let inner = self.inner.read();

        // Vector search.
        let vector_results = if inner.index.is_empty() {
            Vec::new()
        } else {
            let fetch = (k * 2 + inner.id_map.tombstone_count()).min(inner.index.len());
            let raw = inner.index.search(query, fetch).map_err(DbError::Core)?;
            raw.into_iter()
                .filter(|r| {
                    inner.id_map.is_live(&r.id)
                        && filter
                            .map(|f| f.matches(r.metadata.as_ref()))
                            .unwrap_or(true)
                })
                .take(k * 2)
                .collect::<Vec<_>>()
        };

        // BM25 search.
        let tokens = inner.tokenizer.tokenize(query_text);
        let keyword_results = if tokens.is_empty() {
            Vec::new()
        } else {
            let fetch =
                (k.saturating_mul(2) + inner.id_map.tombstone_count()).min(inner.text_index.len());
            let raw = inner.text_index.search(&tokens, fetch);
            raw.into_iter()
                .filter(|(pos, _)| {
                    inner
                        .id_map
                        .id_at(*pos)
                        .map(|id| {
                            inner.id_map.is_live(id)
                                && filter
                                    .map(|f| {
                                        inner
                                            .documents
                                            .get(*pos)
                                            .map(|d| f.matches(d.metadata.as_ref()))
                                            .unwrap_or(false)
                                    })
                                    .unwrap_or(true)
                        })
                        .unwrap_or(false)
                })
                .take(k * 2)
                .collect::<Vec<_>>()
        };

        // Merge.
        let doc_lookup = |pos: usize| -> Option<SearchResult> {
            let doc = inner.documents.get(pos)?;
            Some(SearchResult {
                id: doc.id.clone(),
                content: doc.content.clone(),
                score: 0.0,
                metadata: doc.metadata.clone(),
            })
        };

        Ok(hybrid::merge_results(
            &vector_results,
            &keyword_results,
            &doc_lookup,
            k,
            config,
        ))
    }

    // ── private helpers ─────────────────────────────────────────────

    /// Unfiltered search: query HNSW, exclude tombstones.
    fn search_unfiltered(
        &self,
        inner: &CollectionInner,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Over-fetch to compensate for tombstones, clamped to index size
        // to avoid requesting more results than the index can return.
        let fetch = (k + inner.id_map.tombstone_count()).min(inner.index.len());
        let raw = inner.index.search(query, fetch).map_err(DbError::Core)?;

        let results: Vec<SearchResult> = raw
            .into_iter()
            .filter(|r| inner.id_map.is_live(&r.id))
            .take(k)
            .collect();

        Ok(results)
    }

    /// Filtered search: progressive over-fetch (2x, 4x, 8x, then full scan).
    fn search_filtered(
        &self,
        inner: &CollectionInner,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let fetch_sizes = [k * 2, k * 4, k * 8, inner.index.len()];

        for fetch in fetch_sizes {
            let raw = inner.index.search(query, fetch).map_err(DbError::Core)?;

            let results: Vec<SearchResult> = raw
                .into_iter()
                .filter(|r| inner.id_map.is_live(&r.id) && filter.matches(r.metadata.as_ref()))
                .take(k)
                .collect();

            if results.len() >= k || fetch >= inner.index.len() {
                return Ok(results);
            }
        }

        Ok(Vec::new())
    }

    /// Collect all live (non-tombstoned) documents, deduplicating by ID.
    /// Reverse-iterates so the latest version of a re-inserted ID wins.
    fn collect_live_documents(&self, inner: &CollectionInner) -> Vec<Document> {
        let mut seen = std::collections::HashSet::new();
        let mut live: Vec<Document> = inner
            .documents
            .iter()
            .rev()
            .filter(|doc| inner.id_map.is_live(&doc.id) && seen.insert(doc.id.clone()))
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
            let inner = self.inner.read();
            let live_docs = self.collect_live_documents(&inner);
            let doc_count = live_docs.len();

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
    fn compact_preserves_all_live_documents() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();
        col.delete("b").unwrap();

        col.compact().unwrap();

        assert_eq!(col.len(), 2);
        assert!(col.get("a").unwrap().is_some());
        assert!(col.get("b").unwrap().is_none());
        assert!(col.get("c").unwrap().is_some());

        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results.iter().any(|r| r.id == "b"));
    }

    #[test]
    fn delete_is_durable_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
            col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
                .unwrap();
            col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
                .unwrap();
            col.delete("a").unwrap();
            col.flush().unwrap();
        }
        {
            let col = Collection::open("test", dir.path(), cfg(3)).unwrap();
            assert_eq!(col.len(), 1);
            assert!(col.get("a").unwrap().is_none());
            assert!(col.get("b").unwrap().is_some());
        }
    }

    #[test]
    fn search_text_basic() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "gw".into(),
            "gateway service running on port 8080".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "db".into(),
            "database connection pool exhausted".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "gw2".into(),
            "gateway timeout after 30 seconds".into(),
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        let results = col.search_text("gateway", 10, None).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"gw"));
        assert!(ids.contains(&"gw2"));
    }

    #[test]
    fn search_text_with_filter() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service alpha".into(),
            vec![1.0, 0.0, 0.0],
            Some(json!({"env": "prod"})),
        )
        .unwrap();
        col.insert(
            "b".into(),
            "gateway service beta".into(),
            vec![0.0, 1.0, 0.0],
            Some(json!({"env": "staging"})),
        )
        .unwrap();

        let filter = Filter::eq("env", "prod");
        let results = col.search_text("gateway", 10, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn search_text_after_delete() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway running".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "gateway stopped".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        col.delete("a").unwrap();

        let results = col.search_text("gateway", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn search_hybrid_basic() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        // Doc with matching embedding AND keyword.
        col.insert(
            "both".into(),
            "gateway service running".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        // Doc with matching embedding only.
        col.insert(
            "vec_only".into(),
            "database connection pool".into(),
            vec![0.9, 0.1, 0.0],
            None,
        )
        .unwrap();
        // Doc with matching keyword only.
        col.insert(
            "kw_only".into(),
            "gateway timeout error".into(),
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway", 10, None, None)
            .unwrap();

        // All 3 docs should appear (vector finds "both" + "vec_only", keyword finds "both" + "kw_only").
        assert_eq!(results.len(), 3);
        // "both" appears in both signals → should be ranked first.
        assert_eq!(results[0].id, "both");
    }

    #[test]
    fn text_index_survives_compact_and_reopen() {
        let dir = TempDir::new().unwrap();

        {
            let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
            col.insert(
                "a".into(),
                "gateway service".into(),
                vec![1.0, 0.0, 0.0],
                None,
            )
            .unwrap();
            col.insert(
                "b".into(),
                "database pool".into(),
                vec![0.0, 1.0, 0.0],
                None,
            )
            .unwrap();
            col.insert(
                "c".into(),
                "gateway timeout".into(),
                vec![0.0, 0.0, 1.0],
                None,
            )
            .unwrap();

            col.delete("b").unwrap();
            col.compact().unwrap();

            // Text search works after compact.
            let results = col.search_text("gateway", 10, None).unwrap();
            assert_eq!(results.len(), 2);

            col.flush().unwrap();
        }

        // Reopen → recovery rebuilds text index.
        {
            let col = Collection::open("test", dir.path(), cfg(3)).unwrap();
            let results = col.search_text("gateway", 10, None).unwrap();
            assert_eq!(results.len(), 2);
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            assert!(ids.contains(&"a"));
            assert!(ids.contains(&"c"));
        }
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

    // ── Edge-case tests for text/hybrid search after mutations ──────

    #[test]
    fn search_text_after_reinsert() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service running".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();

        // Re-insert same ID with completely different content.
        col.insert(
            "a".into(),
            "database connection pool".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        // Old terms ("gateway") must NOT match.
        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(
            results.is_empty(),
            "old terms should be gone after re-insert, got {:?}",
            results.iter().map(|r| &r.id).collect::<Vec<_>>()
        );

        // New terms ("database") must match.
        let results = col.search_text("database", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn search_text_scores_correct_after_delete() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service alpha".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "gateway service beta".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "c".into(),
            "gateway service gamma".into(),
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        col.delete("b").unwrap();

        let results = col.search_text("gateway", 10, None).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"c"));
        assert!(!ids.contains(&"b"));
    }

    #[test]
    fn search_hybrid_after_reinsert() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "database pool".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        // Re-insert "a" with new content and embedding.
        col.insert(
            "a".into(),
            "database connection".into(),
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        let results = col
            .search_hybrid(&[0.0, 0.0, 1.0], "database", 10, None, None)
            .unwrap();

        // "a" should appear (matched on "database" keyword and vector).
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
    }

    #[test]
    fn search_hybrid_after_delete() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "gateway timeout".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        col.delete("a").unwrap();

        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway", 10, None, None)
            .unwrap();

        // Deleted doc must not appear.
        assert!(
            results.iter().all(|r| r.id != "a"),
            "deleted doc should not appear in hybrid results"
        );
    }

    #[test]
    fn search_text_k_zero() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();

        let results = col.search_text("gateway", 0, None).unwrap();
        assert!(results.is_empty(), "k=0 should return empty");
    }

    #[test]
    fn search_text_empty_query() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();

        let results = col.search_text("", 10, None).unwrap();
        assert!(results.is_empty(), "empty query should return empty");
    }

    #[test]
    fn search_hybrid_empty_text_query() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "database pool".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        // Empty text query → only vector contributes.
        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "", 10, None, None)
            .unwrap();

        assert!(
            !results.is_empty(),
            "vector arm should still produce results"
        );
        // "a" should rank first by vector similarity.
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn search_text_no_matching_terms() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();

        let results = col.search_text("xyznonexistent", 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn reinsert_then_compact_preserves_text_search() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service alpha".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "database pool beta".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        // Re-insert "a" with new text.
        col.insert(
            "a".into(),
            "database connection gamma".into(),
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        col.compact().unwrap();

        // After compact, "gateway" should not match (old content).
        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(results.is_empty(), "old terms should be gone after compact");

        // "database" should match both docs.
        let results = col.search_text("database", 10, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn delete_all_then_search_text() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.insert(
            "b".into(),
            "gateway timeout".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        col.delete("a").unwrap();
        col.delete("b").unwrap();

        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(results.is_empty(), "all deleted → empty results");
    }

    #[test]
    fn text_index_rebuilt_on_recovery() {
        let dir = TempDir::new().unwrap();

        // Session 1: insert, re-insert, delete, flush.
        {
            let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
            col.insert(
                "a".into(),
                "gateway service alpha".into(),
                vec![1.0, 0.0, 0.0],
                None,
            )
            .unwrap();
            col.insert(
                "b".into(),
                "database pool beta".into(),
                vec![0.0, 1.0, 0.0],
                None,
            )
            .unwrap();
            col.insert(
                "c".into(),
                "gateway timeout gamma".into(),
                vec![0.0, 0.0, 1.0],
                None,
            )
            .unwrap();
            // Re-insert "a" with different text.
            col.insert(
                "a".into(),
                "database connection delta".into(),
                vec![0.5, 0.5, 0.0],
                None,
            )
            .unwrap();
            col.delete("c").unwrap();
            col.flush().unwrap();
        }

        // Session 2: reopen → recovery rebuilds text index.
        {
            let col = Collection::open("test", dir.path(), cfg(3)).unwrap();
            assert_eq!(col.len(), 2);

            // "gateway" should not match (a was re-inserted, c deleted).
            let results = col.search_text("gateway", 10, None).unwrap();
            assert!(
                results.is_empty(),
                "gateway should not match after recovery"
            );

            // "database" should match both a and b.
            let results = col.search_text("database", 10, None).unwrap();
            assert_eq!(results.len(), 2);
        }
    }

    // ── P2.1: Input validation edge tests ───────────────────────────

    #[test]
    fn search_k_zero_returns_empty() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 0, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_hybrid_k_zero_returns_empty() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        col.insert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();

        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway", 0, None, None)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_exceeds_collection_size() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 100, None).unwrap();
        assert_eq!(results.len(), 2);
    }
}
