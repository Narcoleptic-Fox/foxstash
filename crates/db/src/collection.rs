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
    mutation_lock: Mutex<()>,
}

impl Collection {
    /// Open or create a collection at the given path.
    pub fn open(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        Self::reject_incremental_quantized_storage(&config)?;

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
            mutation_lock: Mutex::new(()),
        })
    }

    /// Create a fresh (empty) collection at the given path.
    ///
    /// Returns an error if the path already contains checkpoint or WAL data.
    /// Use [`open`](Self::open) to recover an existing collection.
    pub fn create(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        Self::reject_incremental_quantized_storage(&config)?;

        // Guard: refuse to create over existing data.
        if path.join("manifest.json").exists() {
            return Err(DbError::Validation(format!(
                "path already contains data: {}. Use Collection::open() instead",
                path.display()
            )));
        }

        let storage =
            IncrementalStorage::new(path, config.storage.clone()).map_err(DbError::Core)?;

        Ok(Self {
            name: name.to_string(),
            inner: RwLock::new(CollectionInner {
                index: HNSWIndex::new(config.embedding_dim, config.hnsw.clone()),
                id_map: IdMap::new(),
                documents: Vec::new(),
                text_index: InvertedIndex::with_config(config.bm25.clone()),
                tokenizer: SimpleTokenizer::new(),
            }),
            config,
            storage: Mutex::new(storage),
            mutation_lock: Mutex::new(()),
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
        let _mutation_guard = self.mutation_lock.lock();

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

        self.maybe_auto_checkpoint_locked()?;
        Ok(())
    }

    /// Insert or update a document. Explicit upsert semantics.
    ///
    /// If the ID already exists, the previous version is removed (WAL-safe)
    /// and replaced with the new content/embedding/metadata. Equivalent to
    /// calling [`insert`](Self::insert), but communicates intent more clearly.
    pub fn upsert(
        &self,
        id: String,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.insert(id, content, embedding, metadata)
    }

    /// Soft-delete a document by ID. Returns `true` if the document existed.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let _mutation_guard = self.mutation_lock.lock();

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

    /// Search multiple queries in parallel with an optional metadata filter.
    ///
    /// Uses thread-local reusable contexts for the ANN search phase, preserving
    /// db-level tombstone filtering semantics. When a filter is provided, applies
    /// progressive over-fetch: `2×`, `4×`, `8×`, then full scan.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        if k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        self.validate_query_batch_dims(queries)?;

        let inner = self.inner.read();
        if inner.index.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        match filter {
            None => {
                let fetch = self.unfiltered_fetch_count(&inner, k);
                let raw_batch = inner
                    .index
                    .search_batch(queries, fetch)
                    .map_err(DbError::Core)?;

                Ok(raw_batch
                    .into_iter()
                    .map(|raw| {
                        raw.into_iter()
                            .filter(|r| inner.id_map.is_live(&r.id))
                            .take(k)
                            .collect()
                    })
                    .collect())
            }
            Some(filter) => self.search_batch_filtered_impl(&inner, queries, k, filter),
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
        let _mutation_guard = self.mutation_lock.lock();

        // Hold write lock for the entire operation to prevent concurrent mutations
        // from being silently dropped during the swap.
        let mut inner = self.inner.write();

        let live_docs = self.collect_live_documents(&inner);
        let doc_count = live_docs.len();

        let mut new_index = HNSWIndex::new(self.config.embedding_dim, self.config.hnsw.clone());
        let mut new_id_map = IdMap::new();
        let mut new_text_index = InvertedIndex::with_config(self.config.bm25.clone());

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

        // Fall back to the collection's CONFIGURED hybrid settings, not a fresh default. This
        // used to build a throwaway `HybridConfig::default()` and defer to that, which made
        // `DbConfig::hybrid` — a public field with a public `with_hybrid()` builder — dead:
        // nothing in the workspace ever read it. A caller who configured custom weights and
        // then made the documented call (`search_hybrid(.., None)`) silently got the defaults.
        let config = config.unwrap_or(&self.config.hybrid);

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

    /// Reject configs whose HNSW storage needs a codebook that `Collection` cannot supply.
    ///
    /// `Storage::SQ8` and `Storage::RaBitQ` both require a codebook fitted on a corpus
    /// sample before the first vector is encoded — `HNSWIndex::build`/`build_parallel` fit
    /// it up front from the whole corpus; `HNSWIndex::new` (what `Collection` uses, since it
    /// ingests one document at a time via `insert`) leaves it empty. Encoding the first
    /// vector against an empty codebook panics (`hnsw.rs`: `q_scale[d]` / `q_min[d]` index
    /// out of bounds for SQ8, or an explicit `.expect()` for RaBitQ) — this turns that panic
    /// into a constructor-time `Err` instead.
    ///
    /// Called from both `create` and `open`, before either touches the filesystem, so a bad
    /// config fails immediately with no partial state left behind. `config.hnsw` is never
    /// mutated after construction (it isn't exposed), so a `Collection` that passes this
    /// check can never end up with quantized storage later — including in `compact()`, which
    /// rebuilds the index via this same `HNSWIndex::new` path.
    ///
    /// TODO: once `HNSWIndex::train(&mut self, sample)` lands (fits a codebook from a
    /// calibration sample instead of requiring the full corpus up front), `Collection` can
    /// train from an initial batch and this restriction can be lifted for that case.
    fn reject_incremental_quantized_storage(config: &DbConfig) -> Result<()> {
        if config.hnsw.storage != foxstash_core::index::Storage::F32 {
            return Err(DbError::UnsupportedIncrementalStorage {
                storage: config.hnsw.storage,
            });
        }
        Ok(())
    }

    #[inline]
    fn unfiltered_fetch_count(&self, inner: &CollectionInner, k: usize) -> usize {
        k.saturating_add(inner.id_map.tombstone_count())
            .min(inner.index.len())
    }

    #[inline]
    fn validate_query_batch_dims(&self, queries: &[Vec<f32>]) -> Result<()> {
        for query in queries {
            if query.len() != self.config.embedding_dim {
                return Err(DbError::DimensionMismatch {
                    expected: self.config.embedding_dim,
                    actual: query.len(),
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn filtered_fetch_sizes(&self, inner: &CollectionInner, k: usize) -> Vec<usize> {
        let index_len = inner.index.len();
        let candidates = [
            k.saturating_mul(2).min(index_len),
            k.saturating_mul(4).min(index_len),
            k.saturating_mul(8).min(index_len),
            index_len,
        ];

        let mut fetch_sizes = Vec::with_capacity(4);
        for fetch in candidates {
            if !fetch_sizes.contains(&fetch) {
                fetch_sizes.push(fetch);
            }
        }
        fetch_sizes
    }

    fn search_batch_filtered_impl(
        &self,
        inner: &CollectionInner,
        queries: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let fetch_sizes = self.filtered_fetch_sizes(inner, k);
        let mut results: Vec<Option<Vec<SearchResult>>> = vec![None; queries.len()];
        let mut pending: Vec<usize> = (0..queries.len()).collect();

        for fetch in fetch_sizes {
            if pending.is_empty() {
                break;
            }

            let pending_queries: Vec<Vec<f32>> = pending
                .iter()
                .map(|&query_idx| queries[query_idx].clone())
                .collect();

            let raw_batch = inner
                .index
                .search_batch(&pending_queries, fetch)
                .map_err(DbError::Core)?;

            let mut next_pending = Vec::new();
            let is_last_round = fetch >= inner.index.len();

            for (query_idx, raw) in pending.into_iter().zip(raw_batch) {
                let filtered: Vec<SearchResult> = raw
                    .into_iter()
                    .filter(|r| inner.id_map.is_live(&r.id) && filter.matches(r.metadata.as_ref()))
                    .take(k)
                    .collect();

                if filtered.len() >= k || is_last_round {
                    results[query_idx] = Some(filtered);
                } else {
                    next_pending.push(query_idx);
                }
            }

            pending = next_pending;
        }

        Ok(results
            .into_iter()
            .map(|result| result.unwrap_or_default())
            .collect())
    }

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

    fn maybe_auto_checkpoint_locked(&self) -> Result<()> {
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
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
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

    // Reproduces https://github.com/foxstash/foxstash (internal): a Collection configured
    // with Storage::SQ8 used to panic on its first insert (HNSWIndex::new leaves the SQ8
    // codebook untrained; push_node indexes into it unconditionally). Confirmed via
    // `cargo test -p foxstash-db repro_sq8_storage_panics_on_insert -- --nocapture` before
    // the guard existed:
    //   thread '...' panicked at crates/core/src/index/hnsw.rs:978:41:
    //   index out of bounds: the len is 0 but the index is 0
    // `Collection::create`/`open` now reject non-F32 storage up front, so this asserts the
    // clean `Err` instead.
    #[test]
    fn sq8_storage_rejected_at_construction() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::SQ8;

        match Collection::create("test", dir.path(), config) {
            Err(DbError::UnsupportedIncrementalStorage {
                storage: foxstash_core::index::Storage::SQ8,
            }) => {}
            Err(e) => panic!("expected UnsupportedIncrementalStorage, got {e:?}"),
            Ok(_) => panic!("expected UnsupportedIncrementalStorage, got Ok"),
        }
    }

    #[test]
    fn rabitq_storage_rejected_at_construction() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::RaBitQ;

        match Collection::create("test", dir.path(), config) {
            Err(DbError::UnsupportedIncrementalStorage {
                storage: foxstash_core::index::Storage::RaBitQ,
            }) => {}
            Err(e) => panic!("expected UnsupportedIncrementalStorage, got {e:?}"),
            Ok(_) => panic!("expected UnsupportedIncrementalStorage, got Ok"),
        }
    }

    #[test]
    fn sq8_storage_rejected_at_open() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::SQ8;

        match Collection::open("test", dir.path(), config) {
            Err(DbError::UnsupportedIncrementalStorage { .. }) => {}
            Err(e) => panic!("expected UnsupportedIncrementalStorage, got {e:?}"),
            Ok(_) => panic!("expected UnsupportedIncrementalStorage, got Ok"),
        }
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

    // `DbConfig::hybrid` was dead code. `search_hybrid` built a throwaway
    // `HybridConfig::default()` and deferred to *that*, so the collection's configured merge
    // settings were silently discarded on the documented call. `with_hybrid()` was a public
    // builder wired to nothing — `grep -rn 'config\.hybrid' crates/` returned zero hits, and
    // no test called it. Same shape as every other bug this repo has shipped: a public knob
    // whose tests exercise it without being able to tell whether it does anything.
    //
    // The fixture is chosen to DISCRIMINATE. The two strategies put the top score two orders
    // of magnitude apart: `WeightedSum` with `vector_weight = 1.0` min-max normalizes the best
    // vector hit to exactly 1.0, while the default `Rrf` scores it 0.7/(60 + 0 + 1) ≈ 0.0115.
    // The first assertion below proves that gap is real on THIS data before the second one
    // relies on it — otherwise the test would be as vacuous as the ones it replaces.
    /// `DbConfig::bm25` must actually reach the `InvertedIndex`.
    ///
    /// `BM25Config` was public and `InvertedIndex::with_config` was public and NOTHING CONNECTED
    /// THEM: every construction site -- `Collection::create`, `Collection::compact`,
    /// `recovery::recover` -- called `InvertedIndex::new()`, so `k1` and `b` were unreachable from
    /// any public API. Same shape as `DbConfig::hybrid`, which was dead for a release, and same
    /// shape as four options the parallel index builder ignored. A knob you cannot turn is not a
    /// knob, and its passing test (`with_config` round-trips a struct) could never say so.
    ///
    /// Discriminates on `k1`, the term-frequency saturation parameter. At `k1 = 0` BM25 ignores
    /// term frequency entirely -- a document containing a term ten times scores exactly as a
    /// document containing it once. At a high `k1` the repetition is rewarded. So the same corpus
    /// and the same query must produce a different ranking, and if they do not, the config never
    /// arrived.
    #[test]
    fn a_collections_configured_bm25_config_reaches_the_text_index() {
        let scores_with = |k1: f32| -> Vec<(String, f32)> {
            let dir = TempDir::new().unwrap();
            let mut config = cfg(3);
            config.bm25 = crate::inverted_index::BM25Config { k1, b: 0.75 };
            let col = Collection::create("test", dir.path(), config).unwrap();

            // `rare` appears once in `once` and five times in `often`. Under k1 = 0 the two are
            // indistinguishable to BM25; under a large k1, `often` pulls ahead.
            col.insert(
                "once".into(),
                "rare filler filler filler filler".into(),
                vec![1.0, 0.0, 0.0],
                None,
            )
            .unwrap();
            col.insert(
                "often".into(),
                "rare rare rare rare rare".into(),
                vec![1.0, 0.0, 0.0],
                None,
            )
            .unwrap();

            let hits = col.search_text("rare", 2, None).unwrap();
            hits.into_iter().map(|h| (h.id, h.score)).collect()
        };

        let flat = scores_with(0.0);
        let saturating = scores_with(4.0);

        let get = |v: &[(String, f32)], id: &str| -> f32 {
            v.iter().find(|(i, _)| i == id).map(|(_, s)| *s).unwrap()
        };

        // Control: the fixture discriminates. With k1 = 0, term frequency is ignored, so the
        // doc containing `rare` five times must score the SAME as the one containing it once.
        // If this control fails the fixture is not exercising k1 at all and the assertion below
        // would be meaningless.
        assert!(
            (get(&flat, "once") - get(&flat, "often")).abs() < 1e-4,
            "control failed: at k1 = 0, BM25 must ignore term frequency, so `once` ({:.4}) and \
             `often` ({:.4}) should score identically. They do not, so this fixture is not \
             measuring k1 and the real assertion below proves nothing.",
            get(&flat, "once"),
            get(&flat, "often")
        );

        // The real assertion: raising k1 must reward the repeated term.
        assert!(
            get(&saturating, "often") > get(&saturating, "once") * 1.2,
            "at k1 = 4.0 the document repeating `rare` five times ({:.4}) must outscore the one \
             containing it once ({:.4}). It does not, so `DbConfig::bm25` never reached the \
             InvertedIndex -- it is going into the struct and quietly nowhere.",
            get(&saturating, "often"),
            get(&saturating, "once")
        );
    }

    #[test]
    fn a_collections_configured_hybrid_config_is_used_when_none_is_passed() {
        let weighted = HybridConfig::default()
            .with_strategy(crate::hybrid::MergeStrategy::WeightedSum)
            .with_weights(1.0, 0.0);

        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hybrid = weighted.clone();
        let col = Collection::create("test", dir.path(), config).unwrap();

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
            vec![0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        // Control: the fixture discriminates. Forcing the DEFAULT (Rrf) via an explicit
        // override must score the top hit far below what WeightedSum gives it. If this fails,
        // the two configs are indistinguishable here and the real assertion proves nothing.
        let rrf = col
            .search_hybrid(
                &[1.0, 0.0, 0.0],
                "gateway",
                10,
                None,
                Some(&HybridConfig::default()),
            )
            .unwrap();
        assert!(
            rrf[0].score < 0.1,
            "fixture does not discriminate: Rrf top score {} is not clearly below WeightedSum's 1.0",
            rrf[0].score
        );

        // The real assertion: passing `None` must use the COLLECTION's config (WeightedSum),
        // not a fresh default. Before the fix this returned the ~0.0115 Rrf score above.
        let got = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway", 10, None, None)
            .unwrap();
        assert!(
            got[0].score > 0.9,
            "search_hybrid(.., None) ignored the collection's configured HybridConfig: \
             top score {} looks like Rrf, not WeightedSum",
            got[0].score
        );
    }

    // `DbConfig.auto_checkpoint`, flagged VACUOUS in the public-option audit: the only prior
    // test, `concurrent_inserts_with_auto_checkpoint_persist_after_reopen`, calls `col.flush()`
    // before dropping the collection and then reopens it — WAL replay on reopen recovers every
    // document regardless of whether any checkpoint ever fired, so that test cannot distinguish
    // "checkpoint fired automatically" from "checkpoint never fired, WAL replay did all the
    // work". This test never calls `flush()` or reopens: it checks for a checkpoint FILE on disk
    // immediately after a single insert, which only `maybe_auto_checkpoint_locked` can produce.
    //
    // NOT COMPILED — the team lead will compile and sabotage-verify this directly.
    //
    // Sabotage this catches: hardcode `maybe_auto_checkpoint_locked` to always return early (as
    // if `auto_checkpoint` were always `false`) — the `true` config below would then also leave
    // no `checkpoint_*.bin` file on disk after the insert.
    #[test]
    fn auto_checkpoint_true_writes_a_checkpoint_file_without_flush_or_reopen() {
        let has_checkpoint_file = |dir: &std::path::Path| -> bool {
            std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .any(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with("checkpoint_") && name.ends_with(".bin")
                })
        };

        let dir_on = TempDir::new().unwrap();
        let config_on = DbConfig {
            embedding_dim: 3,
            auto_checkpoint: true,
            storage: IncrementalConfig::default().with_checkpoint_threshold(1),
            ..Default::default()
        };
        let col_on = Collection::create("test", dir_on.path(), config_on).unwrap();
        col_on
            .insert("a".into(), "content".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        assert!(
            has_checkpoint_file(dir_on.path()),
            "auto_checkpoint: true with checkpoint_threshold: 1 should have written a \
             checkpoint file after a single insert, and none was found"
        );

        let dir_off = TempDir::new().unwrap();
        let config_off = DbConfig {
            embedding_dim: 3,
            auto_checkpoint: false,
            storage: IncrementalConfig::default().with_checkpoint_threshold(1),
            ..Default::default()
        };
        let col_off = Collection::create("test", dir_off.path(), config_off).unwrap();
        col_off
            .insert("a".into(), "content".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        assert!(
            !has_checkpoint_file(dir_off.path()),
            "auto_checkpoint: false must not write a checkpoint file on insert, but one was \
             found — auto_checkpoint is being ignored"
        );
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

    #[test]
    fn search_batch_matches_single_search() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();
        col.insert("d".into(), "delta".into(), vec![0.9, 0.1, 0.0], None)
            .unwrap();

        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch = col.search_batch(&queries, 2, None).unwrap();
        assert_eq!(batch.len(), queries.len());

        for (query, batch_results) in queries.iter().zip(batch.iter()) {
            let single = col.search(query, 2, None).unwrap();
            let single_ids: Vec<&str> = single.iter().map(|r| r.id.as_str()).collect();
            let batch_ids: Vec<&str> = batch_results.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(batch_ids, single_ids);
        }
    }

    #[test]
    fn search_batch_with_filter_matches_single_filtered_search() {
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
            vec![0.0, 1.0, 0.0],
            Some(json!({"scope": "session"})),
        )
        .unwrap();
        col.insert(
            "c".into(),
            "gamma".into(),
            vec![0.0, 0.0, 1.0],
            Some(json!({"scope": "workspace"})),
        )
        .unwrap();
        col.insert(
            "d".into(),
            "delta".into(),
            vec![0.9, 0.1, 0.0],
            Some(json!({"scope": "workspace"})),
        )
        .unwrap();

        let filter = Filter::eq("scope", "workspace");
        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch = col.search_batch(&queries, 3, Some(&filter)).unwrap();
        assert_eq!(batch.len(), queries.len());

        for (query, batch_results) in queries.iter().zip(batch.iter()) {
            let single = col.search(query, 3, Some(&filter)).unwrap();
            let single_ids: Vec<&str> = single.iter().map(|r| r.id.as_str()).collect();
            let batch_ids: Vec<&str> = batch_results.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(batch_ids, single_ids);
        }
    }

    #[test]
    fn search_batch_filter_excludes_non_matching_docs() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        for i in 0..48usize {
            let mut emb = vec![0.0f32; 3];
            emb[i % 3] = 1.0;
            emb[(i + 1) % 3] = (i as f32) * 0.001;
            let scope = if i % 2 == 0 { "workspace" } else { "session" };
            col.insert(
                format!("doc-{i}"),
                format!("doc content {i}"),
                emb,
                Some(json!({ "scope": scope })),
            )
            .unwrap();
        }

        let filter = Filter::eq("scope", "workspace");
        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.8, 0.2, 0.0],
            vec![0.2, 0.8, 0.0],
        ];

        let results = col.search_batch(&queries, 6, Some(&filter)).unwrap();

        for per_query in &results {
            assert!(per_query.iter().all(|r| {
                r.metadata
                    .as_ref()
                    .and_then(|m| m.get("scope"))
                    .and_then(|v| v.as_str())
                    == Some("workspace")
            }));
        }
    }

    #[test]
    fn search_batch_excludes_tombstones() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();
        col.delete("b").unwrap();

        let queries = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let batch = col.search_batch(&queries, 3, None).unwrap();

        assert!(batch.iter().flatten().all(|r| r.id != "b"));
    }

    // ── P2.2: Create guard ──────────────────────────────────────────

    #[test]
    fn create_on_existing_data_returns_error() {
        let dir = TempDir::new().unwrap();

        // Session 1: create, insert, flush to produce WAL/checkpoint files.
        {
            let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
            col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
                .unwrap();
            col.flush().unwrap();
        }

        // Session 2: create() on same path should fail.
        let result = Collection::create("test", dir.path(), cfg(3));
        assert!(result.is_err(), "create() over existing data should error");
    }

    // ── P3.2: Upsert semantics ──────────────────────────────────────

    #[test]
    fn upsert_creates_new_document() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.upsert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(col.len(), 1);

        let doc = col.get("a").unwrap().unwrap();
        assert_eq!(doc.content, "alpha");
    }

    #[test]
    fn upsert_updates_existing_document() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.upsert("a".into(), "old".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.upsert(
            "a".into(),
            "new".into(),
            vec![0.0, 1.0, 0.0],
            Some(json!({"updated": true})),
        )
        .unwrap();

        assert_eq!(col.len(), 1);
        let doc = col.get("a").unwrap().unwrap();
        assert_eq!(doc.content, "new");
        assert_eq!(doc.metadata.unwrap()["updated"], true);
    }

    #[test]
    fn upsert_changes_search_results() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

        col.upsert(
            "a".into(),
            "gateway service".into(),
            vec![1.0, 0.0, 0.0],
            None,
        )
        .unwrap();
        col.upsert(
            "a".into(),
            "database pool".into(),
            vec![0.0, 1.0, 0.0],
            None,
        )
        .unwrap();

        // Old text gone.
        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(results.is_empty());

        // New text found.
        let results = col.search_text("database", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn concurrent_inserts_with_auto_checkpoint_persist_after_reopen() {
        let dir = TempDir::new().unwrap();
        let config = DbConfig {
            embedding_dim: 3,
            auto_checkpoint: true,
            storage: IncrementalConfig::default().with_checkpoint_threshold(1),
            ..Default::default()
        };
        let col = Arc::new(Collection::create("test", dir.path(), config.clone()).unwrap());

        let threads = 4usize;
        let inserts_per_thread = 50usize;
        let start = Arc::new(Barrier::new(threads));
        let mut handles = Vec::new();

        for t in 0..threads {
            let col = Arc::clone(&col);
            let start = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                start.wait();
                for i in 0..inserts_per_thread {
                    let id = format!("t{t}-{i}");
                    col.insert(
                        id,
                        format!("content-{t}-{i}"),
                        vec![1.0, 0.0, 0.0],
                        Some(json!({ "thread": t, "idx": i })),
                    )
                    .unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        col.flush().unwrap();
        drop(col);

        let reopened = Collection::open("test", dir.path(), config).unwrap();
        assert_eq!(reopened.len(), threads * inserts_per_thread);
        for t in 0..threads {
            for i in 0..inserts_per_thread {
                let id = format!("t{t}-{i}");
                assert!(
                    reopened.get(&id).unwrap().is_some(),
                    "missing persisted document {id}"
                );
            }
        }
    }

    #[test]
    fn compact_during_concurrent_inserts_keeps_all_writes() {
        let dir = TempDir::new().unwrap();
        let config = cfg(3);
        let col = Arc::new(Collection::create("test", dir.path(), config.clone()).unwrap());

        for i in 0..20 {
            col.insert(
                format!("seed-{i}"),
                format!("seed-content-{i}"),
                vec![1.0, 0.0, 0.0],
                None,
            )
            .unwrap();
        }

        let writer_col = Arc::clone(&col);
        let writer = thread::spawn(move || {
            for i in 0..120 {
                writer_col
                    .insert(
                        format!("live-{i}"),
                        format!("live-content-{i}"),
                        vec![1.0, 0.0, 0.0],
                        None,
                    )
                    .unwrap();
                if i % 8 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });

        for _ in 0..24 {
            col.compact().unwrap();
            thread::sleep(Duration::from_millis(1));
        }

        writer.join().unwrap();
        col.flush().unwrap();
        drop(col);

        let reopened = Collection::open("test", dir.path(), config).unwrap();
        assert_eq!(reopened.len(), 140);
        for i in 0..20 {
            assert!(reopened.get(&format!("seed-{i}")).unwrap().is_some());
        }
        for i in 0..120 {
            assert!(reopened.get(&format!("live-{i}")).unwrap().is_some());
        }
    }
}
