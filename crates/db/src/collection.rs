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

/// Proof that the holder has exclusive right to mutate a [`Collection`].
///
/// # Why this type exists
///
/// `Collection` holds three locks and **acquires two of them in inconsistent
/// order**:
///
/// ```text
/// insert           mutation -> storage -> inner.write
/// delete, compact  mutation -> inner.write -> storage
/// ```
///
/// That is textbook deadlock shape. The only reason it is safe is that
/// `mutation_lock` serializes every mutator, so two of them can never interleave
/// and the cycle cannot form. The ordering inconsistency is *survivable only
/// because* the mutation lock is always taken first, by every path, without
/// exception.
///
/// That invariant used to live in a naming convention (`_locked` suffixes) and a
/// comment. It cost two real bugs in one day — an auto-fit that deadlocked
/// against itself by re-taking a lock it already held, and a compaction path that
/// looked correct in isolation. So it is a type now: a function that requires the
/// mutation lock takes `&MutationGuard`, which can only be produced by
/// [`Collection::begin_mutation`]. A caller that already holds it passes the guard
/// along instead of locking again, which is what makes the self-deadlock
/// unrepresentable rather than merely discouraged.
///
/// Readers take `inner.read()` alone and need no guard: they never touch
/// `storage`, so they cannot participate in a cycle.
struct MutationGuard<'a> {
    _inner: parking_lot::MutexGuard<'a, ()>,
    #[cfg(debug_assertions)]
    owner: &'a std::sync::atomic::AtomicU64,
}

impl Drop for MutationGuard<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.owner.store(0, std::sync::atomic::Ordering::Release);
    }
}

/// A named collection of documents with vector search.
pub struct Collection {
    name: String,
    /// Base directory. Retained so the graph snapshot can be written beside the
    /// checkpoint it belongs to — `IncrementalStorage` does not expose its path.
    base_path: std::path::PathBuf,
    config: DbConfig,
    inner: RwLock<CollectionInner>,
    storage: Mutex<IncrementalStorage>,
    mutation_lock: Mutex<()>,
    /// Thread currently holding `mutation_lock`, for re-entrance detection.
    ///
    /// `parking_lot::Mutex` is not reentrant, so a path that holds the mutation
    /// lock and takes it again deadlocks **silently and forever** — no panic, no
    /// timeout, just a hung process. That happened: auto-fit called the public
    /// `fit()` from inside `insert`, and it shipped, because a hang looks like
    /// slowness and the test that would have caught it had the feature disabled.
    ///
    /// Debug builds record the owner so re-entrance panics with a clear message
    /// instead. Costs an atomic store per mutation and vanishes in release.
    #[cfg(debug_assertions)]
    mutation_owner: std::sync::atomic::AtomicU64,
}

impl Collection {
    /// Take the mutation lock. Every mutating path must start here — see
    /// [`MutationGuard`] for why the lock ordering depends on it.
    fn begin_mutation(&self) -> MutationGuard<'_> {
        #[cfg(debug_assertions)]
        {
            let me = Self::current_thread_id();
            assert_ne!(
                self.mutation_owner
                    .load(std::sync::atomic::Ordering::Acquire),
                me,
                "re-entrant mutation lock: this thread already holds it. \
                 parking_lot::Mutex is not reentrant, so proceeding would deadlock \
                 forever with no diagnostic. A path that already holds the lock must \
                 pass its &MutationGuard along instead of calling the public method."
            );
        }
        let inner = self.mutation_lock.lock();
        #[cfg(debug_assertions)]
        self.mutation_owner.store(
            Self::current_thread_id(),
            std::sync::atomic::Ordering::Release,
        );
        MutationGuard {
            _inner: inner,
            #[cfg(debug_assertions)]
            owner: &self.mutation_owner,
        }
    }

    #[cfg(debug_assertions)]
    fn current_thread_id() -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut h);
        // 0 means "unheld", so never hand it out as a real id.
        h.finish() | 1
    }

    /// Open or create a collection at the given path.
    pub fn open(name: &str, path: &Path, config: DbConfig) -> Result<Self> {
        Self::reject_incremental_quantized_storage(&config)?;

        let storage =
            IncrementalStorage::new(path, config.storage.clone()).map_err(DbError::Core)?;

        let state = recovery::recover(&storage, &config, path)?;

        debug!(
            name,
            live = state.id_map.live_count(),
            tombstoned = state.id_map.tombstone_count(),
            "collection opened"
        );

        Ok(Self {
            name: name.to_string(),
            base_path: path.to_path_buf(),
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
            #[cfg(debug_assertions)]
            mutation_owner: std::sync::atomic::AtomicU64::new(0),
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
            base_path: path.to_path_buf(),
            inner: RwLock::new(CollectionInner {
                index: HNSWIndex::new(config.embedding_dim, Self::staging_hnsw(&config)),
                id_map: IdMap::new(),
                documents: Vec::new(),
                text_index: InvertedIndex::with_config(config.bm25.clone()),
                tokenizer: SimpleTokenizer::new(),
            }),
            config,
            storage: Mutex::new(storage),
            mutation_lock: Mutex::new(()),
            #[cfg(debug_assertions)]
            mutation_owner: std::sync::atomic::AtomicU64::new(0),
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
        let mutation = self.begin_mutation();

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

        // Validate BEFORE the WAL write, not after.
        //
        // The WAL goes first for crash safety, so anything rejected later is
        // already durable. A non-finite embedding used to be caught only inside
        // `add` — after the entry was on disk — and `serde_json` writes NaN as
        // `null`, which never reads back as f32. One rejected insert made the
        // collection permanently unopenable. Core owns the rules; this asks it.
        {
            let inner = self.inner.read();
            inner
                .index
                .validate_embedding_for_add(&doc.embedding)
                .map_err(DbError::Core)?;
        }

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
            inner.index.add_borrowed(&doc).map_err(DbError::Core)?;
            let pos = inner.id_map.insert(id);
            let tokens = inner.tokenizer.tokenize(&doc.content);
            inner.text_index.add(pos, &tokens);
            inner.documents.push(doc);
        }

        self.maybe_auto_fit_locked(&mutation)?;
        self.maybe_auto_checkpoint_locked(&mutation)?;
        Ok(())
    }

    /// Insert many documents at once, building the graph in parallel.
    ///
    /// `insert` adds one document at a time, which is core's slowest entry point:
    /// measured, sequential `HNSWIndex::add` is **88% of ingest**, and
    /// `build_parallel` does the same corpus **7.3x faster**. A collection built
    /// incrementally cannot use it — but a bulk load can, and bulk load is where
    /// the cost actually shows up (importing a corpus, restoring an export).
    ///
    /// Semantics match `insert`: every document is WAL-logged before any
    /// in-memory state changes, so a crash mid-call replays cleanly.
    ///
    /// Only takes the parallel path on an **empty** collection — `build_parallel`
    /// constructs a whole graph rather than adding to one, so merging into an
    /// existing index would mean rebuilding it. A non-empty collection falls back
    /// to sequential inserts, which is correct but no faster; compact() is the
    /// tool for rebuilding an existing collection.
    pub fn insert_many(&self, documents: Vec<Document>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        // Validate EVERY document before writing ANY of them to the WAL. Same
        // hazard as `insert`: the WAL is durable, so a document rejected later is
        // already on disk, and a non-finite embedding serializes to `null` and can
        // never be read back — bricking the collection. All-or-nothing here also
        // means a partially-rejected batch never half-applies.
        {
            let inner = self.inner.read();
            for doc in &documents {
                if doc.embedding.len() != self.config.embedding_dim {
                    return Err(DbError::DimensionMismatch {
                        expected: self.config.embedding_dim,
                        actual: doc.embedding.len(),
                    });
                }
                inner
                    .index
                    .validate_embedding_for_add(&doc.embedding)
                    .map_err(DbError::Core)?;
            }
        }

        let mutation = self.begin_mutation();

        // WAL first, all of them, before any in-memory mutation.
        {
            let mut storage = self.storage.lock();
            for doc in &documents {
                storage.log_add(doc).map_err(DbError::Core)?;
            }
        }

        {
            let mut inner = self.inner.write();
            if inner.id_map.live_count() == 0 && inner.documents.is_empty() {
                // Empty: build the whole graph in parallel.
                inner.index = HNSWIndex::build_parallel_from_documents(
                    documents.clone(),
                    self.config.hnsw.clone(),
                );
                // The index permutes internally, so id_map/text_index positions
                // are assigned here in insertion order, exactly as the sequential
                // path would — they index `documents`, not index slots.
                for doc in documents {
                    let pos = inner.id_map.insert(doc.id.clone());
                    let tokens = inner.tokenizer.tokenize(&doc.content);
                    inner.text_index.add(pos, &tokens);
                    inner.documents.push(doc);
                }
            } else {
                for doc in documents {
                    if let Some(old_pos) = inner.id_map.get(&doc.id) {
                        inner.text_index.remove(old_pos);
                    }
                    inner.id_map.remove(&doc.id);
                    inner.index.add_borrowed(&doc).map_err(DbError::Core)?;
                    let pos = inner.id_map.insert(doc.id.clone());
                    let tokens = inner.tokenizer.tokenize(&doc.content);
                    inner.text_index.add(pos, &tokens);
                    inner.documents.push(doc);
                }
            }
        }

        self.maybe_auto_fit_locked(&mutation)?;
        self.maybe_auto_checkpoint_locked(&mutation)?;
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
        // Held for exclusion only — this path calls nothing that needs proof.
        let _mutation = self.begin_mutation();

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
        // Held for exclusion only — this path calls nothing that needs proof.
        let _mutation = self.begin_mutation();

        // Hold write lock for the entire operation to prevent concurrent mutations
        // from being silently dropped during the swap.
        let mut inner = self.inner.write();

        let live_docs = self.collect_live_documents(&inner);
        let doc_count = live_docs.len();

        // Rebuild at the storage the collection is CURRENTLY using. Using the
        // configured target would quietly quantize a collection still in staging,
        // making compaction a semantic change rather than a cleanup.
        let rebuild_cfg = if self.is_fitted(&inner) {
            self.config.hnsw.clone()
        } else {
            Self::staging_hnsw(&self.config)
        };
        // Build the whole graph at once rather than inserting one at a time.
        //
        // Required, not merely faster: a fitted collection rebuilds at a QUANTIZED
        // storage, and `HNSWIndex::new` + `add` cannot do that — `add` needs a
        // codebook, and a fresh index has none, so it fails with NotTrained.
        // `build_parallel_from_documents` trains internally from the corpus it is
        // given, which is exactly what compaction has in hand. (This was a real
        // bug: compact() on a fitted collection failed outright.)
        let mut new_index =
            HNSWIndex::build_parallel_from_documents(live_docs.clone(), rebuild_cfg);
        if live_docs.is_empty() {
            // build_parallel_from_documents returns a dimensionless index for an
            // empty corpus; keep the collection's own dimension so later inserts
            // still validate. An empty collection is by definition unfitted.
            new_index = HNSWIndex::new(self.config.embedding_dim, Self::staging_hnsw(&self.config));
        }
        let mut new_id_map = IdMap::new();
        let mut new_text_index = InvertedIndex::with_config(self.config.bm25.clone());

        for doc in &live_docs {
            let pos = new_id_map.insert(doc.id.clone());
            let tokens = inner.tokenizer.tokenize(&doc.content);
            new_text_index.add(pos, &tokens);
        }

        let checkpoint_meta = {
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
                .map_err(DbError::Core)?
        };

        // Snapshot the freshly compacted graph, not the pre-compaction one.
        {
            let path = self.graph_snapshot_path(checkpoint_meta.id);
            if let Err(err) = new_index.snapshot_to_file(&path) {
                debug!(?err, ?path, "graph snapshot not written; open will rebuild");
            }
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
    /// Reject only the storage that genuinely cannot work incrementally.
    ///
    /// This used to reject *every* quantized storage, because a quantizer needs a
    /// codebook fitted on a corpus sample before the first vector is encoded and a
    /// collection has no sample at construction. That is solved by staging — see
    /// [`Self::fit`] — so SQ8, RaBitQ and TurboRabit are now reachable.
    ///
    /// `Warren` is different and still refused: it retains no f32 at all, so core's
    /// incremental `add()` cannot compute the exact distances edge selection needs.
    /// It is a bulk-build-only mode, and no amount of staging changes that.
    fn reject_incremental_quantized_storage(config: &DbConfig) -> Result<()> {
        if config.hnsw.storage == foxstash_core::index::Storage::Warren {
            return Err(DbError::UnsupportedIncrementalStorage {
                storage: config.hnsw.storage,
            });
        }
        Ok(())
    }

    /// HNSW config for the **staging** index: the caller's settings, forced to F32.
    ///
    /// Everything else — `m`, `ef_construction`, metric, reordering — is preserved,
    /// so fitting later changes only how vectors are stored.
    fn staging_hnsw(config: &DbConfig) -> foxstash_core::index::HNSWConfig {
        let mut hnsw = config.hnsw.clone();
        hnsw.storage = foxstash_core::index::Storage::F32;
        hnsw
    }

    /// Whether this collection is configured to quantize at all.
    fn has_quantized_target(&self) -> bool {
        self.config.hnsw.storage != foxstash_core::index::Storage::F32
    }

    /// True once the live index is using the configured target storage.
    fn is_fitted(&self, inner: &CollectionInner) -> bool {
        inner.index.config().storage == self.config.hnsw.storage
    }

    /// Quantize the collection: fit a codebook over what it currently holds and
    /// re-encode every vector into the configured storage.
    ///
    /// This is the transition a quantizer needs and an incrementally-written
    /// collection cannot do at construction. `build_parallel_from_documents` trains
    /// the codebook internally from the full corpus, so the sample is exactly the
    /// data — not an approximation of it.
    ///
    /// Idempotent: a no-op if the collection is already fitted, or if the target is
    /// F32 (nothing to fit). Safe to call at any size, though quantizing a handful
    /// of vectors buys nothing — exact search on a small collection is already the
    /// right answer, which is why `fit_threshold` exists.
    pub fn fit(&self) -> Result<()> {
        if !self.has_quantized_target() {
            return Ok(());
        }
        let mutation = self.begin_mutation();
        self.fit_locked(&mutation)
    }

    /// Body of [`Self::fit`], for callers **already holding the mutation lock**.
    ///
    /// `parking_lot::Mutex` is not reentrant, so a path that holds the mutation
    /// lock and then calls the public `fit()` deadlocks against itself. `insert`
    /// holds it for its whole body and then auto-fits, which is exactly that
    /// shape — it hung on every collection reaching `fit_threshold`, i.e. the
    /// default configuration. Same `_locked` convention as
    /// `maybe_auto_checkpoint_locked`.
    fn fit_locked(&self, _mutation: &MutationGuard<'_>) -> Result<()> {
        let mut inner = self.inner.write();
        if self.is_fitted(&inner) {
            return Ok(());
        }
        let live_docs = self.collect_live_documents(&inner);
        if live_docs.is_empty() {
            return Ok(());
        }
        // Rebuild at the target storage. Trains from the whole corpus.
        inner.index = HNSWIndex::build_parallel_from_documents(live_docs, self.config.hnsw.clone());
        debug!(
            storage = ?self.config.hnsw.storage,
            docs = inner.index.len(),
            "collection fitted"
        );
        Ok(())
    }

    /// Fit once the collection is large enough, if it is configured to.
    /// Fit once the collection is large enough, if it is configured to.
    ///
    /// Callers must already hold the mutation lock — see [`Self::fit_locked`].
    fn maybe_auto_fit_locked(&self, mutation: &MutationGuard<'_>) -> Result<()> {
        if self.config.fit_threshold == 0 || !self.has_quantized_target() {
            return Ok(());
        }
        let due = {
            let inner = self.inner.read();
            !self.is_fitted(&inner) && inner.id_map.live_count() >= self.config.fit_threshold
        };
        if due {
            self.fit_locked(mutation)?;
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

    fn search_batch_filtered_impl(
        &self,
        inner: &CollectionInner,
        queries: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        // One native filtered graph walk per query, fanned across rayon workers by core — same
        // predicate as the single-query path. Replaces the former progressive over-fetch (which
        // re-ran the whole batch at 2×/4×/8×/full scan until every query filled `k`).
        inner
            .index
            .search_batch_filtered_by(queries, k, |id, metadata| {
                inner.id_map.is_live(id) && filter.matches(metadata)
            })
            .map_err(DbError::Core)
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

    /// Filtered search: a single graph walk that admits only live documents passing `filter`,
    /// collecting up to `k` directly via [`HNSWIndex::search_filtered_by`]. The predicate is
    /// evaluated during traversal, on visited nodes only.
    ///
    /// Replaces the former progressive over-fetch (`2×` → `4×` → `8×` → full scan), which re-ran the
    /// walk up to four times and degraded to a full brute-force scan on selective filters. The graph
    /// now filters natively in one pass — no over-fetch, no repeated walks.
    fn search_filtered(
        &self,
        inner: &CollectionInner,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        inner
            .index
            .search_filtered_by(query, k, |id, metadata| {
                inner.id_map.is_live(id) && filter.matches(metadata)
            })
            .map_err(DbError::Core)
    }

    /// Collect all live (non-tombstoned) documents, deduplicating by ID.
    /// Reverse-iterates so the latest version of a re-inserted ID wins.
    /// Path of the graph snapshot belonging to checkpoint `id`.
    ///
    /// Named for the checkpoint it was written with, so a snapshot can never be
    /// paired with a different checkpoint — a crash between the two writes leaves
    /// an orphan that simply is not found, rather than a mismatched graph.
    fn graph_snapshot_path(&self, checkpoint_id: u64) -> std::path::PathBuf {
        self.base_path
            .join(format!("graph_{checkpoint_id:05}.snapshot"))
    }

    /// Persist the graph beside the checkpoint so the next open can load it
    /// instead of rebuilding.
    ///
    /// Best-effort by design: the snapshot is a **same-version cache**, and
    /// recovery falls back to rebuilding whenever it is missing, stale or written
    /// by another build. So a failure here must not fail the checkpoint — the
    /// durable data is the checkpoint, and this only ever saves time.
    fn write_graph_snapshot(&self, inner: &CollectionInner, checkpoint_id: u64) {
        let path = self.graph_snapshot_path(checkpoint_id);
        if let Err(err) = inner.index.snapshot_to_file(&path) {
            debug!(?err, ?path, "graph snapshot not written; open will rebuild");
        }
    }

    /// Live documents as **borrows** — no allocation beyond the pointer vector.
    ///
    /// The checkpoint path only serializes, so it has no business cloning the
    /// whole collection to do it. Measured: 78% of resident memory is allocator
    /// retention rather than live data, and this was the largest single source —
    /// every checkpoint allocated N ids, N contents and N embeddings, then freed
    /// them, leaving the pages behind.
    ///
    /// Dedup keys on `&str` rather than a cloned `String`, which removes a second
    /// allocation per document.
    ///
    /// Iterates in reverse so the LAST version of a re-inserted id wins, then
    /// restores insertion order — `documents` is append-only, so an id may appear
    /// more than once and only the newest is live.
    fn collect_live_document_refs<'a>(&self, inner: &'a CollectionInner) -> Vec<&'a Document> {
        let mut seen = std::collections::HashSet::new();
        let mut live: Vec<&Document> = inner
            .documents
            .iter()
            .rev()
            .filter(|doc| inner.id_map.is_live(&doc.id) && seen.insert(doc.id.as_str()))
            .collect();
        live.reverse();
        live
    }

    /// Owned copies, for callers that must keep them (compaction rebuilds the
    /// index and replaces the document store). Delegates so there is one
    /// definition of "live".
    fn collect_live_documents(&self, inner: &CollectionInner) -> Vec<Document> {
        self.collect_live_document_refs(inner)
            .into_iter()
            .cloned()
            .collect()
    }

    fn maybe_auto_checkpoint_locked(&self, _mutation: &MutationGuard<'_>) -> Result<()> {
        if !self.config.auto_checkpoint {
            return Ok(());
        }

        let needs = {
            let storage = self.storage.lock();
            storage.needs_checkpoint()
        };

        if needs {
            let inner = self.inner.read();
            // Borrowed: this path serializes and nothing more.
            let live_docs = self.collect_live_document_refs(&inner);
            let doc_count = live_docs.len();

            let meta = {
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
                    .map_err(DbError::Core)?
            };
            self.write_graph_snapshot(&inner, meta.id);
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
    // These three used to assert that SQ8/RaBitQ were REJECTED at construction —
    // the old contract, when db was F32-only because a quantizer cannot encode
    // before its codebook exists. Staging solves that (see `fit`), so the contract
    // is now: quantized targets are ACCEPTED and start in F32. Rewritten rather
    // than deleted, so the change of behaviour is asserted rather than merely
    // untested.
    #[test]
    fn sq8_storage_is_accepted_and_starts_in_staging() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::SQ8;

        let col = Collection::create("test", dir.path(), config)
            .expect("SQ8 is reachable now: the collection stages in F32 and fits later");
        col.insert("a".into(), "content".into(), vec![1.0, 2.0, 3.0], None)
            .expect("staging accepts inserts with no codebook");
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn rabitq_storage_is_accepted_and_starts_in_staging() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::RaBitQ;

        let col = Collection::create("test", dir.path(), config).expect("RaBitQ stages too");
        col.insert("a".into(), "content".into(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn sq8_storage_is_accepted_at_open() {
        let dir = TempDir::new().unwrap();
        let mut config = cfg(3);
        config.hnsw.storage = foxstash_core::index::Storage::SQ8;
        Collection::open("test", dir.path(), config).expect("opening an SQ8 collection works");
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

    /// The graph snapshot is a cache: reopening must produce identical results
    /// whether it is present, absent, or corrupt.
    ///
    /// Written this way on purpose. A test that only checks the happy path would
    /// pass even if the fallback were broken — and the fallback is the part that
    /// carries correctness, since the snapshot refuses to load across versions by
    /// design and will therefore be missing on every upgrade.
    #[test]
    fn reopen_is_correct_with_a_present_missing_or_corrupt_graph_snapshot() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();
        let config = DbConfig::default().with_embedding_dim(4);

        let expected: Vec<(String, Vec<f32>)> = (0..40)
            .map(|i| {
                let f = i as f32;
                (format!("d{i}"), vec![f, f + 1.0, f + 2.0, f + 3.0])
            })
            .collect();

        {
            let c = Collection::create("c", &path, config.clone()).unwrap();
            for (id, v) in &expected {
                c.insert(id.clone(), format!("content {id}"), v.clone(), None)
                    .unwrap();
            }
            // compact() checkpoints, which is what writes the snapshot.
            c.compact().unwrap();
        }

        let snapshots = || -> Vec<std::path::PathBuf> {
            std::fs::read_dir(&path)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("snapshot"))
                .collect()
        };
        assert!(
            !snapshots().is_empty(),
            "checkpoint should have written a graph snapshot"
        );

        // Every id must come back, and search must find each vector, in all three cases.
        let verify = |case: &str| {
            let c = Collection::open("c", &path, config.clone()).unwrap();
            assert_eq!(c.len(), expected.len(), "{case}: document count");
            for (id, v) in &expected {
                let hits = c.search(v, 1, None).unwrap();
                assert_eq!(
                    hits.first().map(|h| h.id.as_str()),
                    Some(id.as_str()),
                    "{case}: nearest neighbour of {id}'s own vector should be {id}"
                );
            }
        };

        verify("snapshot present");

        // Corrupt it: the loader must reject rather than misread, then rebuild.
        for p in snapshots() {
            std::fs::write(&p, b"not a snapshot").unwrap();
        }
        verify("snapshot corrupt");

        // Absent — the state after any version upgrade.
        for p in snapshots() {
            std::fs::remove_file(&p).unwrap();
        }
        assert!(snapshots().is_empty());
        verify("snapshot absent");
    }

    /// `insert_many` must be observably identical to a loop of `insert` —
    /// same documents retrievable, same search answers, and durable across a
    /// reopen (i.e. actually WAL-logged, not just placed in memory).
    #[test]
    fn insert_many_matches_insert_and_survives_reopen() {
        let docs: Vec<Document> = (0..120)
            .map(|i| {
                let mut e = vec![0.0f32; 8];
                // Dense and distinct: the default metric is Cosine, so vectors
                // differing only in magnitude would be identical in direction.
                for (k, v) in e.iter_mut().enumerate() {
                    *v = ((i * 7 + k * 13) % 29) as f32 - 14.0;
                }
                e[i % 8] += 40.0;
                Document {
                    id: format!("d{i}"),
                    content: format!("content {i}"),
                    embedding: e,
                    metadata: None,
                }
            })
            .collect();

        let cfg = DbConfig::default().with_embedding_dim(8);

        // Reference: one at a time.
        let one_dir = TempDir::new().unwrap();
        let one_path = one_dir.path().join("c");
        std::fs::create_dir_all(&one_path).unwrap();
        let one = Collection::create("c", &one_path, cfg.clone()).unwrap();
        for d in &docs {
            one.insert(d.id.clone(), d.content.clone(), d.embedding.clone(), None)
                .unwrap();
        }

        // Bulk.
        let many_dir = TempDir::new().unwrap();
        let many_path = many_dir.path().join("c");
        std::fs::create_dir_all(&many_path).unwrap();
        {
            let many = Collection::create("c", &many_path, cfg.clone()).unwrap();
            many.insert_many(docs.clone()).unwrap();
            assert_eq!(many.len(), docs.len());

            for d in &docs {
                assert_eq!(
                    many.get(&d.id).unwrap().map(|g| g.content),
                    Some(d.content.clone()),
                    "{} should be retrievable by id",
                    d.id
                );
                let bulk = many.search(&d.embedding, 1, None).unwrap();
                let seq = one.search(&d.embedding, 1, None).unwrap();
                assert_eq!(
                    bulk.first().map(|h| &h.id),
                    seq.first().map(|h| &h.id),
                    "{} : bulk and sequential must agree on the nearest neighbour",
                    d.id
                );
            }
        }

        // Durability: the WAL must carry it, so a reopen sees everything.
        let reopened = Collection::open("c", &many_path, cfg).unwrap();
        assert_eq!(reopened.len(), docs.len(), "bulk insert must be durable");
        for d in &docs {
            assert_eq!(
                reopened
                    .search(&d.embedding, 1, None)
                    .unwrap()
                    .first()
                    .map(|h| h.id.clone()),
                Some(d.id.clone()),
                "{} should retrieve itself after reopen",
                d.id
            );
        }
    }

    /// A quantized collection must stage in F32, fit on demand, keep serving
    /// correctly across the transition, and still accept writes afterwards.
    ///
    /// The last part is the one worth asserting: the whole reason db was F32-only
    /// was that a quantizer cannot encode before its codebook exists. Fitting is
    /// only useful if inserts keep working *after* it.
    #[test]
    fn a_quantized_collection_stages_then_fits_and_still_accepts_writes() {
        use foxstash_core::index::Storage;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();

        let hnsw = foxstash_core::index::HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        };
        let cfg = DbConfig {
            hnsw,
            fit_threshold: 0, // fit explicitly, so the test controls the transition
            ..DbConfig::default()
        }
        .with_embedding_dim(8);

        // Deterministic dense pseudo-random. NOT modular arithmetic: a first pass
        // used `(i*31 + k*17) % 61`, which makes i and i+61 produce IDENTICAL
        // vectors — d99 and d38 collided and the test failed in staging, before
        // quantization was even involved. Under a magnitude-invariant metric
        // (Cosine, the default) proportional vectors are indistinguishable too, so
        // any fixture with structure risks ties. Use noise.
        let vector = |i: usize| -> Vec<f32> {
            let mut state = (i as u64).wrapping_mul(6_364_136_223_846_793_005) + 1;
            (0..8)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / (1u32 << 31) as f32) - 0.5
                })
                .collect()
        };

        let c = Collection::create("c", &path, cfg).unwrap();
        for i in 0..200 {
            c.insert(format!("d{i}"), format!("content {i}"), vector(i), None)
                .unwrap();
        }

        // Staging: inserts worked at all, which they could not have under the old
        // blanket rejection.
        assert_eq!(c.len(), 200);
        for i in [0usize, 99, 199] {
            let hits = c.search(&vector(i), 1, None).unwrap();
            assert_eq!(
                hits[0].id,
                format!("d{i}"),
                "staging: d{i} should find itself"
            );
        }

        c.fit().unwrap();

        // Fitted: still correct, and still writable — quantized search is lossy, so
        // check membership in the top-k rather than demanding rank 1.
        for i in [0usize, 99, 199] {
            let hits = c.search(&vector(i), 5, None).unwrap();
            assert!(
                hits.iter().any(|h| h.id == format!("d{i}")),
                "fitted: d{i} should be in its own top-5, got {:?}",
                hits.iter().map(|h| &h.id).collect::<Vec<_>>()
            );
        }

        c.insert("post-fit".into(), "after".into(), vector(500), None)
            .unwrap();
        assert_eq!(c.len(), 201, "inserts must work after fitting");
        let hits = c.search(&vector(500), 5, None).unwrap();
        assert!(
            hits.iter().any(|h| h.id == "post-fit"),
            "a document inserted after fitting must be searchable"
        );

        // fit() is idempotent.
        c.fit().unwrap();
        assert_eq!(c.len(), 201);
    }

    /// Warren is bulk-build-only — it retains no f32, so no amount of staging lets
    /// it accept incremental inserts. It must still be refused up front.
    #[test]
    fn warren_storage_is_still_rejected_for_collections() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();
        let hnsw = foxstash_core::index::HNSWConfig {
            storage: foxstash_core::index::Storage::Warren,
            ..Default::default()
        };
        let cfg = DbConfig {
            hnsw,
            ..DbConfig::default()
        }
        .with_embedding_dim(8);
        assert!(Collection::create("c", &path, cfg).is_err());
    }

    #[test]
    fn a_fitted_collection_survives_reopen() {
        use foxstash_core::index::Storage;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();
        let hnsw = foxstash_core::index::HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        };
        let cfg = DbConfig {
            hnsw,
            fit_threshold: 0,
            ..DbConfig::default()
        }
        .with_embedding_dim(8);
        let vector = |i: usize| -> Vec<f32> {
            let mut st = (i as u64).wrapping_mul(6_364_136_223_846_793_005) + 1;
            (0..8)
                .map(|_| {
                    st = st
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((st >> 33) as f32 / (1u32 << 31) as f32) - 0.5
                })
                .collect()
        };
        {
            let c = Collection::create("c", &path, cfg.clone()).unwrap();
            for i in 0..200 {
                c.insert(format!("d{i}"), format!("c{i}"), vector(i), None)
                    .unwrap();
            }
            c.fit().unwrap();
            c.compact().unwrap(); // forces a checkpoint + graph snapshot of the FITTED index
        }
        let re = Collection::open("c", &path, cfg).unwrap();
        assert_eq!(re.len(), 200, "document count after reopen");
        let mut found = 0;
        for i in 0..200 {
            if re
                .search(&vector(i), 5, None)
                .unwrap()
                .iter()
                .any(|h| h.id == format!("d{i}"))
            {
                found += 1;
            }
        }
        assert!(
            found >= 190,
            "only {found}/200 documents retrieved themselves after reopen"
        );
        re.insert("post".into(), "x".into(), vector(999), None)
            .unwrap();
        assert_eq!(re.len(), 201, "must still accept writes after reopen");
    }

    #[test]
    fn auto_fit_triggers_from_insert_without_deadlocking() {
        use foxstash_core::index::Storage;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();
        let hnsw = foxstash_core::index::HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        };
        // Threshold LOW so a plain insert crosses it — the default code path.
        let cfg = DbConfig {
            hnsw,
            fit_threshold: 50,
            ..DbConfig::default()
        }
        .with_embedding_dim(8);
        let vector = |i: usize| -> Vec<f32> {
            let mut st = (i as u64).wrapping_mul(6_364_136_223_846_793_005) + 1;
            (0..8)
                .map(|_| {
                    st = st
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((st >> 33) as f32 / (1u32 << 31) as f32) - 0.5
                })
                .collect()
        };
        let c = Collection::create("c", &path, cfg).unwrap();
        for i in 0..60 {
            c.insert(format!("d{i}"), format!("c{i}"), vector(i), None)
                .unwrap();
        }
        assert_eq!(c.len(), 60);
    }

    /// Auto-fit under concurrent writers.
    ///
    /// `fit` swaps the whole index while other threads are inserting, and it is
    /// reached from inside `insert` while the mutation lock is held. That is the
    /// shape that produced a self-deadlock, so this exercises it under contention
    /// rather than from a single thread with auto-fit disabled — which is how the
    /// deadlock survived its first test.
    #[test]
    fn auto_fit_under_concurrent_writers_loses_nothing() {
        use foxstash_core::index::Storage;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c");
        std::fs::create_dir_all(&path).unwrap();
        let hnsw = foxstash_core::index::HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        };
        let cfg = DbConfig {
            hnsw,
            fit_threshold: 40,
            ..DbConfig::default()
        }
        .with_embedding_dim(8);

        let col = Arc::new(Collection::create("c", &path, cfg).unwrap());
        let threads = 4usize;
        let per_thread = 40usize;
        let start = Arc::new(Barrier::new(threads + 1));
        let mut handles = Vec::new();

        for t in 0..threads {
            let col = Arc::clone(&col);
            let start = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                start.wait();
                for i in 0..per_thread {
                    let seed = (t * per_thread + i) as u64;
                    let mut st = seed.wrapping_mul(6_364_136_223_846_793_005) + 1;
                    let v: Vec<f32> = (0..8)
                        .map(|_| {
                            st = st
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            ((st >> 33) as f32 / (1u32 << 31) as f32) - 0.5
                        })
                        .collect();
                    col.insert(format!("t{t}-{i}"), format!("c{t}{i}"), v, None)
                        .unwrap();
                }
            }));
        }

        // A concurrent reader, to catch a search racing the index swap.
        let reader = {
            let col = Arc::clone(&col);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                for _ in 0..200 {
                    let _ = col.search(&[0.1; 8], 5, None);
                    let _ = col.len();
                }
            })
        };

        for h in handles {
            h.join().expect("writer panicked or deadlocked");
        }
        reader.join().expect("reader panicked");

        assert_eq!(
            col.len(),
            threads * per_thread,
            "every concurrent write must survive the fit"
        );
        for t in 0..threads {
            for i in 0..per_thread {
                assert!(
                    col.get(&format!("t{t}-{i}")).unwrap().is_some(),
                    "lost t{t}-{i} across the fit"
                );
            }
        }
    }

    /// The `documents` / `id_map` positional invariant, under a mixed workload.
    ///
    /// `IdMap` maintains its own `next_pos` counter with no structural link to
    /// `documents.len()`; four separate write paths (insert, insert_many, compact,
    /// recovery) each have to remember to advance both in lockstep. Nothing
    /// enforces it, so this asserts it — after every phase, and especially after
    /// re-inserts and deletes, which are where a position can be assigned without
    /// a matching document.
    #[test]
    fn document_and_id_map_positions_stay_in_lockstep() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(4)).unwrap();

        let check = |phase: &str| {
            let inner = col.inner.read();
            assert_eq!(
                inner.documents.len(),
                inner.id_map.next_pos_for_test(),
                "{phase}: documents.len() and id_map's next position diverged"
            );
            for id in inner.id_map.live_ids() {
                let pos = inner
                    .id_map
                    .get(id)
                    .unwrap_or_else(|| panic!("{phase}: live id {id} has no position"));
                let doc = inner.documents.get(pos).unwrap_or_else(|| {
                    panic!(
                        "{phase}: id {id} points at position {pos}, out of {} documents",
                        inner.documents.len()
                    )
                });
                assert_eq!(
                    doc.id, *id,
                    "{phase}: id {id} maps to position {pos}, which holds {}",
                    doc.id
                );
            }
        };

        for i in 0..30 {
            col.insert(
                format!("d{i}"),
                format!("c{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                None,
            )
            .unwrap();
        }
        check("after inserts");

        // Re-insert existing ids: tombstones the old position and assigns a new one.
        for i in 0..10 {
            col.insert(
                format!("d{i}"),
                format!("v2-{i}"),
                vec![i as f32, 1.0, 0.0, 0.0],
                None,
            )
            .unwrap();
        }
        check("after re-inserts");

        for i in 20..30 {
            col.delete(&format!("d{i}")).unwrap();
        }
        check("after deletes");

        for i in 30..40 {
            col.insert(
                format!("d{i}"),
                format!("c{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                None,
            )
            .unwrap();
        }
        check("after inserting past the deletes");

        col.compact().unwrap();
        check("after compaction");

        // And the content must be the LATEST version, not a resurrected old one.
        for i in 0..10 {
            assert_eq!(
                col.get(&format!("d{i}")).unwrap().map(|d| d.content),
                Some(format!("v2-{i}")),
                "d{i} should hold its re-inserted content"
            );
        }
    }

    #[test]
    fn a_rejected_insert_does_not_break_reopen() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(4)).unwrap();
        col.insert("good".into(), "c".into(), vec![1.0, 2.0, 3.0, 4.0], None)
            .unwrap();
        // Right dimension, but not finite — db validates dim, core rejects NaN.
        let bad = col.insert(
            "bad".into(),
            "c".into(),
            vec![f32::NAN, 0.0, 0.0, 0.0],
            None,
        );
        assert!(bad.is_err(), "a non-finite embedding must be rejected");
        col.flush().unwrap();
        drop(col);
        let re = Collection::open("test", dir.path(), cfg(4))
            .expect("a rejected insert must not make the collection unopenable");
        assert!(
            re.get("good").unwrap().is_some(),
            "the good document must survive"
        );
        assert!(
            re.get("bad").unwrap().is_none(),
            "the rejected one must not appear"
        );
    }

    #[test]
    fn a_rejected_batch_does_not_half_apply_or_break_reopen() {
        let dir = TempDir::new().unwrap();
        let col = Collection::create("test", dir.path(), cfg(4)).unwrap();
        let mut batch: Vec<Document> = (0..10)
            .map(|i| Document {
                id: format!("d{i}"),
                content: format!("c{i}"),
                embedding: vec![i as f32, 1.0, 2.0, 3.0],
                metadata: None,
            })
            .collect();
        // One bad document, in the middle.
        batch[5].embedding = vec![f32::INFINITY, 0.0, 0.0, 0.0];

        assert!(
            col.insert_many(batch).is_err(),
            "the batch must be rejected"
        );
        assert_eq!(col.len(), 0, "a rejected batch must not half-apply");
        col.flush().unwrap();
        drop(col);
        Collection::open("test", dir.path(), cfg(4))
            .expect("a rejected batch must not make the collection unopenable");
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
