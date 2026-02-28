//! Top-level vector store managing multiple named collections.
//!
//! Filesystem layout:
//! ```text
//! <base_path>/
//!   collections/
//!     <name>/          ← one IncrementalStorage dir per collection
//!       manifest.json
//!       checkpoint_*.bin
//!       wal_*.log
//! ```

use crate::collection::Collection;
use crate::{DbConfig, DbError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// A vector store managing multiple named collections on disk.
///
/// All collections within a single `VectorStore` share the same
/// `embedding_dim` from the [`DbConfig`] used to open the store. Attempting
/// to insert a document whose embedding length differs from `config.embedding_dim`
/// will return a [`DbError::DimensionMismatch`] error.
///
/// If you need to work with embeddings of different dimensions (produced by
/// different models), open separate `VectorStore` instances in separate
/// directories — one per embedding model.
pub struct VectorStore {
    base_path: PathBuf,
    config: DbConfig,
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl VectorStore {
    fn validate_collection_name(name: &str) -> Result<()> {
        let mut components = Path::new(name).components();
        let valid_single_segment =
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();

        if valid_single_segment {
            Ok(())
        } else {
            Err(DbError::Validation(format!(
                "invalid collection name '{name}': must be a single path segment"
            )))
        }
    }

    /// Open a store, recovering all existing collections from disk.
    pub fn open(path: impl AsRef<Path>, config: DbConfig) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        let collections_dir = base_path.join("collections");
        std::fs::create_dir_all(&collections_dir)?;

        let mut map = HashMap::new();

        // Scan for existing collections.
        for entry in std::fs::read_dir(&collections_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                let col_path = entry.path();
                let col = Collection::open(&name, &col_path, config.clone())?;
                map.insert(name, Arc::new(col));
            }
        }

        info!(
            path = %base_path.display(),
            collections = map.len(),
            "store opened"
        );

        Ok(Self {
            base_path,
            config,
            collections: RwLock::new(map),
        })
    }

    /// Get or create a collection by name.
    ///
    /// If the collection already exists it is returned from the registry.
    /// Otherwise a new empty collection is created on disk and registered.
    pub fn get_or_create_collection(&self, name: &str) -> Result<Arc<Collection>> {
        Self::validate_collection_name(name)?;

        // Fast path: read lock.
        {
            let map = self.collections.read();
            if let Some(col) = map.get(name) {
                return Ok(Arc::clone(col));
            }
        }

        // Slow path: create.
        let mut map = self.collections.write();
        // Double-check under write lock.
        if let Some(col) = map.get(name) {
            return Ok(Arc::clone(col));
        }

        let col_path = self.base_path.join("collections").join(name);
        std::fs::create_dir_all(&col_path)?;
        let col = Arc::new(Collection::create(name, &col_path, self.config.clone())?);
        map.insert(name.to_string(), Arc::clone(&col));

        info!(name, "collection created");
        Ok(col)
    }

    /// Create a new collection. Returns error if it already exists.
    pub fn create_collection(&self, name: &str) -> Result<Arc<Collection>> {
        Self::validate_collection_name(name)?;

        let mut map = self.collections.write();
        if map.contains_key(name) {
            return Err(DbError::CollectionExists(name.to_string()));
        }

        let col_path = self.base_path.join("collections").join(name);
        std::fs::create_dir_all(&col_path)?;
        let col = Arc::new(Collection::create(name, &col_path, self.config.clone())?);
        map.insert(name.to_string(), Arc::clone(&col));

        info!(name, "collection created (explicit)");
        Ok(col)
    }

    /// Get an existing collection. Returns error if not found.
    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>> {
        Self::validate_collection_name(name)?;

        let map = self.collections.read();
        map.get(name)
            .cloned()
            .ok_or_else(|| DbError::CollectionNotFound(name.to_string()))
    }

    /// List all collection names.
    pub fn collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Unload a collection, removing it from the in-memory registry.
    ///
    /// **Does not** delete files from disk — the collection can be recovered
    /// by calling [`get_or_create_collection`] again. To permanently delete,
    /// use [`delete_collection`] or remove the directory manually after unloading.
    pub fn unload_collection(&self, name: &str) -> Result<()> {
        Self::validate_collection_name(name)?;

        let mut map = self.collections.write();
        if map.remove(name).is_none() {
            return Err(DbError::CollectionNotFound(name.to_string()));
        }
        info!(name, "collection unloaded");
        Ok(())
    }

    /// Permanently delete a collection: unload from registry and remove files from disk.
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        Self::validate_collection_name(name)?;

        {
            let mut map = self.collections.write();
            if map.remove(name).is_none() {
                return Err(DbError::CollectionNotFound(name.to_string()));
            }
        }
        let col_path = self.base_path.join("collections").join(name);
        if col_path.exists() {
            std::fs::remove_dir_all(&col_path)?;
        }
        info!(name, "collection deleted");
        Ok(())
    }

    /// Flush all collections to disk.
    pub fn flush_all(&self) -> Result<()> {
        let map = self.collections.read();
        for col in map.values() {
            col.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cfg() -> DbConfig {
        DbConfig {
            embedding_dim: 3,
            auto_checkpoint: false,
            ..Default::default()
        }
    }

    #[test]
    fn open_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        assert!(store.collections().is_empty());
    }

    #[test]
    fn create_and_use_collection() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();

        let col = store.get_or_create_collection("memories").unwrap();
        col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(col.len(), 1);

        // Getting the same collection returns the cached instance.
        let col2 = store.get_or_create_collection("memories").unwrap();
        assert_eq!(col2.len(), 1);
    }

    #[test]
    fn list_collections() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();

        store.get_or_create_collection("alpha").unwrap();
        store.get_or_create_collection("beta").unwrap();

        let mut names = store.collections();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn unload_collection() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();

        store.get_or_create_collection("temp").unwrap();
        assert_eq!(store.collections().len(), 1);

        store.unload_collection("temp").unwrap();
        assert!(store.collections().is_empty());
    }

    #[test]
    fn unload_nonexistent() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        assert!(store.unload_collection("nope").is_err());
    }

    #[test]
    fn get_collection_not_found() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        assert!(store.get_collection("nope").is_err());
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = TempDir::new().unwrap();

        // Session 1.
        {
            let store = VectorStore::open(dir.path(), cfg()).unwrap();
            let col = store.get_or_create_collection("memories").unwrap();
            col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0], None)
                .unwrap();
            col.insert("b".into(), "world".into(), vec![0.0, 1.0, 0.0], None)
                .unwrap();
            store.flush_all().unwrap();
        }

        // Session 2.
        {
            let store = VectorStore::open(dir.path(), cfg()).unwrap();
            assert_eq!(store.collections(), vec!["memories"]);

            let col = store.get_collection("memories").unwrap();
            assert_eq!(col.len(), 2);

            let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
            assert_eq!(results[0].id, "a");
        }
    }

    #[test]
    fn full_lifecycle() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();

        // Create collection, insert, search.
        let col = store.get_or_create_collection("test").unwrap();
        col.insert("x".into(), "ex".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("y".into(), "why".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.insert("z".into(), "zee".into(), vec![0.0, 0.0, 1.0], None)
            .unwrap();

        // Delete.
        col.delete("y").unwrap();
        assert_eq!(col.len(), 2);

        // Search.
        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!results.iter().any(|r| r.id == "y"));

        // Compact.
        col.compact().unwrap();
        assert_eq!(col.len(), 2);

        // Flush.
        store.flush_all().unwrap();
    }

    #[test]
    fn hybrid_search_lifecycle() {
        use crate::filter::Filter;
        use serde_json::json;

        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        let col = store.get_or_create_collection("hybrid").unwrap();

        // ── Phase 1: Insert 10 docs with varied content + embeddings + metadata ──
        let docs = [
            (
                "d0",
                "gateway service running on port 8080",
                [1.0, 0.0, 0.0],
                "infra",
            ),
            (
                "d1",
                "database connection pool exhausted",
                [0.0, 1.0, 0.0],
                "infra",
            ),
            (
                "d2",
                "gateway timeout after 30 seconds",
                [0.9, 0.1, 0.0],
                "infra",
            ),
            (
                "d3",
                "user authentication failed for admin",
                [0.0, 0.0, 1.0],
                "auth",
            ),
            (
                "d4",
                "session token expired and gateway rejected request",
                [0.8, 0.1, 0.1],
                "auth",
            ),
            (
                "d5",
                "memory usage exceeded threshold on gateway node",
                [0.7, 0.2, 0.1],
                "infra",
            ),
            (
                "d6",
                "ssl certificate renewal pending",
                [0.1, 0.1, 0.8],
                "security",
            ),
            (
                "d7",
                "load balancer health check failed",
                [0.5, 0.5, 0.0],
                "infra",
            ),
            (
                "d8",
                "gateway dns resolution error",
                [0.6, 0.3, 0.1],
                "infra",
            ),
            (
                "d9",
                "disk space running low on primary node",
                [0.2, 0.7, 0.1],
                "infra",
            ),
        ];

        for (id, content, emb, cat) in &docs {
            col.insert(
                id.to_string(),
                content.to_string(),
                emb.to_vec(),
                Some(json!({ "category": cat })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 10);

        // ── Phase 2: Text search for "gateway service" ──
        let results = col.search_text("gateway service", 10, None).unwrap();
        assert!(!results.is_empty());
        // All results should contain "gateway" or "service".
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"d0")); // has both terms

        // ── Phase 3: Hybrid search ──
        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway service", 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // ── Phase 4: Delete docs, verify exclusion ──
        col.delete("d0").unwrap();
        col.delete("d2").unwrap();

        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(!results.iter().any(|r| r.id == "d0"));
        assert!(!results.iter().any(|r| r.id == "d2"));

        let results = col
            .search_hybrid(&[1.0, 0.0, 0.0], "gateway", 10, None, None)
            .unwrap();
        assert!(!results.iter().any(|r| r.id == "d0"));
        assert!(!results.iter().any(|r| r.id == "d2"));

        // ── Phase 5: Compact, text search still works ──
        col.compact().unwrap();
        let results = col.search_text("gateway", 10, None).unwrap();
        assert!(!results.is_empty());
        assert!(!results.iter().any(|r| r.id == "d0"));

        // ── Phase 6: Flush, reopen, text search works (recovery) ──
        store.flush_all().unwrap();
        drop(store);

        let store2 = VectorStore::open(dir.path(), cfg()).unwrap();
        let col2 = store2.get_collection("hybrid").unwrap();
        assert_eq!(col2.len(), 8);

        let results = col2.search_text("gateway", 10, None).unwrap();
        assert!(!results.is_empty());
        assert!(!results.iter().any(|r| r.id == "d0"));
        assert!(!results.iter().any(|r| r.id == "d2"));

        // Text search with filter.
        let filter = Filter::eq("category", "auth");
        let results = col2.search_text("gateway", 10, Some(&filter)).unwrap();
        // Only d4 has "gateway" AND category=auth.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "d4");
    }

    #[test]
    fn integration_full_lifecycle() {
        use crate::filter::Filter;
        use serde_json::json;

        let dir = TempDir::new().unwrap();

        // ── Phase 1: Create, bulk insert 12 docs with metadata ─────
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        let col = store.get_or_create_collection("docs").unwrap();

        let categories = ["rust", "python", "go"];
        for i in 0..12 {
            let cat = categories[i % 3];
            let mut emb = vec![0.0f32; 3];
            emb[i % 3] = 1.0;
            emb[(i + 1) % 3] = (i as f32) * 0.01;

            col.insert(
                format!("doc-{i}"),
                format!("content-{i}"),
                emb,
                Some(json!({ "lang": cat, "idx": i })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 12);

        // ── Phase 2: Unfiltered search ─────────────────────────────
        let results = col.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);

        // ── Phase 3: Filtered search (lang == "rust") ──────────────
        let filter = Filter::eq("lang", "rust");
        let results = col.search(&[1.0, 0.0, 0.0], 10, Some(&filter)).unwrap();
        assert!(results.iter().all(|r| {
            r.metadata
                .as_ref()
                .and_then(|m| m.get("lang"))
                .and_then(|v| v.as_str())
                == Some("rust")
        }));
        assert_eq!(results.len(), 4); // docs 0, 3, 6, 9

        // ── Phase 4: Delete 3 docs, verify exclusion ───────────────
        col.delete("doc-0").unwrap();
        col.delete("doc-1").unwrap();
        col.delete("doc-2").unwrap();
        assert_eq!(col.len(), 9);

        let results = col.search(&[1.0, 0.0, 0.0], 12, None).unwrap();
        assert!(!results.iter().any(|r| r.id == "doc-0"));
        assert!(!results.iter().any(|r| r.id == "doc-1"));
        assert!(!results.iter().any(|r| r.id == "doc-2"));

        // ── Phase 5: Compact, verify counts ────────────────────────
        col.compact().unwrap();
        assert_eq!(col.len(), 9);

        // ── Phase 6: get() for live and deleted docs ───────────────
        let live = col.get("doc-3").unwrap();
        assert!(live.is_some());
        assert_eq!(live.unwrap().content, "content-3");

        let deleted = col.get("doc-0").unwrap();
        assert!(deleted.is_none());

        // ── Phase 7: Flush, drop, reopen ───────────────────────────
        store.flush_all().unwrap();
        drop(store);

        let store2 = VectorStore::open(dir.path(), cfg()).unwrap();
        assert!(store2.collections().contains(&"docs".to_string()));

        let col2 = store2.get_collection("docs").unwrap();
        assert_eq!(col2.len(), 9);

        // Deleted docs still gone after reopen.
        assert!(col2.get("doc-0").unwrap().is_none());
        assert!(col2.get("doc-1").unwrap().is_none());
        assert!(col2.get("doc-2").unwrap().is_none());

        // Content and metadata correct after reopen.
        let doc5 = col2.get("doc-5").unwrap().unwrap();
        assert_eq!(doc5.content, "content-5");
        assert_eq!(doc5.metadata.as_ref().unwrap()["lang"], "go");

        // Search still works.
        let results = col2.search(&[0.0, 1.0, 0.0], 2, None).unwrap();
        assert!(!results.is_empty());
    }

    // ── P2.3 + P3.1: Lifecycle and access API tests ─────────────────

    #[test]
    fn create_collection_errors_if_exists() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        store.create_collection("alpha").unwrap();
        assert!(store.create_collection("alpha").is_err());
    }

    #[test]
    fn delete_collection_removes_files() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        let col = store.get_or_create_collection("temp").unwrap();
        col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.flush().unwrap();

        store.delete_collection("temp").unwrap();
        assert!(store.collections().is_empty());

        let col_path = dir.path().join("collections").join("temp");
        assert!(!col_path.exists());
    }

    #[test]
    fn unload_collection_leaves_files() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        let col = store.get_or_create_collection("temp").unwrap();
        col.insert("a".into(), "hello".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.flush().unwrap();

        store.unload_collection("temp").unwrap();
        assert!(store.collections().is_empty());

        let col_path = dir.path().join("collections").join("temp");
        assert!(col_path.exists());
    }

    #[test]
    fn delete_nonexistent_collection() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();
        assert!(store.delete_collection("nope").is_err());
    }

    #[test]
    fn rejects_path_traversal_collection_names() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path(), cfg()).unwrap();

        let invalid = [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "../outside",
            "..\\outside",
            "/absolute",
        ];

        for name in invalid {
            match store.create_collection(name) {
                Err(DbError::Validation(_)) => {}
                Err(other) => panic!("expected Validation for name '{name}', got {other:?}"),
                Ok(_) => panic!("expected error for invalid collection name '{name}'"),
            }
        }
    }
}
