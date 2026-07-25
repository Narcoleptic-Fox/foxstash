//! Recovery: rebuild index state from checkpoint + WAL replay.

use crate::id_map::IdMap;
use crate::inverted_index::InvertedIndex;
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::{DbConfig, DbError, Result};
use foxstash_core::index::HNSWIndex;
use foxstash_core::storage::incremental::{IncrementalStorage, RecoveryHelper, WalOperation};
use foxstash_core::Document;
use tracing::{debug, info};

/// State recovered from storage.
pub struct RecoveredState {
    pub index: HNSWIndex,
    pub id_map: IdMap,
    pub documents: Vec<Document>,
    pub text_index: InvertedIndex,
}

/// Re-insert every document to reconstruct the graph. The slow path, used when no
/// usable snapshot exists.
fn rebuild_index(index: &mut HNSWIndex, docs: &[Document]) -> Result<()> {
    for doc in docs {
        index.add(doc.clone()).map_err(DbError::Core)?;
    }
    Ok(())
}

/// Load a checkpoint (if any) and replay the WAL to produce a fully recovered state.
pub fn recover(
    storage: &IncrementalStorage,
    config: &DbConfig,
    base_path: &std::path::Path,
) -> Result<RecoveredState> {
    let mut index = HNSWIndex::new(config.embedding_dim, config.hnsw.clone());
    let mut id_map = IdMap::new();
    let mut documents: Vec<Document> = Vec::new();

    // Phase 1: Load checkpoint.
    if let Some((checkpoint_docs, meta)) = storage
        .load_checkpoint::<Vec<Document>>()
        .map_err(DbError::Core)?
    {
        info!(
            checkpoint_id = meta.id,
            doc_count = checkpoint_docs.len(),
            "loaded checkpoint"
        );

        if meta.embedding_dim != config.embedding_dim {
            return Err(DbError::DimensionMismatch {
                expected: config.embedding_dim,
                actual: meta.embedding_dim,
            });
        }

        for doc in &checkpoint_docs {
            id_map.insert(doc.id.clone());
        }

        // Prefer the graph snapshot written alongside this checkpoint. Loading it
        // is a read; rebuilding is a full HNSW construction — measured at 88% of
        // the entire open, because re-inserting one document at a time is core's
        // slowest entry point.
        //
        // The snapshot is a **same-version cache** by core's design: it refuses to
        // load if written by another build, rather than misreading. Every failure
        // mode — absent, orphaned by a crash between the two writes, or written by
        // a different version — lands in the same fallback, so correctness never
        // depends on it being there.
        let snapshot_path = base_path.join(format!("graph_{:05}.snapshot", meta.id));
        match HNSWIndex::snapshot_from_file(&snapshot_path) {
            Ok(loaded) if loaded.len() == checkpoint_docs.len() => {
                info!(
                    docs = checkpoint_docs.len(),
                    "graph loaded from snapshot; skipping rebuild"
                );
                index = loaded;
            }
            Ok(loaded) => {
                // Present but disagrees with the checkpoint — treat as stale.
                debug!(
                    snapshot_docs = loaded.len(),
                    checkpoint_docs = checkpoint_docs.len(),
                    "graph snapshot disagrees with checkpoint; rebuilding"
                );
                rebuild_index(&mut index, &checkpoint_docs)?;
            }
            Err(err) => {
                debug!(?err, "no usable graph snapshot; rebuilding");
                rebuild_index(&mut index, &checkpoint_docs)?;
            }
        }

        documents = checkpoint_docs;
    }

    // Phase 2: Replay WAL entries after the checkpoint.
    let helper = RecoveryHelper::new(storage);
    let replayed = helper.replay_wal(|op| match op {
        WalOperation::Add(doc) => {
            // Tombstone old position if re-adding (no-op if absent).
            id_map.remove(&doc.id);
            index.add(doc.clone())?;
            id_map.insert(doc.id.clone());
            documents.push(doc.clone());
            Ok(())
        }
        WalOperation::Remove(id) => {
            id_map.remove(id);
            Ok(())
        }
        WalOperation::Clear => {
            index.clear();
            id_map.clear();
            documents.clear();
            Ok(())
        }
        WalOperation::Checkpoint { .. } => Ok(()),
    })?;

    debug!(replayed, "WAL entries replayed");

    // Phase 3: Rebuild inverted index from live documents.
    let tokenizer = SimpleTokenizer::new();
    let mut text_index = InvertedIndex::with_config(config.bm25.clone());
    for id in id_map.live_ids() {
        let pos = id_map.get(id).ok_or_else(|| {
            DbError::Recovery(format!("live ID '{id}' has no position in id_map"))
        })?;
        let doc = documents.get(pos).ok_or_else(|| {
            DbError::Recovery(format!(
                "position {pos} out of bounds (len={})",
                documents.len()
            ))
        })?;
        let tokens = tokenizer.tokenize(&doc.content);
        text_index.add(pos, &tokens);
    }

    debug!(text_docs = text_index.len(), "text index rebuilt");

    Ok(RecoveredState {
        index,
        id_map,
        documents,
        text_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use foxstash_core::storage::incremental::{IncrementalConfig, IndexMetadata};
    use tempfile::TempDir;

    fn test_doc(id: &str, dim: usize) -> Document {
        Document {
            id: id.into(),
            content: format!("content-{id}"),
            embedding: vec![0.1; dim],
            metadata: None,
        }
    }

    fn test_config(dim: usize) -> DbConfig {
        DbConfig::default().with_embedding_dim(dim)
    }

    #[test]
    fn recover_empty_storage() {
        let dir = TempDir::new().unwrap();
        let storage = IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        let state = recover(&storage, &config, dir.path()).unwrap();
        assert_eq!(state.index.len(), 0);
        assert_eq!(state.id_map.live_count(), 0);
    }

    #[test]
    fn recover_wal_only() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        storage.log_add(&test_doc("a", 4)).unwrap();
        storage.log_add(&test_doc("b", 4)).unwrap();
        storage.log_remove("a").unwrap();
        storage.sync().unwrap();

        let state = recover(&storage, &config, dir.path()).unwrap();
        assert_eq!(state.index.len(), 2); // HNSW still has both nodes
        assert_eq!(state.id_map.live_count(), 1); // only "b" is live
        assert!(state.id_map.is_tombstoned("a"));
    }

    #[test]
    fn recover_checkpoint_plus_wal() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        // Add docs and checkpoint.
        storage.log_add(&test_doc("a", 4)).unwrap();
        storage.log_add(&test_doc("b", 4)).unwrap();

        let docs = vec![test_doc("a", 4), test_doc("b", 4)];
        storage
            .checkpoint(
                &docs,
                IndexMetadata {
                    document_count: 2,
                    embedding_dim: 4,
                    index_type: "hnsw".into(),
                },
            )
            .unwrap();

        // Post-checkpoint ops.
        storage.log_add(&test_doc("c", 4)).unwrap();
        storage.log_remove("a").unwrap();
        storage.sync().unwrap();

        let state = recover(&storage, &config, dir.path()).unwrap();
        assert_eq!(state.id_map.live_count(), 2); // b + c
        assert!(state.id_map.is_tombstoned("a"));
        assert!(state.id_map.get("b").is_some());
        assert!(state.id_map.get("c").is_some());
    }

    #[test]
    fn recover_with_clear_in_wal() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        storage.log_add(&test_doc("a", 4)).unwrap();
        storage.log_clear().unwrap();
        storage.log_add(&test_doc("b", 4)).unwrap();
        storage.sync().unwrap();

        let state = recover(&storage, &config, dir.path()).unwrap();
        assert_eq!(state.id_map.live_count(), 1);
        assert!(state.id_map.get("b").is_some());
        assert!(state.id_map.get("a").is_none());
    }

    #[test]
    fn recover_wal_remove_nonexistent_is_noop() {
        // Removing an ID that was never added should not panic during recovery.
        let dir = TempDir::new().unwrap();
        let mut storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        storage.log_add(&test_doc("a", 4)).unwrap();
        storage.log_remove("nonexistent").unwrap();
        storage.sync().unwrap();

        let state = recover(&storage, &config, dir.path()).unwrap();
        assert_eq!(state.id_map.live_count(), 1);
        assert!(state.id_map.get("a").is_some());
    }
}
