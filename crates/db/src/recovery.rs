//! Recovery: rebuild index state from checkpoint + WAL replay.

use crate::id_map::IdMap;
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
}

/// Load a checkpoint (if any) and replay the WAL to produce a fully recovered state.
pub fn recover(storage: &IncrementalStorage, config: &DbConfig) -> Result<RecoveredState> {
    let mut index = HNSWIndex::new(config.embedding_dim, config.hnsw.clone());
    let mut id_map = IdMap::new();
    let mut documents: Vec<Document> = Vec::new();

    // Phase 1: Load checkpoint.
    if let Some((checkpoint_docs, meta)) =
        storage.load_checkpoint::<Vec<Document>>().map_err(DbError::Core)?
    {
        info!(
            checkpoint_id = meta.id,
            doc_count = checkpoint_docs.len(),
            "loaded checkpoint"
        );

        for doc in &checkpoint_docs {
            index.add(doc.clone()).map_err(DbError::Core)?;
            id_map.insert(doc.id.clone());
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

    Ok(RecoveredState {
        index,
        id_map,
        documents,
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
        let storage =
            IncrementalStorage::new(dir.path(), IncrementalConfig::default()).unwrap();
        let config = test_config(4);

        let state = recover(&storage, &config).unwrap();
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

        let state = recover(&storage, &config).unwrap();
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

        let state = recover(&storage, &config).unwrap();
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

        let state = recover(&storage, &config).unwrap();
        assert_eq!(state.id_map.live_count(), 1);
        assert!(state.id_map.get("b").is_some());
        assert!(state.id_map.get("a").is_none());
    }
}
