//! IndexedDB persistence for RAG indices
//!
//! This module provides IndexedDB storage capabilities for persisting RAG indices
//! in the browser. It supports saving and loading both Flat and HNSW indices with
//! all their documents and metadata.
//!
//! # Usage
//!
//! ```javascript
//! import init, { IndexedDBStore } from './foxstash_wasm.js';
//!
//! await init();
//!
//! // Create a store
//! const store = new IndexedDBStore();
//!
//! // Save data
//! const data = { /* your serialized index */ };
//! await store.save("my-index", data);
//!
//! // Load data
//! const loaded = await store.load("my-index");
//!
//! // List all keys
//! const keys = await store.list_keys();
//!
//! // Delete an index
//! await store.delete("my-index");
//!
//! // Clear all data
//! await store.clear();
//! ```

use foxstash_core::Document;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    IdbDatabase, IdbOpenDbRequest, IdbTransaction,
    IdbTransactionMode, IdbVersionChangeEvent,
};

const DB_NAME: &str = "foxstash-db";
const STORE_NAME: &str = "indices";
const DB_VERSION: u32 = 1;

/// Serializable representation of a Flat index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedFlatIndex {
    pub embedding_dim: usize,
    pub documents: Vec<Document>,
}

/// Serializable representation of HNSW connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedHNSWNode {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
    pub connections: Vec<Vec<usize>>, // Layer -> list of neighbor indices
}

/// Serializable representation of HNSW configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedHNSWConfig {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f32,
}

/// Serializable representation of an HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedHNSWIndex {
    pub embedding_dim: usize,
    pub config: SerializedHNSWConfig,
    pub nodes: Vec<SerializedHNSWNode>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
}

/// Unified serializable index format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SerializedIndex {
    Flat(SerializedFlatIndex),
    Hnsw(SerializedHNSWIndex),
}

/// Metadata stored alongside the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub name: String,
    pub created_at: f64,
    pub updated_at: f64,
    pub document_count: usize,
    pub index_type: String,
}

/// Complete persisted index with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedIndex {
    pub metadata: IndexMetadata,
    pub index: SerializedIndex,
}

/// IndexedDB store manager for RAG indices
///
/// Provides async methods to save, load, delete, and list RAG indices
/// in the browser's IndexedDB storage.
///
/// # Example
///
/// ```javascript
/// const store = new IndexedDBStore();
///
/// // Save an index
/// await store.save("my-index", indexData);
///
/// // Load it back
/// const data = await store.load("my-index");
///
/// // List all saved indices
/// const keys = await store.list_keys();
/// console.log("Saved indices:", keys);
/// ```
#[wasm_bindgen]
pub struct IndexedDBStore {
    db_name: String,
    store_name: String,
}

#[wasm_bindgen]
impl IndexedDBStore {
    /// Creates a new IndexedDB store
    ///
    /// # Arguments
    /// * `db_name` - Optional custom database name (defaults to "foxstash-db")
    ///
    /// # Returns
    /// A new IndexedDBStore instance
    ///
    /// # Example
    ///
    /// ```javascript
    /// // Use default database name
    /// const store1 = new IndexedDBStore();
    ///
    /// // Use custom database name
    /// const store2 = new IndexedDBStore("my-custom-db");
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: Option<String>) -> Self {
        Self {
            db_name: db_name.unwrap_or_else(|| DB_NAME.to_string()),
            store_name: STORE_NAME.to_string(),
        }
    }

    /// Saves an index to IndexedDB
    ///
    /// # Arguments
    /// * `key` - Unique identifier for this index
    /// * `data` - Serialized index data as JsValue
    ///
    /// # Returns
    /// Promise that resolves when save is complete
    ///
    /// # Errors
    /// Returns JsValue error if:
    /// - Database cannot be opened
    /// - Data cannot be serialized
    /// - Write transaction fails
    /// - Quota is exceeded
    ///
    /// # Example
    ///
    /// ```javascript
    /// try {
    ///   await store.save("my-index", indexData);
    ///   console.log("Saved successfully");
    /// } catch (error) {
    ///   console.error("Save failed:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub async fn save(&self, key: &str, data: JsValue) -> Result<(), JsValue> {
        let db = self.open_db().await?;

        // Create a readwrite transaction
        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create transaction: {:?}", e))
            })?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get object store: {:?}", e)))?;

        // Put the data with the key
        let request = store
            .put_with_key(&data, &JsValue::from_str(key))
            .map_err(|e| JsValue::from_str(&format!("Failed to put data: {:?}", e)))?;

        // Wait for the request to complete
        JsFuture::from(request.unchecked_into::<js_sys::Promise>()).await?;

        // Wait for transaction to complete
        Self::wait_for_transaction(&transaction).await?;

        Ok(())
    }

    /// Loads an index from IndexedDB
    ///
    /// # Arguments
    /// * `key` - Unique identifier of the index to load
    ///
    /// # Returns
    /// Promise that resolves to the serialized index data
    ///
    /// # Errors
    /// Returns JsValue error if:
    /// - Database cannot be opened
    /// - Key does not exist
    /// - Read transaction fails
    ///
    /// # Example
    ///
    /// ```javascript
    /// try {
    ///   const data = await store.load("my-index");
    ///   console.log("Loaded:", data);
    /// } catch (error) {
    ///   console.error("Load failed:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub async fn load(&self, key: &str) -> Result<JsValue, JsValue> {
        let db = self.open_db().await?;

        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readonly)
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create transaction: {:?}", e))
            })?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .get(&JsValue::from_str(key))
            .map_err(|e| JsValue::from_str(&format!("Failed to get data: {:?}", e)))?;

        let result = JsFuture::from(request.unchecked_into::<js_sys::Promise>()).await?;

        if result.is_undefined() {
            return Err(JsValue::from_str(&format!("Index '{}' not found", key)));
        }

        Ok(result)
    }

    /// Deletes an index from IndexedDB
    ///
    /// # Arguments
    /// * `key` - Unique identifier of the index to delete
    ///
    /// # Returns
    /// Promise that resolves when deletion is complete
    ///
    /// # Errors
    /// Returns JsValue error if delete operation fails
    ///
    /// # Example
    ///
    /// ```javascript
    /// try {
    ///   await store.delete("my-index");
    ///   console.log("Deleted successfully");
    /// } catch (error) {
    ///   console.error("Delete failed:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub async fn delete(&self, key: &str) -> Result<(), JsValue> {
        let db = self.open_db().await?;

        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create transaction: {:?}", e))
            })?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .delete(&JsValue::from_str(key))
            .map_err(|e| JsValue::from_str(&format!("Failed to delete data: {:?}", e)))?;

        JsFuture::from(request.unchecked_into::<js_sys::Promise>()).await?;
        Self::wait_for_transaction(&transaction).await?;

        Ok(())
    }

    /// Lists all keys in the IndexedDB store
    ///
    /// # Returns
    /// Promise that resolves to an array of key names
    ///
    /// # Errors
    /// Returns JsValue error if operation fails
    ///
    /// # Example
    ///
    /// ```javascript
    /// const keys = await store.list_keys();
    /// console.log("Available indices:", keys);
    /// ```
    #[wasm_bindgen]
    pub async fn list_keys(&self) -> Result<Vec<String>, JsValue> {
        let db = self.open_db().await?;

        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readonly)
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create transaction: {:?}", e))
            })?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get object store: {:?}", e)))?;

        // Get all keys
        let request = store
            .get_all_keys()
            .map_err(|e| JsValue::from_str(&format!("Failed to get keys: {:?}", e)))?;

        let result = JsFuture::from(request.unchecked_into::<js_sys::Promise>()).await?;

        // Convert DomStringList to Vec<String>
        let keys = js_sys::Array::from(&result);
        let mut key_list = Vec::new();

        for i in 0..keys.length() {
            if let Some(key) = keys.get(i).as_string() {
                key_list.push(key);
            }
        }

        Ok(key_list)
    }

    /// Clears all indices from the IndexedDB store
    ///
    /// # Returns
    /// Promise that resolves when all data is cleared
    ///
    /// # Errors
    /// Returns JsValue error if clear operation fails
    ///
    /// # Example
    ///
    /// ```javascript
    /// try {
    ///   await store.clear();
    ///   console.log("All data cleared");
    /// } catch (error) {
    ///   console.error("Clear failed:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub async fn clear(&self) -> Result<(), JsValue> {
        let db = self.open_db().await?;

        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| {
                JsValue::from_str(&format!("Failed to create transaction: {:?}", e))
            })?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .clear()
            .map_err(|e| JsValue::from_str(&format!("Failed to clear store: {:?}", e)))?;

        JsFuture::from(request.unchecked_into::<js_sys::Promise>()).await?;
        Self::wait_for_transaction(&transaction).await?;

        Ok(())
    }
}

impl IndexedDBStore {
    /// Opens or creates the IndexedDB database
    async fn open_db(&self) -> Result<IdbDatabase, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        let idb_factory = window
            .indexed_db()
            .map_err(|e| JsValue::from_str(&format!("IndexedDB not supported: {:?}", e)))?
            .ok_or_else(|| JsValue::from_str("IndexedDB not available"))?;

        let open_request = idb_factory
            .open_with_u32(&self.db_name, DB_VERSION)
            .map_err(|e| JsValue::from_str(&format!("Failed to open database: {:?}", e)))?;

        // Setup upgrade handler
        let store_name = self.store_name.clone();
        let onupgradeneeded = Closure::once(move |event: IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let db = request.result().unwrap().dyn_into::<IdbDatabase>().unwrap();

            // Create object store if it doesn't exist
            if !db.object_store_names().contains(&store_name) {
                let _ = db.create_object_store(&store_name);
            }
        });

        open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
        onupgradeneeded.forget();

        // Wait for database to open
        let db_value = JsFuture::from(open_request.unchecked_into::<js_sys::Promise>()).await?;
        let db: IdbDatabase = db_value
            .dyn_into()
            .map_err(|_| JsValue::from_str("Failed to cast to IdbDatabase"))?;

        Ok(db)
    }

    /// Waits for a transaction to complete
    async fn wait_for_transaction(transaction: &IdbTransaction) -> Result<(), JsValue> {
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let oncomplete = Closure::once(move || {
                resolve.call0(&JsValue::NULL).unwrap();
            });

            let onerror = Closure::once(move || {
                reject.call0(&JsValue::NULL).unwrap();
            });

            transaction.set_oncomplete(Some(oncomplete.as_ref().unchecked_ref()));
            transaction.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            oncomplete.forget();
            onerror.forget();
        });

        JsFuture::from(promise).await?;
        Ok(())
    }
}

// ============================================================================
// Helper Functions for Serialization
// ============================================================================

/// Creates a PersistedIndex with metadata
pub fn create_persisted_index(
    name: String,
    index: SerializedIndex,
    document_count: usize,
) -> PersistedIndex {
    let now = js_sys::Date::now();
    let index_type = match &index {
        SerializedIndex::Flat(_) => "flat".to_string(),
        SerializedIndex::Hnsw(_) => "hnsw".to_string(),
    };

    PersistedIndex {
        metadata: IndexMetadata {
            name,
            created_at: now,
            updated_at: now,
            document_count,
            index_type,
        },
        index,
    }
}

/// Serializes a PersistedIndex to JsValue
pub fn to_js_value(persisted: &PersistedIndex) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(persisted)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize index: {:?}", e)))
}

/// Deserializes a JsValue to PersistedIndex
pub fn from_js_value(value: JsValue) -> Result<PersistedIndex, JsValue> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize index: {:?}", e)))
}

/// Creates a SerializedIndex from a flat index's documents
pub fn create_flat_index(embedding_dim: usize, documents: Vec<Document>) -> SerializedIndex {
    SerializedIndex::Flat(SerializedFlatIndex {
        embedding_dim,
        documents,
    })
}

/// Creates a SerializedIndex from HNSW index components
pub fn create_hnsw_index(
    embedding_dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f32,
    nodes: Vec<SerializedHNSWNode>,
    entry_point: Option<usize>,
    max_layer: usize,
) -> SerializedIndex {
    SerializedIndex::Hnsw(SerializedHNSWIndex {
        embedding_dim,
        config: SerializedHNSWConfig {
            m,
            m0,
            ef_construction,
            ef_search,
            ml,
        },
        nodes,
        entry_point,
        max_layer,
    })
}

/// Serializes a FlatIndex to SerializedFlatIndex
pub fn serialize_flat_index(
    index: &foxstash_core::index::FlatIndex,
    embedding_dim: usize,
) -> Result<SerializedFlatIndex, String> {
    Ok(SerializedFlatIndex {
        embedding_dim,
        documents: index.get_all_documents(),
    })
}

/// Serializes an HNSWIndex to SerializedHNSWIndex
pub fn serialize_hnsw_index(
    index: &foxstash_core::index::HNSWIndex,
    _embedding_dim: usize,
) -> Result<SerializedHNSWIndex, String> {
    // Get all documents from the index
    let documents = index.get_all_documents();

    // Create serialized nodes from documents
    // Note: We don't serialize the graph connections because we'll rebuild them on deserialization
    let nodes: Vec<SerializedHNSWNode> = documents
        .into_iter()
        .map(|doc| SerializedHNSWNode {
            id: doc.id,
            content: doc.content,
            embedding: doc.embedding,
            metadata: doc.metadata,
            connections: Vec::new(), // Will be rebuilt on deserialization
        })
        .collect();

    let config = index.config();

    Ok(SerializedHNSWIndex {
        embedding_dim: index.embedding_dim(),
        config: SerializedHNSWConfig {
            m: config.m,
            m0: config.m0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            ml: config.ml,
        },
        nodes,
        entry_point: index.entry_point(),
        max_layer: index.max_layer(),
    })
}

/// Deserializes a SerializedFlatIndex to FlatIndex
pub fn deserialize_flat_index(data: SerializedFlatIndex) -> Result<foxstash_core::index::FlatIndex, String> {
    let mut index = foxstash_core::index::FlatIndex::new(data.embedding_dim);

    for doc in data.documents {
        index.add(doc).map_err(|e| e.to_string())?;
    }

    Ok(index)
}

/// Deserializes a SerializedHNSWIndex to HNSWIndex
pub fn deserialize_hnsw_index(data: SerializedHNSWIndex) -> Result<foxstash_core::index::HNSWIndex, String> {
    use foxstash_core::index::HNSWConfig;
    use foxstash_core::Document;

    let config = HNSWConfig {
        m: data.config.m,
        m0: data.config.m0,
        ef_construction: data.config.ef_construction,
        ef_search: data.config.ef_search,
        ml: data.config.ml,
    };

    let mut index = foxstash_core::index::HNSWIndex::new(data.embedding_dim, config);

    // Re-add all documents from nodes
    // This rebuilds the HNSW graph rather than restoring it exactly
    for node in &data.nodes {
        let doc = Document {
            id: node.id.clone(),
            content: node.content.clone(),
            embedding: node.embedding.clone(),
            metadata: node.metadata.clone(),
        };
        index.add(doc).map_err(|e| e.to_string())?;
    }

    Ok(index)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_flat_index() {
        let documents = vec![Document {
            id: "doc1".to_string(),
            content: "Test content".to_string(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: None,
        }];

        let serialized = create_flat_index(3, documents);

        match serialized {
            SerializedIndex::Flat(flat) => {
                assert_eq!(flat.embedding_dim, 3);
                assert_eq!(flat.documents.len(), 1);
                assert_eq!(flat.documents[0].id, "doc1");
            }
            _ => panic!("Expected Flat index"),
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_create_persisted_index() {
        let documents = vec![];
        let serialized = create_flat_index(384, documents);

        let persisted = create_persisted_index("test-index".to_string(), serialized, 0);

        assert_eq!(persisted.metadata.name, "test-index");
        assert_eq!(persisted.metadata.document_count, 0);
        assert_eq!(persisted.metadata.index_type, "flat");
        assert!(persisted.metadata.created_at > 0.0);
    }

    #[test]
    fn test_serialized_index_json_roundtrip() {
        let documents = vec![];
        let serialized = create_flat_index(384, documents);

        let json = serde_json::to_string(&serialized).unwrap();
        let deserialized: SerializedIndex = serde_json::from_str(&json).unwrap();

        match deserialized {
            SerializedIndex::Flat(flat) => {
                assert_eq!(flat.embedding_dim, 384);
                assert_eq!(flat.documents.len(), 0);
            }
            _ => panic!("Expected Flat index"),
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_persisted_index_json_roundtrip() {
        let documents = vec![
            Document {
                id: "doc1".to_string(),
                content: "Content 1".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
                metadata: Some(serde_json::json!({"key": "value"})),
            },
            Document {
                id: "doc2".to_string(),
                content: "Content 2".to_string(),
                embedding: vec![0.0, 1.0, 0.0],
                metadata: None,
            },
        ];

        let serialized = create_flat_index(3, documents);
        let persisted = create_persisted_index("test".to_string(), serialized, 2);

        let json = serde_json::to_string(&persisted).unwrap();
        let deserialized: PersistedIndex = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.metadata.name, "test");
        assert_eq!(deserialized.metadata.document_count, 2);

        match deserialized.index {
            SerializedIndex::Flat(flat) => {
                assert_eq!(flat.documents.len(), 2);
                assert_eq!(flat.documents[0].id, "doc1");
                assert_eq!(flat.documents[1].id, "doc2");
            }
            _ => panic!("Expected Flat index"),
        }
    }
}

// WASM-specific tests
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_indexeddb_store_creation() {
        let store = IndexedDBStore::new(Some("test-db".to_string()));
        assert_eq!(store.db_name, "test-db");
        assert_eq!(store.store_name, STORE_NAME);
    }

    #[wasm_bindgen_test]
    fn test_indexeddb_store_default() {
        let store = IndexedDBStore::new(None);
        assert_eq!(store.db_name, DB_NAME);
    }

    #[wasm_bindgen_test]
    async fn test_save_and_load() {
        let store = IndexedDBStore::new(Some("test-save-load".to_string()));

        let documents = vec![];
        let serialized = create_flat_index(384, documents);
        let persisted = create_persisted_index("test".to_string(), serialized, 0);
        let js_value = to_js_value(&persisted).unwrap();

        // Save
        store.save("test-key", js_value).await.unwrap();

        // Load
        let loaded_value = store.load("test-key").await.unwrap();
        let loaded: PersistedIndex = from_js_value(loaded_value).unwrap();

        assert_eq!(loaded.metadata.name, "test");
    }

    #[wasm_bindgen_test]
    async fn test_delete() {
        let store = IndexedDBStore::new(Some("test-delete".to_string()));

        let documents = vec![];
        let serialized = create_flat_index(384, documents);
        let persisted = create_persisted_index("test".to_string(), serialized, 0);
        let js_value = to_js_value(&persisted).unwrap();

        // Save
        store.save("delete-key", js_value).await.unwrap();

        // Delete
        store.delete("delete-key").await.unwrap();

        // Try to load - should fail
        let result = store.load("delete-key").await;
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    async fn test_list_keys() {
        let store = IndexedDBStore::new(Some("test-list".to_string()));

        // Clear any existing data
        store.clear().await.unwrap();

        let documents = vec![];
        let serialized = create_flat_index(384, documents);

        // Save multiple indices
        for i in 1..=3 {
            let persisted = create_persisted_index(format!("test-{}", i), serialized.clone(), 0);
            let js_value = to_js_value(&persisted).unwrap();
            store
                .save(&format!("key-{}", i), js_value)
                .await
                .unwrap();
        }

        let keys = store.list_keys().await.unwrap();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"key-1".to_string()));
        assert!(keys.contains(&"key-2".to_string()));
        assert!(keys.contains(&"key-3".to_string()));
    }

    #[wasm_bindgen_test]
    async fn test_clear() {
        let store = IndexedDBStore::new(Some("test-clear".to_string()));

        let documents = vec![];
        let serialized = create_flat_index(384, documents);
        let persisted = create_persisted_index("test".to_string(), serialized, 0);
        let js_value = to_js_value(&persisted).unwrap();

        // Save some data
        store.save("clear-key", js_value).await.unwrap();

        // Clear all
        store.clear().await.unwrap();

        // Check it's gone
        let keys = store.list_keys().await.unwrap();
        assert_eq!(keys.len(), 0);
    }
}
