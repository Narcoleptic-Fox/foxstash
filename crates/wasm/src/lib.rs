//! WebAssembly bindings for Foxstash
//!
//! This crate provides JavaScript-friendly interfaces to the core RAG functionality,
//! enabling vector search and document management in browser environments.
//!
//! ## Usage
//!
//! ```javascript
//! import init, { LocalRAG, JsDocument } from './foxstash_wasm.js';
//!
//! await init();
//!
//! // Create a RAG instance
//! const rag = new LocalRAG(384, true); // 384-dim embeddings, use HNSW
//!
//! // Add a document
//! const doc = new JsDocument(
//!   "doc1",
//!   "Hello world",
//!   new Float32Array(384),
//!   null
//! );
//! rag.add_document(doc);
//!
//! // Search
//! const results = rag.search(new Float32Array(384), 5);
//! ```

use wasm_bindgen::prelude::*;
use foxstash_core::{Document, SearchResult, index::{FlatIndex, HNSWIndex}};

pub mod persistence;

/// Initialize the WASM module
///
/// This sets up panic hooks to provide better error messages in JavaScript.
/// Call this once when your module loads.
///
/// # Example
///
/// ```javascript
/// import init, { init_panic_hook } from './foxstash_wasm.js';
///
/// await init();
/// init_panic_hook();
/// ```
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// JavaScript-compatible document wrapper
///
/// Represents a document with its content, embedding vector, and optional metadata.
#[wasm_bindgen]
#[derive(Clone)]
pub struct JsDocument {
    id: String,
    content: String,
    embedding: Vec<f32>,
    metadata: Option<serde_json::Value>,
}

#[wasm_bindgen]
impl JsDocument {
    /// Create a new document
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the document
    /// * `content` - Text content of the document
    /// * `embedding` - Embedding vector as Float32Array or Array
    /// * `metadata` - Optional metadata as a JavaScript object (will be serialized to JSON)
    ///
    /// # Returns
    ///
    /// A new JsDocument instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding cannot be converted to a Vec<f32>
    /// - The metadata cannot be serialized to JSON
    ///
    /// # Example
    ///
    /// ```javascript
    /// const doc = new JsDocument(
    ///   "doc1",
    ///   "Hello world",
    ///   new Float32Array([0.1, 0.2, 0.3]),
    ///   { source: "test", timestamp: Date.now() }
    /// );
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        id: String,
        content: String,
        embedding: JsValue,
        metadata: JsValue,
    ) -> Result<JsDocument, JsValue> {
        // Convert embedding from JsValue to Vec<f32>
        let embedding: Vec<f32> = serde_wasm_bindgen::from_value(embedding)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse embedding: {}", e)))?;

        // Convert metadata from JsValue to Option<serde_json::Value>
        let metadata: Option<serde_json::Value> = if metadata.is_null() || metadata.is_undefined() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(metadata)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse metadata: {}", e)))?)
        };

        Ok(JsDocument {
            id,
            content,
            embedding,
            metadata,
        })
    }

    /// Get the document ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get the document content
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    /// Get the embedding vector
    ///
    /// Returns a JavaScript array of numbers
    #[wasm_bindgen(getter)]
    pub fn embedding(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.embedding)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize embedding: {}", e)))
    }

    /// Get the document metadata
    ///
    /// Returns a JavaScript object or null if no metadata
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsValue> {
        if let Some(ref meta) = self.metadata {
            serde_wasm_bindgen::to_value(meta)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize metadata: {}", e)))
        } else {
            Ok(JsValue::NULL)
        }
    }

    /// Get the embedding dimension
    #[wasm_bindgen]
    pub fn embedding_dim(&self) -> usize {
        self.embedding.len()
    }
}

impl JsDocument {
    /// Convert to core Document type
    fn to_core(&self) -> Document {
        Document {
            id: self.id.clone(),
            content: self.content.clone(),
            embedding: self.embedding.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// JavaScript-compatible search result wrapper
///
/// Represents a search result with the document content, similarity score, and metadata.
#[wasm_bindgen]
#[derive(Clone)]
pub struct JsSearchResult {
    id: String,
    content: String,
    score: f32,
    metadata: Option<serde_json::Value>,
}

#[wasm_bindgen]
impl JsSearchResult {
    /// Get the document ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get the document content
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    /// Get the similarity score
    ///
    /// Score is between -1 and 1 for cosine similarity,
    /// where 1 means identical and 0 means orthogonal.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Get the document metadata
    ///
    /// Returns a JavaScript object or null if no metadata
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsValue> {
        if let Some(ref meta) = self.metadata {
            serde_wasm_bindgen::to_value(meta)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize metadata: {}", e)))
        } else {
            Ok(JsValue::NULL)
        }
    }
}

impl JsSearchResult {
    /// Create from core SearchResult type
    fn from_core(result: SearchResult) -> Self {
        JsSearchResult {
            id: result.id,
            content: result.content,
            score: result.score,
            metadata: result.metadata,
        }
    }
}

/// Internal index type selector
enum IndexType {
    Flat(FlatIndex),
    HNSW(HNSWIndex),
}

/// Local RAG system for in-browser vector search
///
/// This is the main interface for managing documents and performing similarity search.
/// It wraps either a flat index (exact search) or HNSW index (approximate search)
/// depending on the configuration.
///
/// # Example
///
/// ```javascript
/// // Create with HNSW for better performance with large datasets
/// const rag = new LocalRAG(384, true);
///
/// // Add documents
/// const doc = new JsDocument("doc1", "content", embedding, null);
/// rag.add_document(doc);
///
/// // Search
/// const results = rag.search(query_embedding, 10);
/// console.log(`Found ${results.length} results`);
/// ```
#[wasm_bindgen]
pub struct LocalRAG {
    index: IndexType,
    embedding_dim: usize,
}

#[wasm_bindgen]
impl LocalRAG {
    /// Create a new LocalRAG instance
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of embedding vectors (e.g., 384 for MiniLM)
    /// * `use_hnsw` - If true, use HNSW index (approximate but fast); if false, use flat index (exact but slower)
    ///
    /// # Returns
    ///
    /// A new LocalRAG instance
    ///
    /// # Example
    ///
    /// ```javascript
    /// // For small datasets (< 1000 docs), use flat index
    /// const rag_small = new LocalRAG(384, false);
    ///
    /// // For large datasets (> 1000 docs), use HNSW
    /// const rag_large = new LocalRAG(384, true);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dim: usize, use_hnsw: bool) -> LocalRAG {
        let index = if use_hnsw {
            IndexType::HNSW(HNSWIndex::with_defaults(embedding_dim))
        } else {
            IndexType::Flat(FlatIndex::new(embedding_dim))
        };

        LocalRAG {
            index,
            embedding_dim,
        }
    }

    /// Add a document to the index
    ///
    /// # Arguments
    ///
    /// * `document` - Document to add with its embedding
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The document embedding dimension doesn't match the index dimension
    /// - Internal indexing error occurs
    ///
    /// # Example
    ///
    /// ```javascript
    /// const embedding = new Float32Array(384);
    /// // ... fill embedding with values ...
    ///
    /// const doc = new JsDocument(
    ///   "doc1",
    ///   "Document content",
    ///   embedding,
    ///   { category: "example" }
    /// );
    ///
    /// try {
    ///   rag.add_document(doc);
    ///   console.log("Document added successfully");
    /// } catch (error) {
    ///   console.error("Failed to add document:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn add_document(&mut self, document: JsDocument) -> Result<(), JsValue> {
        let core_doc = document.to_core();

        match &mut self.index {
            IndexType::Flat(index) => index.add(core_doc),
            IndexType::HNSW(index) => index.add(core_doc),
        }
        .map_err(|e| JsValue::from_str(&format!("Failed to add document: {}", e)))
    }

    /// Search for similar documents
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding as Float32Array or Array
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Array of search results sorted by similarity score (highest first)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query dimension doesn't match the index dimension
    /// - The query cannot be parsed as a number array
    /// - Internal search error occurs
    ///
    /// # Example
    ///
    /// ```javascript
    /// const query = new Float32Array(384);
    /// // ... set query values ...
    ///
    /// try {
    ///   const results = rag.search(query, 10);
    ///   for (const result of results) {
    ///     console.log(`${result.id}: ${result.score}`);
    ///     console.log(`Content: ${result.content}`);
    ///   }
    /// } catch (error) {
    ///   console.error("Search failed:", error);
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn search(&self, query: JsValue, k: usize) -> Result<Vec<JsSearchResult>, JsValue> {
        // Convert query from JsValue to Vec<f32>
        let query_vec: Vec<f32> = serde_wasm_bindgen::from_value(query)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse query: {}", e)))?;

        // Perform search based on index type
        let results = match &self.index {
            IndexType::Flat(index) => index.search(&query_vec, k),
            IndexType::HNSW(index) => index.search(&query_vec, k),
        }
        .map_err(|e| JsValue::from_str(&format!("Search failed: {}", e)))?;

        // Convert to JS-compatible results
        Ok(results.into_iter().map(JsSearchResult::from_core).collect())
    }

    /// Remove a document from the index by ID
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the document to remove
    ///
    /// # Returns
    ///
    /// `true` if the document was found and removed, `false` otherwise
    ///
    /// # Example
    ///
    /// ```javascript
    /// if (rag.remove_document("doc1")) {
    ///   console.log("Document removed");
    /// } else {
    ///   console.log("Document not found");
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn remove_document(&mut self, id: &str) -> bool {
        match &mut self.index {
            IndexType::Flat(index) => index.remove(id).is_some(),
            IndexType::HNSW(_) => {
                // HNSW doesn't support removal in this implementation
                // This is a known limitation of HNSW indices
                false
            }
        }
    }

    /// Get the number of documents in the index
    ///
    /// # Returns
    ///
    /// Number of documents currently indexed
    ///
    /// # Example
    ///
    /// ```javascript
    /// console.log(`Index contains ${rag.document_count()} documents`);
    /// ```
    #[wasm_bindgen]
    pub fn document_count(&self) -> usize {
        match &self.index {
            IndexType::Flat(index) => index.len(),
            IndexType::HNSW(index) => index.len(),
        }
    }

    /// Clear all documents from the index
    ///
    /// # Example
    ///
    /// ```javascript
    /// rag.clear();
    /// console.log("All documents removed");
    /// ```
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        match &mut self.index {
            IndexType::Flat(index) => index.clear(),
            IndexType::HNSW(index) => index.clear(),
        }
    }

    /// Get the embedding dimension of this index
    ///
    /// # Returns
    ///
    /// The expected dimension of embeddings
    ///
    /// # Example
    ///
    /// ```javascript
    /// console.log(`This index expects ${rag.embedding_dim()}-dimensional embeddings`);
    /// ```
    #[wasm_bindgen]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Check if the index is using HNSW
    ///
    /// # Returns
    ///
    /// `true` if using HNSW index, `false` if using flat index
    ///
    /// # Example
    ///
    /// ```javascript
    /// if (rag.is_hnsw()) {
    ///   console.log("Using approximate search (HNSW)");
    /// } else {
    ///   console.log("Using exact search (Flat)");
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn is_hnsw(&self) -> bool {
        matches!(self.index, IndexType::HNSW(_))
    }

    /// Serialize the current index state to JSON
    ///
    /// # Returns
    ///
    /// A JavaScript object representing the serialized index
    ///
    /// # Example
    ///
    /// ```javascript
    /// const data = rag.to_json();
    /// localStorage.setItem('my-index', JSON.stringify(data));
    /// ```
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        match &self.index {
            IndexType::Flat(index) => {
                let serialized = persistence::serialize_flat_index(index, self.embedding_dim)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                serde_wasm_bindgen::to_value(&serialized)
                    .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
            }
            IndexType::HNSW(index) => {
                let serialized = persistence::serialize_hnsw_index(index, self.embedding_dim)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                serde_wasm_bindgen::to_value(&serialized)
                    .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
            }
        }
    }

    /// Restore an index from serialized JSON data
    ///
    /// # Arguments
    ///
    /// * `data` - JavaScript object containing serialized index data
    ///
    /// # Returns
    ///
    /// A new LocalRAG instance with the restored index
    ///
    /// # Example
    ///
    /// ```javascript
    /// const data = JSON.parse(localStorage.getItem('my-index'));
    /// const rag = LocalRAG.from_json(data);
    /// ```
    #[wasm_bindgen]
    pub fn from_json(data: JsValue) -> Result<LocalRAG, JsValue> {
        use persistence::SerializedIndex;

        let serialized: SerializedIndex = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Deserialization failed: {}", e)))?;

        match serialized {
            SerializedIndex::Flat(flat_data) => {
                let index = persistence::deserialize_flat_index(flat_data)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let embedding_dim = index.embedding_dim();
                Ok(LocalRAG {
                    index: IndexType::Flat(index),
                    embedding_dim,
                })
            }
            SerializedIndex::Hnsw(hnsw_data) => {
                let index = persistence::deserialize_hnsw_index(hnsw_data)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let embedding_dim = index.embedding_dim();
                Ok(LocalRAG {
                    index: IndexType::HNSW(index),
                    embedding_dim,
                })
            }
        }
    }

    /// Save the index to IndexedDB
    ///
    /// # Arguments
    ///
    /// * `store` - IndexedDBStore instance
    /// * `key` - Storage key for this index
    ///
    /// # Returns
    ///
    /// A Promise that resolves when the save is complete
    ///
    /// # Example
    ///
    /// ```javascript
    /// const store = new IndexedDBStore();
    /// await rag.save_to_db(store, "my-index");
    /// console.log("Index saved!");
    /// ```
    #[wasm_bindgen]
    pub async fn save_to_db(&self, store: &persistence::IndexedDBStore, key: &str) -> Result<(), JsValue> {
        let data = self.to_json()?;
        store.save(key, data).await
    }

    /// Load an index from IndexedDB
    ///
    /// # Arguments
    ///
    /// * `store` - IndexedDBStore instance
    /// * `key` - Storage key for the index
    ///
    /// # Returns
    ///
    /// A Promise that resolves to a LocalRAG instance
    ///
    /// # Example
    ///
    /// ```javascript
    /// const store = new IndexedDBStore();
    /// const rag = await LocalRAG.load_from_db(store, "my-index");
    /// console.log(`Loaded index with ${rag.document_count()} documents`);
    /// ```
    #[wasm_bindgen]
    pub async fn load_from_db(store: &persistence::IndexedDBStore, key: &str) -> Result<LocalRAG, JsValue> {
        let data = store.load(key).await?;
        Self::from_json(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn create_test_embedding(dim: usize, value: f32) -> JsValue {
        let vec: Vec<f32> = vec![value; dim];
        serde_wasm_bindgen::to_value(&vec).unwrap()
    }

    #[wasm_bindgen_test]
    fn test_js_document_creation() {
        let embedding = create_test_embedding(3, 0.5);
        let metadata = JsValue::NULL;

        let doc = JsDocument::new(
            "test_id".to_string(),
            "test content".to_string(),
            embedding,
            metadata,
        );

        assert!(doc.is_ok());
        let doc = doc.unwrap();
        assert_eq!(doc.id(), "test_id");
        assert_eq!(doc.content(), "test content");
        assert_eq!(doc.embedding_dim(), 3);
    }

    #[wasm_bindgen_test]
    fn test_js_document_with_metadata() {
        let embedding = create_test_embedding(3, 0.5);

        // Create metadata as a JavaScript object
        let metadata = js_sys::Object::new();
        js_sys::Reflect::set(&metadata, &JsValue::from_str("key"), &JsValue::from_str("value")).unwrap();

        let doc = JsDocument::new(
            "test_id".to_string(),
            "test content".to_string(),
            embedding,
            metadata.into(),
        );

        assert!(doc.is_ok());
        let doc = doc.unwrap();

        let meta = doc.metadata().unwrap();
        assert!(!meta.is_null());
    }

    #[wasm_bindgen_test]
    fn test_local_rag_creation_flat() {
        let rag = LocalRAG::new(384, false);
        assert_eq!(rag.embedding_dim(), 384);
        assert_eq!(rag.document_count(), 0);
        assert!(!rag.is_hnsw());
    }

    #[wasm_bindgen_test]
    fn test_local_rag_creation_hnsw() {
        let rag = LocalRAG::new(384, true);
        assert_eq!(rag.embedding_dim(), 384);
        assert_eq!(rag.document_count(), 0);
        assert!(rag.is_hnsw());
    }

    #[wasm_bindgen_test]
    fn test_add_document() {
        let mut rag = LocalRAG::new(3, false);

        let embedding = create_test_embedding(3, 0.5);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding,
            JsValue::NULL,
        ).unwrap();

        let result = rag.add_document(doc);
        assert!(result.is_ok());
        assert_eq!(rag.document_count(), 1);
    }

    #[wasm_bindgen_test]
    fn test_add_document_dimension_mismatch() {
        let mut rag = LocalRAG::new(5, false);

        let embedding = create_test_embedding(3, 0.5); // Wrong dimension
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding,
            JsValue::NULL,
        ).unwrap();

        let result = rag.add_document(doc);
        assert!(result.is_err());
        assert_eq!(rag.document_count(), 0);
    }

    #[wasm_bindgen_test]
    fn test_search_empty_index() {
        let rag = LocalRAG::new(3, false);

        let query = create_test_embedding(3, 1.0);
        let results = rag.search(query, 5);

        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 0);
    }

    #[wasm_bindgen_test]
    fn test_search_with_results() {
        let mut rag = LocalRAG::new(3, false);

        // Add a document
        let embedding = create_test_embedding(3, 1.0);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding.clone(),
            JsValue::NULL,
        ).unwrap();

        rag.add_document(doc).unwrap();

        // Search with same embedding (should get exact match)
        let results = rag.search(embedding, 5);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id(), "doc1");
        assert_eq!(results[0].content(), "test content");
        // Score should be close to 1.0 for exact match
        assert!((results[0].score() - 1.0).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn test_search_dimension_mismatch() {
        let mut rag = LocalRAG::new(3, false);

        let embedding = create_test_embedding(3, 1.0);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding,
            JsValue::NULL,
        ).unwrap();

        rag.add_document(doc).unwrap();

        // Search with wrong dimension
        let query = create_test_embedding(5, 1.0);
        let results = rag.search(query, 5);

        assert!(results.is_err());
    }

    #[wasm_bindgen_test]
    fn test_remove_document_flat() {
        let mut rag = LocalRAG::new(3, false);

        let embedding = create_test_embedding(3, 1.0);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding,
            JsValue::NULL,
        ).unwrap();

        rag.add_document(doc).unwrap();
        assert_eq!(rag.document_count(), 1);

        let removed = rag.remove_document("doc1");
        assert!(removed);
        assert_eq!(rag.document_count(), 0);
    }

    #[wasm_bindgen_test]
    fn test_remove_nonexistent_document() {
        let mut rag = LocalRAG::new(3, false);

        let removed = rag.remove_document("nonexistent");
        assert!(!removed);
    }

    #[wasm_bindgen_test]
    fn test_clear() {
        let mut rag = LocalRAG::new(3, false);

        // Add multiple documents
        for i in 0..5 {
            let embedding = create_test_embedding(3, i as f32);
            let doc = JsDocument::new(
                format!("doc{}", i),
                format!("content {}", i),
                embedding,
                JsValue::NULL,
            ).unwrap();

            rag.add_document(doc).unwrap();
        }

        assert_eq!(rag.document_count(), 5);

        rag.clear();

        assert_eq!(rag.document_count(), 0);
    }

    #[wasm_bindgen_test]
    fn test_multiple_documents_search_ordering() {
        let mut rag = LocalRAG::new(3, false);

        // Add documents with different embeddings
        let docs = vec![
            (vec![1.0, 0.0, 0.0], "doc1"),
            (vec![0.9, 0.1, 0.0], "doc2"),
            (vec![0.0, 1.0, 0.0], "doc3"),
        ];

        for (embedding_vec, id) in docs {
            let embedding = serde_wasm_bindgen::to_value(&embedding_vec).unwrap();
            let doc = JsDocument::new(
                id.to_string(),
                format!("content {}", id),
                embedding,
                JsValue::NULL,
            ).unwrap();

            rag.add_document(doc).unwrap();
        }

        // Query with [1.0, 0.0, 0.0] - should rank doc1 highest
        let query = serde_wasm_bindgen::to_value(&vec![1.0, 0.0, 0.0]).unwrap();
        let results = rag.search(query, 10).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id(), "doc1");
        assert_eq!(results[1].id(), "doc2");
        assert_eq!(results[2].id(), "doc3");

        // Scores should be descending
        assert!(results[0].score() > results[1].score());
        assert!(results[1].score() > results[2].score());
    }

    #[wasm_bindgen_test]
    fn test_hnsw_index_basic_operations() {
        let mut rag = LocalRAG::new(3, true);

        // Add document
        let embedding = create_test_embedding(3, 1.0);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding.clone(),
            JsValue::NULL,
        ).unwrap();

        rag.add_document(doc).unwrap();
        assert_eq!(rag.document_count(), 1);

        // Search
        let results = rag.search(embedding, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id(), "doc1");
    }

    #[wasm_bindgen_test]
    fn test_hnsw_remove_not_supported() {
        let mut rag = LocalRAG::new(3, true);

        let embedding = create_test_embedding(3, 1.0);
        let doc = JsDocument::new(
            "doc1".to_string(),
            "test content".to_string(),
            embedding,
            JsValue::NULL,
        ).unwrap();

        rag.add_document(doc).unwrap();

        // HNSW doesn't support removal
        let removed = rag.remove_document("doc1");
        assert!(!removed);
        assert_eq!(rag.document_count(), 1);
    }
}
