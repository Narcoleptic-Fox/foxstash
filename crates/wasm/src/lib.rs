//! WebAssembly bindings for Foxstash
//!
//! This crate provides JavaScript-friendly interfaces to the core RAG functionality,
//! enabling vector search, keyword search, and hybrid search in browser environments.
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
//! // Vector search
//! const results = rag.search(new Float32Array(384), 5);
//!
//! // Text search (BM25)
//! const textResults = rag.search_text("hello world", 5);
//!
//! // Hybrid search (vector + BM25)
//! const hybridResults = rag.search_hybrid(new Float32Array(384), "hello world", 5);
//! ```

use std::collections::HashMap;

use foxstash_core::{
    index::{FlatIndex, HNSWIndex},
    Document, SearchResult,
};
use foxstash_db::hybrid::{self, HybridConfig, MergeStrategy};
use foxstash_db::inverted_index::InvertedIndex;
use foxstash_db::tokenizer::{SimpleTokenizer, Tokenizer};
use wasm_bindgen::prelude::*;

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
            Some(
                serde_wasm_bindgen::from_value(metadata)
                    .map_err(|e| JsValue::from_str(&format!("Failed to parse metadata: {}", e)))?,
            )
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

/// Internal document store entry for text search lookups.
///
/// We need to maintain a parallel mapping from position IDs (usize) back to
/// document data so the hybrid merge function can resolve BM25 hits.
struct DocEntry {
    id: String,
    content: String,
    metadata: Option<serde_json::Value>,
}

/// Local RAG system for in-browser vector search
///
/// This is the main interface for managing documents and performing similarity search.
/// It wraps either a flat index (exact search) or HNSW index (approximate search)
/// depending on the configuration.
///
/// Supports three search modes:
/// - **Vector search**: Find documents by embedding similarity
/// - **Text search**: Find documents by BM25 keyword matching
/// - **Hybrid search**: Combine vector and keyword scores via RRF or weighted sum
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
/// // Vector search
/// const results = rag.search(query_embedding, 10);
///
/// // Text search (BM25 keyword matching)
/// const textResults = rag.search_text("keyword query", 10);
///
/// // Hybrid search (vector + text combined)
/// const hybridResults = rag.search_hybrid(query_embedding, "keyword query", 10);
/// ```
#[wasm_bindgen]
pub struct LocalRAG {
    index: IndexType,
    embedding_dim: usize,
    /// BM25 inverted index for keyword search.
    text_index: InvertedIndex,
    /// Tokenizer for converting text to search terms.
    tokenizer: SimpleTokenizer,
    /// Document store keyed by position ID (assigned sequentially).
    /// Used to resolve BM25 results back to full document data.
    doc_store: HashMap<usize, DocEntry>,
    /// Mapping from document ID to its position in the text index.
    id_to_pos: HashMap<String, usize>,
    /// Next position ID to assign.
    next_pos: usize,
    /// Hybrid search configuration (RRF/WeightedSum, weights).
    hybrid_config: HybridConfig,
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
            text_index: InvertedIndex::new(),
            tokenizer: SimpleTokenizer::new(),
            doc_store: HashMap::new(),
            id_to_pos: HashMap::new(),
            next_pos: 0,
            hybrid_config: HybridConfig::default(),
        }
    }

    /// Add a document to the index
    ///
    /// The document is indexed for both vector similarity search and BM25 keyword
    /// search. The text content is tokenized and added to the inverted index.
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

        // Add to vector index.
        match &mut self.index {
            IndexType::Flat(index) => index.add(core_doc.clone()),
            IndexType::HNSW(index) => index.add(core_doc.clone()),
        }
        .map_err(|e| JsValue::from_str(&format!("Failed to add document: {}", e)))?;

        // If this document ID already exists in the text index, remove the old entry.
        if let Some(&old_pos) = self.id_to_pos.get(&core_doc.id) {
            self.text_index.remove(old_pos);
            self.doc_store.remove(&old_pos);
        }

        // Add to text index.
        let pos = self.next_pos;
        self.next_pos += 1;

        let tokens = self.tokenizer.tokenize(&core_doc.content);
        self.text_index.add(pos, &tokens);

        self.id_to_pos.insert(core_doc.id.clone(), pos);
        self.doc_store.insert(
            pos,
            DocEntry {
                id: core_doc.id,
                content: core_doc.content,
                metadata: core_doc.metadata,
            },
        );

        Ok(())
    }

    /// Search for similar documents by vector similarity
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

    /// Search for documents by keyword matching (BM25)
    ///
    /// Tokenizes the query text and searches the inverted index using BM25 scoring.
    /// Stop words are filtered and terms are lowercased automatically.
    ///
    /// # Arguments
    ///
    /// * `query` - Text query string
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Array of search results sorted by BM25 score (highest first)
    ///
    /// # Example
    ///
    /// ```javascript
    /// const results = rag.search_text("gateway service error", 10);
    /// for (const result of results) {
    ///   console.log(`${result.id}: ${result.score}`);
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn search_text(&self, query: &str, k: usize) -> Vec<JsSearchResult> {
        let tokens = self.tokenizer.tokenize(query);
        if tokens.is_empty() || k == 0 {
            return Vec::new();
        }

        let bm25_results = self.text_index.search(&tokens, k);

        bm25_results
            .into_iter()
            .filter_map(|(pos, score)| {
                self.doc_store.get(&pos).map(|entry| JsSearchResult {
                    id: entry.id.clone(),
                    content: entry.content.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                })
            })
            .collect()
    }

    /// Search using both vector similarity and keyword matching (hybrid search)
    ///
    /// Combines results from vector search and BM25 keyword search using the
    /// configured merge strategy (default: RRF with 0.7 vector / 0.3 keyword weights).
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - Query embedding as Float32Array or Array
    /// * `text_query` - Text query string for keyword matching
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Array of search results sorted by combined score (highest first).
    /// Documents appearing in both result sets are boosted.
    ///
    /// # Errors
    ///
    /// Returns an error if the query embedding cannot be parsed or has wrong dimensions
    ///
    /// # Example
    ///
    /// ```javascript
    /// const embedding = new Float32Array(384);
    /// // ... set embedding values ...
    ///
    /// const results = rag.search_hybrid(embedding, "gateway service", 10);
    /// for (const result of results) {
    ///   console.log(`${result.id}: ${result.score}`);
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn search_hybrid(
        &self,
        query_embedding: JsValue,
        text_query: &str,
        k: usize,
    ) -> Result<Vec<JsSearchResult>, JsValue> {
        if k == 0 {
            return Ok(Vec::new());
        }

        // Vector search: retrieve more candidates for better merge quality.
        let query_vec: Vec<f32> = serde_wasm_bindgen::from_value(query_embedding)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse query: {}", e)))?;

        let fetch_k = k * 3;
        let vector_results = match &self.index {
            IndexType::Flat(index) => index.search(&query_vec, fetch_k),
            IndexType::HNSW(index) => index.search(&query_vec, fetch_k),
        }
        .map_err(|e| JsValue::from_str(&format!("Vector search failed: {}", e)))?;

        // Keyword search.
        let tokens = self.tokenizer.tokenize(text_query);
        let keyword_results = self.text_index.search(&tokens, fetch_k);

        // Build the doc_lookup closure for merge_results.
        let doc_lookup = |pos: usize| -> Option<SearchResult> {
            self.doc_store.get(&pos).map(|entry| SearchResult {
                id: entry.id.clone(),
                content: entry.content.clone(),
                score: 0.0, // Score is set by merge logic.
                metadata: entry.metadata.clone(),
            })
        };

        let merged =
            hybrid::merge_results(&vector_results, &keyword_results, &doc_lookup, k, &self.hybrid_config);

        Ok(merged.into_iter().map(JsSearchResult::from_core).collect())
    }

    /// Configure hybrid search weights and strategy
    ///
    /// # Arguments
    ///
    /// * `vector_weight` - Weight for vector similarity scores (0.0 to 1.0+)
    /// * `keyword_weight` - Weight for BM25 keyword scores (0.0 to 1.0+)
    /// * `use_rrf` - If true, use Reciprocal Rank Fusion; if false, use weighted sum
    ///
    /// # Example
    ///
    /// ```javascript
    /// // Equal weighting of vector and keyword results
    /// rag.set_hybrid_config(0.5, 0.5, true);
    ///
    /// // Favor keyword results
    /// rag.set_hybrid_config(0.3, 0.7, true);
    ///
    /// // Use weighted sum instead of RRF
    /// rag.set_hybrid_config(0.6, 0.4, false);
    /// ```
    #[wasm_bindgen]
    pub fn set_hybrid_config(
        &mut self,
        vector_weight: f32,
        keyword_weight: f32,
        use_rrf: bool,
    ) {
        let strategy = if use_rrf {
            MergeStrategy::Rrf
        } else {
            MergeStrategy::WeightedSum
        };

        self.hybrid_config = HybridConfig::default()
            .with_weights(vector_weight, keyword_weight)
            .with_strategy(strategy);
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
        let vector_removed = match &mut self.index {
            IndexType::Flat(index) => index.remove(id).is_some(),
            IndexType::HNSW(_) => {
                // HNSW doesn't support removal in this implementation
                false
            }
        };

        // Always try to remove from text index regardless of vector index type.
        let text_removed = if let Some(&pos) = self.id_to_pos.get(id) {
            self.text_index.remove(pos);
            self.doc_store.remove(&pos);
            self.id_to_pos.remove(id);
            true
        } else {
            false
        };

        vector_removed || text_removed
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

    /// Get the number of documents in the text index
    ///
    /// This may differ from `document_count()` if documents were removed
    /// from an HNSW index (which doesn't support removal from the vector index).
    ///
    /// # Returns
    ///
    /// Number of documents in the text/keyword index
    #[wasm_bindgen]
    pub fn text_document_count(&self) -> usize {
        self.text_index.len()
    }

    /// Clear all documents from the index
    ///
    /// Removes all documents from both the vector index and the text index.
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

        self.text_index.clear();
        self.doc_store.clear();
        self.id_to_pos.clear();
        self.next_pos = 0;
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
    /// The text index is automatically rebuilt from the deserialized documents.
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
                let documents = index.get_all_documents();

                let mut rag = LocalRAG {
                    index: IndexType::Flat(index),
                    embedding_dim,
                    text_index: InvertedIndex::new(),
                    tokenizer: SimpleTokenizer::new(),
                    doc_store: HashMap::new(),
                    id_to_pos: HashMap::new(),
                    next_pos: 0,
                    hybrid_config: HybridConfig::default(),
                };

                // Rebuild text index from documents.
                rag.rebuild_text_index(&documents);
                Ok(rag)
            }
            SerializedIndex::Hnsw(hnsw_data) => {
                let index = persistence::deserialize_hnsw_index(hnsw_data)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                let embedding_dim = index.embedding_dim();
                let documents = index.get_all_documents();

                let mut rag = LocalRAG {
                    index: IndexType::HNSW(index),
                    embedding_dim,
                    text_index: InvertedIndex::new(),
                    tokenizer: SimpleTokenizer::new(),
                    doc_store: HashMap::new(),
                    id_to_pos: HashMap::new(),
                    next_pos: 0,
                    hybrid_config: HybridConfig::default(),
                };

                // Rebuild text index from documents.
                rag.rebuild_text_index(&documents);
                Ok(rag)
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
    pub async fn save_to_db(
        &self,
        store: &persistence::IndexedDBStore,
        key: &str,
    ) -> Result<(), JsValue> {
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
    pub async fn load_from_db(
        store: &persistence::IndexedDBStore,
        key: &str,
    ) -> Result<LocalRAG, JsValue> {
        let data = store.load(key).await?;
        Self::from_json(data)
    }
}

impl LocalRAG {
    /// Rebuild the text index from a list of documents.
    ///
    /// Used after deserialization to restore text search state.
    fn rebuild_text_index(&mut self, documents: &[Document]) {
        self.text_index.clear();
        self.doc_store.clear();
        self.id_to_pos.clear();
        self.next_pos = 0;

        for doc in documents {
            let pos = self.next_pos;
            self.next_pos += 1;

            let tokens = self.tokenizer.tokenize(&doc.content);
            self.text_index.add(pos, &tokens);

            self.id_to_pos.insert(doc.id.clone(), pos);
            self.doc_store.insert(
                pos,
                DocEntry {
                    id: doc.id.clone(),
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                },
            );
        }
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
        js_sys::Reflect::set(
            &metadata,
            &JsValue::from_str("key"),
            &JsValue::from_str("value"),
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
            )
            .unwrap();

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
            )
            .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

        rag.add_document(doc).unwrap();

        // HNSW doesn't support vector removal, but text index should be cleaned up.
        // remove_document returns true because text index removal succeeds.
        let removed = rag.remove_document("doc1");
        assert!(removed);
        // Vector count is still 1 (HNSW can't remove), but text count is 0.
        assert_eq!(rag.document_count(), 1);
        assert_eq!(rag.text_document_count(), 0);
    }
}

// Non-wasm unit tests for text/hybrid search logic (run with `cargo test`).
#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod native_tests {
    use super::*;
    use foxstash_db::inverted_index::InvertedIndex;
    use foxstash_db::tokenizer::{SimpleTokenizer, Tokenizer};

    #[test]
    fn text_index_basic_search() {
        let tokenizer = SimpleTokenizer::new();
        let mut idx = InvertedIndex::new();

        let tokens_a = tokenizer.tokenize("gateway service running on port 8080");
        let tokens_b = tokenizer.tokenize("database connection pool exhausted");
        let tokens_c = tokenizer.tokenize("gateway timeout after 30 seconds");

        idx.add(0, &tokens_a);
        idx.add(1, &tokens_b);
        idx.add(2, &tokens_c);

        let query = tokenizer.tokenize("gateway");
        let results = idx.search(&query, 10);

        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[test]
    fn text_index_remove_and_search() {
        let tokenizer = SimpleTokenizer::new();
        let mut idx = InvertedIndex::new();

        let tokens_a = tokenizer.tokenize("gateway service");
        let tokens_b = tokenizer.tokenize("database service");

        idx.add(0, &tokens_a);
        idx.add(1, &tokens_b);
        idx.remove(0);

        let query = tokenizer.tokenize("gateway");
        assert!(idx.search(&query, 10).is_empty());

        let query = tokenizer.tokenize("service");
        let results = idx.search(&query, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn hybrid_merge_rrf_basic() {
        let vector = vec![
            SearchResult { id: "a".into(), content: "alpha".into(), score: 0.9, metadata: None },
            SearchResult { id: "b".into(), content: "beta".into(), score: 0.8, metadata: None },
        ];
        let keyword = vec![(10, 5.0), (20, 3.0)];

        let doc_map: HashMap<usize, SearchResult> = vec![
            (10, SearchResult { id: "c".into(), content: "gamma".into(), score: 5.0, metadata: None }),
            (20, SearchResult { id: "a".into(), content: "alpha".into(), score: 3.0, metadata: None }),
        ]
        .into_iter()
        .collect();

        let lookup = |pos: usize| -> Option<SearchResult> { doc_map.get(&pos).cloned() };

        let config = HybridConfig::default();
        let results = hybrid::merge_results(&vector, &keyword, &lookup, 10, &config);

        // "a" appears in both lists and should be boosted to the top.
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn hybrid_config_builder() {
        let config = HybridConfig::default()
            .with_weights(0.5, 0.5)
            .with_strategy(MergeStrategy::WeightedSum);

        assert!((config.vector_weight() - 0.5).abs() < f32::EPSILON);
        assert!((config.keyword_weight() - 0.5).abs() < f32::EPSILON);
        assert!(matches!(config.merge_strategy(), MergeStrategy::WeightedSum));
    }

    #[test]
    fn text_search_empty_query() {
        let idx = InvertedIndex::new();
        let results = idx.search(&[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn text_search_no_matching_terms() {
        let tokenizer = SimpleTokenizer::new();
        let mut idx = InvertedIndex::new();

        let tokens = tokenizer.tokenize("alpha beta gamma");
        idx.add(0, &tokens);

        let query = tokenizer.tokenize("delta epsilon");
        let results = idx.search(&query, 10);
        assert!(results.is_empty());
    }
}
