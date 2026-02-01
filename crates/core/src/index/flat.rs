//! Flat (brute-force) index implementation
//!
//! The FlatIndex provides exact nearest neighbor search by computing similarity
//! between the query and all stored documents. While this is computationally
//! expensive for large datasets (O(n) complexity), it guarantees exact results
//! and is suitable for:
//! - Small to medium datasets (< 10,000 documents)
//! - Validation and testing of approximate search methods
//! - Use cases where accuracy is more important than speed

use crate::{Document, Result, SearchResult};
use crate::vector::cosine_similarity;
use std::collections::HashMap;

/// Flat index using brute-force search
///
/// This index stores documents in a HashMap and performs exhaustive search
/// by computing cosine similarity with all documents. It provides O(1) insertion
/// and deletion, but O(n) search complexity.
///
/// # Examples
/// ```
/// use nexus_core::index::FlatIndex;
/// use nexus_core::Document;
///
/// let mut index = FlatIndex::new(384);
///
/// let doc = Document {
///     id: "doc1".to_string(),
///     content: "Hello world".to_string(),
///     embedding: vec![0.1; 384],
///     metadata: None,
/// };
///
/// index.add(doc).unwrap();
/// assert_eq!(index.len(), 1);
///
/// let query = vec![0.1; 384];
/// let results = index.search(&query, 5).unwrap();
/// assert_eq!(results.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct FlatIndex {
    /// Expected dimension of embeddings
    embedding_dim: usize,
    /// Documents stored by ID
    documents: HashMap<String, Document>,
}

impl FlatIndex {
    /// Create a new flat index
    ///
    /// # Arguments
    /// * `embedding_dim` - Expected dimension of document embeddings
    ///
    /// # Returns
    /// A new empty FlatIndex
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    ///
    /// let index = FlatIndex::new(384);
    /// assert_eq!(index.len(), 0);
    /// assert!(index.is_empty());
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            documents: HashMap::new(),
        }
    }

    /// Add a document to the index
    ///
    /// # Arguments
    /// * `document` - Document to add
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    ///
    /// # Errors
    /// * `DimensionMismatch` - If document embedding dimension doesn't match index dimension
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    /// use nexus_core::Document;
    ///
    /// let mut index = FlatIndex::new(384);
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test document".to_string(),
    ///     embedding: vec![0.5; 384],
    ///     metadata: None,
    /// };
    ///
    /// index.add(doc).unwrap();
    /// assert_eq!(index.len(), 1);
    /// ```
    pub fn add(&mut self, document: Document) -> Result<()> {
        // Validate embedding dimension
        if document.embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
            });
        }

        // Insert document (will replace if ID already exists)
        self.documents.insert(document.id.clone(), document);
        Ok(())
    }

    /// Add multiple documents to the index
    ///
    /// # Arguments
    /// * `documents` - Vector of documents to add
    ///
    /// # Returns
    /// * `Result<()>` - Ok if all documents were added successfully
    ///
    /// # Errors
    /// * `DimensionMismatch` - If any document embedding dimension doesn't match
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    /// use nexus_core::Document;
    ///
    /// let mut index = FlatIndex::new(384);
    /// let docs = vec![
    ///     Document {
    ///         id: "doc1".to_string(),
    ///         content: "First document".to_string(),
    ///         embedding: vec![0.1; 384],
    ///         metadata: None,
    ///     },
    ///     Document {
    ///         id: "doc2".to_string(),
    ///         content: "Second document".to_string(),
    ///         embedding: vec![0.2; 384],
    ///         metadata: None,
    ///     },
    /// ];
    ///
    /// index.add_batch(docs).unwrap();
    /// assert_eq!(index.len(), 2);
    /// ```
    pub fn add_batch(&mut self, documents: Vec<Document>) -> Result<()> {
        for document in documents {
            self.add(document)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// Performs exhaustive search by computing cosine similarity between the query
    /// and all documents in the index. Results are sorted by similarity score in
    /// descending order.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// * `Result<Vec<SearchResult>>` - Up to k search results, sorted by score (descending)
    ///
    /// # Errors
    /// * `DimensionMismatch` - If query dimension doesn't match index dimension
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    /// use nexus_core::Document;
    ///
    /// let mut index = FlatIndex::new(3);
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test".to_string(),
    ///     embedding: vec![1.0, 0.0, 0.0],
    ///     metadata: None,
    /// };
    ///
    /// index.add(doc).unwrap();
    ///
    /// let query = vec![1.0, 0.0, 0.0];
    /// let results = index.search(&query, 5).unwrap();
    /// assert_eq!(results.len(), 1);
    /// assert!((results[0].score - 1.0).abs() < 1e-6);
    /// ```
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Validate query dimension
        if query.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        // Compute similarity scores for all documents
        let mut scored_docs: Vec<(f32, &Document)> = self
            .documents
            .values()
            .map(|doc| {
                let score = cosine_similarity(query, &doc.embedding)
                    .unwrap_or(0.0); // Should never fail since we validated dimensions
                (score, doc)
            })
            .collect();

        // Sort by score in descending order
        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k results and convert to SearchResult
        let results = scored_docs
            .into_iter()
            .take(k)
            .map(|(score, doc)| SearchResult {
                id: doc.id.clone(),
                content: doc.content.clone(),
                score,
                metadata: doc.metadata.clone(),
            })
            .collect();

        Ok(results)
    }

    /// Remove a document from the index
    ///
    /// # Arguments
    /// * `id` - ID of the document to remove
    ///
    /// # Returns
    /// * `Option<Document>` - The removed document, or None if not found
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    /// use nexus_core::Document;
    ///
    /// let mut index = FlatIndex::new(384);
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test".to_string(),
    ///     embedding: vec![0.5; 384],
    ///     metadata: None,
    /// };
    ///
    /// index.add(doc.clone()).unwrap();
    /// assert_eq!(index.len(), 1);
    ///
    /// let removed = index.remove("doc1");
    /// assert!(removed.is_some());
    /// assert_eq!(index.len(), 0);
    ///
    /// let not_found = index.remove("doc1");
    /// assert!(not_found.is_none());
    /// ```
    pub fn remove(&mut self, id: &str) -> Option<Document> {
        self.documents.remove(id)
    }

    /// Get the number of documents in the index
    ///
    /// # Returns
    /// Number of documents
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    ///
    /// let index = FlatIndex::new(384);
    /// assert_eq!(index.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty
    ///
    /// # Returns
    /// `true` if the index contains no documents
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    ///
    /// let index = FlatIndex::new(384);
    /// assert!(index.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear all documents from the index
    ///
    /// # Examples
    /// ```
    /// use nexus_core::index::FlatIndex;
    /// use nexus_core::Document;
    ///
    /// let mut index = FlatIndex::new(384);
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test".to_string(),
    ///     embedding: vec![0.5; 384],
    ///     metadata: None,
    /// };
    ///
    /// index.add(doc).unwrap();
    /// assert_eq!(index.len(), 1);
    ///
    /// index.clear();
    /// assert_eq!(index.len(), 0);
    /// assert!(index.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.documents.clear();
    }

    /// Get all documents in the index
    ///
    /// Returns a vector containing clones of all documents in the index.
    /// Useful for serialization and persistence.
    ///
    /// # Returns
    /// * `Vec<Document>` - Vector of all documents
    ///
    /// # Examples
    /// ```
    /// use foxstash_core::index::FlatIndex;
    /// use foxstash_core::Document;
    ///
    /// let mut index = FlatIndex::new(3);
    /// let doc = Document {
    ///     id: "doc1".to_string(),
    ///     content: "Test".to_string(),
    ///     embedding: vec![1.0, 0.0, 0.0],
    ///     metadata: None,
    /// };
    ///
    /// index.add(doc).unwrap();
    /// let all_docs = index.get_all_documents();
    /// assert_eq!(all_docs.len(), 1);
    /// assert_eq!(all_docs[0].id, "doc1");
    /// ```
    pub fn get_all_documents(&self) -> Vec<Document> {
        self.documents.values().cloned().collect()
    }

    /// Get the embedding dimension
    ///
    /// # Returns
    /// * `usize` - Embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Test document {}", id),
            embedding,
            metadata: None,
        }
    }

    #[test]
    fn test_new_index() {
        let index = FlatIndex::new(384);
        assert_eq!(index.embedding_dim, 384);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_document() {
        let mut index = FlatIndex::new(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);

        let result = index.add(doc);
        assert!(result.is_ok());
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_add_document_dimension_mismatch() {
        let mut index = FlatIndex::new(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0]); // Wrong dimension

        let result = index.add(doc);
        assert!(result.is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_add_duplicate_id_replaces() {
        let mut index = FlatIndex::new(3);
        let doc1 = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        let mut doc2 = create_test_document("doc1", vec![0.0, 1.0, 0.0]);
        doc2.content = "Updated content".to_string();

        index.add(doc1).unwrap();
        assert_eq!(index.len(), 1);

        index.add(doc2).unwrap();
        assert_eq!(index.len(), 1); // Still 1 document

        // Verify the document was replaced
        let query = vec![0.0, 1.0, 0.0];
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results[0].content, "Updated content");
    }

    #[test]
    fn test_add_batch() {
        let mut index = FlatIndex::new(3);
        let docs = vec![
            create_test_document("doc1", vec![1.0, 0.0, 0.0]),
            create_test_document("doc2", vec![0.0, 1.0, 0.0]),
            create_test_document("doc3", vec![0.0, 0.0, 1.0]),
        ];

        let result = index.add_batch(docs);
        assert!(result.is_ok());
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_add_batch_partial_failure() {
        let mut index = FlatIndex::new(3);
        let docs = vec![
            create_test_document("doc1", vec![1.0, 0.0, 0.0]),
            create_test_document("doc2", vec![0.0, 1.0]), // Wrong dimension
            create_test_document("doc3", vec![0.0, 0.0, 1.0]),
        ];

        let result = index.add_batch(docs);
        assert!(result.is_err());
        // First document should be added before error
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_exact_match() {
        let mut index = FlatIndex::new(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_multiple_results_sorted() {
        let mut index = FlatIndex::new(3);

        // Add documents with varying similarity to query [1, 0, 0]
        index.add(create_test_document("exact", vec![1.0, 0.0, 0.0])).unwrap();
        index.add(create_test_document("close", vec![0.9, 0.1, 0.0])).unwrap();
        index.add(create_test_document("far", vec![0.0, 1.0, 0.0])).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "exact");
        assert_eq!(results[1].id, "close");
        assert_eq!(results[2].id, "far");

        // Verify scores are in descending order
        assert!(results[0].score > results[1].score);
        assert!(results[1].score > results[2].score);
    }

    #[test]
    fn test_search_limit_k() {
        let mut index = FlatIndex::new(3);

        // Add 5 documents
        for i in 0..5 {
            let embedding = vec![i as f32, 0.0, 0.0];
            index.add(create_test_document(&format!("doc{}", i), embedding)).unwrap();
        }

        let query = vec![10.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        // Should only return 3 results even though 5 documents exist
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let mut index = FlatIndex::new(3);
        index.add(create_test_document("doc1", vec![1.0, 0.0, 0.0])).unwrap();

        let query = vec![1.0, 0.0]; // Wrong dimension
        let result = index.search(&query, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_search_empty_index() {
        let index = FlatIndex::new(3);
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_remove_existing() {
        let mut index = FlatIndex::new(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc.clone()).unwrap();

        let removed = index.remove("doc1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "doc1");
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_remove_non_existing() {
        let mut index = FlatIndex::new(3);
        let removed = index.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear() {
        let mut index = FlatIndex::new(3);

        // Add multiple documents
        for i in 0..5 {
            let doc = create_test_document(&format!("doc{}", i), vec![i as f32, 0.0, 0.0]);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 5);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_orthogonal_vectors() {
        let mut index = FlatIndex::new(3);

        index.add(create_test_document("x_axis", vec![1.0, 0.0, 0.0])).unwrap();
        index.add(create_test_document("y_axis", vec![0.0, 1.0, 0.0])).unwrap();
        index.add(create_test_document("z_axis", vec![0.0, 0.0, 1.0])).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "x_axis");
        assert!((results[0].score - 1.0).abs() < 1e-6);

        // Orthogonal vectors should have similarity close to 0
        assert!(results[1].score.abs() < 1e-6);
        assert!(results[2].score.abs() < 1e-6);
    }

    #[test]
    fn test_with_metadata() {
        let mut index = FlatIndex::new(3);

        let mut doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        doc.metadata = Some(serde_json::json!({
            "source": "test",
            "timestamp": 123456789
        }));

        index.add(doc.clone()).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.is_some());
        assert_eq!(
            results[0].metadata.as_ref().unwrap()["source"],
            "test"
        );
    }

    #[test]
    fn test_negative_similarity() {
        let mut index = FlatIndex::new(3);

        index.add(create_test_document("positive", vec![1.0, 0.0, 0.0])).unwrap();
        index.add(create_test_document("negative", vec![-1.0, 0.0, 0.0])).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "positive");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert_eq!(results[1].id, "negative");
        assert!((results[1].score + 1.0).abs() < 1e-6); // Similarity should be -1
    }

    #[test]
    fn test_large_batch() {
        let mut index = FlatIndex::new(128);
        let mut docs = Vec::new();

        // Create 1000 documents
        for i in 0..1000 {
            let mut embedding = vec![0.0; 128];
            embedding[0] = i as f32;
            docs.push(create_test_document(&format!("doc{}", i), embedding));
        }

        let result = index.add_batch(docs);
        assert!(result.is_ok());
        assert_eq!(index.len(), 1000);

        // Search should still work efficiently enough
        let mut query = vec![0.0; 128];
        query[0] = 500.0;
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }
}
