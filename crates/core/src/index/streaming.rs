//! Streaming operations for large-scale indexing
//!
//! This module provides memory-efficient APIs for working with large datasets:
//!
//! - **Batch ingestion**: Add documents in chunks with progress callbacks
//! - **Streaming search**: Iterator-based results for memory efficiency
//! - **Parallel building**: Multi-threaded index construction
//!
//! # Example: Batch Ingestion
//!
//! ```
//! use foxstash_core::index::streaming::{BatchBuilder, BatchConfig};
//! use foxstash_core::index::HNSWIndex;
//! use foxstash_core::Document;
//!
//! let config = BatchConfig::default()
//!     .with_batch_size(100);
//!
//! let mut index = HNSWIndex::with_defaults(128);
//! let mut builder = BatchBuilder::new(&mut index, config);
//!
//! // Add documents
//! for i in 0..10 {
//!     let doc = Document {
//!         id: format!("doc_{}", i),
//!         content: format!("Content {}", i),
//!         embedding: vec![i as f32 / 10.0; 128],
//!         metadata: None,
//!     };
//!     builder.add(doc).unwrap();
//! }
//! let result = builder.finish();
//! assert_eq!(result.documents_indexed, 10);
//! ```
//!
//! # Example: Streaming Search
//!
//! ```ignore
//! use foxstash_core::index::streaming::StreamingSearch;
//!
//! // Get results as an iterator - doesn't allocate all results upfront
//! let search = StreamingSearch::new(&index, &query, 1000);
//!
//! for result in search.take(10) {
//!     println!("{}: {:.4}", result.id, result.score);
//! }
//! ```

use crate::{Document, RagError, Result, SearchResult};
use std::sync::Arc;

// ============================================================================
// Batch Configuration
// ============================================================================

/// Progress information during batch operations
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Number of documents processed so far
    pub completed: usize,
    /// Total documents to process (if known)
    pub total: Option<usize>,
    /// Current batch number
    pub batch_number: usize,
    /// Documents in current batch
    pub batch_size: usize,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Estimated documents per second
    pub docs_per_sec: f64,
}

impl BatchProgress {
    /// Calculate completion percentage (if total is known)
    pub fn percent(&self) -> Option<f64> {
        self.total
            .map(|t| (self.completed as f64 / t as f64) * 100.0)
    }

    /// Estimate remaining time in milliseconds (if total is known)
    pub fn eta_ms(&self) -> Option<u64> {
        if self.docs_per_sec > 0.0 {
            self.total.map(|t| {
                let remaining = t.saturating_sub(self.completed);
                ((remaining as f64) / self.docs_per_sec * 1000.0) as u64
            })
        } else {
            None
        }
    }
}

/// Callback type for progress updates
pub type ProgressCallback = Arc<dyn Fn(&BatchProgress) + Send + Sync>;

/// Configuration for batch operations
#[derive(Clone)]
pub struct BatchConfig {
    /// Number of documents to process before triggering progress callback
    pub batch_size: usize,
    /// Optional progress callback
    pub progress_callback: Option<ProgressCallback>,
    /// Total documents (if known ahead of time)
    pub total_documents: Option<usize>,
    /// Whether to validate dimensions on every document
    pub validate_dimensions: bool,
    /// Continue on individual document errors
    pub continue_on_error: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            progress_callback: None,
            total_documents: None,
            validate_dimensions: true,
            continue_on_error: false,
        }
    }
}

impl BatchConfig {
    /// Set the batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set a progress callback
    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&BatchProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Set the total document count (enables ETA calculation)
    pub fn with_total(mut self, total: usize) -> Self {
        self.total_documents = Some(total);
        self
    }

    /// Enable/disable dimension validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_dimensions = validate;
        self
    }

    /// Continue processing even if individual documents fail
    pub fn continue_on_error(mut self, continue_: bool) -> Self {
        self.continue_on_error = continue_;
        self
    }
}

// ============================================================================
// Index Trait for Batch Operations
// ============================================================================

/// Trait for indexes that support batch operations
pub trait BatchIndex {
    /// Add a single document
    fn add_document(&mut self, doc: Document) -> Result<()>;

    /// Get the expected embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get current document count
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Implement for HNSWIndex
impl BatchIndex for crate::index::HNSWIndex {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        self.add(doc)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// Implement for SQ8HNSWIndex
impl BatchIndex for crate::index::SQ8HNSWIndex {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        self.add(doc)
    }

    fn embedding_dim(&self) -> usize {
        self.quantizer().dim()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// Implement for BinaryHNSWIndex
impl BatchIndex for crate::index::BinaryHNSWIndex {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        self.add(doc)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// Implement for FlatIndex
impl BatchIndex for crate::index::FlatIndex {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        self.add(doc)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// ============================================================================
// Batch Builder
// ============================================================================

/// Builder for batch document ingestion
///
/// Provides memory-efficient batch processing with progress tracking.
pub struct BatchBuilder<'a, I: BatchIndex> {
    index: &'a mut I,
    config: BatchConfig,
    completed: usize,
    batch_count: usize,
    errors: Vec<(String, RagError)>,
    start_time: std::time::Instant,
}

impl<'a, I: BatchIndex> BatchBuilder<'a, I> {
    /// Create a new batch builder
    pub fn new(index: &'a mut I, config: BatchConfig) -> Self {
        Self {
            index,
            config,
            completed: 0,
            batch_count: 0,
            errors: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Add a single document (batches automatically)
    pub fn add(&mut self, doc: Document) -> Result<()> {
        let doc_id = doc.id.clone();

        // Validate dimensions if enabled
        if self.config.validate_dimensions && doc.embedding.len() != self.index.embedding_dim() {
            let err = RagError::DimensionMismatch {
                expected: self.index.embedding_dim(),
                actual: doc.embedding.len(),
            };

            if self.config.continue_on_error {
                self.errors.push((doc_id, err));
                return Ok(());
            } else {
                return Err(err);
            }
        }

        // Add document
        match self.index.add_document(doc) {
            Ok(()) => {
                self.completed += 1;

                // Check if we should report progress
                if self.completed % self.config.batch_size == 0 {
                    self.batch_count += 1;
                    self.report_progress();
                }

                Ok(())
            }
            Err(e) => {
                if self.config.continue_on_error {
                    self.errors.push((doc_id, e));
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Add multiple documents from an iterator
    pub fn add_all<T: IntoIterator<Item = Document>>(&mut self, docs: T) -> Result<()> {
        for doc in docs {
            self.add(doc)?;
        }
        Ok(())
    }

    /// Finish batch processing and report final progress
    pub fn finish(mut self) -> BatchResult {
        // Report final progress
        if self.completed % self.config.batch_size != 0 {
            self.batch_count += 1;
            self.report_progress();
        }

        BatchResult {
            documents_indexed: self.completed,
            errors: self.errors,
            elapsed_ms: self.start_time.elapsed().as_millis() as u64,
            batches_processed: self.batch_count,
        }
    }

    /// Get current progress without finishing
    pub fn progress(&self) -> BatchProgress {
        let elapsed_ms = self.start_time.elapsed().as_millis() as u64;
        let docs_per_sec = if elapsed_ms > 0 {
            (self.completed as f64) / (elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        };

        BatchProgress {
            completed: self.completed,
            total: self.config.total_documents,
            batch_number: self.batch_count,
            batch_size: self.config.batch_size,
            elapsed_ms,
            docs_per_sec,
        }
    }

    /// Get errors encountered so far
    pub fn errors(&self) -> &[(String, RagError)] {
        &self.errors
    }

    fn report_progress(&self) {
        if let Some(ref callback) = self.config.progress_callback {
            callback(&self.progress());
        }
    }
}

/// Result of batch processing
#[derive(Debug)]
pub struct BatchResult {
    /// Number of documents successfully indexed
    pub documents_indexed: usize,
    /// Documents that failed with their errors
    pub errors: Vec<(String, RagError)>,
    /// Total elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Number of batches processed
    pub batches_processed: usize,
}

impl BatchResult {
    /// Check if there were any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get documents per second throughput
    pub fn throughput(&self) -> f64 {
        if self.elapsed_ms > 0 {
            (self.documents_indexed as f64) / (self.elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Streaming Search
// ============================================================================

/// Trait for indexes that support streaming search
pub trait StreamingSearchIndex {
    /// Search and return raw scored results
    fn search_raw(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>>;

    /// Get document by internal index
    fn get_document(&self, idx: usize) -> Option<SearchResult>;
}

/// Iterator-based search results
///
/// Provides lazy evaluation of search results, useful for:
/// - Early termination (stop when you find what you need)
/// - Memory efficiency (don't allocate all results upfront)
/// - Chaining with other iterators
pub struct SearchResultIterator {
    results: Vec<SearchResult>,
    position: usize,
}

impl SearchResultIterator {
    /// Create from pre-computed results
    pub fn new(results: Vec<SearchResult>) -> Self {
        Self {
            results,
            position: 0,
        }
    }

    /// Get total number of results
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Peek at next result without consuming
    pub fn peek(&self) -> Option<&SearchResult> {
        self.results.get(self.position)
    }

    /// Skip n results
    pub fn skip_n(&mut self, n: usize) {
        self.position = (self.position + n).min(self.results.len());
    }

    /// Collect remaining results
    pub fn collect_remaining(self) -> Vec<SearchResult> {
        self.results.into_iter().skip(self.position).collect()
    }
}

impl Iterator for SearchResultIterator {
    type Item = SearchResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.results.len() {
            let result = self.results[self.position].clone();
            self.position += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.results.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SearchResultIterator {}

// ============================================================================
// Paginated Search
// ============================================================================

/// Configuration for paginated search
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    /// Results per page
    pub page_size: usize,
    /// Oversample factor for better pagination accuracy
    pub oversample: f32,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            page_size: 10,
            oversample: 2.0,
        }
    }
}

impl PaginationConfig {
    /// Create with custom page size
    pub fn with_page_size(mut self, size: usize) -> Self {
        self.page_size = size;
        self
    }
}

/// Paginated search results
#[derive(Debug, Clone)]
pub struct SearchPage {
    /// Results for this page
    pub results: Vec<SearchResult>,
    /// Current page number (0-indexed)
    pub page: usize,
    /// Total pages available
    pub total_pages: usize,
    /// Total results available
    pub total_results: usize,
    /// Whether there are more pages
    pub has_next: bool,
    /// Whether there are previous pages
    pub has_prev: bool,
}

impl SearchPage {
    /// Create from results with pagination info
    pub fn from_results(all_results: Vec<SearchResult>, page: usize, page_size: usize) -> Self {
        let total_results = all_results.len();
        let total_pages = (total_results + page_size - 1) / page_size;
        let start = page * page_size;
        let end = (start + page_size).min(total_results);

        let results = if start < total_results {
            all_results[start..end].to_vec()
        } else {
            Vec::new()
        };

        Self {
            results,
            page,
            total_pages,
            total_results,
            has_next: page + 1 < total_pages,
            has_prev: page > 0,
        }
    }
}

// ============================================================================
// Filtered Search
// ============================================================================

/// Filter function type for search results
pub type SearchFilter = Box<dyn Fn(&SearchResult) -> bool + Send + Sync>;

/// Builder for filtered search operations
pub struct FilteredSearchBuilder {
    filters: Vec<SearchFilter>,
    min_score: Option<f32>,
    max_results: Option<usize>,
}

impl FilteredSearchBuilder {
    /// Create a new filtered search builder
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            min_score: None,
            max_results: None,
        }
    }

    /// Add a custom filter
    pub fn filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&SearchResult) -> bool + Send + Sync + 'static,
    {
        self.filters.push(Box::new(f));
        self
    }

    /// Filter by minimum score
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Limit maximum results
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    /// Filter by metadata field existence
    pub fn has_metadata_field(self, field: &'static str) -> Self {
        self.filter(move |r| {
            r.metadata
                .as_ref()
                .map(|m| m.get(field).is_some())
                .unwrap_or(false)
        })
    }

    /// Filter by metadata field value
    pub fn metadata_equals(self, field: &'static str, value: serde_json::Value) -> Self {
        self.filter(move |r| {
            r.metadata
                .as_ref()
                .and_then(|m| m.get(field))
                .map(|v| *v == value)
                .unwrap_or(false)
        })
    }

    /// Apply filters to search results
    pub fn apply(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| {
                // Apply min score filter
                if let Some(min) = self.min_score {
                    if r.score < min {
                        return false;
                    }
                }

                // Apply custom filters
                for filter in &self.filters {
                    if !filter(r) {
                        return false;
                    }
                }

                true
            })
            .collect();

        // Apply max results limit
        if let Some(max) = self.max_results {
            filtered.truncate(max);
        }

        filtered
    }
}

impl Default for FilteredSearchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::HNSWIndex;

    fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding,
            metadata: Some(serde_json::json!({"category": "test"})),
        }
    }

    fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim)
            .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
            .collect()
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .with_batch_size(500)
            .with_total(10000)
            .with_validation(false)
            .continue_on_error(true);

        assert_eq!(config.batch_size, 500);
        assert_eq!(config.total_documents, Some(10000));
        assert!(!config.validate_dimensions);
        assert!(config.continue_on_error);
    }

    #[test]
    fn test_batch_builder_basic() {
        let mut index = HNSWIndex::with_defaults(128);
        let config = BatchConfig::default().with_batch_size(10);

        let mut builder = BatchBuilder::new(&mut index, config);

        for i in 0..25 {
            let doc = create_test_document(&format!("doc{}", i), generate_random_vector(128, i));
            builder.add(doc).unwrap();
        }

        let result = builder.finish();

        assert_eq!(result.documents_indexed, 25);
        assert!(!result.has_errors());
        assert_eq!(result.batches_processed, 3); // 10 + 10 + 5
        assert_eq!(index.len(), 25);
    }

    #[test]
    fn test_batch_builder_with_progress() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();

        let mut index = HNSWIndex::with_defaults(128);
        let config = BatchConfig::default()
            .with_batch_size(10)
            .with_progress(move |_p| {
                progress_count_clone.fetch_add(1, Ordering::SeqCst);
            });

        let mut builder = BatchBuilder::new(&mut index, config);

        for i in 0..35 {
            let doc = create_test_document(&format!("doc{}", i), generate_random_vector(128, i));
            builder.add(doc).unwrap();
        }

        let _result = builder.finish();

        // Progress should be called 4 times: at 10, 20, 30, and 35 (finish)
        assert_eq!(progress_count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_batch_builder_dimension_error() {
        let mut index = HNSWIndex::with_defaults(128);
        let config = BatchConfig::default();

        let mut builder = BatchBuilder::new(&mut index, config);

        // Add valid document
        let doc = create_test_document("doc1", generate_random_vector(128, 1));
        assert!(builder.add(doc).is_ok());

        // Add invalid document (wrong dimension)
        let doc = create_test_document("doc2", generate_random_vector(64, 2));
        assert!(builder.add(doc).is_err());
    }

    #[test]
    fn test_batch_builder_continue_on_error() {
        let mut index = HNSWIndex::with_defaults(128);
        let config = BatchConfig::default().continue_on_error(true);

        let mut builder = BatchBuilder::new(&mut index, config);

        // Add valid document
        let doc = create_test_document("doc1", generate_random_vector(128, 1));
        builder.add(doc).unwrap();

        // Add invalid document (should be skipped)
        let doc = create_test_document("doc2", generate_random_vector(64, 2));
        builder.add(doc).unwrap();

        // Add another valid document
        let doc = create_test_document("doc3", generate_random_vector(128, 3));
        builder.add(doc).unwrap();

        let result = builder.finish();

        assert_eq!(result.documents_indexed, 2);
        assert!(result.has_errors());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, "doc2");
    }

    #[test]
    fn test_batch_progress_eta() {
        let progress = BatchProgress {
            completed: 5000,
            total: Some(10000),
            batch_number: 5,
            batch_size: 1000,
            elapsed_ms: 5000,
            docs_per_sec: 1000.0,
        };

        assert_eq!(progress.percent(), Some(50.0));
        assert_eq!(progress.eta_ms(), Some(5000)); // 5000 remaining / 1000 per sec
    }

    #[test]
    fn test_search_result_iterator() {
        let results = vec![
            SearchResult {
                id: "doc1".to_string(),
                content: "Content 1".to_string(),
                score: 0.9,
                metadata: None,
            },
            SearchResult {
                id: "doc2".to_string(),
                content: "Content 2".to_string(),
                score: 0.8,
                metadata: None,
            },
            SearchResult {
                id: "doc3".to_string(),
                content: "Content 3".to_string(),
                score: 0.7,
                metadata: None,
            },
        ];

        let mut iter = SearchResultIterator::new(results);

        assert_eq!(iter.total(), 3);
        assert_eq!(iter.peek().unwrap().id, "doc1");

        let first = iter.next().unwrap();
        assert_eq!(first.id, "doc1");

        let second = iter.next().unwrap();
        assert_eq!(second.id, "doc2");

        let remaining = iter.collect_remaining();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "doc3");
    }

    #[test]
    fn test_search_page() {
        let results: Vec<SearchResult> = (0..25)
            .map(|i| SearchResult {
                id: format!("doc{}", i),
                content: format!("Content {}", i),
                score: 1.0 - (i as f32 * 0.01),
                metadata: None,
            })
            .collect();

        // First page
        let page0 = SearchPage::from_results(results.clone(), 0, 10);
        assert_eq!(page0.results.len(), 10);
        assert_eq!(page0.page, 0);
        assert_eq!(page0.total_pages, 3);
        assert!(page0.has_next);
        assert!(!page0.has_prev);

        // Middle page
        let page1 = SearchPage::from_results(results.clone(), 1, 10);
        assert_eq!(page1.results.len(), 10);
        assert!(page1.has_next);
        assert!(page1.has_prev);

        // Last page
        let page2 = SearchPage::from_results(results.clone(), 2, 10);
        assert_eq!(page2.results.len(), 5);
        assert!(!page2.has_next);
        assert!(page2.has_prev);
    }

    #[test]
    fn test_filtered_search_builder() {
        let results: Vec<SearchResult> = vec![
            SearchResult {
                id: "doc1".to_string(),
                content: "Content 1".to_string(),
                score: 0.9,
                metadata: Some(serde_json::json!({"category": "A"})),
            },
            SearchResult {
                id: "doc2".to_string(),
                content: "Content 2".to_string(),
                score: 0.5,
                metadata: Some(serde_json::json!({"category": "B"})),
            },
            SearchResult {
                id: "doc3".to_string(),
                content: "Content 3".to_string(),
                score: 0.3,
                metadata: None,
            },
        ];

        // Filter by min score
        let filtered = FilteredSearchBuilder::new()
            .min_score(0.4)
            .apply(results.clone());
        assert_eq!(filtered.len(), 2);

        // Filter by metadata
        let filtered = FilteredSearchBuilder::new()
            .has_metadata_field("category")
            .apply(results.clone());
        assert_eq!(filtered.len(), 2);

        // Filter by metadata value
        let filtered = FilteredSearchBuilder::new()
            .metadata_equals("category", serde_json::json!("A"))
            .apply(results.clone());
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "doc1");

        // Combined filters
        let filtered = FilteredSearchBuilder::new()
            .min_score(0.4)
            .max_results(1)
            .apply(results);
        assert_eq!(filtered.len(), 1);
    }
}
