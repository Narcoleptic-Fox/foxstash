//! Cached embedder with LRU caching for improved performance
//!
//! This module provides a caching layer on top of the ONNX embedder to avoid
//! redundant embedding computations for frequently used texts.

#[cfg(feature = "onnx")]
use lru::LruCache;
#[cfg(feature = "onnx")]
use std::num::NonZeroUsize;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use crate::{Result, RagError};
use super::OnnxEmbedder;

/// A cached embedder that wraps an ONNX embedder with an LRU cache
///
/// The cache stores embeddings keyed by the input text, significantly improving
/// performance when the same texts are embedded multiple times. This is common
/// in RAG scenarios where queries may be repeated or similar.
///
/// # Thread Safety
///
/// The cache is protected by a `Mutex`, making the `CachedEmbedder` safe to use
/// across multiple threads. However, contention may occur under heavy concurrent
/// access patterns.
///
/// # Memory Usage
///
/// Each cache entry stores the input text as a `String` and the embedding as a
/// `Vec<f32>`. For a typical 384-dimensional embedding, this is approximately:
/// - Key: text length in bytes
/// - Value: 384 * 4 bytes = 1.5KB
///
/// With the default cache size of 10,000 entries, expect ~15MB for embeddings
/// plus the overhead of stored text.
///
/// # Example
///
/// ```ignore
/// use foxstash_core::embedding::{OnnxEmbedder, CachedEmbedder};
///
/// let embedder = OnnxEmbedder::new("model.onnx", "tokenizer.json")?;
/// let cached = CachedEmbedder::new(embedder, 10_000);
///
/// // First call will compute and cache
/// let embedding1 = cached.embed("Hello, world!")?;
///
/// // Second call will use cached result
/// let embedding2 = cached.embed("Hello, world!")?;
///
/// // Check cache statistics
/// let (hits, misses) = cached.cache_stats();
/// assert_eq!(hits, 1);
/// assert_eq!(misses, 1);
/// ```
#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct CachedEmbedder {
    /// The underlying ONNX embedder (mutable for inference)
    embedder: Mutex<OnnxEmbedder>,
    /// LRU cache storing text -> embedding mappings
    cache: Mutex<LruCache<String, Vec<f32>>>,
    /// Cache hit counter
    hits: Mutex<usize>,
    /// Cache miss counter
    misses: Mutex<usize>,
}

#[cfg(feature = "onnx")]
impl CachedEmbedder {
    /// Creates a new cached embedder
    ///
    /// # Arguments
    ///
    /// * `embedder` - The underlying ONNX embedder to use for computing embeddings
    /// * `max_cache_size` - Maximum number of embeddings to cache (LRU eviction)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = OnnxEmbedder::new("model.onnx", "tokenizer.json")?;
    /// let cached = CachedEmbedder::new(embedder, 10_000);
    /// ```
    pub fn new(embedder: OnnxEmbedder, max_cache_size: usize) -> Self {
        let cache_size = NonZeroUsize::new(max_cache_size)
            .unwrap_or_else(|| NonZeroUsize::new(1).unwrap());

        Self {
            embedder: Mutex::new(embedder),
            cache: Mutex::new(LruCache::new(cache_size)),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Creates a new cached embedder with default cache size (10,000 entries)
    ///
    /// # Arguments
    ///
    /// * `embedder` - The underlying ONNX embedder to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = OnnxEmbedder::new("model.onnx", "tokenizer.json")?;
    /// let cached = CachedEmbedder::with_default_size(embedder);
    /// ```
    pub fn with_default_size(embedder: OnnxEmbedder) -> Self {
        Self::new(embedder, 10_000)
    }

    /// Embeds a single text, using cache if available
    ///
    /// This method first checks the cache for the given text. If found, it returns
    /// the cached embedding and increments the hit counter. Otherwise, it computes
    /// the embedding using the underlying embedder, caches it, and increments the
    /// miss counter.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector of floats representing the embedding
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying embedder fails to compute the embedding
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedding = cached.embed("Hello, world!")?;
    /// assert_eq!(embedding.len(), 384); // For MiniLM-L6-v2
    /// ```
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Try to get from cache first
        {
            let mut cache = self.cache.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Cache lock poisoned: {}", e)))?;

            if let Some(embedding) = cache.get(text) {
                // Cache hit
                let mut hits = self.hits.lock()
                    .map_err(|e| RagError::EmbeddingError(format!("Hits lock poisoned: {}", e)))?;
                *hits += 1;
                return Ok(embedding.clone());
            }
        }

        // Cache miss - compute embedding
        let embedding = {
            let mut embedder = self.embedder.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Embedder lock poisoned: {}", e)))?;
            embedder.embed(text)?
        };

        // Store in cache
        {
            let mut cache = self.cache.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Cache lock poisoned: {}", e)))?;
            cache.put(text.to_string(), embedding.clone());

            let mut misses = self.misses.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Misses lock poisoned: {}", e)))?;
            *misses += 1;
        }

        Ok(embedding)
    }

    /// Embeds multiple texts in batch, utilizing cache for individual texts
    ///
    /// This method processes each text independently, checking the cache first
    /// before computing embeddings. Texts are batched together for efficient
    /// computation when cache misses occur.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text references to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input text in the same order
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying embedder fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let texts = ["Hello", "World", "RAG"];
    /// let embeddings = cached.embed_batch(&texts)?;
    /// assert_eq!(embeddings.len(), 3);
    /// ```
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut to_compute = Vec::new();
        let mut to_compute_indices = Vec::new();

        // Check cache for each text
        {
            let mut cache = self.cache.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Cache lock poisoned: {}", e)))?;
            let mut hits = self.hits.lock()
                .map_err(|e| RagError::EmbeddingError(format!("Hits lock poisoned: {}", e)))?;

            for (idx, &text) in texts.iter().enumerate() {
                if let Some(embedding) = cache.get(text) {
                    // Cache hit - store result at correct index
                    results.push((idx, embedding.clone()));
                    *hits += 1;
                } else {
                    // Cache miss - mark for computation
                    to_compute.push(text);
                    to_compute_indices.push(idx);
                }
            }
        }

        // Compute embeddings for cache misses
        if !to_compute.is_empty() {
            let computed = {
                let mut embedder = self.embedder.lock()
                    .map_err(|e| RagError::EmbeddingError(format!("Embedder lock poisoned: {}", e)))?;
                embedder.embed_batch(&to_compute)?
            };

            // Cache the newly computed embeddings
            {
                let mut cache = self.cache.lock()
                    .map_err(|e| RagError::EmbeddingError(format!("Cache lock poisoned: {}", e)))?;
                let mut misses = self.misses.lock()
                    .map_err(|e| RagError::EmbeddingError(format!("Misses lock poisoned: {}", e)))?;

                for (text, embedding) in to_compute.iter().zip(computed.iter()) {
                    cache.put(text.to_string(), embedding.clone());
                    *misses += 1;
                }
            }

            // Add computed results
            for (idx, embedding) in to_compute_indices.into_iter().zip(computed.into_iter()) {
                results.push((idx, embedding));
            }
        }

        // Sort by original index and extract embeddings
        results.sort_by_key(|(idx, _)| *idx);
        Ok(results.into_iter().map(|(_, emb)| emb).collect())
    }

    /// Clears all cached embeddings
    ///
    /// This method removes all entries from the cache and resets the hit/miss
    /// counters. Use this to free memory or reset statistics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// cached.clear_cache();
    /// assert_eq!(cached.cache_size(), 0);
    /// ```
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut hits) = self.hits.lock() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.misses.lock() {
            *misses = 0;
        }
    }

    /// Returns the current number of entries in the cache
    ///
    /// # Example
    ///
    /// ```ignore
    /// let size = cached.cache_size();
    /// println!("Cache contains {} embeddings", size);
    /// ```
    pub fn cache_size(&self) -> usize {
        self.cache.lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    /// Returns cache hit and miss statistics
    ///
    /// # Returns
    ///
    /// A tuple of `(hits, misses)` where:
    /// - `hits`: Number of times an embedding was found in cache
    /// - `misses`: Number of times an embedding had to be computed
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (hits, misses) = cached.cache_stats();
    /// let hit_rate = hits as f64 / (hits + misses) as f64;
    /// println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
    /// ```
    pub fn cache_stats(&self) -> (usize, usize) {
        let hits = self.hits.lock().map(|h| *h).unwrap_or(0);
        let misses = self.misses.lock().map(|m| *m).unwrap_or(0);
        (hits, misses)
    }

    /// Returns the maximum cache size
    ///
    /// This is the capacity configured at creation time, not the current
    /// number of entries (use `cache_size()` for that).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let max_size = cached.max_cache_size();
    /// println!("Cache capacity: {}", max_size);
    /// ```
    pub fn max_cache_size(&self) -> usize {
        self.cache.lock()
            .map(|cache| cache.cap().get())
            .unwrap_or(0)
    }

    /// Returns the cache hit rate as a percentage
    ///
    /// # Returns
    ///
    /// The hit rate as a value between 0.0 and 1.0, or None if no operations
    /// have been performed yet.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(rate) = cached.hit_rate() {
    ///     println!("Hit rate: {:.2}%", rate * 100.0);
    /// }
    /// ```
    pub fn hit_rate(&self) -> Option<f64> {
        let (hits, misses) = self.cache_stats();
        let total = hits + misses;

        if total == 0 {
            None
        } else {
            Some(hits as f64 / total as f64)
        }
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    // Tests for CachedEmbedder functionality
    // Most tests are marked #[ignore] because they require actual ONNX model files.
    // Run with: cargo test -- --ignored --test-threads=1

    /// Test cache creation and basic properties
    #[test]
    #[ignore]
    fn test_cache_creation() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        // Test with custom size
        let cached = CachedEmbedder::new(embedder, 42);
        assert_eq!(cached.max_cache_size(), 42);
        assert_eq!(cached.cache_size(), 0);
        assert_eq!(cached.cache_stats(), (0, 0));
        assert_eq!(cached.hit_rate(), None);
    }

    /// Test default cache size
    #[test]
    #[ignore]
    fn test_with_default_size() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::with_default_size(embedder);
        assert_eq!(cached.max_cache_size(), 10_000);
    }

    /// Test that zero cache size is handled (converted to 1)
    #[test]
    #[ignore]
    fn test_zero_cache_size() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 0);
        assert_eq!(cached.max_cache_size(), 1);
    }

    /// Test cache hit - same text embedded twice
    #[test]
    #[ignore]
    fn test_cache_hit() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // First call - cache miss
        let embedding1 = cached.embed("Hello, world!").unwrap();
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
        assert_eq!(cached.cache_size(), 1);

        // Second call - cache hit
        let embedding2 = cached.embed("Hello, world!").unwrap();
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(cached.cache_size(), 1);

        // Embeddings should be identical
        assert_eq!(embedding1, embedding2);
        assert_eq!(cached.hit_rate(), Some(0.5));
    }

    /// Test cache misses - different texts
    #[test]
    #[ignore]
    fn test_cache_miss() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // Different texts should all be cache misses
        let emb1 = cached.embed("Hello").unwrap();
        let emb2 = cached.embed("World").unwrap();
        let emb3 = cached.embed("RAG").unwrap();

        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 3);
        assert_eq!(cached.cache_size(), 3);
        assert_eq!(cached.hit_rate(), Some(0.0));

        // Embeddings should be different
        assert_ne!(emb1, emb2);
        assert_ne!(emb2, emb3);
    }

    /// Test cache eviction with LRU policy
    #[test]
    #[ignore]
    fn test_cache_eviction() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 3); // Small cache

        // Fill cache to capacity
        let _emb1 = cached.embed("text1").unwrap();
        let _emb2 = cached.embed("text2").unwrap();
        let _emb3 = cached.embed("text3").unwrap();
        assert_eq!(cached.cache_size(), 3);

        // Add one more - should evict oldest (text1) via LRU
        let _emb4 = cached.embed("text4").unwrap();
        assert_eq!(cached.cache_size(), 3);

        // Access text2 and text3 (hits)
        let _ = cached.embed("text2").unwrap();
        let _ = cached.embed("text3").unwrap();
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 4);

        // text1 was evicted, so accessing it should be a miss
        let _ = cached.embed("text1").unwrap();
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 5);
    }

    /// Test clearing the cache
    #[test]
    #[ignore]
    fn test_clear_cache() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // Add some embeddings
        let _emb1 = cached.embed("Hello").unwrap();
        let _emb2 = cached.embed("World").unwrap();
        assert_eq!(cached.cache_size(), 2);
        assert_eq!(cached.cache_stats(), (0, 2));

        // Clear cache
        cached.clear_cache();
        assert_eq!(cached.cache_size(), 0);
        assert_eq!(cached.cache_stats(), (0, 0));
        assert_eq!(cached.hit_rate(), None);

        // Embeddings should be computed again
        let _emb1_again = cached.embed("Hello").unwrap();
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }

    /// Test cache statistics tracking
    #[test]
    #[ignore]
    fn test_cache_statistics() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // Initial state
        assert_eq!(cached.cache_stats(), (0, 0));
        assert_eq!(cached.hit_rate(), None);

        // First embedding - miss
        let _emb = cached.embed("test").unwrap();
        assert_eq!(cached.cache_stats(), (0, 1));
        assert_eq!(cached.hit_rate(), Some(0.0));

        // Second embedding - hit
        let _emb = cached.embed("test").unwrap();
        assert_eq!(cached.cache_stats(), (1, 1));
        assert_eq!(cached.hit_rate(), Some(0.5));

        // Third embedding - hit
        let _emb = cached.embed("test").unwrap();
        assert_eq!(cached.cache_stats(), (2, 1));
        assert!((cached.hit_rate().unwrap() - 0.6666).abs() < 0.01);
    }

    /// Test batch embedding with all cached entries
    #[test]
    #[ignore]
    fn test_batch_embed_all_cached() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);
        let texts = ["Hello", "World", "RAG"];

        // First batch - all misses
        let embeddings1 = cached.embed_batch(&texts).unwrap();
        assert_eq!(embeddings1.len(), 3);
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 3);

        // Second batch - all hits
        let embeddings2 = cached.embed_batch(&texts).unwrap();
        assert_eq!(embeddings2.len(), 3);
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 3);
        assert_eq!(misses, 3);

        // Embeddings should be identical
        assert_eq!(embeddings1, embeddings2);
    }

    /// Test batch embedding with mixed cache hits and misses
    #[test]
    #[ignore]
    fn test_batch_embed_mixed() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // Pre-cache some texts
        let _emb1 = cached.embed("Hello").unwrap();
        let _emb2 = cached.embed("World").unwrap();
        let initial_stats = cached.cache_stats();
        assert_eq!(initial_stats, (0, 2));

        // Batch with mix of cached and uncached
        let texts = ["Hello", "World", "RAG", "System"];
        let embeddings = cached.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 4);

        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 2); // Hello and World
        assert_eq!(misses, 4); // Initial 2 + RAG + System
    }

    /// Test that batch embedding preserves input order
    #[test]
    #[ignore]
    fn test_batch_embed_order_preserved() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        // Pre-cache in different order
        let emb_rag = cached.embed("RAG").unwrap();
        let emb_hello = cached.embed("Hello").unwrap();

        // Batch request in different order
        let texts = ["Hello", "World", "RAG"];
        let embeddings = cached.embed_batch(&texts).unwrap();

        // Verify order is preserved
        assert_eq!(embeddings[0], emb_hello);
        assert_eq!(embeddings[2], emb_rag);

        // World should be computed
        let emb_world = cached.embed("World").unwrap();
        assert_eq!(embeddings[1], emb_world);
    }

    /// Test concurrent access to the cache
    #[test]
    #[ignore]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = Arc::new(CachedEmbedder::new(embedder, 100));
        let mut handles = vec![];

        // Spawn multiple threads accessing the cache
        for i in 0..10 {
            let cached_clone = Arc::clone(&cached);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let text = format!("text_{}", (i + j) % 5);
                    let _ = cached_clone.embed(&text).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify cache has entries and all operations completed
        assert!(cached.cache_size() > 0);
        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits + misses, 100); // 10 threads * 10 operations

        // Since we're using modulo 5, there should be many cache hits
        assert!(hits > 0, "Expected cache hits with repeated texts");
    }

    /// Test embedding dimension matches
    #[test]
    #[ignore]
    fn test_embedding_dimension() {
        let embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json"
        ).expect("Failed to create embedder");

        let cached = CachedEmbedder::new(embedder, 100);

        let embedding = cached.embed("Test text").unwrap();
        assert_eq!(embedding.len(), 384, "MiniLM-L6-v2 should produce 384-dim embeddings");

        // Verify normalization
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be L2-normalized");
    }
}
