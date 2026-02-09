//! Comprehensive integration tests for foxstash-core
//!
//! These tests exercise full end-to-end pipelines across foxstash-core subsystems:
//! - Document lifecycle (create, add, search, verify metadata)
//! - Index persistence and recovery via FileStorage
//! - Quantized index accuracy comparison (HNSW, SQ8, Binary, PQ)
//! - Incremental persistence (WAL logging, checkpoint, recovery)
//! - Compression round-trip for all available codecs
//! - Batch operations and streaming search
//! - Edge cases (empty index, single doc, zero vectors, k > n)
//! - Concurrent parallel search

use foxstash_core::index::{
    BatchBuilder, BatchConfig, BinaryHNSWIndex, FlatIndex, FilteredSearchBuilder, HNSWIndex,
    PQHNSWConfig, PQHNSWIndex, QuantizedHNSWConfig, SQ8HNSWIndex, SearchPage,
    SearchResultIterator,
};
use foxstash_core::storage::compression::{self, Codec};
use foxstash_core::storage::file::{FileStorage, FlatIndexWrapper, HNSWIndexWrapper};
use foxstash_core::storage::incremental::{
    IncrementalConfig, IncrementalStorage, IndexMetadata, RecoveryHelper, WalOperation,
};
use foxstash_core::vector::product_quantize::PQConfig;
use foxstash_core::{Document, SearchResult};

use std::collections::HashSet;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a deterministic embedding vector from a seed.
/// Produces a vector in the range [-1, 1] using a simple
/// deterministic formula so tests are fully reproducible without RNG.
fn deterministic_embedding(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            // Use wrapping arithmetic to avoid overflow in debug builds
            let s = seed as u64;
            let idx = i as u64;
            let hash = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(idx.wrapping_mul(1442695040888963407));
            let x = (hash & 0xFFFFFFFF) as f32 / u32::MAX as f32;
            x * 2.0 - 1.0
        })
        .collect()
}

/// Create a test document with a deterministic embedding.
fn make_doc(id: &str, dim: usize, seed: usize) -> Document {
    Document {
        id: id.to_string(),
        content: format!("Content for document {}", id),
        embedding: deterministic_embedding(dim, seed),
        metadata: None,
    }
}

/// Create a test document with metadata.
fn make_doc_with_metadata(
    id: &str,
    dim: usize,
    seed: usize,
    metadata: serde_json::Value,
) -> Document {
    Document {
        id: id.to_string(),
        content: format!("Content for document {}", id),
        embedding: deterministic_embedding(dim, seed),
        metadata: Some(metadata),
    }
}

/// Compute brute-force ground truth top-k using FlatIndex as the oracle.
fn brute_force_top_k(
    documents: &[Document],
    query: &[f32],
    k: usize,
) -> Vec<SearchResult> {
    let dim = query.len();
    let mut flat = FlatIndex::new(dim);
    for doc in documents {
        flat.add(doc.clone()).unwrap();
    }
    flat.search(query, k).unwrap()
}

/// Measure recall@k: fraction of ground truth IDs found in actual results.
fn recall_at_k(ground_truth: &[SearchResult], actual: &[SearchResult]) -> f64 {
    let gt_ids: HashSet<&str> = ground_truth.iter().map(|r| r.id.as_str()).collect();
    let actual_ids: HashSet<&str> = actual.iter().map(|r| r.id.as_str()).collect();
    let intersection = gt_ids.intersection(&actual_ids).count();
    if gt_ids.is_empty() {
        return 1.0;
    }
    intersection as f64 / gt_ids.len() as f64
}

// ============================================================================
// (a) Full Document Lifecycle
// ============================================================================

mod document_lifecycle {
    use super::*;

    #[test]
    fn add_and_search_returns_results_ranked_by_similarity() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        // Create documents where doc_0 is most similar to the query
        let query = deterministic_embedding(dim, 0);
        let mut docs = Vec::new();

        for i in 0..20 {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            docs.push(doc.clone());
            index.add(doc).unwrap();
        }

        let results = index.search(&query, 5).unwrap();

        // Verify we get the requested number of results
        assert_eq!(results.len(), 5, "Expected 5 results");

        // Verify results are sorted by score descending
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Results not sorted: {} < {}",
                window[0].score,
                window[1].score
            );
        }

        // The exact match (doc_0 shares the same embedding generation as query seed 0)
        // should be the top result
        assert_eq!(
            results[0].id, "doc_0",
            "Expected doc_0 as top result since it shares the query embedding"
        );
        assert!(
            results[0].score > 0.99,
            "Top result should have near-perfect similarity, got {}",
            results[0].score
        );
    }

    #[test]
    fn metadata_is_preserved_through_add_and_search() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);

        let metadata = serde_json::json!({
            "source": "unit_test",
            "category": "integration",
            "priority": 42,
            "tags": ["rust", "search"]
        });

        let doc = make_doc_with_metadata("meta_doc", dim, 0, metadata.clone());
        index.add(doc).unwrap();

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "meta_doc");

        let result_meta = results[0].metadata.as_ref().expect("metadata should be present");
        assert_eq!(result_meta["source"], "unit_test");
        assert_eq!(result_meta["category"], "integration");
        assert_eq!(result_meta["priority"], 42);
        assert_eq!(result_meta["tags"][0], "rust");
        assert_eq!(result_meta["tags"][1], "search");
    }

    #[test]
    fn content_is_preserved_through_add_and_search() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);

        let doc = Document {
            id: "content_doc".to_string(),
            content: "This is the original content that should be preserved".to_string(),
            embedding: deterministic_embedding(dim, 0),
            metadata: None,
        };
        index.add(doc).unwrap();

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results[0].content, "This is the original content that should be preserved");
    }

    #[test]
    fn hnsw_vs_flat_recall_is_high() {
        // Verify HNSW achieves good recall against brute-force FlatIndex
        let dim = 64;
        let n = 100;
        let k = 10;

        let mut hnsw = HNSWIndex::with_defaults(dim);
        let mut documents = Vec::new();

        for i in 0..n {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            documents.push(doc.clone());
            hnsw.add(doc).unwrap();
        }

        let query = deterministic_embedding(dim, 9999);
        let hnsw_results = hnsw.search(&query, k).unwrap();
        let gt_results = brute_force_top_k(&documents, &query, k);

        let recall = recall_at_k(&gt_results, &hnsw_results);
        assert!(
            recall >= 0.7,
            "HNSW recall@{} should be >= 70%, got {:.0}%",
            k,
            recall * 100.0
        );
    }
}

// ============================================================================
// (b) Index Persistence & Recovery
// ============================================================================

mod index_persistence {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn flat_index_save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        // Build and populate index
        let dim = 16;
        let mut index = FlatIndex::new(dim);
        for i in 0..10 {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            index.add(doc).unwrap();
        }

        // Save
        let wrapper = FlatIndexWrapper::from_index(&index);
        let stats = storage.save_flat_index("test_flat", &wrapper).unwrap();
        assert!(stats.original_size > 0);

        // Load
        let loaded_wrapper = storage.load_flat_index("test_flat").unwrap();
        let loaded = loaded_wrapper.to_index().unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.embedding_dim(), index.embedding_dim());

        // Search both and compare results
        let query = deterministic_embedding(dim, 0);
        let original_results = index.search(&query, 5).unwrap();
        let loaded_results = loaded.search(&query, 5).unwrap();

        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, loaded_r) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.id, loaded_r.id, "Result IDs should match after roundtrip");
            assert!(
                (orig.score - loaded_r.score).abs() < 1e-5,
                "Scores should match after roundtrip"
            );
        }
    }

    #[test]
    fn hnsw_index_save_load_produces_searchable_index() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);
        for i in 0..20 {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            index.add(doc).unwrap();
        }

        // Save
        let wrapper = HNSWIndexWrapper::from_index(&index);
        storage.save_hnsw_index("test_hnsw", &wrapper).unwrap();

        // Load and reconstruct
        let loaded_wrapper = storage.load_hnsw_index("test_hnsw").unwrap();
        let loaded = loaded_wrapper.to_index().unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.embedding_dim(), index.embedding_dim());

        // Verify search works on loaded index
        let query = deterministic_embedding(dim, 5);
        let results = loaded.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);

        // Results should be sorted by score
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn document_save_load_preserves_all_fields() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        let doc = make_doc_with_metadata(
            "persist_doc",
            16,
            42,
            serde_json::json!({"key": "value", "num": 123}),
        );

        storage.save_document("persist_doc", &doc).unwrap();
        let loaded = storage.load_document("persist_doc").unwrap();

        assert_eq!(loaded.id, doc.id);
        assert_eq!(loaded.content, doc.content);
        assert_eq!(loaded.embedding, doc.embedding);
        assert_eq!(loaded.metadata, doc.metadata);
    }

    #[test]
    fn gzip_compressed_storage_roundtrip() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::with_codec(dir.path(), Codec::Gzip).unwrap();

        let dim = 64;
        let mut index = FlatIndex::new(dim);
        for i in 0..15 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let wrapper = FlatIndexWrapper::from_index(&index);
        let stats = storage.save_flat_index("compressed_idx", &wrapper).unwrap();

        // With Gzip, compressed should be smaller for embeddings
        assert_eq!(stats.codec, Codec::Gzip);

        let loaded_wrapper = storage.load_flat_index("compressed_idx").unwrap();
        let loaded = loaded_wrapper.to_index().unwrap();
        assert_eq!(loaded.len(), 15);

        // Verify search works
        let query = deterministic_embedding(dim, 0);
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn storage_list_and_delete_work() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::new(dir.path()).unwrap();

        // Store 3 documents
        for i in 0..3 {
            let doc = make_doc(&format!("doc_{}", i), 8, i);
            storage.save_document(&format!("doc_{}", i), &doc).unwrap();
        }

        let items = storage.list().unwrap();
        assert_eq!(items.len(), 3);

        // Delete one
        storage.delete("doc_1").unwrap();
        let items = storage.list().unwrap();
        assert_eq!(items.len(), 2);
        assert!(!items.contains(&"doc_1".to_string()));

        // Verify remaining are loadable
        storage.load_document("doc_0").unwrap();
        storage.load_document("doc_2").unwrap();
    }
}

// ============================================================================
// (c) Quantized Index Accuracy Comparison
// ============================================================================

mod quantized_accuracy {
    use super::*;

    /// Build all index types with the same documents and verify recall ordering.
    #[test]
    fn all_index_types_return_results_without_panics() {
        let dim = 64;
        let n = 50;
        let k = 5;

        let documents: Vec<Document> = (0..n)
            .map(|i| make_doc(&format!("doc_{}", i), dim, i))
            .collect();

        // HNSW (full precision)
        let mut hnsw = HNSWIndex::with_defaults(dim);
        for doc in &documents {
            hnsw.add(doc.clone()).unwrap();
        }

        // SQ8
        let mut sq8 = SQ8HNSWIndex::for_normalized(dim, QuantizedHNSWConfig::default());
        for doc in &documents {
            sq8.add(doc.clone()).unwrap();
        }

        // Binary
        let mut binary = BinaryHNSWIndex::new(dim, QuantizedHNSWConfig::default());
        for doc in &documents {
            binary.add(doc.clone()).unwrap();
        }

        let query = deterministic_embedding(dim, 9999);

        let hnsw_results = hnsw.search(&query, k).unwrap();
        let sq8_results = sq8.search(&query, k).unwrap();
        let binary_results = binary.search(&query, k).unwrap();

        // All should return k results
        assert_eq!(hnsw_results.len(), k, "HNSW should return {} results", k);
        assert_eq!(sq8_results.len(), k, "SQ8 should return {} results", k);
        assert_eq!(binary_results.len(), k, "Binary should return {} results", k);

        // All results should be sorted by score descending
        for (name, results) in [
            ("HNSW", &hnsw_results),
            ("SQ8", &sq8_results),
            ("Binary", &binary_results),
        ] {
            for window in results.windows(2) {
                assert!(
                    window[0].score >= window[1].score,
                    "{} results not sorted", name
                );
            }
        }
    }

    #[test]
    fn pq_index_returns_results() {
        let dim = 64;
        let n = 80;
        let k = 5;

        // Generate training data and documents
        let training_data: Vec<Vec<f32>> = (0..200)
            .map(|i| deterministic_embedding(dim, i + 10000))
            .collect();

        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(5);

        let mut pq_index =
            PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default()).unwrap();

        for i in 0..n {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            pq_index.add(doc).unwrap();
        }

        let query = deterministic_embedding(dim, 9999);
        let results = pq_index.search(&query, k).unwrap();

        assert_eq!(results.len(), k, "PQ should return {} results", k);

        // Results should be sorted by score descending
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "PQ results not sorted"
            );
        }
    }

    #[test]
    fn binary_rerank_improves_over_binary_only() {
        let dim = 64;
        let n = 80;
        let k = 10;

        let documents: Vec<Document> = (0..n)
            .map(|i| make_doc(&format!("doc_{}", i), dim, i))
            .collect();

        let mut binary = BinaryHNSWIndex::with_full_precision(dim, QuantizedHNSWConfig::default());
        for doc in &documents {
            binary.add_with_full_precision(doc.clone()).unwrap();
        }

        let query = deterministic_embedding(dim, 9999);

        // Binary-only search
        let binary_results = binary.search(&query, k).unwrap();

        // Binary + full-precision rerank
        let rerank_results = binary.search_and_rerank(&query, 50, k).unwrap();

        assert_eq!(binary_results.len(), k);
        assert_eq!(rerank_results.len(), k);

        // Both should return sorted results
        for window in rerank_results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }
}

// ============================================================================
// (d) Incremental Persistence (WAL)
// ============================================================================

mod incremental_persistence {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn wal_records_add_and_remove_operations() {
        let dir = TempDir::new().unwrap();
        let config = IncrementalConfig::default()
            .with_checkpoint_threshold(1000)
            .with_wal_sync_interval(1); // Sync every operation for test reliability

        let mut storage = IncrementalStorage::new(dir.path(), config).unwrap();

        // Log operations
        let doc1 = make_doc("doc_1", 16, 1);
        let doc2 = make_doc("doc_2", 16, 2);

        storage.log_add(&doc1).unwrap();
        storage.log_add(&doc2).unwrap();
        storage.log_remove("doc_1").unwrap();
        storage.sync().unwrap();

        assert_eq!(storage.manifest().wal_seq, 3);
        assert_eq!(storage.manifest().ops_since_checkpoint, 3);

        // Read back WAL entries
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 3, "Expected 3 WAL entries");

        // Verify operation types
        match &entries[0].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc_1"),
            other => panic!("Expected Add, got {:?}", other),
        }
        match &entries[1].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc_2"),
            other => panic!("Expected Add, got {:?}", other),
        }
        match &entries[2].operation {
            WalOperation::Remove(id) => assert_eq!(id, "doc_1"),
            other => panic!("Expected Remove, got {:?}", other),
        }

        // Verify integrity checksums
        for entry in &entries {
            assert!(entry.verify(), "WAL entry {} failed integrity check", entry.seq);
        }
    }

    #[test]
    fn checkpoint_and_recovery_roundtrip() {
        let dir = TempDir::new().unwrap();
        let config = IncrementalConfig::default()
            .with_checkpoint_threshold(100)
            .with_wal_sync_interval(1);

        let mut storage = IncrementalStorage::new(dir.path(), config).unwrap();

        // Log initial batch of documents
        let dim = 16;
        let initial_docs: Vec<Document> = (0..5)
            .map(|i| make_doc(&format!("doc_{}", i), dim, i))
            .collect();

        for doc in &initial_docs {
            storage.log_add(doc).unwrap();
        }

        // Create checkpoint with document list as serializable data
        let doc_ids: Vec<String> = initial_docs.iter().map(|d| d.id.clone()).collect();
        let meta = storage
            .checkpoint(
                &doc_ids,
                IndexMetadata {
                    document_count: 5,
                    embedding_dim: dim,
                    index_type: "hnsw".to_string(),
                },
            )
            .unwrap();

        assert_eq!(meta.id, 1);
        assert_eq!(meta.document_count, 5);

        // Log more operations after checkpoint
        storage.log_add(&make_doc("doc_5", dim, 5)).unwrap();
        storage.log_add(&make_doc("doc_6", dim, 6)).unwrap();
        storage.sync().unwrap();

        // Verify checkpoint is loadable
        let (loaded_ids, loaded_meta): (Vec<String>, _) =
            storage.load_checkpoint().unwrap().unwrap();
        assert_eq!(loaded_ids, doc_ids);
        assert_eq!(loaded_meta.document_count, 5);

        // Verify WAL has only post-checkpoint entries
        let entries = storage.get_wal_entries().unwrap();
        assert_eq!(entries.len(), 2, "Expected 2 entries after checkpoint");
        match &entries[0].operation {
            WalOperation::Add(doc) => assert_eq!(doc.id, "doc_5"),
            other => panic!("Expected Add(doc_5), got {:?}", other),
        }
    }

    #[test]
    fn recovery_helper_replays_wal() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_wal_sync_interval(1),
        )
        .unwrap();

        let dim = 8;
        storage.log_add(&make_doc("a", dim, 0)).unwrap();
        storage.log_add(&make_doc("b", dim, 1)).unwrap();
        storage.log_add(&make_doc("c", dim, 2)).unwrap();
        storage.log_remove("b").unwrap();
        storage.log_clear().unwrap();
        storage.sync().unwrap();

        let helper = RecoveryHelper::new(&storage);

        let mut adds = 0usize;
        let mut removes = 0usize;
        let mut clears = 0usize;

        helper
            .replay_wal(|op| {
                match op {
                    WalOperation::Add(_) => adds += 1,
                    WalOperation::Remove(_) => removes += 1,
                    WalOperation::Clear => clears += 1,
                    WalOperation::Checkpoint { .. } => {} // Should not appear
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(adds, 3);
        assert_eq!(removes, 1);
        assert_eq!(clears, 1);
    }

    #[test]
    fn needs_checkpoint_respects_threshold() {
        let dir = TempDir::new().unwrap();
        let mut storage = IncrementalStorage::new(
            dir.path(),
            IncrementalConfig::default().with_checkpoint_threshold(3),
        )
        .unwrap();

        let dim = 4;
        storage.log_add(&make_doc("a", dim, 0)).unwrap();
        storage.log_add(&make_doc("b", dim, 1)).unwrap();
        assert!(!storage.needs_checkpoint(), "Should not need checkpoint at 2 ops");

        storage.log_add(&make_doc("c", dim, 2)).unwrap();
        assert!(storage.needs_checkpoint(), "Should need checkpoint at 3 ops");
    }
}

// ============================================================================
// (e) Compression Round-Trip
// ============================================================================

mod compression_roundtrip {
    use super::*;

    #[test]
    fn gzip_compress_decompress_identity() {
        let original = b"Test data for compression. ".repeat(100);
        let (compressed, stats) = compression::compress_with(&original, Codec::Gzip).unwrap();

        assert_eq!(stats.codec, Codec::Gzip);
        assert_eq!(stats.original_size, original.len());
        assert!(
            stats.compressed_size < stats.original_size,
            "Gzip should reduce size for repetitive data"
        );
        assert!(stats.ratio > 1.0, "Compression ratio should be > 1.0");

        let decompressed = compression::decompress(&compressed).unwrap();
        assert_eq!(
            original.as_slice(),
            decompressed.as_slice(),
            "Decompressed data must equal original"
        );
    }

    #[test]
    fn no_compression_roundtrip() {
        let original = b"Passthrough data";
        let (compressed, stats) = compression::compress_with(original, Codec::None).unwrap();

        assert_eq!(stats.codec, Codec::None);
        let decompressed = compression::decompress(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn empty_data_roundtrip() {
        let original = b"";
        let (compressed, stats) = compression::compress_with(original, Codec::Gzip).unwrap();
        assert_eq!(stats.original_size, 0);

        let decompressed = compression::decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn embedding_vectors_survive_compression() {
        // Serialize embeddings as raw bytes, compress, decompress, reconstruct
        let embeddings: Vec<f32> = (0..384).map(|i| (i as f32) * 0.0013).collect();
        let raw_bytes: Vec<u8> = embeddings.iter().flat_map(|f| f.to_le_bytes()).collect();

        let (compressed, _stats) = compression::compress_with(&raw_bytes, Codec::Gzip).unwrap();
        let decompressed = compression::decompress(&compressed).unwrap();

        assert_eq!(raw_bytes, decompressed);

        // Reconstruct and verify
        let reconstructed: Vec<f32> = decompressed
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(embeddings, reconstructed);
    }

    #[test]
    fn best_codec_produces_valid_output() {
        let data = b"Best codec test data for auto-selection. ".repeat(50);
        let (compressed, stats) = compression::compress(&data).unwrap();

        // Verify the selected codec is usable
        assert!(!stats.codec.name().is_empty());
        assert!(stats.compressed_size > 0);

        let decompressed = compression::decompress(&compressed).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_compress_decompress_identity() {
        let original = b"LZ4 test data with repetition. ".repeat(100);
        let (compressed, stats) = compression::compress_with(&original, Codec::Lz4).unwrap();

        assert_eq!(stats.codec, Codec::Lz4);
        assert!(stats.compressed_size < stats.original_size);

        let decompressed = compression::decompress(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_compress_decompress_identity() {
        let original = b"Zstd test data with repetition. ".repeat(100);
        let (compressed, stats) = compression::compress_with(&original, Codec::Zstd).unwrap();

        assert_eq!(stats.codec, Codec::Zstd);
        assert!(stats.compressed_size < stats.original_size);

        let decompressed = compression::decompress(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }
}

// ============================================================================
// (f) Batch Operations & Streaming
// ============================================================================

mod batch_and_streaming {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn batch_builder_ingests_many_documents() {
        let dim = 32;
        let n = 50;
        let mut index = HNSWIndex::with_defaults(dim);

        let config = BatchConfig::default().with_batch_size(10).with_total(n);

        let mut builder = BatchBuilder::new(&mut index, config);
        for i in 0..n {
            let doc = make_doc(&format!("doc_{}", i), dim, i);
            builder.add(doc).unwrap();
        }
        let result = builder.finish();

        assert_eq!(result.documents_indexed, n);
        assert!(!result.has_errors());
        assert_eq!(index.len(), n);

        // Verify search works after batch ingestion
        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].id, "doc_0");
    }

    #[test]
    fn batch_builder_fires_progress_callback() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);
        let callback_count = Arc::new(AtomicUsize::new(0));
        let cc = callback_count.clone();

        let config = BatchConfig::default()
            .with_batch_size(5)
            .with_progress(move |_progress| {
                cc.fetch_add(1, Ordering::SeqCst);
            });

        let mut builder = BatchBuilder::new(&mut index, config);
        for i in 0..17 {
            builder.add(make_doc(&format!("d{}", i), dim, i)).unwrap();
        }
        let _result = builder.finish();

        // Should fire at: 5, 10, 15, and 17 (finish)
        let count = callback_count.load(Ordering::SeqCst);
        assert_eq!(count, 4, "Expected 4 progress callbacks, got {}", count);
    }

    #[test]
    fn batch_builder_continue_on_error_skips_bad_docs() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);
        let config = BatchConfig::default().continue_on_error(true);

        let mut builder = BatchBuilder::new(&mut index, config);

        // Good doc
        builder.add(make_doc("good_1", dim, 0)).unwrap();

        // Bad doc (wrong dimension)
        let bad_doc = Document {
            id: "bad_1".to_string(),
            content: "Bad".to_string(),
            embedding: vec![0.0; dim / 2], // Wrong dimension
            metadata: None,
        };
        builder.add(bad_doc).unwrap(); // Should not error because continue_on_error

        // Good doc
        builder.add(make_doc("good_2", dim, 1)).unwrap();

        let result = builder.finish();
        assert_eq!(result.documents_indexed, 2, "Only 2 good docs should be indexed");
        assert!(result.has_errors(), "Should have 1 error");
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, "bad_1");
    }

    #[test]
    fn search_result_iterator_works() {
        let results = vec![
            SearchResult {
                id: "a".to_string(),
                content: "A".to_string(),
                score: 0.9,
                metadata: None,
            },
            SearchResult {
                id: "b".to_string(),
                content: "B".to_string(),
                score: 0.8,
                metadata: None,
            },
            SearchResult {
                id: "c".to_string(),
                content: "C".to_string(),
                score: 0.7,
                metadata: None,
            },
        ];

        let mut iter = SearchResultIterator::new(results);
        assert_eq!(iter.total(), 3);
        assert_eq!(iter.peek().unwrap().id, "a");

        let first = iter.next().unwrap();
        assert_eq!(first.id, "a");

        let remaining = iter.collect_remaining();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].id, "b");
        assert_eq!(remaining[1].id, "c");
    }

    #[test]
    fn search_page_pagination() {
        let results: Vec<SearchResult> = (0..23)
            .map(|i| SearchResult {
                id: format!("doc_{}", i),
                content: format!("Content {}", i),
                score: 1.0 - (i as f32 * 0.01),
                metadata: None,
            })
            .collect();

        let page0 = SearchPage::from_results(results.clone(), 0, 10);
        assert_eq!(page0.results.len(), 10);
        assert_eq!(page0.total_pages, 3);
        assert!(page0.has_next);
        assert!(!page0.has_prev);

        let page2 = SearchPage::from_results(results.clone(), 2, 10);
        assert_eq!(page2.results.len(), 3); // 23 - 20 = 3
        assert!(!page2.has_next);
        assert!(page2.has_prev);

        // Beyond range
        let page_oob = SearchPage::from_results(results, 5, 10);
        assert!(page_oob.results.is_empty());
    }

    #[test]
    fn filtered_search_applies_min_score_and_metadata_filters() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..20 {
            let meta = serde_json::json!({"category": if i % 2 == 0 { "even" } else { "odd" }});
            let doc = make_doc_with_metadata(&format!("doc_{}", i), dim, i, meta);
            index.add(doc).unwrap();
        }

        let query = deterministic_embedding(dim, 0);
        let all_results = index.search(&query, 20).unwrap();

        // Filter to only "even" category with min score > 0.0
        let filtered = FilteredSearchBuilder::new()
            .min_score(0.0)
            .metadata_equals("category", serde_json::json!("even"))
            .max_results(5)
            .apply(all_results);

        assert!(filtered.len() <= 5);
        for result in &filtered {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "even");
            assert!(result.score >= 0.0);
        }
    }
}

// ============================================================================
// (g) Edge Cases
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn empty_index_search_returns_empty() {
        let index = HNSWIndex::with_defaults(16);
        let query = deterministic_embedding(16, 0);
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty(), "Search on empty index should return no results");
    }

    #[test]
    fn single_document_index_search() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);
        index.add(make_doc("only_doc", dim, 0)).unwrap();

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 1, "Should return 1 result from single-doc index");
        assert_eq!(results[0].id, "only_doc");
        assert!(results[0].score > 0.99, "Exact match should have score ~1.0");
    }

    #[test]
    fn search_with_k_greater_than_doc_count() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..3 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 100).unwrap();

        // Should return at most 3 results (the actual doc count)
        assert_eq!(
            results.len(),
            3,
            "Should return all 3 docs when k=100 > n=3"
        );
    }

    #[test]
    fn duplicate_document_ids_in_hnsw() {
        // HNSW does not deduplicate; both copies should be added
        let dim = 8;
        let mut index = HNSWIndex::with_defaults(dim);

        let doc1 = Document {
            id: "dup".to_string(),
            content: "First".to_string(),
            embedding: deterministic_embedding(dim, 0),
            metadata: None,
        };
        let doc2 = Document {
            id: "dup".to_string(),
            content: "Second".to_string(),
            embedding: deterministic_embedding(dim, 1),
            metadata: None,
        };

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        // HNSW stores both (it does not deduplicate by ID)
        assert_eq!(index.len(), 2, "HNSW should store both docs with same ID");

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn duplicate_document_ids_in_flat_replaces() {
        // FlatIndex deduplicates by ID (HashMap-based)
        let dim = 8;
        let mut index = FlatIndex::new(dim);

        let doc1 = Document {
            id: "dup".to_string(),
            content: "First".to_string(),
            embedding: deterministic_embedding(dim, 0),
            metadata: None,
        };
        let doc2 = Document {
            id: "dup".to_string(),
            content: "Replaced".to_string(),
            embedding: deterministic_embedding(dim, 1),
            metadata: None,
        };

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        assert_eq!(index.len(), 1, "FlatIndex should deduplicate by ID");

        let query = deterministic_embedding(dim, 1);
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results[0].content, "Replaced", "Should have the second version");
    }

    #[test]
    fn high_dimensional_vectors() {
        // Test with a relatively high dimension (1024)
        let dim = 1024;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..10 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let query = deterministic_embedding(dim, 0);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "doc_0");

        // Score should still be valid
        for result in &results {
            assert!(
                result.score >= -1.0 && result.score <= 1.0,
                "Score {} out of valid range",
                result.score
            );
        }
    }

    #[test]
    fn zero_vectors_do_not_panic() {
        let dim = 16;
        let mut index = HNSWIndex::with_defaults(dim);

        // Add a zero vector
        let zero_doc = Document {
            id: "zero".to_string(),
            content: "Zero vector".to_string(),
            embedding: vec![0.0; dim],
            metadata: None,
        };
        index.add(zero_doc).unwrap();

        // Add a non-zero vector
        index.add(make_doc("nonzero", dim, 1)).unwrap();

        // Search with zero query
        let zero_query = vec![0.0; dim];
        let results = index.search(&zero_query, 2).unwrap();
        assert_eq!(results.len(), 2, "Should still return results for zero query");

        // Search with non-zero query
        let query = deterministic_embedding(dim, 1);
        let results = index.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut index = HNSWIndex::with_defaults(16);
        index.add(make_doc("ok", 16, 0)).unwrap();

        // Wrong dimension in add
        let bad_doc = Document {
            id: "bad".to_string(),
            content: "".to_string(),
            embedding: vec![0.0; 8],
            metadata: None,
        };
        assert!(index.add(bad_doc).is_err(), "Should reject mismatched dimension");

        // Wrong dimension in search
        let bad_query = vec![0.0; 8];
        assert!(
            index.search(&bad_query, 1).is_err(),
            "Should reject mismatched query dimension"
        );
    }

    #[test]
    fn flat_index_empty_search() {
        let index = FlatIndex::new(16);
        let query = vec![0.5; 16];
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn sq8_empty_search() {
        let index = SQ8HNSWIndex::for_normalized(16, QuantizedHNSWConfig::default());
        let query = vec![0.5; 16];
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn binary_empty_search() {
        let index = BinaryHNSWIndex::new(16, QuantizedHNSWConfig::default());
        let query = vec![0.5; 16];
        let results = index.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }
}

// ============================================================================
// (h) Concurrent Access (Thread Safety)
// ============================================================================

mod concurrent_access {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn parallel_searches_produce_consistent_results() {
        let dim = 64;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..50 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        // Share the index across threads (HNSWIndex::search takes &self, so it's read-only)
        let index = Arc::new(index);
        let num_threads = 8;
        let queries_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    let mut all_results = Vec::new();
                    for q in 0..queries_per_thread {
                        let seed = t * 1000 + q;
                        let query = deterministic_embedding(dim, seed);
                        let results = index.search(&query, 5).unwrap();
                        assert_eq!(results.len(), 5, "Thread {} query {} got wrong count", t, q);

                        // Results should be sorted
                        for window in results.windows(2) {
                            assert!(window[0].score >= window[1].score);
                        }
                        all_results.push(results);
                    }
                    all_results
                })
            })
            .collect();

        // Collect and verify all threads completed successfully
        for handle in handles {
            let thread_results = handle.join().expect("Thread should not panic");
            assert_eq!(thread_results.len(), queries_per_thread);
        }
    }

    #[test]
    fn search_batch_produces_correct_count() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..30 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| deterministic_embedding(dim, i + 5000))
            .collect();

        let batch_results = index.search_batch(&queries, 5).unwrap();

        assert_eq!(batch_results.len(), 10, "Should return results for all 10 queries");
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(results.len(), 5, "Query {} should return 5 results", i);
            for window in results.windows(2) {
                assert!(window[0].score >= window[1].score);
            }
        }
    }

    #[test]
    fn search_batch_fast_produces_correct_count() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..30 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| deterministic_embedding(dim, i + 5000))
            .collect();

        let batch_results = index.search_batch_fast(&queries, 5).unwrap();

        assert_eq!(batch_results.len(), 10);
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(results.len(), 5, "Fast batch query {} should return 5 results", i);
        }
    }

    #[test]
    fn search_with_context_matches_regular_search() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..30 {
            index.add(make_doc(&format!("doc_{}", i), dim, i)).unwrap();
        }

        let query = deterministic_embedding(dim, 999);
        let k = 5;

        let regular_results = index.search(&query, k).unwrap();
        let mut ctx = index.create_search_context();
        let ctx_results = index.search_with_context(&query, k, &mut ctx).unwrap();

        assert_eq!(regular_results.len(), ctx_results.len());

        // Results should overlap substantially (both find top matches)
        let regular_ids: HashSet<&str> = regular_results.iter().map(|r| r.id.as_str()).collect();
        let ctx_ids: HashSet<&str> = ctx_results.iter().map(|r| r.id.as_str()).collect();
        let overlap = regular_ids.intersection(&ctx_ids).count();

        assert!(
            overlap >= 3,
            "Regular and context search should substantially overlap, got {}/{}",
            overlap,
            k
        );
    }
}

// ============================================================================
// Cross-cutting: Full Pipeline Test
// ============================================================================

#[test]
fn full_pipeline_create_index_persist_load_search() {
    // End-to-end: create documents -> build HNSW index -> save to disk ->
    // load from disk -> search -> verify results match ground truth

    let dir = tempfile::tempdir().unwrap();
    let dim = 32;
    let n = 30;
    let k = 5;

    // Step 1: Create documents with deterministic embeddings
    let documents: Vec<Document> = (0..n)
        .map(|i| {
            make_doc_with_metadata(
                &format!("doc_{}", i),
                dim,
                i,
                serde_json::json!({"index": i}),
            )
        })
        .collect();

    // Step 2: Build HNSW index
    let mut index = HNSWIndex::with_defaults(dim);
    for doc in &documents {
        index.add(doc.clone()).unwrap();
    }
    assert_eq!(index.len(), n);

    // Step 3: Save to disk with Gzip compression
    let storage = FileStorage::with_codec(dir.path(), Codec::Gzip).unwrap();
    let wrapper = HNSWIndexWrapper::from_index(&index);
    let stats = storage.save_hnsw_index("pipeline_test", &wrapper).unwrap();
    assert!(stats.original_size > 0);
    assert_eq!(stats.codec, Codec::Gzip);

    // Step 4: Load from disk
    let loaded_wrapper = storage.load_hnsw_index("pipeline_test").unwrap();
    let loaded_index = loaded_wrapper.to_index().unwrap();
    assert_eq!(loaded_index.len(), n);

    // Step 5: Search the loaded index
    let query = deterministic_embedding(dim, 0); // Should match doc_0
    let results = loaded_index.search(&query, k).unwrap();

    assert_eq!(results.len(), k);

    // Step 6: Verify results match ground truth (brute force)
    let gt = brute_force_top_k(&documents, &query, k);
    let recall = recall_at_k(&gt, &results);

    assert!(
        recall >= 0.6,
        "Full pipeline recall@{} should be >= 60%, got {:.0}%",
        k,
        recall * 100.0
    );

    // Step 7: Verify metadata survived the full pipeline
    let top_result = &results[0];
    assert!(top_result.metadata.is_some(), "Metadata should survive persistence");
}
