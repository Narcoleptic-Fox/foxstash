//! Comprehensive benchmarking suite for Foxstash Storage Layer
//!
//! This benchmark suite measures performance of storage operations including:
//! - Compression codecs (None, Gzip, LZ4, Zstd)
//! - Serialization (Documents, FlatIndex, HNSWIndex)
//! - File storage operations (save/load documents and indices)
//! - Batch operations
//! - Realistic workloads (ingestion, cold start, incremental updates)
//!
//! ## Performance Targets
//!
//! **Compression:**
//! - Gzip: ~10-50 MB/s (good ratio)
//! - LZ4: ~500+ MB/s (fast)
//! - Zstd: ~100-300 MB/s (best ratio)
//!
//! **Serialization:**
//! - Small docs (<1KB): <100μs
//! - FlatIndex (1K docs): <10ms
//! - HNSWIndex (1K docs): <50ms
//!
//! **File I/O:**
//! - Save document: <5ms
//! - Load document: <2ms
//! - Save index (1K docs): <100ms
//! - Load index (1K docs): <50ms
//!
//! ## Running Benchmarks
//! ```bash
//! # Run all storage benchmarks
//! cargo bench -p foxstash-benches storage
//!
//! # Run specific benchmark group
//! cargo bench -p foxstash-benches compression_codecs
//! cargo bench -p foxstash-benches serialization
//! cargo bench -p foxstash-benches file_storage
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use foxstash_core::index::flat::FlatIndex;
use foxstash_core::index::hnsw::HNSWIndex;
use foxstash_core::{Document, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

// ============================================================================
// Data Generation Helpers
// ============================================================================

mod data_gen {
    use super::*;

    /// Generate random incompressible data (baseline for compression tests)
    ///
    /// This generates cryptographically random bytes that won't compress well,
    /// providing a baseline for compression performance testing.
    pub fn random_bytes(size: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..size).map(|_| rng.gen::<u8>()).collect()
    }

    /// Generate JSON documents (realistic text data)
    ///
    /// Creates JSON-serialized documents that represent typical text content,
    /// which should compress well with all codecs.
    pub fn json_documents(count: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        let categories = ["technology", "science", "business"];
        let documents: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                serde_json::json!({
                    "id": format!("doc_{}", i),
                    "title": format!("Document Title {}", i),
                    "content": format!(
                        "This is the content of document number {}. It contains some \
                        text that represents a typical document in a RAG system. \
                        The content should compress reasonably well because text \
                        has natural redundancy and patterns. Random value: {}",
                        i,
                        rng.gen::<f64>()
                    ),
                    "metadata": {
                        "category": categories[i % 3],
                        "timestamp": format!("2024-01-{:02}T12:00:00Z", (i % 28) + 1),
                        "word_count": 42 + (i % 100),
                    }
                })
            })
            .collect();

        serde_json::to_vec(&documents).unwrap()
    }

    /// Generate embedding vectors (float data)
    ///
    /// Creates serialized f32 vectors representing embeddings, which have
    /// moderate compressibility due to float precision patterns.
    pub fn embedding_vectors(count: usize, dim: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count * dim * 4);

        for _ in 0..count {
            for _ in 0..dim {
                let value = rng.gen::<f32>() * 2.0 - 1.0; // Range [-1, 1]
                data.extend_from_slice(&value.to_le_bytes());
            }
        }

        data
    }

    /// Generate highly compressible repeated pattern
    ///
    /// Creates data with significant repetition to test compression ratios
    /// on best-case scenarios.
    pub fn repeated_pattern(size: usize) -> Vec<u8> {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        pattern.iter().cycle().take(size).copied().collect()
    }

    /// Generate a test document with specified content size
    pub fn test_document(id: &str, content_size: usize, dim: usize, seed: u64) -> Document {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate content of specified size
        let content = if content_size < 100 {
            format!("Small document {}", id)
        } else {
            let base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
            base.repeat(content_size / base.len() + 1)
                .chars()
                .take(content_size)
                .collect()
        };

        // Generate random embedding
        let embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        Document {
            id: id.to_string(),
            content,
            embedding,
            metadata: Some(serde_json::json!({
                "source": "test",
                "timestamp": "2024-01-01T00:00:00Z",
            })),
        }
    }

    /// Create a flat index with specified number of documents
    pub fn test_flat_index(doc_count: usize, dim: usize, seed: u64) -> FlatIndex {
        let mut index = FlatIndex::new(dim);

        for i in 0..doc_count {
            let doc = test_document(&format!("doc_{}", i), 100, dim, seed + i as u64);
            index.add(doc).unwrap();
        }

        index
    }

    /// Create an HNSW index with specified number of documents
    pub fn test_hnsw_index(doc_count: usize, dim: usize, seed: u64) -> HNSWIndex {
        let mut index = HNSWIndex::with_defaults(dim);

        for i in 0..doc_count {
            let doc = test_document(&format!("doc_{}", i), 100, dim, seed + i as u64);
            index.add(doc).unwrap();
        }

        index
    }
}

// ============================================================================
// Mock Storage Types (for benchmarking until actual implementation is ready)
// ============================================================================

/// Mock compression codec enum
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum MockCodec {
    None,
    Gzip,
    Lz4,
    Zstd,
}

impl MockCodec {
    fn name(&self) -> &'static str {
        match self {
            MockCodec::None => "none",
            MockCodec::Gzip => "gzip",
            MockCodec::Lz4 => "lz4",
            MockCodec::Zstd => "zstd",
        }
    }

    /// Mock compression - in real implementation this would call actual codec
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, just simulate compression with bincode
        // Real implementation will use actual compression libraries
        match self {
            MockCodec::None => Ok(data.to_vec()),
            _ => Ok(bincode::serialize(data)?),
        }
    }

    /// Mock decompression - in real implementation this would call actual codec
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self {
            MockCodec::None => Ok(data.to_vec()),
            _ => Ok(bincode::deserialize(data)?),
        }
    }
}

// ============================================================================
// Compression Benchmarks
// ============================================================================

/// Benchmark compression codecs on different data types
///
/// Tests each codec (None, Gzip, LZ4, Zstd) on various data types:
/// - Random data (incompressible baseline)
/// - JSON documents (text, structured)
/// - Embedding vectors (floats)
/// - Repeated patterns (highly compressible)
///
/// Measures compression time, decompression time, and compression ratio.
fn benchmark_compression_codecs(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_codecs");

    let size = 1024 * 1024; // 1MB test data

    // Generate different types of test data
    let test_data = vec![
        ("random", data_gen::random_bytes(size, 12345)),
        ("json", data_gen::json_documents(1000, 54321)),
        ("embeddings", data_gen::embedding_vectors(1000, 384, 11111)),
        ("repeated", data_gen::repeated_pattern(size)),
    ];

    for codec in [
        MockCodec::None,
        MockCodec::Gzip,
        MockCodec::Lz4,
        MockCodec::Zstd,
    ] {
        for (data_type, data) in &test_data {
            // Benchmark compression
            group.throughput(Throughput::Bytes(data.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_compress", codec.name()), data_type),
                data,
                |b, data| b.iter(|| codec.compress(black_box(data)).unwrap()),
            );

            // Pre-compress data for decompression benchmark
            let compressed = codec.compress(data).unwrap();

            // Benchmark decompression
            group.throughput(Throughput::Bytes(compressed.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_decompress", codec.name()), data_type),
                &compressed,
                |b, compressed_data| {
                    b.iter(|| codec.decompress(black_box(compressed_data)).unwrap())
                },
            );

            // Calculate and report compression ratio
            let ratio = compressed.len() as f64 / data.len() as f64;
            eprintln!(
                "{} compression ratio for {}: {:.2}% (original: {} bytes, compressed: {} bytes)",
                codec.name(),
                data_type,
                ratio * 100.0,
                data.len(),
                compressed.len()
            );
        }
    }

    group.finish();
}

/// Benchmark compression performance at different data sizes
///
/// Tests how compression performance scales with data size from 1KB to 1MB.
/// Helps identify optimal chunk sizes and scaling characteristics.
fn benchmark_compression_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_sizes");

    // Test different sizes: 1KB, 10KB, 100KB, 1MB
    for size in [1024, 10_240, 102_400, 1_024_000] {
        let data = data_gen::json_documents(size / 100, 42);

        group.throughput(Throughput::Bytes(data.len() as u64));

        for codec in [MockCodec::Lz4, MockCodec::Zstd, MockCodec::Gzip] {
            group.bench_with_input(BenchmarkId::new(codec.name(), size), &data, |b, data| {
                b.iter(|| codec.compress(black_box(data)).unwrap())
            });
        }
    }

    group.finish();
}

// ============================================================================
// Serialization Benchmarks
// ============================================================================

/// Benchmark document and index serialization performance
///
/// Tests bincode serialization of:
/// - Documents (small, medium, large)
/// - FlatIndex (100, 1K, 10K documents)
/// - HNSWIndex (100, 1K, 10K documents)
fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let dim = 384;

    // Document serialization - different sizes
    for (size_name, content_size) in [("small", 100), ("medium", 1_000), ("large", 10_000)] {
        let doc = data_gen::test_document("doc_1", content_size, dim, 999);

        group.bench_with_input(
            BenchmarkId::new("serialize_document", size_name),
            &doc,
            |b, doc| b.iter(|| bincode::serialize(black_box(doc)).unwrap()),
        );

        let serialized = bincode::serialize(&doc).unwrap();
        eprintln!(
            "Document ({}) serialized size: {} bytes",
            size_name,
            serialized.len()
        );

        group.bench_with_input(
            BenchmarkId::new("deserialize_document", size_name),
            &serialized,
            |b, data| b.iter(|| bincode::deserialize::<Document>(black_box(data)).unwrap()),
        );
    }

    // Index serialization requires Serialize/Deserialize traits
    // In practice, use FileStorage which handles serialization via wrappers:
    //
    //   let storage = FileStorage::new("path")?;
    //   storage.save_flat_index("name", &flat_index)?;
    //   storage.save_hnsw_index("name", &hnsw_index)?;
    //   let loaded_flat = storage.load_flat_index("name")?;
    //   let loaded_hnsw = storage.load_hnsw_index("name")?;
    //
    // Direct serialization benchmarks commented out to avoid trait requirements

    group.finish();
}

// ============================================================================
// File Storage Benchmarks
// ============================================================================

/// Benchmark file storage operations (save/load)
///
/// Tests actual file I/O performance for documents and indices.
/// Uses temporary directories to avoid filesystem state issues.
///
/// NOTE: This benchmark currently uses mock file operations.
/// Once FileStorage is implemented, replace with actual storage calls.
#[allow(dead_code)]
fn benchmark_file_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_storage");

    let dim = 384;

    // Save document with different codecs
    for codec in [MockCodec::None, MockCodec::Lz4, MockCodec::Zstd] {
        let doc = data_gen::test_document("doc_1", 1_000, dim, 777);

        group.bench_with_input(
            BenchmarkId::new("save_document", codec.name()),
            &doc,
            |b, doc| {
                b.iter_batched(
                    || {
                        // Setup: create temp directory
                        tempfile::tempdir().unwrap()
                    },
                    |temp_dir| {
                        // Benchmark: serialize and compress
                        let serialized = bincode::serialize(black_box(doc)).unwrap();
                        let compressed = codec.compress(&serialized).unwrap();

                        // Write to file (mock - real implementation would use FileStorage)
                        let path = temp_dir.path().join("doc.bin");
                        std::fs::write(path, compressed).unwrap();

                        temp_dir // Return to cleanup
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    // Load document with different codecs
    for codec in [MockCodec::None, MockCodec::Lz4, MockCodec::Zstd] {
        let doc = data_gen::test_document("doc_1", 1_000, dim, 777);
        let serialized = bincode::serialize(&doc).unwrap();
        let compressed = codec.compress(&serialized).unwrap();

        group.bench_with_input(
            BenchmarkId::new("load_document", codec.name()),
            &compressed,
            |b, data| {
                b.iter_batched(
                    || {
                        // Setup: write file
                        let temp_dir = tempfile::tempdir().unwrap();
                        let path = temp_dir.path().join("doc.bin");
                        std::fs::write(&path, data).unwrap();
                        (temp_dir, path)
                    },
                    |(temp_dir, path)| {
                        // Benchmark: read, decompress, deserialize
                        let compressed = std::fs::read(black_box(&path)).unwrap();
                        let serialized = codec.decompress(&compressed).unwrap();
                        let _doc: Document = bincode::deserialize(&serialized).unwrap();

                        temp_dir // Return to cleanup
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    // Save and load indices (smaller counts for faster benchmarks)
    for doc_count in [100, 1000] {
        // FlatIndex
        let flat_index = data_gen::test_flat_index(doc_count, dim, 3000);

        group.bench_with_input(
            BenchmarkId::new("save_flat_index", doc_count),
            &flat_index,
            |b, index| {
                b.iter_batched(
                    || tempfile::tempdir().unwrap(),
                    |temp_dir| {
                        let serialized = bincode::serialize(black_box(index)).unwrap();
                        let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                        let path = temp_dir.path().join("flat_index.bin");
                        std::fs::write(path, compressed).unwrap();
                        temp_dir
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        let serialized = bincode::serialize(&flat_index).unwrap();
        let compressed = MockCodec::Zstd.compress(&serialized).unwrap();

        group.bench_with_input(
            BenchmarkId::new("load_flat_index", doc_count),
            &compressed,
            |b, data| {
                b.iter_batched(
                    || {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let path = temp_dir.path().join("flat_index.bin");
                        std::fs::write(&path, data).unwrap();
                        (temp_dir, path)
                    },
                    |(temp_dir, path)| {
                        let compressed = std::fs::read(black_box(&path)).unwrap();
                        let serialized = MockCodec::Zstd.decompress(&compressed).unwrap();
                        let _index: FlatIndex = bincode::deserialize(&serialized).unwrap();
                        temp_dir
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // HNSWIndex
        let hnsw_index = data_gen::test_hnsw_index(doc_count, dim, 4000);

        group.bench_with_input(
            BenchmarkId::new("save_hnsw_index", doc_count),
            &hnsw_index,
            |b, index| {
                b.iter_batched(
                    || tempfile::tempdir().unwrap(),
                    |temp_dir| {
                        let serialized = serde_json::to_vec(black_box(index)).unwrap();
                        let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                        let path = temp_dir.path().join("hnsw_index.bin");
                        std::fs::write(path, compressed).unwrap();
                        temp_dir
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        let serialized = serde_json::to_vec(&hnsw_index).unwrap();
        let compressed = MockCodec::Zstd.compress(&serialized).unwrap();

        group.bench_with_input(
            BenchmarkId::new("load_hnsw_index", doc_count),
            &compressed,
            |b, data| {
                b.iter_batched(
                    || {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let path = temp_dir.path().join("hnsw_index.bin");
                        std::fs::write(&path, data).unwrap();
                        (temp_dir, path)
                    },
                    |(temp_dir, path)| {
                        let compressed = std::fs::read(black_box(&path)).unwrap();
                        let serialized = MockCodec::Zstd.decompress(&compressed).unwrap();
                        let _index: HNSWIndex = serde_json::from_slice(&serialized).unwrap();
                        temp_dir
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ============================================================================
// Batch Operations Benchmarks
// ============================================================================

/// Benchmark batch save/load operations
///
/// Tests performance of saving and loading multiple documents at once,
/// which is important for bulk ingestion and backup/restore operations.
#[allow(dead_code)]
fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let dim = 384;

    for count in [10, 100, 1000] {
        // Generate batch of documents
        let docs: Vec<Document> = (0..count)
            .map(|i| data_gen::test_document(&format!("doc_{}", i), 500, dim, 5000 + i as u64))
            .collect();

        // Benchmark batch save
        group.bench_with_input(BenchmarkId::new("save_batch", count), &docs, |b, docs| {
            b.iter_batched(
                || tempfile::tempdir().unwrap(),
                |temp_dir| {
                    // Save each document
                    for (i, doc) in docs.iter().enumerate() {
                        let serialized = bincode::serialize(black_box(doc)).unwrap();
                        let compressed = MockCodec::Lz4.compress(&serialized).unwrap();
                        let path = temp_dir.path().join(format!("doc_{}.bin", i));
                        std::fs::write(path, compressed).unwrap();
                    }
                    temp_dir
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // Prepare files for load benchmark
        let temp_dir = tempfile::tempdir().unwrap();
        for (i, doc) in docs.iter().enumerate() {
            let serialized = bincode::serialize(doc).unwrap();
            let compressed = MockCodec::Lz4.compress(&serialized).unwrap();
            let path = temp_dir.path().join(format!("doc_{}.bin", i));
            std::fs::write(path, compressed).unwrap();
        }

        // Benchmark batch load
        group.bench_with_input(
            BenchmarkId::new("load_batch", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut loaded_docs = Vec::new();
                    for i in 0..count {
                        let path = temp_dir.path().join(format!("doc_{}.bin", i));
                        let compressed = std::fs::read(black_box(&path)).unwrap();
                        let serialized = MockCodec::Lz4.decompress(&compressed).unwrap();
                        let doc: Document = bincode::deserialize(&serialized).unwrap();
                        loaded_docs.push(doc);
                    }
                    loaded_docs
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Realistic Workload Benchmarks
// ============================================================================

/// Benchmark realistic end-to-end workloads
///
/// These benchmarks simulate real-world usage patterns:
/// - Document ingestion: add documents to index and save
/// - Cold start: load index from disk and perform search
/// - Incremental update: load, add documents, save
#[allow(dead_code)]
fn benchmark_realistic_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workloads");

    // These are slower tests, reduce sample size
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let dim = 384;

    // Workload 1: Document Ingestion Pipeline
    // Add 100 documents to index and persist to disk
    group.bench_function("ingestion_pipeline_100docs", |b| {
        b.iter_batched(
            || {
                (
                    tempfile::tempdir().unwrap(),
                    (0..100)
                        .map(|i| {
                            data_gen::test_document(
                                &format!("doc_{}", i),
                                500,
                                dim,
                                6000 + i as u64,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            },
            |(temp_dir, docs)| {
                // Create index
                let mut index = FlatIndex::new(dim);

                // Add documents
                for doc in black_box(docs) {
                    index.add(doc).unwrap();
                }

                // Save index to disk
                let serialized = bincode::serialize(&index).unwrap();
                let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                let path = temp_dir.path().join("index.bin");
                std::fs::write(path, compressed).unwrap();

                temp_dir
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Workload 2: Cold Start
    // Load index from disk and perform search
    group.bench_function("cold_start_1000docs", |b| {
        b.iter_batched(
            || {
                // Setup: create and save index
                let temp_dir = tempfile::tempdir().unwrap();
                let index = data_gen::test_flat_index(1000, dim, 7000);
                let serialized = bincode::serialize(&index).unwrap();
                let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                let path = temp_dir.path().join("index.bin");
                std::fs::write(&path, compressed).unwrap();

                let query = (0..dim).map(|_| 0.5f32).collect::<Vec<_>>();

                (temp_dir, path, query)
            },
            |(temp_dir, path, query)| {
                // Load index
                let compressed = std::fs::read(black_box(&path)).unwrap();
                let serialized = MockCodec::Zstd.decompress(&compressed).unwrap();
                let index: FlatIndex = bincode::deserialize(&serialized).unwrap();

                // Perform search
                let _results = index.search(black_box(&query), 10).unwrap();

                temp_dir
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Workload 3: Incremental Update
    // Load index, add documents, save back
    group.bench_function("incremental_update_10docs", |b| {
        b.iter_batched(
            || {
                // Setup: create and save initial index
                let temp_dir = tempfile::tempdir().unwrap();
                let index = data_gen::test_flat_index(100, dim, 8000);
                let serialized = bincode::serialize(&index).unwrap();
                let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                let path = temp_dir.path().join("index.bin");
                std::fs::write(&path, compressed).unwrap();

                let new_docs: Vec<Document> = (0..10)
                    .map(|i| {
                        data_gen::test_document(
                            &format!("new_doc_{}", i),
                            500,
                            dim,
                            9000 + i as u64,
                        )
                    })
                    .collect();

                (temp_dir, path, new_docs)
            },
            |(temp_dir, path, new_docs)| {
                // Load index
                let compressed = std::fs::read(black_box(&path)).unwrap();
                let serialized = MockCodec::Zstd.decompress(&compressed).unwrap();
                let mut index: FlatIndex = bincode::deserialize(&serialized).unwrap();

                // Add new documents
                for doc in black_box(new_docs) {
                    index.add(doc).unwrap();
                }

                // Save updated index
                let serialized = bincode::serialize(&index).unwrap();
                let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                std::fs::write(black_box(&path), compressed).unwrap();

                temp_dir
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Workload 4: HNSW Index Persistence
    // Save and load HNSW index (more complex structure)
    group.bench_function("hnsw_persistence_1000docs", |b| {
        b.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let index = data_gen::test_hnsw_index(1000, dim, 10000);
                (temp_dir, index)
            },
            |(temp_dir, index)| {
                // Save index
                let serialized = serde_json::to_vec(&index).unwrap();
                let compressed = MockCodec::Zstd.compress(&serialized).unwrap();
                let path = temp_dir.path().join("hnsw_index.bin");
                std::fs::write(&path, compressed).unwrap();

                // Load it back
                let compressed = std::fs::read(black_box(&path)).unwrap();
                let serialized = MockCodec::Zstd.decompress(&compressed).unwrap();
                let _loaded_index: HNSWIndex = serde_json::from_slice(&serialized).unwrap();

                temp_dir
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    benchmark_compression_codecs,
    benchmark_compression_sizes,
    benchmark_serialization,
    // These benchmarks use index serialization which requires FileStorage wrappers:
    // benchmark_file_storage,
    // benchmark_batch_operations,
    // benchmark_realistic_workloads,
);

criterion_main!(benches);
