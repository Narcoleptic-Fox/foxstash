//! Comprehensive Benchmark Suite for Foxstash
//!
//! This benchmark suite provides complete performance testing for:
//! - Index Construction (HNSW and Flat indices)
//! - Search Performance (varying index sizes and k values)
//! - Vector Operations (cosine similarity, L2 distance, dot product)
//! - Serialization/Deserialization
//! - Storage and Compression
//!
//! ## Performance Targets
//! - Search (10K docs): <20ms (native), <50ms (WASM)
//! - Memory per document: <20KB
//! - Compression ratio: >50%
//! - Index construction: Linear with dataset size
//!
//! ## Running Benchmarks
//! ```bash
//! # Run all comprehensive benchmarks
//! cargo bench -p foxstash-benches --bench full
//!
//! # Run specific benchmark group
//! cargo bench -p foxstash-benches --bench full -- index_construction
//! cargo bench -p foxstash-benches --bench full -- search_performance
//! cargo bench -p foxstash-benches --bench full -- vector_operations
//! cargo bench -p foxstash-benches --bench full -- serialization
//! cargo bench -p foxstash-benches --bench full -- storage
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use foxstash_core::index::{FlatIndex, HNSWConfig, HNSWIndex};
use foxstash_core::storage::{Codec, FileStorage};
use foxstash_core::{cosine_similarity, dot_product, l2_distance, normalize, Document};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use tempfile::TempDir;

// ============================================================================
// Constants
// ============================================================================

const EMBEDDING_DIM: usize = 384; // MiniLM-L6-v2 dimension
const INDEX_SIZES: &[usize] = &[100, 1_000, 5_000, 10_000];
const K_VALUES: &[usize] = &[1, 10, 50];
const VECTOR_DIMS: &[usize] = &[384, 768, 1024];

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a random embedding vector with given dimension
fn generate_random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Generate multiple random embedding vectors
fn generate_random_embeddings(count: usize, dim: usize, seed_offset: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_random_embedding(dim, seed_offset + i as u64))
        .collect()
}

/// Create a test document with random embedding
fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
    Document {
        id: id.to_string(),
        content: format!("Test document content for {}", id),
        embedding,
        metadata: None,
    }
}

/// Create multiple test documents with random embeddings
fn create_test_documents(count: usize, dim: usize, seed_offset: u64) -> Vec<Document> {
    (0..count)
        .map(|i| {
            let embedding = generate_random_embedding(dim, seed_offset + i as u64);
            create_test_document(&format!("doc_{}", i), embedding)
        })
        .collect()
}

/// Build a FlatIndex with given documents
fn build_flat_index(documents: &[Document]) -> FlatIndex {
    let mut index = FlatIndex::new(EMBEDDING_DIM);
    for doc in documents {
        index.add(doc.clone()).unwrap();
    }
    index
}

/// Build an HNSW index with given documents
fn build_hnsw_index(documents: &[Document]) -> HNSWIndex {
    let config = HNSWConfig {
        m: 16,
        m0: 32,
        ef_construction: 200,
        ef_search: 50,
        ml: 1.0 / 16.0_f32.ln(),
    };
    let mut index = HNSWIndex::new(EMBEDDING_DIM, config);
    for doc in documents {
        index.add(doc.clone()).unwrap();
    }
    index
}

// ============================================================================
// A. Index Construction Benchmarks
// ============================================================================

fn benchmark_index_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_construction");
    group.sample_size(10); // Fewer samples for slow operations
    group.measurement_time(Duration::from_secs(30)); // Longer measurement time

    for &size in INDEX_SIZES {
        let documents = create_test_documents(size, EMBEDDING_DIM, 1000);

        // Benchmark HNSW construction
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("hnsw", size), &documents, |b, docs| {
            b.iter(|| {
                let config = HNSWConfig {
                    m: 16,
                    m0: 32,
                    ef_construction: 200,
                    ef_search: 50,
                    ml: 1.0 / 16.0_f32.ln(),
                };
                let mut index = HNSWIndex::new(EMBEDDING_DIM, config);
                for doc in docs {
                    index.add(black_box(doc.clone())).unwrap();
                }
                black_box(index)
            })
        });

        // Benchmark Flat construction
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("flat", size), &documents, |b, docs| {
            b.iter(|| {
                let mut index = FlatIndex::new(EMBEDDING_DIM);
                for doc in docs {
                    index.add(black_box(doc.clone())).unwrap();
                }
                black_box(index)
            })
        });
    }

    group.finish();
}

// ============================================================================
// B. Search Performance Benchmarks
// ============================================================================

fn benchmark_search_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_performance");
    group.sample_size(50);

    for &size in INDEX_SIZES {
        // Prepare indices
        let documents = create_test_documents(size, EMBEDDING_DIM, 2000);
        let flat_index = build_flat_index(&documents);
        let hnsw_index = build_hnsw_index(&documents);

        // Generate query vectors
        let queries = generate_random_embeddings(10, EMBEDDING_DIM, 3000);

        for &k in K_VALUES {
            // Benchmark HNSW search
            group.bench_with_input(
                BenchmarkId::new(format!("hnsw_{}docs", size), k),
                &(k, &hnsw_index, &queries),
                |b, (k, index, queries)| {
                    let mut query_idx = 0;
                    b.iter(|| {
                        let query = &queries[query_idx % queries.len()];
                        query_idx += 1;
                        black_box(index.search(black_box(query), black_box(*k)).unwrap())
                    })
                },
            );

            // Benchmark Flat search
            group.bench_with_input(
                BenchmarkId::new(format!("flat_{}docs", size), k),
                &(k, &flat_index, &queries),
                |b, (k, index, queries)| {
                    let mut query_idx = 0;
                    b.iter(|| {
                        let query = &queries[query_idx % queries.len()];
                        query_idx += 1;
                        black_box(index.search(black_box(query), black_box(*k)).unwrap())
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// C. Vector Operations Benchmarks
// ============================================================================

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");
    group.sample_size(100);

    for &dim in VECTOR_DIMS {
        let vec_a = generate_random_embedding(dim, 4000);
        let vec_b = generate_random_embedding(dim, 4001);

        // Benchmark cosine similarity
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dim),
            &(&vec_a, &vec_b),
            |b, (a, b_vec)| {
                b.iter(|| black_box(cosine_similarity(black_box(a), black_box(b_vec)).unwrap()))
            },
        );

        // Benchmark L2 distance
        group.bench_with_input(
            BenchmarkId::new("l2_distance", dim),
            &(&vec_a, &vec_b),
            |b, (a, b_vec)| {
                b.iter(|| black_box(l2_distance(black_box(a), black_box(b_vec)).unwrap()))
            },
        );

        // Benchmark dot product
        group.bench_with_input(
            BenchmarkId::new("dot_product", dim),
            &(&vec_a, &vec_b),
            |b, (a, b_vec)| {
                b.iter(|| black_box(dot_product(black_box(a), black_box(b_vec)).unwrap()))
            },
        );

        // Benchmark normalization
        group.bench_with_input(BenchmarkId::new("normalize", dim), &vec_a, |b, a| {
            b.iter(|| {
                let mut v = a.clone();
                normalize(black_box(&mut v));
                black_box(v)
            })
        });
    }

    group.finish();
}

// ============================================================================
// D. Serialization Benchmarks
// ============================================================================

fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    group.sample_size(50);

    for &size in &[100, 1_000, 10_000] {
        let documents = create_test_documents(size, EMBEDDING_DIM, 5000);

        // Benchmark document serialization (JSON)
        let doc = &documents[0];
        let serialized_json = serde_json::to_vec(doc).unwrap();
        group.throughput(Throughput::Bytes(serialized_json.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("document_json_serialize", size),
            doc,
            |b, doc| b.iter(|| black_box(serde_json::to_vec(black_box(doc)).unwrap())),
        );

        // Benchmark document deserialization (JSON)
        group.bench_with_input(
            BenchmarkId::new("document_json_deserialize", size),
            &serialized_json,
            |b, data| {
                b.iter(|| black_box(serde_json::from_slice::<Document>(black_box(data)).unwrap()))
            },
        );

        // Benchmark document serialization (bincode)
        let serialized_bincode = bincode::serialize(doc).unwrap();
        group.throughput(Throughput::Bytes(serialized_bincode.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("document_bincode_serialize", size),
            doc,
            |b, doc| b.iter(|| black_box(bincode::serialize(black_box(doc)).unwrap())),
        );

        // Benchmark document deserialization (bincode)
        group.bench_with_input(
            BenchmarkId::new("document_bincode_deserialize", size),
            &serialized_bincode,
            |b, data| {
                b.iter(|| black_box(bincode::deserialize::<Document>(black_box(data)).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// E. Storage Benchmarks
// ============================================================================

fn benchmark_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage");
    group.sample_size(20);

    let codecs = vec![
        Codec::None,
        Codec::Gzip,
        #[cfg(feature = "zstd")]
        Codec::Zstd,
        #[cfg(feature = "lz4")]
        Codec::Lz4,
    ];

    for &size in &[100, 1_000] {
        let documents = create_test_documents(size, EMBEDDING_DIM, 6000);

        for codec in &codecs {
            let codec_name = format!("{:?}", codec).to_lowercase();

            // Create temp directory for this test
            let temp_dir = TempDir::new().unwrap();
            let storage = FileStorage::with_codec(temp_dir.path(), *codec).unwrap();

            // Benchmark document save with compression
            let doc = &documents[0];
            group.bench_with_input(
                BenchmarkId::new(format!("save_document_{}", codec_name), size),
                &(&storage, doc),
                |b, (storage, doc)| {
                    let mut counter = 0;
                    b.iter(|| {
                        let id = format!("doc_{}", counter);
                        counter += 1;
                        black_box(storage.save_document(&id, black_box(doc)).unwrap())
                    })
                },
            );

            // Save a document for load benchmark
            storage.save_document("test_doc", doc).unwrap();

            // Benchmark document load with decompression
            group.bench_with_input(
                BenchmarkId::new(format!("load_document_{}", codec_name), size),
                &storage,
                |b, storage| b.iter(|| black_box(storage.load_document("test_doc").unwrap())),
            );
        }
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    benchmark_index_construction,
    benchmark_search_performance,
    benchmark_vector_operations,
    benchmark_serialization,
    benchmark_storage,
);

criterion_main!(benches);
