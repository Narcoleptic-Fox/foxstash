//! Comprehensive benchmarking suite for Foxstash
//!
//! This benchmark suite measures performance of core RAG operations including:
//! - Vector operations (cosine similarity, L2 distance, normalization)
//! - Flat index operations (add, search)
//! - HNSW index operations (add, search)
//! - Embedding generation (single and batch)
//! - Cache performance
//!
//! ## Performance Targets
//! - Vector ops: <1µs for 384-dim vectors
//! - Flat index search: <1ms for 1,000 docs
//! - HNSW search: <10ms for 10,000 docs
//! - Embedding: <30ms per text (will test when model available)
//!
//! ## Running Benchmarks
//! ```bash
//! # Run all benchmarks (except ignored embedding tests)
//! cargo bench -p foxstash-benches
//!
//! # Run specific benchmark group
//! cargo bench -p foxstash-benches vector_operations
//! cargo bench -p foxstash-benches flat_index
//! cargo bench -p foxstash-benches hnsw_index
//!
//! # Run embedding benchmarks (when model is available)
//! cargo bench -p foxstash-benches -- --ignored
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use foxstash_core::index::flat::FlatIndex;
use foxstash_core::index::hnsw::HNSWIndex;
use foxstash_core::{cosine_similarity, dot_product, l2_distance, normalize, Document};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

// ============================================================================
// Vector Operations Benchmarks
// ============================================================================

/// Benchmark vector operations: cosine similarity, L2 distance, normalization
///
/// Performance target: <1µs for 384-dimensional vectors
fn benchmark_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    // Test with realistic embedding dimension (384 for MiniLM-L6-v2)
    let dim = 384;
    let vec_a = generate_random_embedding(dim, 1);
    let vec_b = generate_random_embedding(dim, 2);

    // Benchmark cosine similarity
    group.bench_function("cosine_similarity_384d", |b| {
        b.iter(|| cosine_similarity(black_box(&vec_a), black_box(&vec_b)).unwrap())
    });

    // Benchmark L2 distance
    group.bench_function("l2_distance_384d", |b| {
        b.iter(|| l2_distance(black_box(&vec_a), black_box(&vec_b)).unwrap())
    });

    // Benchmark dot product
    group.bench_function("dot_product_384d", |b| {
        b.iter(|| dot_product(black_box(&vec_a), black_box(&vec_b)).unwrap())
    });

    // Benchmark normalization (in-place operation)
    group.bench_function("normalize_384d", |b| {
        b.iter(|| {
            let mut vec = black_box(vec_a.clone());
            normalize(black_box(&mut vec));
            vec
        })
    });

    // Also test with smaller dimension for comparison
    let dim_small = 128;
    let vec_a_small = generate_random_embedding(dim_small, 3);
    let vec_b_small = generate_random_embedding(dim_small, 4);

    group.bench_function("cosine_similarity_128d", |b| {
        b.iter(|| cosine_similarity(black_box(&vec_a_small), black_box(&vec_b_small)).unwrap())
    });

    group.bench_function("normalize_128d", |b| {
        b.iter(|| {
            let mut vec = black_box(vec_a_small.clone());
            normalize(black_box(&mut vec));
            vec
        })
    });

    group.finish();
}

// ============================================================================
// Flat Index Benchmarks
// ============================================================================

/// Benchmark Flat index operations with varying document counts
///
/// Performance target: <1ms search for 1,000 documents
fn benchmark_flat_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_index");
    let dim = 384;

    // Test with varying document counts
    for &doc_count in &[100, 1000, 10000] {
        // Benchmark index construction (add operations)
        group.bench_with_input(
            BenchmarkId::new("add", doc_count),
            &doc_count,
            |b, &count| {
                b.iter_batched(
                    || create_test_documents(count, dim, 1000),
                    |docs| {
                        let mut index = FlatIndex::new(dim);
                        for doc in docs {
                            index.add(black_box(doc)).unwrap();
                        }
                        index
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Benchmark search operations
        group.bench_with_input(
            BenchmarkId::new("search_k5", doc_count),
            &doc_count,
            |b, &count| {
                // Setup: create index with documents
                let mut index = FlatIndex::new(dim);
                let docs = create_test_documents(count, dim, 1000);
                for doc in docs {
                    index.add(doc).unwrap();
                }
                let query = generate_random_embedding(dim, 9999);

                // Benchmark search
                b.iter(|| index.search(black_box(&query), black_box(5)).unwrap())
            },
        );

        // Benchmark search with k=10
        group.bench_with_input(
            BenchmarkId::new("search_k10", doc_count),
            &doc_count,
            |b, &count| {
                let mut index = FlatIndex::new(dim);
                let docs = create_test_documents(count, dim, 1000);
                for doc in docs {
                    index.add(doc).unwrap();
                }
                let query = generate_random_embedding(dim, 9999);

                b.iter(|| index.search(black_box(&query), black_box(10)).unwrap())
            },
        );

        // Benchmark batch add
        group.bench_with_input(
            BenchmarkId::new("add_batch", doc_count),
            &doc_count,
            |b, &count| {
                b.iter_batched(
                    || create_test_documents(count, dim, 1000),
                    |docs| {
                        let mut index = FlatIndex::new(dim);
                        index.add_batch(black_box(docs)).unwrap();
                        index
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ============================================================================
// HNSW Index Benchmarks
// ============================================================================

/// Benchmark HNSW index operations with varying document counts
///
/// Performance target: <10ms search for 10,000 documents
fn benchmark_hnsw_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_index");
    let dim = 384;

    // Increase sample size for more stable measurements on larger datasets
    group.sample_size(20);

    // Test with varying document counts
    for &doc_count in &[100, 1000, 10000] {
        // Benchmark index construction (add operations)
        group.bench_with_input(
            BenchmarkId::new("add", doc_count),
            &doc_count,
            |b, &count| {
                b.iter_batched(
                    || create_test_documents(count, dim, 2000),
                    |docs| {
                        let mut index = HNSWIndex::with_defaults(dim);
                        for doc in docs {
                            index.add(black_box(doc)).unwrap();
                        }
                        index
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Benchmark search operations with k=5
        group.bench_with_input(
            BenchmarkId::new("search_k5", doc_count),
            &doc_count,
            |b, &count| {
                // Setup: create index with documents
                let mut index = HNSWIndex::with_defaults(dim);
                let docs = create_test_documents(count, dim, 2000);
                for doc in docs {
                    index.add(doc).unwrap();
                }
                let query = generate_random_embedding(dim, 9999);

                // Benchmark search
                b.iter(|| index.search(black_box(&query), black_box(5)).unwrap())
            },
        );

        // Benchmark search with k=10
        group.bench_with_input(
            BenchmarkId::new("search_k10", doc_count),
            &doc_count,
            |b, &count| {
                let mut index = HNSWIndex::with_defaults(dim);
                let docs = create_test_documents(count, dim, 2000);
                for doc in docs {
                    index.add(doc).unwrap();
                }
                let query = generate_random_embedding(dim, 9999);

                b.iter(|| index.search(black_box(&query), black_box(10)).unwrap())
            },
        );

        // Benchmark search with k=20 (larger recall)
        group.bench_with_input(
            BenchmarkId::new("search_k20", doc_count),
            &doc_count,
            |b, &count| {
                let mut index = HNSWIndex::with_defaults(dim);
                let docs = create_test_documents(count, dim, 2000);
                for doc in docs {
                    index.add(doc).unwrap();
                }
                let query = generate_random_embedding(dim, 9999);

                b.iter(|| index.search(black_box(&query), black_box(20)).unwrap())
            },
        );
    }

    group.finish();
}

// ============================================================================
// Embedding Benchmarks (Marked as #[ignore])
// ============================================================================

/// Benchmark single text embedding with varying text lengths
///
/// Performance target: <30ms per embedding
///
/// Note: This benchmark is marked `#[ignore]` because the embedding module
/// is not yet implemented. Run with `cargo bench -- --ignored` once the
/// ONNX embedder is available.
#[allow(dead_code)]
#[cfg(feature = "onnx")]
fn benchmark_embedding_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    // Short text (10-20 words)
    let short_text = "This is a short test document for embedding generation.";

    // Medium text (50-100 words)
    let medium_text = "This is a medium length document that contains more content \
        and provides a realistic test case for embedding generation performance. \
        The text is long enough to exercise the tokenizer and model properly, \
        simulating real-world usage patterns where documents might contain several \
        sentences of meaningful content that needs to be processed efficiently.";

    // Long text (200+ words)
    let long_text = "This is a longer document that represents a more substantial \
        piece of text that might be encountered in production environments. \
        Documents of this length are common in many applications including \
        knowledge bases, documentation, articles, and other content repositories. \
        The embedding model needs to efficiently process such texts while \
        maintaining good quality vector representations. Performance at this \
        scale is critical for real-world applications where users expect \
        responsive search and retrieval operations. The system must balance \
        speed and accuracy, ensuring that embeddings are generated quickly \
        without sacrificing the quality needed for effective similarity search. \
        This longer text helps us understand how the model performs under \
        more realistic load conditions and whether there are any performance \
        bottlenecks that only appear with larger inputs.";

    // Note: These benchmarks will be implemented when OnnxEmbedder is ready
    // group.bench_function("single_short", |b| {
    //     let embedder = OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap();
    //     b.iter(|| embedder.embed(black_box(short_text)).unwrap())
    // });

    // group.bench_function("single_medium", |b| {
    //     let embedder = OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap();
    //     b.iter(|| embedder.embed(black_box(medium_text)).unwrap())
    // });

    // group.bench_function("single_long", |b| {
    //     let embedder = OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap();
    //     b.iter(|| embedder.embed(black_box(long_text)).unwrap())
    // });

    // Placeholder to prevent unused variable warnings
    let _ = (short_text, medium_text, long_text);

    group.finish();
}

/// Benchmark batch embedding with varying batch sizes
///
/// Performance target: Amortized <30ms per embedding in batch
///
/// Note: This benchmark is marked `#[ignore]` because the embedding module
/// is not yet implemented. Run with `cargo bench -- --ignored` once the
/// ONNX embedder is available.
#[allow(dead_code)]
#[cfg(feature = "onnx")]
fn benchmark_embedding_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    let sample_texts: Vec<&str> = vec![
        "First document about machine learning.",
        "Second document discussing neural networks.",
        "Third document on natural language processing.",
        "Fourth document about vector embeddings.",
        "Fifth document covering information retrieval.",
        "Sixth document on semantic search techniques.",
        "Seventh document about transformer models.",
        "Eighth document on attention mechanisms.",
        "Ninth document discussing BERT and derivatives.",
        "Tenth document about efficient inference.",
    ];

    // Test different batch sizes
    for &batch_size in &[1, 5, 10, 20, 50] {
        // Note: This will be implemented when OnnxEmbedder is ready
        // group.bench_with_input(
        //     BenchmarkId::new("batch", batch_size),
        //     &batch_size,
        //     |b, &size| {
        //         let embedder = OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap();
        //         let texts: Vec<&str> = sample_texts.iter()
        //             .cycle()
        //             .take(size)
        //             .copied()
        //             .collect();
        //
        //         b.iter(|| embedder.embed_batch(black_box(&texts)).unwrap())
        //     },
        // );

        // Placeholder to prevent unused variable warnings
        let _ = batch_size;
    }

    group.finish();
}

/// Benchmark cached vs uncached embedding performance
///
/// This benchmark compares the performance of embedding with and without caching.
///
/// Note: This benchmark is marked `#[ignore]` because the embedding module
/// is not yet implemented. Run with `cargo bench -- --ignored` once the
/// ONNX embedder with caching is available.
#[allow(dead_code)]
#[cfg(feature = "onnx")]
fn benchmark_cached_embedder(c: &mut Criterion) {
    let mut group = c.benchmark_group("caching");

    let test_text = "This is a test document that will be embedded multiple times.";

    // Note: These will be implemented when CachedEmbedder is ready
    // group.bench_function("uncached", |b| {
    //     let embedder = OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap();
    //     b.iter(|| embedder.embed(black_box(test_text)).unwrap())
    // });

    // group.bench_function("cached_hit", |b| {
    //     let mut embedder = CachedEmbedder::new(
    //         OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap(),
    //         1000
    //     );
    //     // Warm up the cache
    //     embedder.embed(test_text).unwrap();
    //
    //     b.iter(|| embedder.embed(black_box(test_text)).unwrap())
    // });

    // group.bench_function("cached_miss", |b| {
    //     let mut embedder = CachedEmbedder::new(
    //         OnnxEmbedder::new("models/minilm-l6-v2.onnx").unwrap(),
    //         1000
    //     );
    //     let mut counter = 0;
    //
    //     b.iter(|| {
    //         let text = format!("{} {}", test_text, counter);
    //         counter += 1;
    //         embedder.embed(black_box(&text)).unwrap()
    //     })
    // });

    // Placeholder to prevent unused variable warnings
    let _ = test_text;

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

// Active benchmarks (run by default)
criterion_group!(
    benches,
    benchmark_vector_ops,
    benchmark_flat_index,
    benchmark_hnsw_index,
);

// Embedding benchmarks (ignored until ONNX module is implemented)
// Uncomment when ready:
// #[cfg(feature = "onnx")]
// criterion_group!(
//     embedding_benches,
//     benchmark_embedding_single,
//     benchmark_embedding_batch,
//     benchmark_cached_embedder,
// );

criterion_main!(benches);
