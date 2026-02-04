//! Comparison Benchmarks: Foxstash vs instant-distance
//!
//! This benchmark compares Foxstash's HNSW implementation against
//! the `instant-distance` crate (another Rust HNSW library).
//!
//! ## Running
//! ```bash
//! cargo bench -p foxstash-benches --bench comparison
//! cargo bench -p foxstash-benches --bench comparison -- --save-baseline main
//! cargo bench -p foxstash-benches --bench comparison -- --baseline main
//! ```
//!
//! ## Test Sizes
//! - 10K vectors @ 128-dim (CI-friendly)
//! - 100K vectors @ 128-dim (full comparison)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use foxstash_core::index::{HNSWConfig, HNSWIndex};
use foxstash_core::Document;
use instant_distance::{Builder, Search};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Duration;

// ============================================================================
// Constants
// ============================================================================

/// SIFT-like dimension (also common for smaller embedding models)
const DIM: usize = 128;

/// Test sizes for CI and full benchmarks
const SIZES: &[usize] = &[10_000, 100_000];

/// Number of neighbors to retrieve
const K_VALUES: &[usize] = &[10, 100];

/// Number of queries per benchmark
const NUM_QUERIES: usize = 100;

// ============================================================================
// Data Generation
// ============================================================================

/// Generate random f32 vectors (SIFT-like distribution)
fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 256.0).collect())
        .collect()
}

/// Create Foxstash documents from vectors
fn create_documents(vectors: &[Vec<f32>]) -> Vec<Document> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| Document {
            id: format!("doc_{}", i),
            content: String::new(),
            embedding: v.clone(),
            metadata: None,
        })
        .collect()
}

/// Wrapper for instant-distance point
#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // L2 distance (squared)
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
    }
}

// ============================================================================
// Index Construction Benchmarks
// ============================================================================

fn benchmark_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in SIZES {
        let vectors = generate_vectors(size, DIM, 42);
        let documents = create_documents(&vectors);
        let points: Vec<Point> = vectors.iter().map(|v| Point(v.clone())).collect();
        // instant-distance requires values alongside points
        let values: Vec<usize> = (0..size).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Foxstash HNSW
        group.bench_with_input(
            BenchmarkId::new("foxstash_hnsw", size),
            &documents,
            |b, docs| {
                b.iter(|| {
                    let config = HNSWConfig::default()
                        .with_ef_construction(200)
                        .with_ef_search(50);
                    let mut index = HNSWIndex::new(DIM, config);
                    for doc in docs {
                        index.add(black_box(doc.clone())).unwrap();
                    }
                    black_box(index)
                })
            },
        );

        // instant-distance
        group.bench_with_input(
            BenchmarkId::new("instant_distance", size),
            &(&points, &values),
            |b, (pts, vals)| {
                b.iter(|| {
                    let builder = Builder::default();
                    let hnsw = builder.build(black_box((*pts).clone()), black_box((*vals).clone()));
                    black_box(hnsw)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Search Benchmarks
// ============================================================================

fn benchmark_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_comparison");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(20));

    for &size in SIZES {
        // Generate data
        let vectors = generate_vectors(size, DIM, 42);
        let documents = create_documents(&vectors);
        let points: Vec<Point> = vectors.iter().map(|v| Point(v.clone())).collect();
        let values: Vec<usize> = (0..size).collect();

        // Build indices
        let config = HNSWConfig::default()
            .with_ef_construction(200)
            .with_ef_search(100);
        let mut foxstash_index = HNSWIndex::new(DIM, config);
        for doc in &documents {
            foxstash_index.add(doc.clone()).unwrap();
        }

        let instant_hnsw = Builder::default().build(points.clone(), values);

        // Generate query vectors
        let queries = generate_vectors(NUM_QUERIES, DIM, 1000);

        for &k in K_VALUES {
            // Foxstash search
            group.bench_with_input(
                BenchmarkId::new(format!("foxstash_search_{}docs", size), k),
                &(&foxstash_index, &queries),
                |b, (index, qs)| {
                    let mut query_idx = 0;
                    b.iter(|| {
                        let query = &qs[query_idx % qs.len()];
                        query_idx += 1;
                        black_box(index.search(black_box(query), black_box(k)).unwrap())
                    })
                },
            );

            // instant-distance search
            let query_points: Vec<Point> = queries.iter().map(|v| Point(v.clone())).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("instant_distance_{}docs", size), k),
                &(&instant_hnsw, &query_points),
                |b, (hnsw, qs)| {
                    let mut query_idx = 0;
                    b.iter(|| {
                        let query = &qs[query_idx % qs.len()];
                        query_idx += 1;
                        let mut search = Search::default();
                        let results: Vec<_> = hnsw.search(black_box(query), &mut search).take(k).collect();
                        black_box(results)
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmark (via serialization size proxy)
// ============================================================================

fn benchmark_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");
    group.sample_size(10);

    // This benchmark measures construction overhead as a proxy for memory efficiency
    
    for &size in &[10_000usize] {
        let vectors = generate_vectors(size, DIM, 42);
        let documents = create_documents(&vectors);

        group.bench_with_input(
            BenchmarkId::new("foxstash_build_overhead", size),
            &documents,
            |b, docs| {
                b.iter(|| {
                    let config = HNSWConfig::default()
                        .with_ef_construction(200)
                        .with_ef_search(50);
                    let mut index = HNSWIndex::new(DIM, config);
                    for doc in docs {
                        index.add(doc.clone()).unwrap();
                    }
                    // Count nodes as memory proxy
                    black_box(index.len())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Recall Measurement (not timed, just for accuracy comparison)
// ============================================================================

/// Compute ground truth using brute force
fn compute_ground_truth(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist: f32 = v.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (i, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(i, _)| *i).collect()
}

/// Measure recall of Foxstash HNSW
fn measure_foxstash_recall(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f32 {
    let documents = create_documents(base);
    
    let config = HNSWConfig::default()
        .with_ef_construction(200)
        .with_ef_search(200);  // Higher for accuracy test
    let mut index = HNSWIndex::new(DIM, config);
    for doc in &documents {
        index.add(doc.clone()).unwrap();
    }
    
    let mut total_recall = 0.0;
    for query in queries {
        let gt = compute_ground_truth(base, query, k);
        let results = index.search(query, k).unwrap();
        let retrieved: std::collections::HashSet<_> = results.iter().map(|r| {
            r.id.strip_prefix("doc_").unwrap().parse::<usize>().unwrap()
        }).collect();
        let gt_set: std::collections::HashSet<_> = gt.into_iter().collect();
        total_recall += retrieved.intersection(&gt_set).count() as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}

/// Measure recall of instant-distance
fn measure_instant_recall(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f32 {
    let points: Vec<Point> = base.iter().map(|v| Point(v.clone())).collect();
    let values: Vec<usize> = (0..base.len()).collect();
    let hnsw = Builder::default().build(points, values);
    
    let mut total_recall = 0.0;
    
    for query in queries {
        let gt = compute_ground_truth(base, query, k);
        let gt_set: std::collections::HashSet<_> = gt.into_iter().collect();
        
        let mut search = Search::default();
        let query_point = Point(query.clone());
        let retrieved: std::collections::HashSet<_> = hnsw
            .search(&query_point, &mut search)
            .take(k)
            .map(|item| *item.value)
            .collect();
        
        total_recall += retrieved.intersection(&gt_set).count() as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}

fn benchmark_recall(c: &mut Criterion) {
    // This is more of a test than a benchmark, but useful for comparison
    let mut group = c.benchmark_group("recall_measurement");
    group.sample_size(10);
    
    let size = 10_000;
    let base = generate_vectors(size, DIM, 42);
    let queries = generate_vectors(100, DIM, 1000);
    
    for &k in &[10, 100] {
        group.bench_function(
            BenchmarkId::new("foxstash_recall", k),
            |b| {
                b.iter(|| {
                    let recall = measure_foxstash_recall(&base, &queries, k);
                    println!("Foxstash Recall@{}: {:.4}", k, recall);
                    black_box(recall)
                })
            },
        );
        
        group.bench_function(
            BenchmarkId::new("instant_distance_recall", k),
            |b| {
                b.iter(|| {
                    let recall = measure_instant_recall(&base, &queries, k);
                    println!("instant-distance Recall@{}: {:.4}", k, recall);
                    black_box(recall)
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    benchmark_construction,
    benchmark_search,
    benchmark_memory,
    benchmark_recall,
);

criterion_main!(benches);
