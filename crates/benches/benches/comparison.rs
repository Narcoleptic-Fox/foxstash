//! Comparative benchmarks: Foxstash vs instant-distance
//!
//! Run with: cargo bench -p foxstash-benches --bench comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use instant_distance::{Builder, Search};
use std::time::Instant;

// Same dataset parameters as Python benchmarks
const NUM_VECTORS: usize = 100_000;
const NUM_QUERIES: usize = 10_000;
const DIM: usize = 128;
const K: usize = 10;

/// Generate deterministic random vectors (same as Python synthetic data)
fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (seed, i, j).hash(&mut hasher);
                    let h = hasher.finish();
                    // Map to 0-255 range like Python synthetic data
                    (h % 256) as f32
                })
                .collect()
        })
        .collect()
}

/// Point type for instant-distance
#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

fn bench_instant_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("instant_distance");
    group.sample_size(10); // Fewer samples for slower benchmarks

    // Generate data
    println!("Generating {} vectors...", NUM_VECTORS);
    let base_vecs = generate_vectors(NUM_VECTORS, DIM, 42);
    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 123);

    let points: Vec<Point> = base_vecs.iter().map(|v| Point(v.clone())).collect();
    let queries: Vec<Point> = query_vecs.iter().map(|v| Point(v.clone())).collect();

    // Values are just indices
    let values: Vec<usize> = (0..NUM_VECTORS).collect();

    // Build benchmark
    group.bench_function("build_100k", |b| {
        b.iter(|| {
            let hnsw = Builder::default().build(black_box(points.clone()), values.clone());
            black_box(hnsw)
        })
    });

    // Build once for search benchmarks
    println!("Building instant-distance index...");
    let start = Instant::now();
    let hnsw = Builder::default().build(points.clone(), values.clone());
    println!("Build time: {:?}", start.elapsed());

    // Search benchmark
    group.throughput(Throughput::Elements(NUM_QUERIES as u64));
    group.bench_function("search_100k_10nn", |b| {
        b.iter(|| {
            let mut search = Search::default();
            for q in &queries {
                let results: Vec<_> = hnsw.search(q, &mut search).take(K).collect();
                black_box(results);
            }
        })
    });

    group.finish();
}

fn bench_foxstash(c: &mut Criterion) {
    use foxstash_core::index::hnsw::{HNSWConfig, HNSWIndex};
    use foxstash_core::Document;

    let mut group = c.benchmark_group("foxstash");
    group.sample_size(10);

    // Generate data
    println!("Generating {} vectors...", NUM_VECTORS);
    let base_vecs = generate_vectors(NUM_VECTORS, DIM, 42);
    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 123);

    // Build benchmark
    group.bench_function("build_100k", |b| {
        b.iter(|| {
            let config = HNSWConfig::default();
            let mut index = HNSWIndex::new(DIM, config);
            for (i, vec) in base_vecs.iter().enumerate() {
                let doc = Document {
                    id: i.to_string(),
                    content: String::new(),
                    embedding: vec.clone(),
                    metadata: None,
                };
                let _ = index.add(doc);
            }
            black_box(index)
        })
    });

    // Build once for search benchmarks
    println!("Building Foxstash index...");
    let config = HNSWConfig::default();
    let mut index = HNSWIndex::new(DIM, config);
    let start = Instant::now();
    for (i, vec) in base_vecs.iter().enumerate() {
        let doc = Document {
            id: i.to_string(),
            content: String::new(),
            embedding: vec.clone(),
            metadata: None,
        };
        let _ = index.add(doc);
    }
    println!("Build time: {:?}", start.elapsed());

    // Search benchmark
    group.throughput(Throughput::Elements(NUM_QUERIES as u64));
    group.bench_function("search_100k_10nn", |b| {
        b.iter(|| {
            for q in &query_vecs {
                let results = index.search(q, K);
                black_box(results);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_instant_distance, bench_foxstash);
criterion_main!(benches);
