//! Benchmarks for SIMD vs scalar vector operations
//!
//! This benchmark suite measures the performance improvement from SIMD acceleration
//! for various vector operations across different dimensions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use foxstash_core::vector::{
    cosine_similarity, cosine_similarity_simd, dot_product, dot_product_simd, l2_distance,
    l2_distance_simd,
};

fn create_test_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
    let b: Vec<f32> = (0..size)
        .map(|i| 1.0 - (i as f32) / (size as f32))
        .collect();
    (a, b)
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [128, 384, 768, 1024, 2048] {
        let (a, b) = create_test_vectors(size);

        // Throughput in terms of number of f32 operations
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let result = dot_product(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| {
                let result = dot_product_simd(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for size in [128, 384, 768, 1024, 2048] {
        let (a, b) = create_test_vectors(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let result = l2_distance(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| {
                let result = l2_distance_simd(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for size in [128, 384, 768, 1024, 2048] {
        let (a, b) = create_test_vectors(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let result = cosine_similarity(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| {
                let result = cosine_similarity_simd(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_l2_distance,
    bench_cosine_similarity
);
criterion_main!(benches);
