//! Benchmarks for vector quantization operations
//!
//! Compares performance of:
//! - Full precision (f32) distance computation
//! - Scalar quantization (SQ8) distance computation
//! - Binary quantization Hamming distance
//! - Product quantization (PQ) distance computation
//!
//! Run with:
//! ```bash
//! cargo bench -p foxstash-core --bench quantization
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use foxstash_core::vector::{cosine_similarity, l2_distance};
use foxstash_core::vector::quantize::{
    BinaryQuantizer, Quantizer, ScalarQuantizer,
    hamming_distance_simd, sq8_l2_distance_simd,
};
use foxstash_core::vector::product_quantize::{PQConfig, ProductQuantizer, PQDistanceCache};
use rand::{Rng, SeedableRng};

fn create_test_vectors(size: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..1000)
        .map(|_| (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn create_single_vector(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

// ============================================================================
// Full Precision Benchmarks (Baseline)
// ============================================================================

fn bench_full_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_precision");

    for dim in [128, 384, 768] {
        let vectors = create_test_vectors(dim, 42);
        let query = create_single_vector(dim, 999);

        group.throughput(Throughput::Elements(1000));

        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for v in &vectors {
                        black_box(cosine_similarity(&query, v).unwrap());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("l2_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for v in &vectors {
                        black_box(l2_distance(&query, v).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scalar Quantization (SQ8) Benchmarks
// ============================================================================

fn bench_sq8(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_quantization");

    for dim in [128, 384, 768] {
        let vectors = create_test_vectors(dim, 42);
        let query = create_single_vector(dim, 999);

        // Create quantizer and quantize vectors
        let sq = ScalarQuantizer::fit(&vectors);
        let quantized: Vec<_> = vectors.iter().map(|v| sq.quantize(v)).collect();
        let query_quantized = sq.quantize(&query);

        group.throughput(Throughput::Elements(1000));

        // Symmetric distance (both quantized)
        group.bench_with_input(
            BenchmarkId::new("symmetric_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for qv in &quantized {
                        black_box(sq.distance_quantized(&query_quantized, qv));
                    }
                });
            },
        );

        // Asymmetric distance (full query vs quantized)
        group.bench_with_input(
            BenchmarkId::new("asymmetric_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for qv in &quantized {
                        black_box(sq.distance_asymmetric(&query, qv));
                    }
                });
            },
        );

        // Raw SIMD L2 distance on quantized bytes
        group.bench_with_input(
            BenchmarkId::new("simd_l2_bytes", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for qv in &quantized {
                        black_box(sq8_l2_distance_simd(&query_quantized.data, &qv.data));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Binary Quantization Benchmarks
// ============================================================================

fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_quantization");

    for dim in [128, 384, 768] {
        let vectors = create_test_vectors(dim, 42);
        let query = create_single_vector(dim, 999);

        // Create quantizer and quantize vectors
        let bq = BinaryQuantizer::new(dim);
        let quantized: Vec<_> = vectors.iter().map(|v| bq.quantize(v)).collect();
        let query_quantized = bq.quantize(&query);

        group.throughput(Throughput::Elements(1000));

        // Hamming distance
        group.bench_with_input(
            BenchmarkId::new("hamming_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for qv in &quantized {
                        black_box(bq.distance_quantized(&query_quantized, qv));
                    }
                });
            },
        );

        // Raw SIMD Hamming
        group.bench_with_input(
            BenchmarkId::new("simd_hamming", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for qv in &quantized {
                        black_box(hamming_distance_simd(&query_quantized.data, &qv.data));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Product Quantization Benchmarks
// ============================================================================

fn bench_pq(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_quantization");
    group.sample_size(50); // PQ training is slow, reduce samples

    for dim in [128, 384] {
        let vectors = create_test_vectors(dim, 42);
        let query = create_single_vector(dim, 999);

        // Train PQ
        let pq_config = PQConfig::new(dim, 8, 8)
            .with_seed(42)
            .with_kmeans_iterations(10);
        let pq = ProductQuantizer::train(&vectors, pq_config).unwrap();

        // Quantize vectors
        let codes: Vec<_> = vectors.iter().map(|v| pq.encode(v)).collect();

        // Build distance cache
        let cache = PQDistanceCache::build(&pq);

        // Precompute distance table for query
        let table = pq.compute_distance_table(&query);

        group.throughput(Throughput::Elements(1000));

        // ADC with precomputed table (fastest for batch)
        group.bench_with_input(
            BenchmarkId::new("adc_with_table", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for code in &codes {
                        black_box(pq.distance_with_table(&table, code));
                    }
                });
            },
        );

        // ADC without table (per-query)
        group.bench_with_input(
            BenchmarkId::new("adc_direct", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    for code in &codes {
                        black_box(pq.asymmetric_distance(&query, code));
                    }
                });
            },
        );

        // Symmetric with cache
        group.bench_with_input(
            BenchmarkId::new("symmetric_cached", dim),
            &dim,
            |bench, _| {
                let query_code = pq.encode(&query);
                bench.iter(|| {
                    for code in &codes {
                        black_box(cache.distance(&query_code, code));
                    }
                });
            },
        );

        // Symmetric without cache
        group.bench_with_input(
            BenchmarkId::new("symmetric_direct", dim),
            &dim,
            |bench, _| {
                let query_code = pq.encode(&query);
                bench.iter(|| {
                    for code in &codes {
                        black_box(pq.symmetric_distance(&query_code, code));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Usage Comparison
// ============================================================================

fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");

    let dim = 384;
    let count = 10_000;

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| create_single_vector(dim, i as u64))
        .collect();

    // Memory comparison (printed during benchmark)
    let full_size = count * dim * 4;
    let sq8_size = count * dim;
    let binary_size = count * ((dim + 7) / 8);
    let pq_size = count * 8;

    println!("\n=== Memory Comparison ({} vectors x {} dims) ===", count, dim);
    println!("Full precision (f32): {:.2} MB", full_size as f64 / 1_000_000.0);
    println!("SQ8: {:.2} MB ({:.0}x compression)", sq8_size as f64 / 1_000_000.0, full_size as f64 / sq8_size as f64);
    println!("Binary: {:.2} MB ({:.0}x compression)", binary_size as f64 / 1_000_000.0, full_size as f64 / binary_size as f64);
    println!("PQ (M=8): {:.2} MB ({:.0}x compression)", pq_size as f64 / 1_000_000.0, full_size as f64 / pq_size as f64);
    println!();

    // Benchmark encode speed
    group.throughput(Throughput::Elements(count as u64));

    let sq = ScalarQuantizer::fit(&vectors[..1000]);
    group.bench_function("sq8_encode", |bench| {
        bench.iter(|| {
            for v in &vectors {
                black_box(sq.quantize(v));
            }
        });
    });

    let bq = BinaryQuantizer::new(dim);
    group.bench_function("binary_encode", |bench| {
        bench.iter(|| {
            for v in &vectors {
                black_box(bq.quantize(v));
            }
        });
    });

    let pq_config = PQConfig::new(dim, 8, 8).with_seed(42).with_kmeans_iterations(10);
    let pq = ProductQuantizer::train(&vectors[..1000], pq_config).unwrap();
    group.bench_function("pq_encode", |bench| {
        bench.iter(|| {
            for v in &vectors {
                black_box(pq.encode(v));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_full_precision,
    bench_sq8,
    bench_binary,
    bench_pq,
    bench_memory_comparison,
);
criterion_main!(benches);
