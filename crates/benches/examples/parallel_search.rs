//! Parallel search benchmark
//!
//! Run with: cargo run -p foxstash-benches --example parallel_search --release

use foxstash_core::index::hnsw::{HNSWConfig, HNSWIndex};
use std::time::Instant;

const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 10_000;
const DIM: usize = 128;
const K: usize = 10;

fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (seed, i, j).hash(&mut hasher);
                    let h = hasher.finish();
                    (h % 256) as f32 - 128.0
                })
                .collect();

            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        })
        .collect()
}

fn main() {
    println!("=== Parallel Search Benchmark ===\n");
    println!(
        "Dataset: {} vectors, {}d, {} queries, top-{}\n",
        NUM_VECTORS, DIM, NUM_QUERIES, K
    );

    // Generate data
    println!("Generating vectors...");
    let base_vecs = generate_vectors(NUM_VECTORS, DIM, 42);
    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 123);

    // Build index
    println!("Building index...");
    let config = HNSWConfig::default();
    let mut index = HNSWIndex::new(DIM, config);

    let start = Instant::now();
    for (i, vec) in base_vecs.iter().enumerate() {
        let _ = index.add_embedding(i.to_string(), vec.clone());
    }
    println!("Build time: {:?}\n", start.elapsed());

    // Sequential search
    println!("--- Sequential Search ---");
    let start = Instant::now();
    for q in &query_vecs {
        let _ = index.search(q, K);
    }
    let seq_time = start.elapsed();
    let seq_qps = NUM_QUERIES as f64 / seq_time.as_secs_f64();
    println!("Time: {:?} ({:.0} QPS)", seq_time, seq_qps);

    // Parallel search
    println!("\n--- Parallel Search (batch) ---");
    let start = Instant::now();
    let _ = index.search_batch(&query_vecs, K);
    let par_time = start.elapsed();
    let par_qps = NUM_QUERIES as f64 / par_time.as_secs_f64();
    println!("Time: {:?} ({:.0} QPS)", par_time, par_qps);

    println!("\n=== SUMMARY ===");
    println!("Sequential: {:.0} QPS", seq_qps);
    println!("Parallel:   {:.0} QPS", par_qps);
    println!("Speedup:    {:.2}x", par_qps / seq_qps);
}
