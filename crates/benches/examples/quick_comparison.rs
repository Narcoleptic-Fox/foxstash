//! Quick comparison benchmark - single run timing
//!
//! Run with: cargo run -p foxstash-benches --example quick_comparison --release

use instant_distance::{Builder, Search};
use std::time::Instant;

const NUM_VECTORS: usize = 100_000;
const NUM_QUERIES: usize = 10_000;
const DIM: usize = 128;
const K: usize = 10;

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
                    (h % 256) as f32
                })
                .collect()
        })
        .collect()
}

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

fn main() {
    println!("=== Foxstash vs instant-distance Comparison ===\n");
    println!("Dataset: {} vectors, {}d, {} queries, top-{}\n", NUM_VECTORS, DIM, NUM_QUERIES, K);
    
    // Generate data
    println!("Generating vectors...");
    let base_vecs = generate_vectors(NUM_VECTORS, DIM, 42);
    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 123);
    
    // === instant-distance ===
    println!("\n--- instant-distance ---");
    let points: Vec<Point> = base_vecs.iter().map(|v| Point(v.clone())).collect();
    let queries_id: Vec<Point> = query_vecs.iter().map(|v| Point(v.clone())).collect();
    let values: Vec<usize> = (0..NUM_VECTORS).collect();
    
    let start = Instant::now();
    let hnsw = Builder::default().build(points.clone(), values);
    let id_build_time = start.elapsed();
    println!("Build time: {:?}", id_build_time);
    
    let start = Instant::now();
    let mut search = Search::default();
    for q in &queries_id {
        let _: Vec<_> = hnsw.search(q, &mut search).take(K).collect();
    }
    let id_search_time = start.elapsed();
    let id_qps = NUM_QUERIES as f64 / id_search_time.as_secs_f64();
    println!("Search time: {:?} ({:.0} QPS)", id_search_time, id_qps);
    
    // === Foxstash ===
    println!("\n--- Foxstash ---");
    use foxstash_core::index::hnsw::{HNSWConfig, HNSWIndex};
    use foxstash_core::Document;
    
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
    let fs_build_time = start.elapsed();
    println!("Build time: {:?}", fs_build_time);
    
    let start = Instant::now();
    for q in &query_vecs {
        let _ = index.search(q, K);
    }
    let fs_search_time = start.elapsed();
    let fs_qps = NUM_QUERIES as f64 / fs_search_time.as_secs_f64();
    println!("Search time: {:?} ({:.0} QPS)", fs_search_time, fs_qps);
    
    // === Summary ===
    println!("\n=== SUMMARY ===");
    println!("{:<20} {:>12} {:>12}", "Library", "Build Time", "QPS");
    println!("{:-<46}", "");
    println!("{:<20} {:>12.2?} {:>12.0}", "instant-distance", id_build_time, id_qps);
    println!("{:<20} {:>12.2?} {:>12.0}", "Foxstash", fs_build_time, fs_qps);
    
    // Speedup
    let build_speedup = id_build_time.as_secs_f64() / fs_build_time.as_secs_f64();
    let search_speedup = fs_qps / id_qps;
    println!("\nFoxstash vs instant-distance:");
    println!("  Build: {:.2}x {}", build_speedup, if build_speedup > 1.0 { "faster" } else { "slower" });
    println!("  Search: {:.2}x {}", search_speedup, if search_speedup > 1.0 { "faster" } else { "slower" });
}
