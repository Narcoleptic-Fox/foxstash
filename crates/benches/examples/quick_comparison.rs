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
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (seed, i, j).hash(&mut hasher);
                    let h = hasher.finish();
                    // Center around 0 for better distribution
                    (h % 256) as f32 - 128.0
                })
                .collect();

            // Normalize to unit length (required for cosine similarity)
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
    println!(
        "Dataset: {} vectors, {}d, {} queries, top-{}\n",
        NUM_VECTORS, DIM, NUM_QUERIES, K
    );

    // Generate data
    println!("Generating vectors...");
    let base_vecs = generate_vectors(NUM_VECTORS, DIM, 42);
    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 123);

    // === instant-distance ===
    println!("\n--- instant-distance (single-threaded, context reuse) ---");
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

    // === Foxstash (parallel build, single-threaded context-reuse search) ===
    println!("\n--- Foxstash (parallel build, single-threaded context reuse) ---");
    use foxstash_core::index::{BuildStrategy, HNSWConfig, HNSWIndex};

    let config = HNSWConfig::default().with_build_strategy(BuildStrategy::Parallel);

    let start = Instant::now();
    let index = HNSWIndex::build(base_vecs.clone(), config);
    let fs_build_time = start.elapsed();
    println!("Build time: {:?}", fs_build_time);

    // Single-threaded search with searcher reuse (fair comparison to instant-distance)
    let mut searcher = index.searcher();
    let start = Instant::now();
    for q in &query_vecs {
        let _ = searcher.search(q, K);
    }
    let fs_st_search_time = start.elapsed();
    let fs_st_qps = NUM_QUERIES as f64 / fs_st_search_time.as_secs_f64();
    println!(
        "Search time: {:?} ({:.0} QPS)",
        fs_st_search_time, fs_st_qps
    );

    // === Foxstash batch (parallel search) ===
    println!("\n--- Foxstash (parallel build, batch search via rayon) ---");
    let start = Instant::now();
    let _ = index.search_batch(&query_vecs, K);
    let fs_batch_search_time = start.elapsed();
    let fs_batch_qps = NUM_QUERIES as f64 / fs_batch_search_time.as_secs_f64();
    println!(
        "Search time: {:?} ({:.0} QPS)",
        fs_batch_search_time, fs_batch_qps
    );

    // === Recall Check ===
    println!("\n--- Recall Check (100 queries, brute-force ground truth) ---");
    let recall_queries = 100;
    let mut foxstash_total_recall = 0.0;
    let mut id_total_recall = 0.0;
    let mut recall_searcher = index.searcher();

    for q in query_vecs.iter().take(recall_queries) {
        // Brute-force ground truth using Euclidean distance (same as instant-distance)
        let mut distances: Vec<(f32, usize)> = base_vecs
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let dist: f32 = q
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (dist, j)
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let ground_truth: std::collections::HashSet<usize> =
            distances.iter().take(K).map(|(_, j)| *j).collect();

        // Foxstash results (with searcher reuse)
        let results = recall_searcher.search(q, K).unwrap();
        let foxstash_ids: std::collections::HashSet<usize> =
            results.iter().map(|r| r.id.parse().unwrap()).collect();

        let overlap = ground_truth.intersection(&foxstash_ids).count();
        foxstash_total_recall += overlap as f32 / K as f32;

        // instant-distance results
        let query_point = Point(q.clone());
        let id_results: Vec<_> = hnsw.search(&query_point, &mut search).take(K).collect();
        let id_ids: std::collections::HashSet<usize> =
            id_results.iter().map(|item| *item.value).collect();

        let id_overlap = ground_truth.intersection(&id_ids).count();
        id_total_recall += id_overlap as f32 / K as f32;
    }

    let foxstash_avg_recall = foxstash_total_recall / recall_queries as f32;
    let id_avg_recall = id_total_recall / recall_queries as f32;
    println!("Foxstash Recall@{}: {:.2}%", K, foxstash_avg_recall * 100.0);
    println!(
        "instant-distance Recall@{}: {:.2}%",
        K,
        id_avg_recall * 100.0
    );

    // === Summary ===
    println!("\n=== SUMMARY ===");
    println!("{:<30} {:>12} {:>12}", "Library", "Build Time", "QPS");
    println!("{:-<56}", "");
    println!(
        "{:<30} {:>12.2?} {:>12.0}",
        "instant-distance (1T)", id_build_time, id_qps
    );
    println!(
        "{:<30} {:>12.2?} {:>12.0}",
        "Foxstash (1T, ctx reuse)", fs_build_time, fs_st_qps
    );
    println!(
        "{:<30} {:>12.2?} {:>12.0}",
        "Foxstash (batch, rayon)", fs_build_time, fs_batch_qps
    );

    // Speedup (single-threaded apples-to-apples)
    let st_speedup = fs_st_qps / id_qps;
    let batch_speedup = fs_batch_qps / id_qps;
    println!("\nFoxstash vs instant-distance:");
    println!(
        "  Single-threaded: {:.2}x {}",
        st_speedup,
        if st_speedup > 1.0 { "faster" } else { "slower" }
    );
    println!(
        "  Batch (rayon):   {:.2}x {}",
        batch_speedup,
        if batch_speedup > 1.0 {
            "faster"
        } else {
            "slower"
        }
    );

    println!("\nNote: Foxstash uses ef_search=100, Python benchmarks use ef_search=64.");
    println!("      Higher ef_search = better recall but lower QPS.");
}
