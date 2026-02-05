use foxstash_core::index::{BuildStrategy, HNSWConfig, HNSWIndex};
use std::time::Instant;
use std::collections::HashSet;

const NUM_VECTORS: usize = 100_000;
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
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { for x in &mut vec { *x /= norm; } }
            vec
        })
        .collect()
}

fn measure_recall(index: &HNSWIndex, queries: &[Vec<f32>], base: &[Vec<f32>], k: usize) -> f32 {
    let mut total = 0.0;
    for q in queries.iter().take(100) {
        let mut distances: Vec<(f32, usize)> = base.iter().enumerate()
            .map(|(j, v)| {
                let dist: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                (dist, j)
            }).collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: HashSet<usize> = distances.iter().take(k).map(|(_, j)| *j).collect();
        
        let results = index.search(q, k).unwrap();
        let found: HashSet<usize> = results.iter().map(|r| r.id.parse().unwrap()).collect();
        total += truth.intersection(&found).count() as f32 / k as f32;
    }
    total / 100.0
}

fn main() {
    println!("=== Foxstash Build Strategy Comparison ===");
    println!("Dataset: {} vectors, {}d\n", NUM_VECTORS, DIM);
    
    let base = generate_vectors(NUM_VECTORS, DIM, 42);
    let queries = generate_vectors(1000, DIM, 123);
    
    for (name, strategy) in [
        ("Sequential", BuildStrategy::Sequential),
        ("Parallel", BuildStrategy::Parallel),
    ] {
        let config = HNSWConfig::default().with_build_strategy(strategy);
        
        let start = Instant::now();
        let index = HNSWIndex::build(base.clone(), config);
        let build_time = start.elapsed();
        
        let start = Instant::now();
        for q in &queries { let _ = index.search(q, K); }
        let search_time = start.elapsed();
        let qps = 1000.0 / search_time.as_secs_f64();
        
        let recall = measure_recall(&index, &queries, &base, K);
        
        println!("{:<12} Build: {:>6.2}s | Search: {:>5.0} QPS | Recall@{}: {:.1}%", 
            name, build_time.as_secs_f64(), qps, K, recall * 100.0);
    }
}
