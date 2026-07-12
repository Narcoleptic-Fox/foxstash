//! Recall/QPS Pareto curve — the only fair way to compare ANN implementations.
//!
//! Run: `cargo run --release -p foxstash-benches --example pareto`
//!
//! Comparing two ANN libraries at one fixed `ef` compares nothing: `ef` is a knob, and two
//! implementations reach a given recall at different settings of it. The comparison that
//! means something is **QPS at matched recall**.
//!
//! Two traps this exists to avoid, both of which caught this project:
//!   - hnswlib's `knn_query` defaults to `num_threads=-1` (every core). Timing that against
//!     a single-threaded Rust loop makes foxstash look ~11x slower than it is. Both sides
//!     here are single-threaded; `search_batch` is reported separately.
//!   - At `ef=500` on a 10k index the search touches ~85% of the graph — HNSW degenerating
//!     into brute force. That is a terrible operating point to tune against.
//!
//! hnswlib reference (single-threaded, M=32, ef_construction=200, k=10, same machine):
//!
//! |  ef | recall@10 |   QPS |
//! |-----|-----------|-------|
//! |  10 |    40.20% | 68483 |
//! |  20 |    57.29% | 41951 |
//! |  50 |    81.95% | 20051 |
//! | 100 |    94.34% | 10850 |
//! | 200 |    99.20% |  6458 |
//! | 500 |    99.98% |  3487 |

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use std::time::Instant;

const K: usize = 10;

/// hnswlib, single-threaded, same machine/config. (ef, recall@10, qps)
const HNSWLIB: &[(usize, f64, f64)] = &[
    (10, 40.20, 68483.0),
    (20, 57.29, 41951.0),
    (50, 81.95, 20051.0),
    (100, 94.34, 10850.0),
    (200, 99.20, 6458.0),
    (500, 99.98, 3487.0),
];

/// QPS hnswlib achieves at `recall`, linearly interpolated along its curve.
/// Returns None if the recall is outside the measured range.
fn hnswlib_qps_at(recall: f64) -> Option<f64> {
    for w in HNSWLIB.windows(2) {
        let (_, r0, q0) = w[0];
        let (_, r1, q1) = w[1];
        if recall >= r0 && recall <= r1 {
            let t = if (r1 - r0).abs() < 1e-9 {
                0.0
            } else {
                (recall - r0) / (r1 - r0)
            };
            return Some(q0 + t * (q1 - q0));
        }
    }
    None
}

fn main() {
    let ds = Dataset::load("benchmarks/data", "sift10k").expect("load sift10k");

    println!("SIFT10K — single-threaded, k={K}, M=32, ef_construction=200");
    println!("foxstash vs hnswlib at MATCHED RECALL (both single-threaded)\n");
    println!(
        "{:>5} {:>11} {:>9} {:>14} {:>10}",
        "ef", "recall@10", "QPS", "hnswlib@same", "ratio"
    );
    println!("{:-<56}", "");

    for &ef in &[10usize, 20, 50, 100, 200, 500] {
        let index = HNSWIndex::build_parallel(
            ds.base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 32,
                m0: 64,
                ef_construction: 200,
                ef_search: ef,
                build_strategy: BuildStrategy::Parallel,
                ..Default::default()
            },
        );

        let mut ctx = index.create_search_context();
        // warm
        for q in ds.queries.iter().take(50) {
            std::hint::black_box(index.search_with_context(q, K, &mut ctx).unwrap());
        }

        let recall = ds.recall_at(K, |q| {
            index
                .search(q, K)
                .unwrap()
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect()
        });

        let t = Instant::now();
        for q in &ds.queries {
            std::hint::black_box(index.search_with_context(q, K, &mut ctx).unwrap());
        }
        let qps = ds.queries.len() as f64 / t.elapsed().as_secs_f64();

        let r = recall as f64 * 100.0;
        match hnswlib_qps_at(r) {
            Some(h) => println!(
                "{:>5} {:>10.2}% {:>9.0} {:>14.0} {:>9.2}x",
                ef,
                r,
                qps,
                h,
                qps / h
            ),
            None => println!(
                "{:>5} {:>10.2}% {:>9.0} {:>14} {:>10}",
                ef, r, qps, "off-curve", "-"
            ),
        }
    }

    // Multi-threaded, at the ef=100 operating point, for comparison with hnswlib's
    // default knn_query (which fans out across every core).
    let index = HNSWIndex::build_parallel(
        ds.base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: 100,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );
    std::hint::black_box(index.search_batch(&ds.queries[..64].to_vec(), K).unwrap());
    let t = Instant::now();
    std::hint::black_box(index.search_batch(&ds.queries, K).unwrap());
    let batch_qps = ds.queries.len() as f64 / t.elapsed().as_secs_f64();
    println!("\nsearch_batch() [all cores, ef=100]: {:.0} QPS", batch_qps);

    println!("\nratio > 1.00x means foxstash serves more QPS than hnswlib at the same recall.");
}
