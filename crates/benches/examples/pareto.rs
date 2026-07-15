//! Recall/QPS Pareto curve — the only fair way to compare ANN implementations.
//!
//! Run: `cargo run --release -p foxstash-benches --example pareto [dataset]`
//!   e.g. `... --example pareto sift1m`   (default: sift10k)
//!
//! Comparing two ANN libraries at one fixed `ef` compares nothing: `ef` is a knob, and two
//! implementations reach a given recall at different settings of it. The comparison that
//! means something is **QPS at matched recall**.
//!
//! Three traps this exists to avoid, all of which caught this project:
//!   - hnswlib's `knn_query` defaults to `num_threads=-1` (every core). Timing that against
//!     a single-threaded Rust loop makes foxstash look ~11x slower than it is. Both sides
//!     here are single-threaded; `search_batch` is reported separately.
//!   - At `ef=500` on a 10k index the search touches ~85% of the graph — HNSW degenerating
//!     into brute force. That is a terrible operating point to tune against.
//!   - The dataset may not be what its directory is named. `benchmarks/data/sift1m/` held a
//!     10,000-vector base. `Dataset::load` now verifies shape against a manifest, and this
//!     bench prints the exact-search control row: if that row is not ~100%, every other row
//!     here is void.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use std::time::Instant;

const K: usize = 10;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift10k".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    println!(
        "{} — {} base x {}d, {} queries — single-threaded, k={K}, M=32, ef_construction=200\n",
        ds.name,
        ds.base.len(),
        ds.dim(),
        ds.queries.len()
    );

    // The control row, first and unconditionally. If exact brute-force search does not
    // score ~100% against the shipped ground truth, the loader or the metric is wrong and
    // every number below is void — so we refuse to print them.
    const CTRL_N: usize = 200;
    let control = ds.exact_control_sampled(K, CTRL_N);
    let sep = ds.separation(K, CTRL_N);
    println!(
        "exact control (brute force, {CTRL_N} queries): {:.2}%  {}",
        control * 100.0,
        if control > 0.99 {
            "PASS"
        } else {
            "*** FAIL ***"
        }
    );
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — the harness is broken, not the index. \
         Refusing to emit a recall table.",
        control * 100.0
    );
    println!(
        "difficulty: d(100th)/d(10th) = {sep:.3}  \
         (near 1.0 = true neighbours buried in a near-tie shell; recall@10 is NOT \
         comparable across datasets with different separation)\n"
    );

    let t = Instant::now();
    // Build ONCE. `ef_search` is a search-time dial and does not touch graph structure,
    // so the whole curve below comes off this one graph.
    let mut index = HNSWIndex::build_parallel(
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
    println!("build: {:.1}s\n", t.elapsed().as_secs_f64());

    let mem = index.memory_breakdown();
    println!(
        "memory: {:.1} MB total ({:.1} MB vectors + {:.1} MB links)\n",
        mem.total() as f64 / 1e6,
        mem.embeddings as f64 / 1e6,
        (mem.layer0_links + mem.upper_layer_links) as f64 / 1e6,
    );

    println!("{:>6} {:>11} {:>10}", "ef", "recall@10", "QPS");
    println!("{:-<30}", "");

    for &ef in &[10usize, 20, 50, 100, 200, 500] {
        index.set_ef_search(ef);

        let mut searcher = index.searcher();
        for q in ds.queries.iter().take(50) {
            std::hint::black_box(searcher.search(q, K).unwrap());
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
        let mut searcher = index.searcher();
        for q in &ds.queries {
            std::hint::black_box(searcher.search(q, K).unwrap());
        }
        let qps = ds.queries.len() as f64 / t.elapsed().as_secs_f64();

        println!("{:>6} {:>10.2}% {:>10.0}", ef, recall * 100.0, qps);
    }

    // Multi-threaded, at ef=100, for comparison with hnswlib's default knn_query
    // (which fans out across every core).
    index.set_ef_search(100);
    std::hint::black_box(index.search_batch(&ds.queries[..64], K).unwrap());
    let t = Instant::now();
    std::hint::black_box(index.search_batch(&ds.queries, K).unwrap());
    println!(
        "\nsearch_batch() [all cores, ef=100]: {:.0} QPS",
        ds.queries.len() as f64 / t.elapsed().as_secs_f64()
    );
}
