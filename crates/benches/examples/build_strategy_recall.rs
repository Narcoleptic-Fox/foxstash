//! Does `BuildStrategy::Parallel` actually lose recall at scale?
//!
//! ANSWER (SIFT1M, idle machine): NO. Parallel builds 5.2x faster (165.8s vs 865.3s) and gives up
//! 0.02-0.31 recall points. `BuildStrategy::Auto` — which routed everything >= 50k vectors to
//! Sequential on the premise that Parallel lost recall at scale — has been DELETED as a result,
//! and `Parallel` is now the default. This example is what settled it; keep it runnable.
//!
//! Run: `cargo run --release -p foxstash-benches --example build_strategy_recall [dataset] [n]`
//!   e.g. `... --example build_strategy_recall sift1m 200000`
//!
//! WHAT THIS SETTLED. `BuildStrategy` used to document Parallel as "good recall at small scale
//! (<50k) / may have lower recall at larger scales (needs more work)", `Auto` routed anything
//! >= 50k to Sequential on that basis, and Sequential was the `#[default]`.
//!
//! The caveat was load-bearing and unsupported. Two things contradicted it before a single number
//! was collected:
//!
//! 1. `storage_pareto` — the example that generates the flagship SIFT1M/GIST1M figures quoted in
//!    the README — builds with `BuildStrategy::Parallel` at **1,000,000** vectors, 20x past the
//!    stated ceiling, and measures 99.5% (SIFT1M) and 98.3% (GIST1M) recall@10. The refutation
//!    was in our own README.
//! 2. The parallel builder *did* have a real recall bug once. It was fixed. The warning outlived
//!    the defect, and nothing in the tree said what "needs more work" still referred to.
//!
//! A stale warning is not free. That one quietly routed every production-sized index onto a
//! builder taking 5x longer, to dodge a bug that no longer existed.
//!
//! NOTE the trap this example is designed to avoid: **never do this on uniform-random vectors.**
//! They have no cluster structure, every ANN scores ~60% on them, and that is precisely how the
//! original parallel-build recall bug hid for a whole release. Use a real corpus with real
//! neighbourhood structure and the dataset's own exact ground truth.
//!
//! Run on an IDLE machine.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use std::time::Instant;

const K: usize = 10;
const EFS: &[usize] = &[50, 100, 200];

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift1m".into());
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));
    let n = n.min(ds.base.len());
    let base: Vec<Vec<f32>> = ds.base[..n].to_vec();

    // The exact control. If this is not ~100%, ground truth does not match the base slice and
    // every recall number below is fiction. (Truncating the base to `n` keeps the shipped GT valid
    // only because the GT indexes the first `n` vectors' neighbours out of the FULL corpus — so
    // for n < len we must recompute. Guard against quoting a wrong number by refusing to run.)
    assert!(
        n == ds.base.len(),
        "n={n} is a prefix of a {}-vector corpus; the shipped ground truth indexes the FULL \
         corpus and is void for a prefix. Either run with the full corpus or recompute GT.",
        ds.base.len()
    );
    let control = ds.exact_control_sampled(K, 200);
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — harness broken",
        control * 100.0
    );

    println!(
        "{} — {n} base x {}d, {} queries, k={K}, M=32/m0=64, ef_c=200\nexact control: {:.2}%  PASS\n",
        ds.name,
        ds.dim(),
        ds.queries.len(),
        control * 100.0
    );
    println!(
        "The removed `Auto` variant sent n >= 50k to Sequential, on the premise that Parallel\n\
         loses recall at scale. n = {n} is {}x that threshold. If the columns match, the premise\n\
         was false — which is what the 2026-07 run found, and why `Auto` is gone.\n",
        n / 50_000
    );

    let mut builds = Vec::new();
    let mut indexes = Vec::new();

    for strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
        let t = Instant::now();
        // `build` — not `build_parallel` — so the dispatch on `config.build_strategy` is the
        // thing under test, exactly as a caller would hit it.
        let index = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 32,
                m0: 64,
                ef_construction: 200,
                build_strategy: strategy,
                ..Default::default()
            },
        );
        builds.push(t.elapsed().as_secs_f64());
        indexes.push(index);
    }

    println!("build:  Sequential {:.1}s   Parallel {:.1}s   ({:.2}x faster)\n",
        builds[0], builds[1], builds[0] / builds[1]);

    println!("{:>6} {:>14} {:>14} {:>10}", "ef", "Sequential", "Parallel", "delta");
    println!("{:-<48}", "");

    for &ef in EFS {
        let mut r = Vec::new();
        for index in indexes.iter_mut() {
            index.set_ef_search(ef);
            r.push(ds.recall_at(K, |q| {
                index
                    .search(q, K)
                    .unwrap()
                    .into_iter()
                    .filter_map(|x| x.id.parse::<usize>().ok())
                    .collect()
            }) * 100.0);
        }
        println!(
            "{:>6} {:>13.2}% {:>13.2}% {:>+9.2}",
            ef,
            r[0],
            r[1],
            r[1] - r[0]
        );
    }

    println!(
        "\nA `delta` at or above roughly -0.5 points means Parallel costs no meaningful recall at\n\
         this scale. That is what the 2026-07 SIFT1M run showed (-0.02 to -0.31, at 5.2x the build\n\
         speed), so `Auto` was deleted and `Parallel` became the default. Re-run this if you ever\n\
         touch the parallel builder: it is the guard on that decision."
    );
}
