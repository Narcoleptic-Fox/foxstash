//! Which storage mode wins, and does the answer depend on the dimension?
//!
//! Run: `cargo run --release -p foxstash-benches --example storage_pareto [dataset]`
//!   e.g. `... --example storage_pareto sift1m`   (128-d)
//!        `... --example storage_pareto gist1m`   (960-d)
//!
//! Search is **memory-latency bound**: a distance computation costs 55-98 ns, essentially one
//! DRAM round-trip. So the lever is bytes moved per node visit, and a node block is
//! `header(m0) + vector`. At m0=64 the header is 272 B regardless of `dim` — which means
//! which half of the block dominates is a pure function of the dimension:
//!
//! ```text
//!  dim    F32 block   SQ8 block   RaBitQ block    vector's share of the SQ8 block
//!  128       784 B       400 B        296 B          32%
//!  384     1,808 B       656 B        320 B          59%
//!  960     4,112 B     1,232 B        392 B          78%
//! 1536     6,416 B     1,808 B        472 B          85%
//! ```
//!
//! On SIFT (128-d) SQ8 wins decisively and RaBitQ loses ~12x: 1-bit codes fight for 104 B out
//! of a 400 B block and wreck the metric to get them, so the walk needs ~10x more hops. The
//! open question is whether that inverts at the dimensionality anyone actually uses — MiniLM
//! is 384-d, OpenAI's embeddings are 1536-d, and *nobody runs RAG on 128-d vectors*. At 960-d
//! RaBitQ's block is 3.1x smaller than SQ8's rather than 1.35x.
//!
//! Concluding "1-bit quantization doesn't work" from SIFT alone would be a
//! dataset-generalization error — the same class of mistake that let a 1.2%-recall quantizer
//! ship for a whole release behind synthetic vectors. This example exists to settle it on
//! evidence.
//!
//! Compare at **matched recall**, never at matched ef. Run on an IDLE machine.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::time::Instant;

const K: usize = 10;
const EFS: &[usize] = &[20, 50, 100, 200, 500];

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift1m".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    // An exact brute-force control. If this does not score ~100%, the harness is broken and
    // every number below is void — print nothing rather than something plausible and wrong.
    let control = ds.exact_control_sampled(K, 200);
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — harness broken, everything below would be fiction",
        control * 100.0
    );

    let dim = ds.dim();
    println!(
        "{} — {} base x {}d, {} queries, k={K}, M=32/m0=64, ef_c=200\nexact control: {:.2}%  PASS\n",
        ds.name,
        ds.base.len(),
        dim,
        ds.queries.len(),
        control * 100.0
    );

    // The block arithmetic this whole comparison turns on, for THIS dataset's dim.
    let hdr = 272usize;
    let blk = |vec_bytes: usize| hdr + vec_bytes;
    println!(
        "node block at {dim}d:  F32 {} B   SQ8 {} B   RaBitQ {} B",
        blk(dim * 4),
        blk(dim),
        blk(dim.div_ceil(8) + 8)
    );
    println!("{:-<78}", "");

    for (label, storage, rerank) in [
        ("F32 (control)", Storage::F32, 0),
        ("SQ8 + rerank", Storage::SQ8, 100),
        ("RaBitQ + rerank", Storage::RaBitQ, 400),
    ] {
        let t = Instant::now();
        let mut index = HNSWIndex::build_parallel(
            ds.base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 32,
                m0: 64,
                ef_construction: 200,
                storage,
                rerank_candidates: rerank,
                build_strategy: BuildStrategy::Parallel,
                ..Default::default()
            },
        );
        let build = t.elapsed();
        let mem = index.memory_breakdown().total() as f64 / 1e6;

        println!(
            "\n=== {label} ===  build {:.0}s, {mem:.0} MB",
            build.as_secs_f64()
        );
        println!(
            "{:>6} {:>11} {:>10} {:>12} {:>9}",
            "ef", "recall@10", "QPS", "dist/query", "ns/dist"
        );

        for &ef in EFS {
            index.set_ef_search(ef);

            let recall = ds.recall_at(K, |q| {
                index
                    .search(q, K)
                    .unwrap()
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect()
            });

            let mut s = index.searcher();
            for q in ds.queries.iter().take(50) {
                std::hint::black_box(s.search(q, K).unwrap());
            }

            let mut s = index.searcher();
            let t = Instant::now();
            for q in &ds.queries {
                std::hint::black_box(s.search(q, K).unwrap());
            }
            let el = t.elapsed();

            let n = ds.queries.len() as f64;
            let d = s.distance_calls() as f64;
            println!(
                "{:>6} {:>10.2}% {:>10.0} {:>12.0} {:>9.1}",
                ef,
                recall * 100.0,
                n / el.as_secs_f64(),
                d / n,
                el.as_nanos() as f64 / d
            );
        }
    }

    println!(
        "\nRead this at MATCHED RECALL, not matched ef. And read `dist/query` next to `ns/dist`:\n\
         they separate the two ways a mode can lose — doing more work (a coarse metric misleads\n\
         the walk) from doing the same work more slowly (a worse kernel). QPS alone cannot."
    );
}
