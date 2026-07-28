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

/// Sweep OUR `M` too, and report our own best-of-M frontier.
///
/// `competitors.py` carries this comment, and it is right:
///
/// > "Sweep M on EVERY library, not just ours. Tuning our own M while pinning theirs at a default
/// > would manufacture a win — which is precisely the class of error that produced this repo's
/// > earlier false headlines."
///
/// And then this example did the exact thing that comment warns against, in reverse: it swept
/// hnswlib's and faiss's M over {16, 32} and PINNED OURS at 32. Every published foxstash number
/// was measured at a single degree, chosen by nobody, while the competition was shown at its best.
///
/// That handicap was real and it was large. Once `config.m0` actually reached the parallel builder
/// (it was hardcoded to 64 — see `BuildStrategy`), m=16/m0=32 turned out to DOMINATE m=32/m0=64 on
/// GIST: 2.5x faster to build (67s vs 167s at 200k) AND ~1.3x faster to query at matched recall.
/// Our default degree was simply twice what it should have been.
///
/// Fairness cuts both ways. A harness rigged against yourself is still a rigged harness.
const MS: &[(usize, usize)] = &[(16, 32), (32, 64)];

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
        "{} — {} base x {}d, {} queries, k={K}, ef_c=200\nexact control: {:.2}%  PASS\n\
         Sweeping M over {MS:?} — our own best-of-M frontier, the same courtesy competitors.py\n\
         extends to hnswlib and faiss. Read ACROSS M at matched recall.\n",
        ds.name,
        ds.base.len(),
        dim,
        ds.queries.len(),
        control * 100.0
    );

    // The block arithmetic this whole comparison turns on, for THIS dataset's dim.
    let hdr = 272usize;
    let blk = |vec_bytes: usize| hdr + vec_bytes;
    // TurboRabit and Warren share a byte-identical arena block — Warren's walk *is*
    // TurboRabit's walk — so they get one column. Warren's extra 8-bit residual lives
    // outside the block, which is why it costs nothing per node visit.
    //
    // Mirrors `node_block_words`' arms exactly rather than restating them in prettier
    // arithmetic, so this table cannot drift away from the layout it claims to
    // describe. Note the nibble pack is a fixed 4 bits wide: the block is the same
    // size at `rabit_bits` 1 through 4, only the code's precision changes.
    let nibble_words = dim.div_ceil(2).div_ceil(4);
    println!(
        "node block at {dim}d:  F32 {} B   SQ8 {} B   RaBitQ {} B   TurboRabit/Warren {} B",
        blk(dim * 4),
        blk(dim.div_ceil(4) * 4),
        blk((2 + dim.div_ceil(32)) * 4),
        blk((2 + nibble_words) * 4)
    );
    println!("{:-<78}", "");

    // Every non-deprecated `Storage` variant appears here.
    //
    // It used to be F32, SQ8 and RaBitQ only, while the enum had six variants. So the
    // two modes the quantizer findings actually recommend — `TurboRabit` at 4 bits, and
    // `Warren` — were absent from the harness that decides which mode wins, and the
    // published Pareto frontier was drawn without them. That is this repo's own lesson
    // about untested public options, in the instrument rather than the library: a
    // variant nothing exercises is a variant nobody has checked.
    //
    // ⚠️ The `Warren` rows are DEAD on this dataset, and the reason is a real defect.
    //
    // Warren's rerank ends in `(2.0 - 2.0 * acc).max(0.0)`, which is `‖q-x‖²` only when
    // both sides are unit-norm. SIFT vectors have ‖x‖ ≈ 500, so `acc` is a large inner
    // product, the expression is negative for every candidate, and `.max(0.0)` flattens
    // the whole pool to distance 0 — every candidate ties and the re-sort carries no
    // information. Recall at M=32/m0=64, recall@10:
    //
    //              rerank=0   rerank=400   rerank=2000
    //   sift10k      85.5%       85.5%        85.5%     <- identical; stage is inert
    //   rustdocs     97.1%       99.99%       99.99%    <- +2.9 points, F32 parity
    //
    // It is NOT a property of 128 dimensions — synthetic *signed* data at 128-d gives a
    // 0.59 gap, and unit-normalizing SIFT (or switching it to Cosine, which normalizes
    // via `rabitq_cosine_input`) restores it. It is a property of the norm.
    //
    // The walk does not share the bug: it computes `dtc² + ‖q-c‖² + f_rescale·(…)`, which
    // is norm-aware. Only the rerank took the unit-norm shortcut, so the two disagree
    // about what distance means on the same index. Pinned by the ignored test
    // `warren_rerank_works_under_l2_on_unnormalized_vectors` in core.
    //
    // So read the Warren rows here as a lower bound on any non-unit-norm dataset. On
    // rustdocs45k, where the vectors *are* unit-norm, they are real: 99.99% recall in
    // 37 MB against F32's 158 MB.
    //
    // rerank=400 and rerank=2000 agree because the pool is `min(rerank_candidates,
    // found.len())` and `found.len() <= ef <= 500`. Both already cover it.
    //
    // Worth stating because none of this was visible until now: Warren and TurboRabit
    // were absent from this harness entirely, so the mode with the best memory/recall
    // trade-off in the enum had never appeared on the frontier it belongs on — and the
    // defect above sat behind a `recall > 0.8` unit test that passed either way.
    for (label, storage, rerank) in [
        ("F32 (control)", Storage::F32, 0),
        ("SQ8 + rerank", Storage::SQ8, 100),
        ("RaBitQ + rerank", Storage::RaBitQ, 400),
        ("TurboRabit b=4 + rerank", Storage::TurboRabit, 400),
        ("Warren (8-bit residual rerank)", Storage::Warren, 400),
    ] {
        for &(m, m0) in MS {
            let t = Instant::now();
            let mut index = HNSWIndex::build_parallel(
                ds.base.clone(),
                HNSWConfig {
                    metric: DistanceMetric::L2,
                    m,
                    m0,
                    ef_construction: 200,
                    storage,
                    rerank_candidates: rerank,
                    build_strategy: BuildStrategy::Parallel,
                    // Only read under TurboRabit/Warren. 4 is what the high-recall RAG
                    // preset uses; the default of 3 would show those modes below their
                    // recommended setting while F32 and SQ8 are shown at theirs.
                    rabit_bits: 4,
                    ..Default::default()
                },
            );
            let build = t.elapsed();
            let mem = index.memory_breakdown().total() as f64 / 1e6;

            println!(
                "\n=== {label}  M={m}/m0={m0} ===  build {:.0}s, {mem:.0} MB",
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
    }

    println!(
        "\nRead this at MATCHED RECALL, not matched ef. And read `dist/query` next to `ns/dist`:\n\
         they separate the two ways a mode can lose — doing more work (a coarse metric misleads\n\
         the walk) from doing the same work more slowly (a worse kernel). QPS alone cannot."
    );
}
