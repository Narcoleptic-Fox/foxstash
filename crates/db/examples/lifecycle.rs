//! Scale harness for `Collection` — ingest, restart, search, delete, compact.
//!
//! # Why this exists
//!
//! `Collection` had **no consumer outside its own unit tests**. The only crate depending on
//! `foxstash-db` is `foxstash-wasm`, and it imports just the pure-algorithm modules (`hybrid`,
//! `inverted_index`, `tokenizer`) — never the database. Nothing had ever opened a collection at
//! scale, measured its memory, or restarted it, which is how several structural problems coexisted
//! in a crate with green CI.
//!
//! This is the instrument. It does not assert; it **reports**, so a change to `db` can be shown to
//! help or hurt instead of argued about.
//!
//! ```sh
//! cargo run -p foxstash-db --release --example lifecycle              # 10k x 128
//! cargo run -p foxstash-db --release --example lifecycle -- 100000 768
//! ```
//!
//! # What each number is for
//!
//! | phase | the finding it measures |
//! |---|---|
//! | ingest | baseline write path |
//! | **resident after ingest** | vectors held twice — `HNSWIndex.full` *and* `documents[].embedding` |
//! | **reopen** | the graph is never persisted; every open re-inserts every document one at a time |
//! | search | read path, and the recall cost of any future quantization |
//! | delete + compact | compaction is a stop-the-world full rebuild |
//!
//! Resident-memory figures are Linux-only (`/proc/self/statm`); elsewhere they print `n/a`.
//! Timings are single-run wall clock, not a criterion distribution — this measures *lifecycle*
//! costs that are too slow to iterate, so treat them as indicative and compare like with like on an
//! idle machine.

use std::path::Path;
use std::time::Instant;

use foxstash_db::{Collection, DbConfig};

/// Resident set size in MiB, or `None` off Linux.
fn rss_mib() -> Option<f64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: f64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    let page = 4096.0; // getpagesize(); 4 KiB on every platform this runs on
    Some(resident_pages * page / (1024.0 * 1024.0))
}

fn fmt_rss(v: Option<f64>) -> String {
    v.map_or_else(|| "n/a".to_string(), |m| format!("{m:8.1}"))
}

/// Deterministic pseudo-random vectors — no `rand` dependency, and reproducible across runs so two
/// measurements are comparable.
fn vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

fn dir_bytes(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            match e.metadata() {
                Ok(m) if m.is_dir() => total += dir_bytes(&e.path()),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let docs: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10_000);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);

    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("bench_collection");
    std::fs::create_dir_all(&path)?;

    // FOXSTASH_AUTO_CHECKPOINT=0 isolates how much of ingest is the periodic
    // whole-collection re-serialization vs the HNSW insert itself.
    let auto_checkpoint = std::env::var("FOXSTASH_AUTO_CHECKPOINT").as_deref() != Ok("0");
    let config = DbConfig::default()
        .with_embedding_dim(dim)
        .with_auto_checkpoint(auto_checkpoint);

    println!("foxstash-db lifecycle harness — {docs} docs x {dim} dim (auto_checkpoint={auto_checkpoint})");
    println!(
        "corpus vectors alone: {:.1} MiB (f32)",
        (docs * dim * 4) as f64 / (1024.0 * 1024.0)
    );
    println!();
    println!(
        "{:<22} {:>10} {:>12} {:>10}",
        "phase", "wall_s", "resident_MiB", "on_disk_MiB"
    );

    let base_rss = rss_mib();
    let report = |phase: &str, secs: f64, disk: u64| {
        println!(
            "{phase:<22} {secs:>10.3} {:>12} {:>10.1}",
            fmt_rss(rss_mib()),
            disk as f64 / (1024.0 * 1024.0)
        );
    };

    // ---- core baseline: raw sequential HNSW insert, no db at all ----------
    // Attribution check. If this is close to db's ingest, the cost is core's
    // sequential `add`, not db's WAL/text-index/document bookkeeping — which
    // decides whether the fix belongs in db or in how db drives core.
    {
        use foxstash_core::index::{HNSWConfig, HNSWIndex};
        let mut index = HNSWIndex::new(dim, HNSWConfig::default());
        let start = Instant::now();
        for i in 0..docs {
            index
                .add(foxstash_core::Document {
                    id: format!("doc-{i}"),
                    content: String::new(),
                    embedding: vector(i as u64, dim),
                    metadata: None,
                })
                .expect("core add");
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:<22} {secs:>10.3} {:>12} {:>10}   <- core HNSWIndex::add only",
            "core baseline",
            fmt_rss(rss_mib()),
            "-"
        );

        // Same corpus, same config, through the parallel builder. Quantifies how
        // much of sequential `add`'s cost is the lack of parallelism versus the
        // per-insert algorithm itself.
        let embeddings: Vec<Vec<f32>> = (0..docs).map(|i| vector(i as u64, dim)).collect();
        let start = Instant::now();
        let built = HNSWIndex::build_parallel(embeddings, HNSWConfig::default());
        let par = start.elapsed().as_secs_f64();
        println!(
            "{:<22} {par:>10.3} {:>12} {:>10}   <- build_parallel, {} nodes, {:.1}x vs add",
            "core build_parallel",
            fmt_rss(rss_mib()),
            "-",
            built.len(),
            secs / par
        );
    }

    // ---- ingest -----------------------------------------------------------
    let start = Instant::now();
    {
        let collection = Collection::create("bench", &path, config.clone())?;
        for i in 0..docs {
            collection.insert(
                format!("doc-{i}"),
                format!("document number {i} about topic {}", i % 97),
                vector(i as u64, dim),
                None,
            )?;
        }
        collection.flush()?;
    }
    report(
        "ingest + flush",
        start.elapsed().as_secs_f64(),
        dir_bytes(&path),
    );

    // ---- bulk ingest into a fresh collection -------------------------------
    // Same corpus through insert_many, which builds the graph in parallel.
    {
        let bulk_dir = tmp.path().join("bulk_collection");
        std::fs::create_dir_all(&bulk_dir)?;
        let docs: Vec<foxstash_core::Document> = (0..docs)
            .map(|i| foxstash_core::Document {
                id: format!("doc-{i}"),
                content: format!("document number {i} about topic {}", i % 97),
                embedding: vector(i as u64, dim),
                metadata: None,
            })
            .collect();
        let start = Instant::now();
        let c = Collection::create("bulk", &bulk_dir, config.clone())?;
        c.insert_many(docs)?;
        c.flush()?;
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:<22} {secs:>10.3} {:>12} {:>10.1}",
            "bulk insert_many",
            fmt_rss(rss_mib()),
            dir_bytes(&bulk_dir) as f64 / (1024.0 * 1024.0)
        );
    }

    // ---- reopen: the O(N) sequential rebuild -------------------------------
    let start = Instant::now();
    let collection = Collection::open("bench", &path, config.clone())?;
    let reopen = start.elapsed().as_secs_f64();
    report("reopen", reopen, dir_bytes(&path));
    println!(
        "{:<22} {:>10} {:>12.1} us/doc rebuilt",
        "",
        "",
        reopen * 1e6 / docs as f64
    );

    // ---- search -----------------------------------------------------------
    let queries = 200.min(docs);
    let start = Instant::now();
    let mut found_self = 0usize;
    for i in 0..queries {
        let hits = collection.search(&vector(i as u64, dim), 10, None)?;
        if hits.first().is_some_and(|h| h.id == format!("doc-{i}")) {
            found_self += 1;
        }
    }
    let search = start.elapsed().as_secs_f64();
    report("search (200 queries)", search, dir_bytes(&path));
    println!(
        "{:<22} {:>10.3} ms/query   self-retrieval {}/{}",
        "",
        search * 1e3 / queries as f64,
        found_self,
        queries
    );

    // ---- delete + compact: the stop-the-world rebuild ----------------------
    let to_delete = docs / 10;
    let start = Instant::now();
    for i in 0..to_delete {
        collection.delete(&format!("doc-{i}"))?;
    }
    report(
        "delete 10%",
        start.elapsed().as_secs_f64(),
        dir_bytes(&path),
    );

    let start = Instant::now();
    collection.compact()?;
    report("compact", start.elapsed().as_secs_f64(), dir_bytes(&path));

    println!();
    println!("live documents: {}", collection.len());
    if let (Some(b), Some(n)) = (base_rss, rss_mib()) {
        let vectors_mib = (docs * dim * 4) as f64 / (1024.0 * 1024.0);
        println!(
            "resident growth: {:.1} MiB for a {:.1} MiB corpus ({:.2}x)",
            n - b,
            vectors_mib,
            (n - b) / vectors_mib
        );
        println!(
            "  (>2x suggests vectors are held in both HNSWIndex.full and documents[].embedding)"
        );
    }
    Ok(())
}
