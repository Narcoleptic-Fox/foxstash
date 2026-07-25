//! Where does a Collection's resident memory actually go?
//!
//! The lifecycle harness reports 12–14× the corpus size resident at 128-dim and
//! ~6.6× at 768-dim. The redesign note originally asserted that removing the
//! duplicated vectors would "roughly halve" that. The arithmetic does not support
//! it — one vector copy is only 8–15% of the total — so this exists to find out
//! what the rest is **before** anyone refactors on the strength of a guess.
//!
//! ```sh
//! cargo run -p foxstash-db --release --example memory_attribution -- 20000 128
//! ```
//!
//! # Method, and what it can and cannot tell you
//!
//! Components are built up **cumulatively** and RSS is sampled after each. Nothing
//! is freed until the end, because freeing does not lower RSS — glibc keeps the
//! arena — so a build-up gives clean deltas while a build-and-drop would not.
//!
//! The final phase drops everything and re-samples. Whatever stays resident after
//! that is **allocator retention**, not live data, and is the difference between
//! "we use too much memory" and "we churn too much memory". Those want completely
//! different fixes, which is the main thing this is here to distinguish.
//!
//! RSS is Linux-only (`/proc/self/statm`) and is process-wide, so it includes the
//! binary, the allocator's own structures and any transient peak. Treat the
//! deltas as attribution, not as exact struct sizes.

use std::time::Instant;

use foxstash_core::index::{HNSWConfig, HNSWIndex};
use foxstash_core::Document;
use foxstash_db::inverted_index::InvertedIndex;
use foxstash_db::tokenizer::{SimpleTokenizer, Tokenizer};

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<f64>().ok())
        .map(|pages| pages * 4096.0 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

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

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);

    let corpus_mib = (n * dim * 4) as f64 / (1024.0 * 1024.0);
    println!("memory attribution — {n} docs x {dim} dim");
    println!("corpus vectors (f32): {corpus_mib:.1} MiB\n");
    println!(
        "{:<34} {:>10} {:>10} {:>12}",
        "component (cumulative)", "RSS_MiB", "delta", "x corpus"
    );

    let base = rss_mib();
    let mut prev = base;
    let mut row = |label: &str| {
        let now = rss_mib();
        println!(
            "{label:<34} {now:>10.1} {:>10.1} {:>12.2}",
            now - prev,
            (now - prev) / corpus_mib
        );
        prev = now;
    };

    // 1. The document store db keeps beside the index: ids, contents, metadata
    //    AND a full copy of every embedding.
    let documents: Vec<Document> = (0..n)
        .map(|i| Document {
            id: format!("doc-{i}"),
            content: format!("document number {i} about topic {}", i % 97),
            embedding: vector(i as u64, dim),
            metadata: None,
        })
        .collect();
    row("documents (ids+content+vectors)");

    // 2. The same vectors again, inside the index — plus the graph arena.
    //
    // Built from a freshly generated Vec that is MOVED in, not from
    // `documents.clone()`. A clone would be freed immediately but stay resident
    // (the allocator keeps the pages), inflating this delta by a whole document
    // store — an artifact that made the index look ~14 MiB larger than it is on a
    // first pass.
    let build_start = Instant::now();
    let index = HNSWIndex::build_parallel_from_documents(
        (0..n)
            .map(|i| Document {
                id: format!("doc-{i}"),
                content: format!("document number {i} about topic {}", i % 97),
                embedding: vector(i as u64, dim),
                metadata: None,
            })
            .collect(),
        HNSWConfig::default(),
    );
    let build_secs = build_start.elapsed().as_secs_f64();
    row("+ HNSWIndex (vectors + graph)");

    // 3. BM25 postings over the same content.
    let tokenizer = SimpleTokenizer::new();
    let mut text_index = InvertedIndex::default();
    for (pos, doc) in documents.iter().enumerate() {
        text_index.add(pos, &tokenizer.tokenize(&doc.content));
    }
    row("+ inverted index (BM25)");

    // 4. id -> position map.
    let id_map: std::collections::HashMap<String, usize> = documents
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.clone(), i))
        .collect();
    row("+ id map");

    let live_total = rss_mib() - base;
    println!(
        "\nlive total: {live_total:.1} MiB = {:.2}x corpus   (index built in {build_secs:.2}s, \
         {} nodes, {} postings, {} ids)",
        live_total / corpus_mib,
        index.len(),
        text_index.len(),
        id_map.len()
    );

    // 5. Drop everything. What remains is the allocator holding pages, not data.
    drop(index);
    drop(documents);
    drop(text_index);
    drop(id_map);
    let after_drop = rss_mib();
    println!("\nafter dropping everything: {after_drop:.1} MiB  (started at {base:.1})",);
    println!(
        "  retained by the allocator: {:.1} MiB — {:.0}% of what was live",
        after_drop - base,
        (after_drop - base) / live_total * 100.0
    );
    println!(
        "\nOne duplicated vector copy is {corpus_mib:.1} MiB = {:.0}% of the live total.",
        corpus_mib / live_total * 100.0
    );
    println!(
        "NOTE: the inverted-index and id-map rows read 0.0 because they fit in arena space the\n\
         allocator already held. Their cost is real but invisible to RSS deltas — this method\n\
         attributes GROWTH, not struct sizes."
    );
}
