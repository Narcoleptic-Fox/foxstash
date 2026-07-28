use super::*;

fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
    Document {
        id: id.to_string(),
        content: format!("Content for {}", id),
        embedding,
        metadata: None,
    }
}

fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

/// Every document must retrieve itself, **with locality reordering on**.
///
/// This is the whole point of the test. `build_parallel` shuffles insertion
/// order and `reorder_for_locality` (the default) permutes nodes afterwards,
/// so a positional attachment mismatches ids to vectors — but only when
/// reordering runs. A test with `reorder_for_locality: false` would pass
/// while the bug was live in every default build.

#[test]
fn diagnostic_build_parallel_id_vector_correspondence() {
    let n = 300;
    let dim = 16;
    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 10.0 + i as f32;
            v
        })
        .collect();
    for reorder in [true, false] {
        let config = HNSWConfig {
            reorder_for_locality: reorder,
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(embeddings.clone(), config);
        let mut wrong = 0;
        for (i, e) in embeddings.iter().enumerate() {
            let hits = index.search(e, 1).unwrap();
            if hits[0].id != i.to_string() {
                wrong += 1;
            }
        }
        println!("DIAG reorder={reorder}: {wrong}/{n} vectors retrieve the wrong id");
    }
}

#[test]
fn build_parallel_from_documents_keeps_ids_with_their_vectors_under_reordering() {
    let n = 300;
    let dim = 16;
    // Dense pseudo-random vectors, deterministic so failures reproduce.
    //
    // NOT one-hot-by-`i % dim`: the default metric is Cosine, which is
    // magnitude-invariant, so `v[i % dim] = 10 + i` makes every vector sharing
    // an axis IDENTICAL in direction. A first pass used exactly that and
    // reported 284/300 "mismatches" that were really ties — the fixture was
    // degenerate under the metric, not the code wrong.
    let vector = |seed: usize| -> Vec<f32> {
        let mut state = (seed as u64).wrapping_mul(6_364_136_223_846_793_005) + 1;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u32 << 31) as f32) - 0.5
            })
            .collect()
    };
    let docs: Vec<Document> = (0..n)
        .map(|i| {
            let embedding = vector(i);
            Document {
                id: format!("doc-{i}"),
                content: format!("content of {i}"),
                embedding,
                metadata: Some(serde_json::json!({ "n": i })),
            }
        })
        .collect();

    for reorder in [true, false] {
        let config = HNSWConfig {
            reorder_for_locality: reorder,
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel_from_documents(docs.clone(), config);
        assert_eq!(index.len(), n, "reorder={reorder}: node count");

        // Ids are the documents' own, not the synthetic "0".."n".
        let ids: std::collections::HashSet<&str> = index.ids.iter().map(|s| s.as_str()).collect();
        assert!(
            ids.contains("doc-0") && ids.contains("doc-299"),
            "reorder={reorder}: real ids should replace the synthetic ones"
        );

        for (i, doc) in docs.iter().enumerate() {
            let hits = index.search(&doc.embedding, 1).unwrap();
            let hit = hits.first().expect("a hit");
            assert_eq!(
                hit.id, doc.id,
                "reorder={reorder}: doc-{i}'s own vector must retrieve doc-{i}, \
                     got {} — id and vector are mismatched",
                hit.id
            );
            assert_eq!(
                hit.content, doc.content,
                "reorder={reorder}: content must travel with the id"
            );
        }
    }
}

#[test]
fn build_parallel_from_documents_handles_an_empty_corpus() {
    let index = HNSWIndex::build_parallel_from_documents(Vec::new(), HNSWConfig::default());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_hnsw_config_default() {
    let config = HNSWConfig::default();
    assert_eq!(config.m, 32); // Changed from 16 to match instant-distance
    assert_eq!(config.m0, 64); // m * 2
    assert_eq!(config.ef_construction, 100);
    assert_eq!(config.ef_search, 100);
    assert!((config.ml - (1.0 / 32_f32.ln())).abs() < 0.01);
    assert!(config.use_heuristic); // Heuristic enabled by default
    assert!(!config.extend_candidates);
    assert!(config.keep_pruned_connections);
}

#[test]
fn test_hnsw_config_builders() {
    let config = HNSWConfig::default()
        .with_m(32)
        .with_ef_search(100)
        .with_ef_construction(400)
        .with_simple_selection()
        .with_extended_candidates();

    assert_eq!(config.m, 32);
    assert_eq!(config.m0, 64);
    assert_eq!(config.ef_search, 100);
    assert_eq!(config.ef_construction, 400);
    assert!(!config.use_heuristic);
    assert!(config.extend_candidates);
}

#[test]
fn test_hnsw_new() {
    let index = HNSWIndex::with_defaults(128);
    assert_eq!(index.embedding_dim, 128);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

/// Filtered search must return the true top-k *within the allowed set* — not the unfiltered
/// top-k with excluded nodes dropped afterward (which would under-fill k), and never an excluded
/// node. Asserts the user-visible OUTPUT against brute force, with `ef_search >= n` so the walk
/// is exhaustive and the comparison is exact (no graph-miss slack to hide a logic bug).
#[test]
fn filtered_search_matches_bruteforce_over_allowed() {
    let n = 300usize;
    let dim = 16usize;
    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_random_vector(dim, i as u64))
        .collect();
    let config = HNSWConfig {
        metric: DistanceMetric::Cosine,
        ef_search: n, // exhaustive walk → exact
        ..HNSWConfig::default()
    };
    let index = HNSWIndex::build(embeddings.clone(), config);

    // Allow even-id nodes only. `build` assigns ids "0".."n-1".
    let allow = |i: usize| i % 2 == 0;
    let mask = index.filter_mask(|id, _content, _meta| allow(id.parse::<usize>().unwrap()));
    assert_eq!(mask.allowed_count(), n / 2);

    let k = 10usize;
    for qseed in [1000u64, 2000, 3000] {
        let q = generate_random_vector(dim, qseed);
        let got = index.search_filtered(&q, k, &mask).unwrap();

        // Never an excluded node; exactly k results (allowed_count >> k here).
        assert_eq!(got.len(), k);
        for r in &got {
            assert!(
                allow(r.id.parse::<usize>().unwrap()),
                "returned excluded id {}",
                r.id
            );
        }

        // Brute-force true top-k over the ALLOWED set (cosine == -distance ordering).
        let qn = crate::vector::simd::norm_simd(&q);
        let mut scored: Vec<(f32, usize)> = (0..n)
            .filter(|&i| allow(i))
            .map(|i| {
                let e = &embeddings[i];
                let en = crate::vector::simd::norm_simd(e);
                let dot: f32 = q.iter().zip(e).map(|(a, b)| a * b).sum();
                (dot / (qn * en), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let want: std::collections::HashSet<usize> =
            scored.iter().take(k).map(|&(_, i)| i).collect();
        let got_ids: std::collections::HashSet<usize> =
            got.iter().map(|r| r.id.parse::<usize>().unwrap()).collect();
        assert_eq!(
            got_ids, want,
            "filtered top-{k} != brute force over allowed (seed {qseed})"
        );
    }
}

/// A filter more selective than `k` yields exactly the allowed nodes ("up to k"), all of them,
/// and `filter_mask_ids` selects by external id.
#[test]
fn filtered_search_fewer_than_k_and_by_id() {
    let n = 100usize;
    let dim = 8usize;
    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_random_vector(dim, i as u64))
        .collect();
    let config = HNSWConfig {
        ef_search: n,
        ..HNSWConfig::default()
    };
    let index = HNSWIndex::build(embeddings, config);

    let allowed: std::collections::HashSet<String> =
        ["7", "42", "99"].iter().map(|s| s.to_string()).collect();
    let mask = index.filter_mask_ids(&allowed);
    assert_eq!(mask.allowed_count(), 3);

    let q = generate_random_vector(dim, 555);
    let got = index.search_filtered(&q, 10, &mask).unwrap();
    // Only 3 allowed → at most 3 back, and exactly the allowed ids.
    assert_eq!(got.len(), 3);
    let got_ids: std::collections::HashSet<String> = got.into_iter().map(|r| r.id).collect();
    assert_eq!(got_ids, allowed);
}

/// Unfiltered `search` must be byte-for-byte unchanged by the filter plumbing: an all-allowed
/// mask returns the same ids as a plain search. Guards the "None ⇒ zero-cost" claim's correctness.
#[test]
fn all_allowed_mask_equals_unfiltered() {
    let n = 200usize;
    let dim = 12usize;
    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_random_vector(dim, i as u64 + 9))
        .collect();
    let config = HNSWConfig {
        metric: DistanceMetric::Cosine,
        ef_search: 120,
        ..HNSWConfig::default()
    };
    let index = HNSWIndex::build(embeddings, config);
    let mask = index.filter_mask(|_, _, _| true);
    assert_eq!(mask.allowed_count(), n);

    let q = generate_random_vector(dim, 77);
    let plain: Vec<String> = index
        .search(&q, 10)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    let filtered: Vec<String> = index
        .search_filtered(&q, 10, &mask)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(plain, filtered);
}

/// `search_filtered_by` (predicate walk) must return the same results as `search_filtered` (mask)
/// for an equivalent filter — the two are one gating mechanism — and both must equal the
/// brute-force top-k over the allowed set. Guards the unification: a one-off predicate filter and
/// a prebuilt mask cannot diverge.
#[test]
fn search_filtered_by_matches_mask_and_bruteforce() {
    let n = 300usize;
    let dim = 16usize;
    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_random_vector(dim, i as u64))
        .collect();
    let config = HNSWConfig {
        metric: DistanceMetric::Cosine,
        ef_search: n, // exhaustive → exact
        ..HNSWConfig::default()
    };
    let index = HNSWIndex::build(embeddings.clone(), config);

    // Same filter expressed two ways: a prebuilt mask, and a live predicate over the id.
    let allow = |i: usize| i % 3 == 0;
    let mask = index.filter_mask(|id, _c, _m| allow(id.parse::<usize>().unwrap()));

    let k = 10usize;
    for qseed in [11u64, 22, 33] {
        let q = generate_random_vector(dim, qseed);
        let via_mask: Vec<String> = index
            .search_filtered(&q, k, &mask)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        let via_pred: Vec<String> = index
            .search_filtered_by(&q, k, |id, _meta| allow(id.parse::<usize>().unwrap()))
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(
            via_pred, via_mask,
            "predicate walk != mask walk (seed {qseed})"
        );
        for id in &via_pred {
            assert!(
                allow(id.parse::<usize>().unwrap()),
                "returned excluded id {id}"
            );
        }

        // ...and both are the true top-k over the allowed set.
        let qn = crate::vector::simd::norm_simd(&q);
        let mut scored: Vec<(f32, usize)> = (0..n)
            .filter(|&i| allow(i))
            .map(|i| {
                let e = &embeddings[i];
                let dot: f32 = q.iter().zip(e).map(|(a, b)| a * b).sum();
                (dot / (qn * crate::vector::simd::norm_simd(e)), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let want: std::collections::HashSet<usize> =
            scored.iter().take(k).map(|&(_, i)| i).collect();
        let got: std::collections::HashSet<usize> =
            via_pred.iter().map(|s| s.parse().unwrap()).collect();
        assert_eq!(
            got, want,
            "predicate walk != brute force over allowed (seed {qseed})"
        );
    }
}

#[test]
fn search_layer_considers_neighbors_beyond_fixed_stack_batch() {
    let mut config = HNSWConfig::default().with_m(64);
    config.m0 = 128;

    let mut index = HNSWIndex::new(2, config);
    let total_nodes = 66usize; // node 0 + 65 neighbors
    index.connections = vec![vec![Vec::new()]; total_nodes];
    index.metadata = vec![None; total_nodes];
    index.entry_point = Some(0);
    index.max_layer = 0;

    for node in 0..total_nodes {
        // Node 65 is the best match for query [1, 0]; the rest are orthogonal to it.
        let v: [f32; 2] = if node == 65 { [1.0, 0.0] } else { [0.0, 1.0] };
        index.push_node(&v);
        index.ids.push(format!("doc-{node}"));
        index.contents.push(String::new());
    }

    // Give node 0 all 65 others as layer-0 neighbours, via the arena's owner.
    let neighbors: Vec<u32> = (1..=65).map(|n| n as u32).collect();
    index.l0_replace(0, &neighbors);

    let mut ctx = SearchContext::new(index.len());
    let qprep = QueryPrep {
        norm: 1.0,
        rabitq: None,
        turboquant: None,
        turborabit: None,
        filter: None,
    };
    let candidates = index.search_layer(&[1.0, 0.0], &[0], 66, 0, &mut ctx, &qprep);
    assert!(
        candidates.iter().any(|&(_, id)| id == 65),
        "best neighbor from position >64 should be considered"
    );
}

#[test]
fn test_add_single_document() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);

    assert!(index.add(doc).is_ok());
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn test_add_dimension_mismatch() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("doc1", vec![1.0, 0.0]); // Wrong dimension

    assert!(index.add(doc).is_err());
}

// ========================================================================
// train() / is_trained() — quantized storages must not panic on add()
// ========================================================================

/// `Storage::SQ8` + `new()` + `add()`, skipping `train()`, must return `Err`, never
/// panic. This is the exact bug that used to crash on the first `add()`: `push_node`
/// indexed into `q_scale`/`q_min`, which stay empty until a codebook is fit.
#[test]
fn sq8_add_without_train_errs_not_panics() {
    let mut index = HNSWIndex::new(
        4,
        HNSWConfig {
            storage: Storage::SQ8,
            rerank_candidates: 100,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    );
    let err = index
        .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
        .expect_err("add() before train() must error, not panic");
    assert!(
        matches!(err, crate::RagError::NotTrained(_)),
        "expected NotTrained, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("train("),
        "error message should name the method to call: {msg}"
    );
}

/// Same bug class, `Storage::RaBitQ`: `push_node` calls `self.rabitq.as_ref().expect(...)`,
/// which would panic identically without this guard.
#[test]
fn rabitq_add_without_train_errs_not_panics() {
    let mut index = HNSWIndex::new(
        4,
        HNSWConfig {
            storage: Storage::RaBitQ,
            rerank_candidates: 100,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    );
    let err = index
        .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
        .expect_err("add() before train() must error, not panic");
    assert!(
        matches!(err, crate::RagError::NotTrained(_)),
        "expected NotTrained, got {err:?}"
    );
}

/// `Storage::F32` needs no codebook, so `add()` without `train()` must keep working
/// exactly as before — `train()` is a no-op there and must never gate it.
#[test]
fn f32_add_works_without_training() {
    let mut index = HNSWIndex::new(4, HNSWConfig::default());
    assert!(index
        .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
        .is_ok());
    assert_eq!(index.len(), 1);
    assert!(index.search(&[0.5, -0.3, 0.8, 0.1], 1).unwrap()[0].id == "doc1");
}

/// End-to-end: `new()` -> `train(sample)` -> incremental `add_embedding()` -> search,
/// for both quantized storages. Ground truth is brute-force exact L2 over the base set;
/// queries are HELD OUT (never added), for the same reason as the build_parallel recall
/// test — self-retrieval can't distinguish a working metric from a broken one.
fn train_then_add_recall(storage: Storage) -> f32 {
    let mut rng = StdRng::seed_from_u64(303);
    let dim = 16;
    let n_clusters = 8;
    let per_cluster = 40;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..40)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
        })
        .collect();

    let mut index = HNSWIndex::new(
        dim,
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 16,
            m0: 32,
            ef_construction: 150,
            ef_search: 150,
            storage,
            rerank_candidates: 50,
            ..Default::default()
        },
    );
    // Train from a sample (need not be the whole corpus — half of it here), then add
    // every vector incrementally, the way a caller without the full corpus up front
    // would use this API.
    index.train(&base[..base.len() / 2]).expect("train");
    for (i, v) in base.iter().enumerate() {
        index
            .add_embedding(i.to_string(), v.clone())
            .expect("add_embedding");
    }

    let k = 10;
    let mut total_recall = 0.0f32;
    for q in &queries {
        let mut exact: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, i)
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

        let got: HashSet<usize> = index
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();
        total_recall += truth.intersection(&got).count() as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}

/// SQ8 via incremental `train()` + `add_embedding()` — not `build_parallel`, so
/// construction itself uses the quantized metric, a strictly harder case than the
/// build-time recall tests. Measured on this exact seed/config: 100%; floor set well
/// below that for margin, not at the measurement (see `lesson_untested_public_options`).
#[test]
fn sq8_train_then_add_retrieves_correctly() {
    let recall = train_then_add_recall(Storage::SQ8);
    assert!(
        recall > 0.6,
        "Storage::SQ8 via train()+add_embedding(): recall@10 = {:.1}%, held-out queries, \
             brute-force ground truth",
        recall * 100.0
    );
}

/// Same as above for RaBitQ. Measured on this exact seed/config: 100%.
#[test]
fn rabitq_train_then_add_retrieves_correctly() {
    let recall = train_then_add_recall(Storage::RaBitQ);
    assert!(
        recall > 0.6,
        "Storage::RaBitQ via train()+add_embedding(): recall@10 = {:.1}%, held-out \
             queries, brute-force ground truth",
        recall * 100.0
    );
}

/// Warren's defining properties: TurboRabit's walk, an 8-bit residual rerank, **no f32**.
///
/// The arena assert is the load-bearing one — Warren's whole QPS case is that its hot path is
/// byte-identical to TurboRabit's. If the block ever diverges, the walk silently stops being
/// the cheap one and the mode's reason to exist is gone.
#[test]
fn warren_walks_like_turborabit_and_reranks_without_f32() {
    let mut rng = StdRng::seed_from_u64(21);
    let (dim, clusters, per) = (32, 6, 70);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 5.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..clusters * per)
        .map(|i| {
            centers[i % clusters]
                .iter()
                .map(|x| x + rng.random::<f32>() * 0.3)
                .collect()
        })
        .collect();
    let cfg = |storage| HNSWConfig {
        metric: DistanceMetric::Cosine,
        m: 12,
        m0: 24,
        ef_construction: 100,
        ef_search: 100,
        storage,
        rabit_bits: 4,
        rerank_candidates: 50,
        seed: Some(4),
        ..Default::default()
    };
    let warren = HNSWIndex::build_parallel(base.clone(), cfg(Storage::Warren));
    let tr = HNSWIndex::build_parallel(base.clone(), cfg(Storage::TurboRabit));

    // The hot block must match TurboRabit's exactly — same stride, same layout.
    assert_eq!(
        warren.stride, tr.stride,
        "Warren's arena block must equal TurboRabit's"
    );
    assert!(warren.full.is_empty(), "Warren must retain no f32");
    assert!(
        !tr.full.is_empty(),
        "TurboRabit does retain f32 (the thing Warren removes)"
    );
    assert_eq!(
        warren.warren_res.len(),
        base.len() * (20 + 2 * dim),
        "one residual block per node"
    );

    // The rerank knob must work despite `full` being empty.
    let mut w = warren;
    w.set_rerank_candidates(100)
        .expect("Warren reranks on its residual, not f32");

    let k = 10;
    let mut total = 0.0f32;
    for (qi, q) in base.iter().enumerate().take(50) {
        let got: HashSet<usize> = w
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();
        assert!(got.contains(&qi), "query {qi} must retrieve itself");
        let mut exact: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                (-d / n.max(1e-12), i)
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();
        total += truth.intersection(&got).count() as f32 / k as f32;
    }
    let recall = total / 50.0;
    assert!(recall > 0.8, "Warren recall@10 = {:.3}", recall);
}

/// Build a clustered corpus with signed, unit-normalizable coordinates.
///
/// Shared by the two rerank-discrimination tests below so they differ only in the mode
/// under test.
fn rerank_fixture(dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(77);
    let (clusters, per) = (12usize, 80usize);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..clusters * per)
        .map(|i| {
            centers[i % clusters]
                .iter()
                .map(|x| x + (rng.random::<f32>() - 0.5) * 0.3)
                .collect()
        })
        .collect();
    let queries = base.iter().take(60).cloned().collect();
    (base, queries)
}

/// Recall@k of `index` against brute-force cosine ground truth over `base`.
fn recall_against_exact(
    index: &HNSWIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let mut total = 0.0;
    for q in queries {
        let got: HashSet<usize> = index
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();
        let mut exact: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                (-d / n.max(1e-12), i)
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();
        total += truth.intersection(&got).count() as f32 / k as f32;
    }
    total / queries.len() as f32
}

/// The rerank stage must **change the answer**, not merely be configurable.
///
/// # Why this is a differential test
///
/// The obvious assertion — "recall is above some threshold" — cannot detect a dead rerank.
/// Warren scores 0.855 on sift10k with its residual stage doing *literally nothing* (recall
/// identical to three decimal places at `rerank_candidates` 0, 400 and 2000), and 0.9999 on
/// real 768-d embeddings with it working. Any fixed bar between those passes in both worlds,
/// so the original `recall > 0.8` here could never have caught the stage being inert.
///
/// So: build **one** index, toggle only `rerank_candidates`, and require the two runs to
/// differ materially. Same graph, same codes, same queries — the rerank is the only variable,
/// which is what makes the gap attributable to it.
#[test]
fn warren_rerank_materially_improves_recall() {
    let dim = 64;
    let (base, queries) = rerank_fixture(dim);
    let mut index = HNSWIndex::build_parallel(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 60,
            storage: Storage::Warren,
            rabit_bits: 4,
            rerank_candidates: 100,
            seed: Some(4),
            ..Default::default()
        },
    );

    index
        .set_rerank_candidates(0)
        .expect("lowering to 0 is always allowed");
    let without = recall_against_exact(&index, &base, &queries, 10);
    index
        .set_rerank_candidates(100)
        .expect("Warren reranks on its residual, not on f32");
    let with = recall_against_exact(&index, &base, &queries, 10);

    assert!(
        with - without > 0.15,
        "Warren's residual rerank changed recall by only {:.4} ({without:.4} -> {with:.4}). \
             The knob is accepted but the stage is not affecting results — which is exactly how \
             it behaves on non-unit-norm L2 data today.",
        with - without
    );

    // A rerank that ran must also produce a real distance spread. Warren under L2 on
    // unnormalized vectors returns `(2 - 2*inner_product).max(0)`, which clamps to 0 for
    // every candidate and hands back score 1.0 ten times over — degenerate output that a
    // recall check alone would not notice.
    let scores: Vec<f32> = index
        .search(&queries[0], 10)
        .expect("search")
        .iter()
        .map(|r| r.score)
        .collect();
    let distinct = scores
        .iter()
        .map(|s| format!("{s:.6}"))
        .collect::<HashSet<_>>()
        .len();
    assert!(
        distinct > 5,
        "reranked scores are degenerate: {distinct} distinct values in {scores:?}"
    );
}

/// The same discrimination for `Storage::TurboRabit`, which reranks against retained f32.
///
/// Kept separate from Warren rather than parameterized: the two reach their accuracy by
/// different mechanisms (exact f32 rescoring vs an 8-bit residual), so a shared test that
/// passed would not tell you *which* one still works.
#[test]
fn turborabit_rerank_materially_improves_recall() {
    let dim = 64;
    let (base, queries) = rerank_fixture(dim);
    let mut index = HNSWIndex::build_parallel(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 60,
            storage: Storage::TurboRabit,
            rabit_bits: 4,
            rerank_candidates: 100,
            seed: Some(4),
            ..Default::default()
        },
    );
    assert!(
        !index.full.is_empty(),
        "TurboRabit retains f32 to rerank against; without it this test proves nothing"
    );

    index.set_rerank_candidates(0).expect("lowering to 0");
    let without = recall_against_exact(&index, &base, &queries, 10);
    index.set_rerank_candidates(100).expect("raising back");
    let with = recall_against_exact(&index, &base, &queries, 10);

    assert!(
        with - without > 0.15,
        "TurboRabit's exact rerank changed recall by only {:.4} ({without:.4} -> {with:.4})",
        with - without
    );
}

/// `Storage::Warren` + `DistanceMetric::L2` on non-unit vectors is refused, loudly.
///
/// # What this is protecting against
///
/// Warren's rerank finishes with `(2.0 - 2.0 * acc).max(0.0)`, which is `‖q-x‖²` **only when
/// `‖q‖ = ‖x‖ = 1`**. At any other scale `acc` is a large inner product, the expression is
/// negative for every candidate, and `.max(0.0)` flattens the pool to distance 0 — every
/// candidate ties, the re-sort carries no information, and the walk's ordering survives.
///
/// Measured on sift10k (‖x‖ ≈ 500) before the guard existed: recall identical to four decimals
/// with the rerank off and on, and a 10-result search returning **one distinct score, 1.0, ten
/// times**. A confident-looking wrong answer, with nothing to alert the caller.
///
/// The guard converts that into a panic. It is not the real fix — the rerank should be
/// norm-aware like the walk already is — but it removes the silent-wrong-answer class.
#[test]
#[should_panic(expected = "requires unit-norm vectors")]
fn warren_l2_on_unnormalized_vectors_is_rejected() {
    let dim = 64;
    let (base, _) = rerank_fixture(dim);
    // Scale well away from unit norm — the condition the rerank silently depends on.
    let base: Vec<Vec<f32>> = base
        .iter()
        .map(|v| v.iter().map(|x| x * 100.0).collect())
        .collect();
    let _ = HNSWIndex::build_parallel(
        base,
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 12,
            m0: 24,
            ef_construction: 100,
            storage: Storage::Warren,
            rabit_bits: 4,
            rerank_candidates: 100,
            seed: Some(4),
            ..Default::default()
        },
    );
}

/// The configuration the guard above leaves legal must actually work.
///
/// Rejecting the broken combination is only half a contract; without this, `Warren` + `L2`
/// could be refused on non-unit input and quietly useless on unit input, and both halves
/// would look fine. This is also the only test that exercises Warren's **L2 rerank arm** at
/// all — every other Warren test runs under `Cosine`, which is a different branch. Mutation
/// testing found all four mutants on that line surviving for exactly that reason.
#[test]
fn warren_rerank_works_under_l2_on_unit_norm_vectors() {
    let dim = 64;
    let (base, queries) = rerank_fixture(dim);
    let unit = |v: &Vec<f32>| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    };
    let base: Vec<Vec<f32>> = base.iter().map(unit).collect();
    let queries: Vec<Vec<f32>> = queries.iter().map(unit).collect();

    let mut index = HNSWIndex::build_parallel(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 12,
            m0: 24,
            ef_construction: 100,
            ef_search: 60,
            storage: Storage::Warren,
            rabit_bits: 4,
            rerank_candidates: 100,
            seed: Some(4),
            ..Default::default()
        },
    );

    index.set_rerank_candidates(0).expect("lowering to 0");
    let without = recall_against_exact(&index, &base, &queries, 10);
    index.set_rerank_candidates(100).expect("raising back");
    let with = recall_against_exact(&index, &base, &queries, 10);
    assert!(
        with - without > 0.15,
        "Warren's L2 rerank changed recall by only {:.4} ({without:.4} -> {with:.4}) on \
         unit-norm vectors, where its `2 - 2*<q,x>` arithmetic is exact",
        with - without
    );

    // Unit-norm L2 distances live in [0, 4]; a clamped-to-zero pool would show one value.
    let scores: Vec<f32> = index
        .search(&queries[0], 10)
        .expect("search")
        .iter()
        .map(|r| r.score)
        .collect();
    let distinct = scores
        .iter()
        .map(|s| format!("{s:.6}"))
        .collect::<HashSet<_>>()
        .len();
    assert!(
        distinct > 5,
        "reranked L2 scores are degenerate: {distinct} distinct in {scores:?}"
    );
}

/// `train()` on a non-empty index must refuse — retraining would desynchronize vectors
/// already encoded under the old codebook.
#[test]
fn train_on_nonempty_index_errs() {
    let mut index = HNSWIndex::new(
        3,
        HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        },
    );
    index.train(&[vec![1.0, 2.0, 3.0]]).expect("first train");
    index
        .add_embedding("0".into(), vec![1.0, 2.0, 3.0])
        .expect("add after train");
    assert!(
        index.train(&[vec![4.0, 5.0, 6.0]]).is_err(),
        "retraining a non-empty index must be rejected"
    );
}

#[test]
fn rejects_inf_embedding() {
    let mut index = HNSWIndex::new(3, HNSWConfig::default());
    let doc = Document {
        id: "inf".to_string(),
        content: "test".to_string(),
        embedding: vec![f32::INFINITY, 0.0, 0.0],
        metadata: None,
    };
    assert!(index.add(doc).is_err());

    let doc_neg = Document {
        id: "neg_inf".to_string(),
        content: "test".to_string(),
        embedding: vec![0.0, f32::NEG_INFINITY, 0.0],
        metadata: None,
    };
    assert!(index.add(doc_neg).is_err());
}

#[test]
fn an_undersized_scratch_context_is_regrown_before_use() {
    // `BitsetVisited` indexes with `get_unchecked`. A context sized for a smaller index
    // than the one it is used against is therefore not merely wrong, it is UB in release.
    // `search_inner` guards that with `if ctx.capacity < self.len()`.
    //
    // A *public* caller can no longer reach this state: `search`/`search_batch` size the
    // scratch at the moment of use, and `Searcher` borrows the index, so the index cannot
    // grow while a searcher is alive — the borrow checker retired the runtime hazard.
    // This test drives the guard directly, from inside the module, because the guard is
    // load-bearing for an `unsafe` block and must not be deleted as "unreachable".
    let mut index = HNSWIndex::new(3, HNSWConfig::default());
    for i in 0..64 {
        index
            .add(Document {
                id: format!("doc-{i}"),
                content: String::new(),
                embedding: vec![(i as f32) * 0.01, 1.0 - (i as f32) * 0.01, 0.0],
                metadata: None,
            })
            .unwrap();
    }

    // Deliberately far too small: one node's worth of bitset for a 64-node index.
    let mut stale = SearchContext::new(1);
    assert!(stale.capacity < index.len());

    let results = index
        .search_inner(&[1.0, 0.0, 0.0], 5, &mut stale, None)
        .unwrap();

    assert_eq!(results.len(), 5);
    assert!(
        stale.capacity >= index.len(),
        "search_inner must regrow an undersized context, not index past the end of its bitset"
    );
}

#[test]
fn test_search_empty_index() {
    let index = HNSWIndex::with_defaults(3);
    let query = vec![1.0, 0.0, 0.0];

    let results = index.search(&query, 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_single_document() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
    index.add(doc).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
    assert!((results[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn test_search_multiple_documents() {
    let mut index = HNSWIndex::with_defaults(3);

    // Add documents with different embeddings
    let docs = vec![
        create_test_document("doc1", vec![1.0, 0.0, 0.0]),
        create_test_document("doc2", vec![0.0, 1.0, 0.0]),
        create_test_document("doc3", vec![0.0, 0.0, 1.0]),
        create_test_document("doc4", vec![1.0, 1.0, 0.0]),
    ];

    for doc in docs {
        index.add(doc).unwrap();
    }

    // Query closest to doc1
    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "doc1");
    assert!(results[0].score > 0.9);
}

#[test]
fn test_search_exact_match() {
    let mut index = HNSWIndex::with_defaults(3);

    let embedding = vec![0.5, 0.5, 0.7072];
    let doc = create_test_document("doc1", embedding.clone());
    index.add(doc).unwrap();

    let results = index.search(&embedding, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert!((results[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn test_clear() {
    let mut index = HNSWIndex::with_defaults(3);

    for i in 0..5 {
        let doc = create_test_document(&format!("doc{}", i), vec![i as f32, 0.0, 0.0]);
        index.add(doc).unwrap();
    }

    assert_eq!(index.len(), 5);

    index.clear();

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[test]
fn test_random_dataset_100_vectors() {
    let dim = 128;
    let mut index = HNSWIndex::with_defaults(dim);

    // Add 100 random vectors
    for i in 0..100 {
        let embedding = generate_random_vector(dim, i);
        let doc = create_test_document(&format!("doc{}", i), embedding);
        index.add(doc).unwrap();
    }

    assert_eq!(index.len(), 100);

    // Search with a random query
    let query = generate_random_vector(dim, 9999);
    let results = index.search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Results should be sorted by score (descending)
    for i in 0..results.len() - 1 {
        assert!(results[i].score >= results[i + 1].score);
    }
}

#[test]
fn test_random_dataset_1000_vectors() {
    let dim = 64;
    let mut index = HNSWIndex::with_defaults(dim);

    // Add 1000 random vectors
    for i in 0..1000 {
        let embedding = generate_random_vector(dim, i);
        let doc = create_test_document(&format!("doc{}", i), embedding);
        index.add(doc).unwrap();
    }

    assert_eq!(index.len(), 1000);

    // Perform multiple searches
    for seed in [111, 222, 333, 444, 555] {
        let query = generate_random_vector(dim, seed);
        let results = index.search(&query, 20).unwrap();

        assert_eq!(results.len(), 20);

        // Verify ordering
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }

        // All scores should be between -1 and 1
        for result in &results {
            assert!(result.score >= -1.0 && result.score <= 1.0);
        }
    }
}

#[test]
fn test_recall_with_known_neighbors() {
    let dim = 32;
    let mut index = HNSWIndex::with_defaults(dim);

    // Create a query vector
    let query = generate_random_vector(dim, 0);

    // Create 100 vectors with varying similarity to query
    for i in 0..100 {
        let mut embedding = generate_random_vector(dim, i + 1);

        // First 10 vectors are more similar to query
        if i < 10 {
            for j in 0..dim {
                embedding[j] = query[j] * 0.9 + embedding[j] * 0.1;
            }
        }

        let doc = create_test_document(&format!("doc{}", i), embedding);
        index.add(doc).unwrap();
    }

    // Search for top 10
    let results = index.search(&query, 10).unwrap();

    // Count how many of the actual top 10 were found
    let mut recall_count = 0;
    for result in &results {
        let doc_num: usize = result.id.strip_prefix("doc").unwrap().parse().unwrap();
        if doc_num < 10 {
            recall_count += 1;
        }
    }

    // HNSW should find most of the true nearest neighbors
    // Expect at least 70% recall
    assert!(recall_count >= 7, "Recall too low: {}/10", recall_count);
}

#[test]
fn test_search_dimension_mismatch() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
    index.add(doc).unwrap();

    let query = vec![1.0, 0.0]; // Wrong dimension
    assert!(index.search(&query, 1).is_err());
}

#[test]
fn test_metadata_preservation() {
    let mut index = HNSWIndex::with_defaults(3);

    let mut doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
    doc.metadata = Some(serde_json::json!({"category": "test", "priority": 5}));

    index.add(doc).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].metadata.is_some());

    let metadata = results[0].metadata.as_ref().unwrap();
    assert_eq!(metadata["category"], "test");
    assert_eq!(metadata["priority"], 5);
}

#[test]
fn test_search_with_nan_query_does_not_panic() {
    let mut index = HNSWIndex::with_defaults(3);
    index
        .add(create_test_document("doc1", vec![1.0, 0.0, 0.0]))
        .unwrap();
    index
        .add(create_test_document("doc2", vec![0.0, 1.0, 0.0]))
        .unwrap();

    let query = vec![f32::NAN, 0.0, 0.0];
    let outcome = std::panic::catch_unwind(|| index.search(&query, 2));

    assert!(outcome.is_ok(), "search panicked when query contains NaN");
}

#[test]
#[should_panic(expected = "All embeddings must have the same dimension")]
fn test_build_rejects_mismatched_dimensions() {
    let _ = HNSWIndex::build(
        vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0]],
        HNSWConfig::default(),
    );
}

#[test]
fn test_add_rejects_nan_embedding() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("nan_doc", vec![1.0, f32::NAN, 0.0]);
    let result = index.add(doc);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("NaN"),
        "Error should mention NaN, got: {}",
        err_msg
    );
}

#[test]
fn test_add_embedding_rejects_nan() {
    let mut index = HNSWIndex::with_defaults(3);
    let result = index.add_embedding("nan_vec".into(), vec![f32::NAN, 0.0, 0.0]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("NaN"),
        "Error should mention NaN, got: {}",
        err_msg
    );
}

#[test]
fn test_add_rejects_all_nan_embedding() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("all_nan", vec![f32::NAN, f32::NAN, f32::NAN]);
    assert!(index.add(doc).is_err());
}

#[test]
fn test_zero_vector_accepted_and_searchable() {
    let mut index = HNSWIndex::with_defaults(3);

    // Zero vectors are valid (they just have zero norm)
    let doc_zero = create_test_document("zero", vec![0.0, 0.0, 0.0]);
    assert!(index.add(doc_zero).is_ok());

    let doc_normal = create_test_document("normal", vec![1.0, 0.0, 0.0]);
    assert!(index.add(doc_normal).is_ok());

    // Search should not panic with zero vectors in the index
    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();
    assert_eq!(results.len(), 2);

    // The normal vector should rank higher than the zero vector
    assert_eq!(results[0].id, "normal");
}

#[test]
fn test_zero_vector_query_does_not_panic() {
    let mut index = HNSWIndex::with_defaults(3);
    let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
    index.add(doc).unwrap();

    // Zero query should not panic (distance_to_node handles zero norm)
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
}

/// Level assignment must never hit `ln(0) = -inf`, AND must come from the seeded stream.
///
/// The old version of this test ran the loop and asserted nothing but the absence of a panic.
/// It therefore could not tell a seeded draw from `rand::rng()` — and `random_level` was in
/// fact calling `rand::rng()`, so `add()` ignored `seed` entirely. The panic property was
/// real and is kept; what it lacked was any way to fail for the reason that mattered.
#[test]
fn random_level_never_panics_and_comes_from_the_seeded_stream() {
    let draws = |seed: Option<u64>| -> Vec<usize> {
        let mut index = HNSWIndex::new(
            3,
            HNSWConfig {
                seed,
                ..Default::default()
            },
        );
        (0..10_000).map(|_| index.random_level()).collect()
    };

    // ln(0) is impossible: a level is finite, so it is small. (`as usize` on -inf saturates
    // to 0 rather than panicking, so assert the shape of the distribution, not just liveness.)
    let a = draws(Some(7));
    assert!(
        a.iter().all(|&l| l < 64),
        "exponential decay must not produce absurd levels"
    );
    assert!(
        a.iter().any(|&l| l > 0),
        "every node landing on layer 0 means ml is not applied"
    );

    assert_eq!(
        a,
        draws(Some(7)),
        "a fixed seed must give a reproducible level sequence"
    );
    assert_ne!(
        a,
        draws(Some(8)),
        "a different seed must give a different level sequence"
    );
}

/// `seed` must reach the INCREMENTAL path too, not just the bulk builders.
///
/// The bulk builders seed their own RNG, so they were reproducible while an index grown by
/// `add_embedding()` was not, at any seed. Found by the generated config x code-path matrix
/// (`xtask/config_matrix.py`): the `seed` row was empty in the `add` column, and an empty cell
/// there means nothing on that path reads the option.
#[test]
fn seed_reaches_the_incremental_add_path() {
    let vecs: Vec<Vec<f32>> = (0..200)
        .map(|i| {
            (0..8)
                .map(|d| ((i * 7 + d * 13) % 40) as f32 * 0.1)
                .collect()
        })
        .collect();

    let grown = |seed: u64| -> Vec<Vec<u32>> {
        let mut ix = HNSWIndex::new(
            8,
            HNSWConfig {
                seed: Some(seed),
                m: 4,
                m0: 8,
                ..Default::default()
            },
        );
        for (i, v) in vecs.iter().enumerate() {
            ix.add_embedding(i.to_string(), v.clone()).unwrap();
        }
        (0..ix.len())
            .map(|i| {
                let mut n = ix.get_neighbors_l0(i).to_vec();
                n.sort_unstable();
                n
            })
            .collect()
    };

    assert_eq!(
        grown(7),
        grown(7),
        "an index grown by add() must be reproducible at a fixed seed -- `add` is inherently \
             sequential, so unlike the parallel bulk builder it has no thread race to blame"
    );
    assert_ne!(
        grown(7),
        grown(9),
        "a different seed must give a different graph"
    );
}

// ========================================================================
// Distance metric
// ========================================================================

/// Cosine and L2 must genuinely disagree, and each must pick its own winner.
///
/// q = [10, 0]:
///   far_same_direction = [100, 0] — cosine distance 0 (identical direction),
///                                   but L2 distance 90 (way off in magnitude)
///   near_off_axis      = [9, 3]   — cosine distance ~0.051 (direction differs),
///                                   but L2 distance ~3.16 (much closer in space)
///
/// A cosine index must return `far_same_direction`; an L2 index must return
/// `near_off_axis`. This is exactly why scoring foxstash's cosine HNSW against
/// SIFT's L2 ground truth read 55% for a graph that is actually 97.7% correct.
#[test]
fn cosine_and_l2_pick_different_neighbors() {
    let query = vec![10.0, 0.0];
    let docs = [
        ("far_same_direction", vec![100.0, 0.0]),
        ("near_off_axis", vec![9.0, 3.0]),
    ];

    let winner = |metric: DistanceMetric| {
        let mut index = HNSWIndex::new(
            2,
            HNSWConfig {
                metric,
                ..Default::default()
            },
        );
        for (id, v) in &docs {
            index
                .add(Document {
                    id: (*id).to_string(),
                    content: String::new(),
                    embedding: v.clone(),
                    metadata: None,
                })
                .unwrap();
        }
        index.search(&query, 1).unwrap()[0].id.clone()
    };

    assert_eq!(winner(DistanceMetric::Cosine), "far_same_direction");
    assert_eq!(winner(DistanceMetric::L2), "near_off_axis");
}

/// L2 scores must stay in (0, 1] and decrease with distance. Cosine's `1 - d`
/// convention would emit large negative scores for unbounded L2 distances.
#[test]
fn l2_scores_are_bounded_and_monotonic() {
    let mut index = HNSWIndex::new(
        2,
        HNSWConfig {
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    );
    for (i, v) in [vec![0.0, 0.0], vec![50.0, 0.0], vec![500.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        index
            .add(Document {
                id: i.to_string(),
                content: String::new(),
                embedding: v,
                metadata: None,
            })
            .unwrap();
    }

    let results = index.search(&[0.0, 0.0], 3).unwrap();
    assert_eq!(results[0].id, "0", "nearest must come first");
    for r in &results {
        assert!(
            r.score > 0.0 && r.score <= 1.0,
            "L2 score {} outside (0, 1]",
            r.score
        );
    }
    for w in results.windows(2) {
        assert!(w[0].score >= w[1].score, "scores must be descending");
    }
}

/// Cosine remains the default, so existing code and persisted indexes are unaffected.
#[test]
fn cosine_is_still_the_default() {
    assert_eq!(HNSWConfig::default().metric, DistanceMetric::Cosine);
    assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
}

/// `keep_pruned_connections` must actually do something in **both** builders.
///
/// It did not. The parallel builder — the default path, and the one every benchmark uses —
/// backfilled each node's neighbour list to `m0` unconditionally: the config flag was named
/// in a comment above the backfill and never read. So the Algorithm-4 diversity heuristic
/// ran, correctly pruned, and had its output immediately refilled with the exact candidates
/// it had just rejected. Every node ended up saturated at `m0` (measured: degree 64.0/64,
/// against faiss's 25.4 at the same M), and every hop paid for it.
///
/// Nothing caught it because no test ever set the flag to `false` — the same DEFAULT-ONLY
/// blind spot that let `BuildStrategy::Sequential` panic on every input for a release.
/// A flag that is only ever exercised at its default is not tested, it is assumed.
#[test]
fn keep_pruned_connections_controls_graph_density_in_both_builders() {
    let mut rng = StdRng::seed_from_u64(11);
    let centers: Vec<Vec<f32>> = (0..12)
        .map(|_| (0..24).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..600)
        .map(|i| {
            let c = &centers[i % 12];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    for strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
        let build = |keep: bool| {
            HNSWIndex::build(
                embeddings.clone(),
                HNSWConfig {
                    m: 16,
                    m0: 32,
                    ef_construction: 100,
                    keep_pruned_connections: keep,
                    build_strategy: strategy,
                    seed: Some(3),
                    ..Default::default()
                },
            )
        };

        let dense = build(true).avg_degree_l0();
        let sparse = build(false).avg_degree_l0();

        assert!(
            sparse < dense,
            "{strategy:?}: keep_pruned_connections has no effect \
                 (degree {sparse:.1} with it off vs {dense:.1} with it on) — \
                 the flag is being ignored and the diversity heuristic's pruning is discarded"
        );
    }
}

/// Every build strategy must produce a working graph.
///
/// `BuildStrategy::Sequential` panicked outright for an entire release: the layer-0
/// refactor made the flat array the sole owner of layer-0 links, and the sequential
/// builder was never taught to grow it. Nothing caught it — every other test and every
/// doctest either forces `Parallel` or takes the default, so `Sequential` had no
/// coverage at all despite being a documented public option.
///
/// This asserts *recall*, not merely absence of a panic: the same refactor left a
/// `build_l0_cache()` call that would have copied an empty nested layer 0 over the real
/// graph, erasing every layer-0 link and failing silently with a still-"working" index.
#[test]
fn every_build_strategy_produces_a_searchable_graph() {
    // Clustered, not uniform-random: random vectors have no structure to recover, and
    // every ANN scores ~60% on them whether or not its graph is intact.
    let mut rng = StdRng::seed_from_u64(7);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..400)
        .map(|i| {
            let c = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.3).collect()
        })
        .collect();

    for strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
        let config = HNSWConfig::default()
            .with_build_strategy(strategy)
            .with_seed(42)
            .with_ef_search(100);
        let index = HNSWIndex::build(embeddings.clone(), config);
        assert_eq!(
            index.len(),
            embeddings.len(),
            "{strategy:?}: wrong node count"
        );

        // Self-retrieval: querying with an indexed vector must return that vector.
        // A graph with its layer-0 links erased fails this immediately.
        let hits = embeddings
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                index
                    .search(e, 1)
                    .expect("search")
                    .first()
                    .is_some_and(|r| r.id == i.to_string())
            })
            .count();
        let recall = hits as f32 / embeddings.len() as f32;
        assert!(
            recall > 0.95,
            "{strategy:?}: self-retrieval recall {:.1}%, graph is broken",
            recall * 100.0,
        );
    }
}

// ========================================================================
// Storage::RaBitQ
// ========================================================================

/// `vec_words`/`node_stride` must match the documented arena layout: 2 scalar words
/// (`dtc_sq`, `est_factor`) plus `dim` packed sign bits, byte-then-word-rounded — not a
/// bare `dim.div_ceil(32)`, which would silently disagree with `RaBitCode::bits`' byte
/// granularity whenever `dim` isn't a multiple of 32 but is a multiple of 8.
#[test]
fn rabitq_vec_words_matches_documented_layout() {
    // dim = 40: divisible by 8 (5 bytes) but not by 32, so a naive dim.div_ceil(32)
    // would round to the same 2 words as dim.div_ceil(8).div_ceil(4) here — pick a case
    // where they'd actually differ: dim = 100.
    // bytes = ceil(100/8) = 13, words = ceil(13/4) = 4.
    assert_eq!(rabitq_bit_words(100), 4);
    assert_eq!(vec_words(Storage::RaBitQ, 100, 0), 2 + 4);

    // dim = 128 (SIFT-adjacent): bytes = 16, words = 4 -> vector region = 24 bytes,
    // matching the doc comment on `Storage`.
    assert_eq!(vec_words(Storage::RaBitQ, 128, 0), 2 + 4);
    assert_eq!(vec_words(Storage::RaBitQ, 128, 0) * 4, 24);
}

/// End-to-end recall gate for `Storage::RaBitQ`, built the same way the benchmarks do
/// (`build_parallel`, which builds the graph in exact f32 space and only quantizes the
/// traversal storage afterward — see `convert_parallel_to_index`).
///
/// Ground truth is exact brute-force L2 over the base set. Queries are HELD OUT: distinct
/// vectors near (not equal to) cluster centers, never inserted into the index. Querying
/// with an indexed vector would re-derive that vector's own RaBitQ code and could score
/// well even against a broken estimator — see `lesson_untestable_by_construction`. Data is
/// clustered, not uniform-random, for the same reason every other recall test here is:
/// uniform-random vectors have no structure to lose and every ANN scores ~60% on them
/// regardless of whether the graph or the metric is correct.
/// End-to-end recall through the real index under [`Storage::TurboQuant`] + cosine — a public
/// storage variant with no integration test is a shipped bug (`lesson_untested_public_options`).
/// Held-out queries only (never self-retrieval), clustered data, and a discriminating-power
/// floor: with the estimator sabotaged this recall collapses (verified separately). Also checks
/// that more bits ⇒ at least as much recall, end to end.
#[test]
#[allow(deprecated)] // exercises Storage::TurboQuant, deprecated until 0.8 removal
fn turboquant_recall_on_clustered_data_with_held_out_queries() {
    let mut rng = StdRng::seed_from_u64(2025);
    let dim = 96;
    let n_clusters = 16;
    let per_cluster = 40;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
        .collect();
    let jitter = |rng: &mut StdRng, c: &[f32]| -> Vec<f32> {
        c.iter()
            .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
            .collect()
    };
    let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
        .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
        .collect();
    let queries: Vec<Vec<f32>> = (0..60)
        .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
        .collect();

    let k = 10;
    // Cosine ground truth (TurboQuant estimates cosine similarity).
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-9)
    };
    let truth: Vec<HashSet<usize>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cos(v, q), i))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            s.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect();

    let recall_for = |bits: usize| -> f32 {
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 200,
            storage: Storage::TurboQuant,
            turbo_bits: bits,
            rerank_candidates: 50,
            seed: Some(9),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);
        let mut total = 0.0f32;
        for (q, gt) in queries.iter().zip(&truth) {
            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            total += gt.intersection(&got).count() as f32 / k as f32;
        }
        total / queries.len() as f32
    };

    let r2 = recall_for(2);
    let r4 = recall_for(4);
    // Non-vacuous floor (sabotaging the estimator collapses this), and more bits never hurt.
    assert!(r2 > 0.6, "TurboQuant b=2 recall too low end-to-end: {r2}");
    assert!(
        r4 >= r2 - 0.05,
        "b=4 ({r4}) unexpectedly far below b=2 ({r2})"
    );
}

/// End-to-end recall through the real index under [`Storage::TurboRabit`] — same contract
/// as the TurboQuant test above (public variant + no integration test = shipped bug), and
/// additionally under **both metrics**: honest L2 support is TurboRabit's differentiator
/// over TurboQuant, so an untested L2 path here would be the untested half of the point.
#[test]
fn turborabit_recall_on_clustered_data_with_held_out_queries() {
    let mut rng = StdRng::seed_from_u64(2026);
    let dim = 96;
    let n_clusters = 16;
    let per_cluster = 40;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
        .collect();
    let jitter = |rng: &mut StdRng, c: &[f32]| -> Vec<f32> {
        c.iter()
            .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
            .collect()
    };
    let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
        .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
        .collect();
    let queries: Vec<Vec<f32>> = (0..60)
        .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
        .collect();

    let k = 10;
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-9)
    };
    let l2 =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let truth_for = |better_first: &dyn Fn(&[f32], &[f32]) -> f32, descending: bool| {
        queries
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (better_first(v, q), i))
                    .collect();
                if descending {
                    s.sort_by(|a, b| b.0.total_cmp(&a.0));
                } else {
                    s.sort_by(|a, b| a.0.total_cmp(&b.0));
                }
                s.into_iter()
                    .take(k)
                    .map(|(_, i)| i)
                    .collect::<HashSet<usize>>()
            })
            .collect::<Vec<_>>()
    };

    let recall_for = |metric: DistanceMetric, bits: usize, truth: &[HashSet<usize>]| -> f32 {
        let config = HNSWConfig {
            metric,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 200,
            storage: Storage::TurboRabit,
            rabit_bits: bits,
            rerank_candidates: 50,
            seed: Some(9),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);
        let mut total = 0.0f32;
        for (q, gt) in queries.iter().zip(truth) {
            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            total += gt.intersection(&got).count() as f32 / k as f32;
        }
        total / queries.len() as f32
    };

    // Cosine: floor + more-bits-never-hurt, same contract as the TurboQuant test.
    let truth_cos = truth_for(&cos, true);
    let r2 = recall_for(DistanceMetric::Cosine, 2, &truth_cos);
    let r4 = recall_for(DistanceMetric::Cosine, 4, &truth_cos);
    assert!(
        r2 > 0.6,
        "TurboRabit b=2 cosine recall too low end-to-end: {r2}"
    );
    assert!(
        r4 >= r2 - 0.05,
        "b=4 ({r4}) unexpectedly far below b=2 ({r2})"
    );

    // L2: the estimator is native squared-L2, no proxy — hold it to the same floor.
    let truth_l2 = truth_for(&l2, false);
    let r3_l2 = recall_for(DistanceMetric::L2, 3, &truth_l2);
    assert!(
        r3_l2 > 0.6,
        "TurboRabit b=3 L2 recall too low end-to-end: {r3_l2}"
    );
}

/// The packed arena walk and the quantizer module are two implementations of one
/// estimator — exactly the shape every 1.0-audit bug had. This pins them together:
/// for every node, `distance_to_node` (arena bit-planes + shared SIMD kernel) must
/// equal `TurboRabitQuantizer::estimate_dist_sq` (reference, allocating) on a fresh
/// encode of the same input. Odd dim stresses the plane-packing tail; both metrics
/// because their dispatch differs.
#[test]
fn turborabit_packed_walk_matches_module_estimator() {
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 97; // not a multiple of 8: partial final byte in every bit-plane
    let base: Vec<Vec<f32>> = (0..80)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    for metric in [DistanceMetric::Cosine, DistanceMetric::L2] {
        let config = HNSWConfig {
            metric,
            storage: Storage::TurboRabit,
            rabit_bits: 3,
            rerank_candidates: 10, // keep `full` so get_embedding works
            seed: Some(5),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);
        let tr = index.turborabit.as_ref().expect("quantizer fitted");
        let prep = index.prepare_turborabit_query(&query).expect("prepared");
        let qprep = QueryPrep {
            norm: crate::vector::simd::norm_simd(&query),
            rabitq: None,
            turboquant: None,
            turborabit: Some(&prep),
            filter: None,
        };
        for node_id in 0..index.len() {
            let packed = index.distance_to_node(&query, node_id, &qprep);
            // Re-encode the same input push_node saw (get_embedding returns the
            // original; the cosine path encodes a unit-normalized copy of it).
            let stored = index.get_embedding(node_id).to_vec();
            let code = tr.encode(&index.rabitq_cosine_input(&stored));
            let raw = tr.estimate_dist_sq(&prep, &code);
            let expected = match metric {
                DistanceMetric::L2 => raw,
                DistanceMetric::Cosine => (raw * 0.5).clamp(0.0, 2.0),
            };
            let rel = (packed - expected).abs() / expected.abs().max(1e-4);
            // 1e-4, not 1e-3: both paths run the same f32 algebra, so the only honest
            // difference is summation order (~1e-5). A loose tolerance let a 1%-of-one-
            // term sabotage through; this one catches it.
            assert!(
                    rel < 1e-4,
                    "{metric:?} node {node_id}: packed walk {packed} != module {expected} (rel {rel:.2e})"
                );
        }
    }
}

/// Same pin for TurboQuant: arena `[gamma][qjl][nibbles]` + LUT/signed-sum kernels
/// must equal `TurboQuantizer::estimate_ip` on a fresh encode. Odd dim stresses the
/// half-used final nibble byte; b=4 exercises the full 8-entry LUT, b=1 the
/// no-nibble-section layout.
#[test]
#[allow(deprecated)] // exercises Storage::TurboQuant, deprecated until 0.8 removal
fn turboquant_packed_walk_matches_module_estimator() {
    let mut rng = StdRng::seed_from_u64(78);
    let dim = 97;
    let base: Vec<Vec<f32>> = (0..80)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    for bits in [1usize, 2, 4] {
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            storage: Storage::TurboQuant,
            turbo_bits: bits,
            rerank_candidates: 10,
            seed: Some(5),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);
        let tq = index.turboquant.as_ref().expect("quantizer fitted");
        let prep = index.prepare_turboquant_query(&query).expect("prepared");
        let qprep = QueryPrep {
            norm: crate::vector::simd::norm_simd(&query),
            rabitq: None,
            turboquant: Some(&prep),
            turborabit: None,
            filter: None,
        };
        for node_id in 0..index.len() {
            let packed = index.distance_to_node(&query, node_id, &qprep);
            let mut unit = index.get_embedding(node_id).to_vec();
            crate::vector::ops::normalize(&mut unit);
            let ip = tq.estimate_ip(&prep, &tq.encode(&unit));
            let expected = (1.0 - ip).clamp(0.0, 2.0);
            let rel = (packed - expected).abs() / expected.abs().max(1e-4);
            assert!(
                    rel < 1e-3,
                    "b={bits} node {node_id}: packed walk {packed} != module {expected} (rel {rel:.2e})"
                );
        }
    }
}

/// `reorder_for_locality: true` is the default and must be **transparent**: a build with it
/// on returns the same search results as one with it off (only faster), while actually
/// changing the internal layout. This pins both halves — the default is applied (arena
/// differs) and it is safe (results identical).
#[test]
fn reorder_default_is_transparent_but_real() {
    let mut rng = StdRng::seed_from_u64(51);
    let base: Vec<Vec<f32>> = (0..400)
        .map(|_| (0..48).map(|_| rng.random::<f32>()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..30)
        .map(|_| (0..48).map(|_| rng.random::<f32>()).collect())
        .collect();
    let cfg = |reorder: bool| HNSWConfig {
        metric: DistanceMetric::Cosine,
        m: 16,
        m0: 32,
        ef_construction: 200,
        ef_search: 100,
        seed: Some(9),
        reorder_for_locality: reorder,
        ..Default::default()
    };
    let plain = HNSWIndex::build_parallel(base.clone(), cfg(false));
    let reordered = HNSWIndex::build_parallel(base.clone(), cfg(true));

    // Real: the default actually relabelled the arena (entry point almost surely moves to a
    // low id under BFS; the arenas are not byte-identical).
    assert_ne!(
        plain.nodes, reordered.nodes,
        "reorder_for_locality: true must change the layout, but the arenas are identical"
    );
    // Transparent: identical results, scores included.
    for q in &queries {
        let a: Vec<(String, u32)> = plain
            .search(q, 10)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.score.to_bits()))
            .collect();
        let b: Vec<(String, u32)> = reordered
            .search(q, 10)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.score.to_bits()))
            .collect();
        assert_eq!(a, b, "default reorder changed a query's results");
    }
}

/// The presets and `with_auto_storage` must produce configs that **build and search**, and
/// the auto-picker must pick sensibly (max bits + rerank on a hostile cone corpus).
#[test]
fn presets_and_auto_storage_are_sane() {
    let mut rng = StdRng::seed_from_u64(61);
    let dim = 64;

    // Cone-shaped (hostile) corpus: big shared offset, tiny residuals → auto wants max bits.
    let offset: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 3.0).collect();
    let cone: Vec<Vec<f32>> = (0..500)
        .map(|_| {
            offset
                .iter()
                .map(|&o| o + (rng.random::<f32>() - 0.5) * 0.1)
                .collect()
        })
        .collect();
    let auto = HNSWConfig {
        rerank_candidates: 0,
        ..Default::default()
    }
    .with_auto_storage(&cone);
    assert_eq!(auto.storage, Storage::TurboRabit);
    assert_eq!(auto.rabit_bits, 4, "cone corpus should auto-pick max bits");
    assert!(
        auto.rerank_candidates > 0,
        "auto must enable rerank for TurboRabit"
    );

    assert_eq!(HNSWConfig::rag_high_recall().storage, Storage::TurboRabit);
    assert_eq!(HNSWConfig::rag_high_recall().rabit_bits, 4);
    assert_eq!(HNSWConfig::rag_throughput().storage, Storage::SQ8);

    // Each config builds a working index. Clustered data + held-out queries, cosine.
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0).collect())
        .collect();
    let mk = |n: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                centers[i % 8]
                    .iter()
                    .map(|x| x + rng.random::<f32>() * 0.3)
                    .collect()
            })
            .collect()
    };
    let base = mk(600, &mut rng);
    let queries = mk(40, &mut rng);
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-9)
    };
    let truth: Vec<HashSet<usize>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cos(v, q), i))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            s.into_iter().take(10).map(|(_, i)| i).collect()
        })
        .collect();

    for (label, mut cfg) in [
        ("high_recall", HNSWConfig::rag_high_recall()),
        ("throughput", HNSWConfig::rag_throughput()),
        ("auto", HNSWConfig::default().with_auto_storage(&base)),
    ] {
        cfg.ef_search = 200;
        cfg.seed = Some(3);
        let idx = HNSWIndex::build_parallel(base.clone(), cfg);
        let hits: usize = queries
            .iter()
            .zip(&truth)
            .map(|(q, gt)| {
                let got: HashSet<usize> = idx
                    .search(q, 10)
                    .unwrap()
                    .into_iter()
                    .filter_map(|r| r.id.parse().ok())
                    .collect();
                gt.intersection(&got).count()
            })
            .sum();
        let recall = hits as f32 / (queries.len() * 10) as f32;
        assert!(
            recall > 0.80,
            "{label} preset recall {recall:.2} too low to be working"
        );
    }
}

/// `reorder_for_locality` is a pure layout change: it must return **byte-identical search
/// results** (same ids, same scores, same order) as the source for every query, in every
/// storage. That is the whole contract — if a relabel is wrong it shows up here as a
/// changed result, not a crash. Also checks the permutation is a bijection (every id
/// still present exactly once).
#[test]
#[allow(deprecated)] // exercises Storage::TurboQuant, deprecated until 0.8 removal
fn reorder_for_locality_preserves_search_results() {
    let mut rng = StdRng::seed_from_u64(41);
    let dim = 80;
    let centers: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..500)
        .map(|i| {
            centers[i % 10]
                .iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            centers[i % 10]
                .iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        })
        .collect();

    for (storage, tb, rb) in [
        (Storage::F32, 2, 3),
        (Storage::SQ8, 2, 3),
        (Storage::TurboRabit, 2, 4),
        (Storage::TurboQuant, 3, 3),
    ] {
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
            storage,
            turbo_bits: tb,
            rabit_bits: rb,
            rerank_candidates: if storage == Storage::F32 { 0 } else { 50 },
            seed: Some(5),
            ..Default::default()
        };
        let src = HNSWIndex::build_parallel(base.clone(), config);
        let re = src.reorder_for_locality();

        assert_eq!(re.len(), src.len(), "{storage:?}: node count changed");
        // Bijection: the multiset of document ids is unchanged.
        let mut a = src.ids.clone();
        let mut b = re.ids.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{storage:?}: reorder is not a bijection over ids");

        // Byte-identical results per query — ids, scores, order.
        for q in &queries {
            let rs = src.search(q, 10).expect("src search");
            let rr = re.search(q, 10).expect("reordered search");
            let sv: Vec<(String, u32)> =
                rs.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
            let rv: Vec<(String, u32)> =
                rr.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
            assert_eq!(sv, rv, "{storage:?}: reorder changed a query's results");
        }
    }
}

/// `requantize` must preserve the graph EXACTLY (that is its whole claim) and produce a
/// working index in every target storage. Graph identity is asserted structurally —
/// layer-0 neighbour lists, upper layers, entry point — not via a recall proxy
/// (measure-the-output applies to the *quantizer*; the graph has an exact answer).
#[test]
#[allow(deprecated)] // exercises Storage::TurboQuant, deprecated until 0.8 removal
fn requantize_preserves_graph_and_searches_in_every_storage() {
    let mut rng = StdRng::seed_from_u64(31);
    let dim = 96;
    let centers: Vec<Vec<f32>> = (0..12)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..600)
        .map(|i| {
            centers[i % 12]
                .iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..40)
        .map(|i| {
            centers[i % 12]
                .iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        })
        .collect();

    let src_config = HNSWConfig {
        metric: DistanceMetric::Cosine,
        m: 16,
        m0: 32,
        ef_construction: 200,
        ef_search: 100,
        storage: Storage::F32,
        rerank_candidates: 0,
        seed: Some(3),
        ..Default::default()
    };
    let src = HNSWIndex::build_parallel(base.clone(), src_config.clone());

    // Exact cosine ground truth for a recall floor per target.
    let k = 10;
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-9)
    };
    let truth: Vec<HashSet<usize>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cos(v, q), i))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            s.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect();

    for (storage, tb, rb, floor) in [
        (Storage::SQ8, 2, 3, 0.85),
        (Storage::RaBitQ, 2, 3, 0.35), // 1-bit is legitimately coarse; floor is non-vacuous
        (Storage::TurboQuant, 3, 3, 0.55),
        (Storage::TurboRabit, 2, 3, 0.80),
    ] {
        let new_config = HNSWConfig {
            storage,
            turbo_bits: tb,
            rabit_bits: rb,
            rerank_candidates: 50,
            ..src_config.clone()
        };
        let re = src.requantize(new_config).expect("requantize");

        // Graph identity — exact, node by node.
        assert_eq!(re.len(), src.len());
        assert_eq!(
            re.entry_point, src.entry_point,
            "{storage:?}: entry point moved"
        );
        assert_eq!(re.max_layer, src.max_layer, "{storage:?}: max layer moved");
        for i in 0..src.len() {
            assert_eq!(
                re.get_neighbors_l0(i),
                src.get_neighbors_l0(i),
                "{storage:?}: node {i} layer-0 links differ"
            );
        }
        assert_eq!(
            re.connections, src.connections,
            "{storage:?}: upper layers differ"
        );

        // And it actually searches.
        let mut hits = 0usize;
        for (q, gt) in queries.iter().zip(&truth) {
            let got: HashSet<usize> = re
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            hits += gt.intersection(&got).count();
        }
        let recall = hits as f32 / (queries.len() * k) as f32;
        assert!(
            recall > floor,
            "{storage:?}: requantized recall {recall} below floor {floor}"
        );
    }
}

/// The contract errors: quantized source, and graph-relevant config changes.
#[test]
fn requantize_rejects_bad_inputs() {
    let mut rng = StdRng::seed_from_u64(32);
    let base: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..32).map(|_| rng.random::<f32>()).collect())
        .collect();
    let config = HNSWConfig {
        m: 8,
        m0: 16,
        seed: Some(1),
        ..Default::default()
    };
    let f32_idx = HNSWIndex::build_parallel(base.clone(), config.clone());

    // Non-F32 source.
    let sq8 = f32_idx
        .requantize(HNSWConfig {
            storage: Storage::SQ8,
            ..config.clone()
        })
        .expect("f32 -> sq8");
    assert!(
        sq8.requantize(HNSWConfig {
            storage: Storage::RaBitQ,
            ..config.clone()
        })
        .is_err(),
        "requantizing a quantized source must be rejected"
    );

    // Graph-relevant change.
    assert!(
        f32_idx
            .requantize(HNSWConfig {
                m: 12,
                storage: Storage::SQ8,
                ..config.clone()
            })
            .is_err(),
        "changing m must be rejected"
    );
}

/// The snapshot's whole claim is *verbatim*: the loaded index is bit-identical where the
/// JSON path is merely equivalent-ish (file.rs re-inserts through `add()`, so the parallel
/// builder hands back a different graph). Every config field is set off its default —
/// the save/load bug-class this guards against is a field silently dropped on one side
/// (the wasm path shipped exactly that: `turbo_bits` was never serialized).
#[test]
#[allow(deprecated)] // exercises Storage::TurboQuant, deprecated until 0.8 removal
fn snapshot_round_trip_is_verbatim_in_every_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 48;
    let base: Vec<Vec<f32>> = (0..400)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();

    for (label, storage) in [
        ("f32", Storage::F32),
        ("sq8", Storage::SQ8),
        ("rabitq", Storage::RaBitQ),
        ("turboquant", Storage::TurboQuant),
        ("turborabit", Storage::TurboRabit),
        // Warren is bulk-build-only (no incremental add), and this test adds a doc — it has
        // tested by the snapshot round-trip tests.
    ] {
        // Every field non-default, so a dropped field cannot hide behind its default.
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 12,
            m0: 24,
            ef_construction: 150,
            ef_search: 80,
            storage,
            turbo_bits: 4,
            rabit_bits: 2,
            rerank_candidates: if storage == Storage::F32 { 0 } else { 40 },
            seed: Some(9),
            ..Default::default()
        };
        let mut src = HNSWIndex::build_parallel(base.clone(), config);
        // One incremental add with metadata, so `metadata` round-trips something real.
        src.add(crate::Document {
            id: "meta-doc".into(),
            content: "has metadata".into(),
            embedding: base[0].clone(),
            metadata: Some(serde_json::json!({"k": 1})),
        })
        .expect("add");

        let path = dir.path().join(format!("{label}.snap"));
        src.snapshot_to_file(&path).expect("snapshot");
        let re = HNSWIndex::snapshot_from_file(&path).expect("load");

        // Verbatim: the arena and every sibling structure, bit for bit.
        assert_eq!(re.nodes, src.nodes, "{label}: arena differs");
        assert_eq!(
            re.connections, src.connections,
            "{label}: upper layers differ"
        );
        assert_eq!(re.stride, src.stride, "{label}: derived stride differs");
        assert_eq!(re.hdr, src.hdr, "{label}: derived hdr differs");
        assert_eq!(
            re.entry_point, src.entry_point,
            "{label}: entry point differs"
        );
        assert_eq!(re.max_layer, src.max_layer, "{label}: max layer differs");
        assert_eq!(re.q_min, src.q_min, "{label}: q_min differs");
        assert_eq!(re.q_scale, src.q_scale, "{label}: q_scale differs");
        assert_eq!(re.full, src.full, "{label}: full vectors differ");
        assert_eq!(re.ids, src.ids, "{label}: ids differ");
        assert_eq!(re.contents, src.contents, "{label}: contents differ");
        assert_eq!(re.metadata, src.metadata, "{label}: metadata differs");
        assert_eq!(re.embedding_dim, src.embedding_dim);

        // And behaviourally: identical results, scores included (same arena, same
        // codebooks, same kernels — any difference is a load bug, not noise).
        for q in &queries {
            let a = src.search(q, 10).expect("src search");
            let b = re.search(q, 10).expect("re search");
            let a: Vec<(String, u32)> = a.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
            let b: Vec<(String, u32)> = b.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
            assert_eq!(a, b, "{label}: search results differ after load");
        }
    }
}

/// A snapshot is a same-version cache: a stamp from any other version (or a truncated
/// arena) must refuse to load with a clear error, never misread.
#[test]
fn snapshot_rejects_version_mismatch_and_corruption() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rng = StdRng::seed_from_u64(78);
    let base: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..16).map(|_| rng.random::<f32>()).collect())
        .collect();
    let src = HNSWIndex::build_parallel(
        base,
        HNSWConfig {
            m: 8,
            m0: 16,
            seed: Some(1),
            ..Default::default()
        },
    );
    let path = dir.path().join("good.snap");
    src.snapshot_to_file(&path).expect("snapshot");

    let good = std::fs::read(&path).expect("read");
    let mut snap: HNSWSnapshot = bincode::deserialize(&good).expect("decode");

    // Wrong crate version.
    snap.crate_version = "0.0.0-other".into();
    let bad = dir.path().join("bad-version.snap");
    std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
    match HNSWIndex::snapshot_from_file(&bad) {
        Err(err) => assert!(
            err.to_string().contains("0.0.0-other"),
            "error should name the offending version, got: {err}"
        ),
        Ok(_) => panic!("wrong crate version must be rejected"),
    }

    // Wrong format version.
    snap.crate_version = env!("CARGO_PKG_VERSION").into();
    snap.format_version = SNAPSHOT_FORMAT_VERSION + 1;
    std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
    assert!(HNSWIndex::snapshot_from_file(&bad).is_err());

    // Truncated arena (valid bincode, wrong length for the config's stride).
    snap.format_version = SNAPSHOT_FORMAT_VERSION;
    snap.nodes.pop();
    std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
    assert!(
        HNSWIndex::snapshot_from_file(&bad).is_err(),
        "arena not a multiple of stride must be rejected"
    );

    // The untampered file still loads.
    assert!(HNSWIndex::snapshot_from_file(&path).is_ok());
}

#[test]
fn rabitq_recall_on_clustered_data_with_held_out_queries() {
    let mut rng = StdRng::seed_from_u64(2024);
    let dim = 32;
    let n_clusters = 16;
    let per_cluster = 50;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 20.0).collect())
        .collect();

    let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.8).collect()
        })
        .collect();

    // Held-out queries: fresh noise around the same centers, drawn from the same RNG
    // stream *after* all base vectors, so none of them coincides with a base vector.
    let n_queries = 60;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.8).collect()
        })
        .collect();

    let config = HNSWConfig {
        metric: DistanceMetric::L2,
        m: 16,
        m0: 32,
        ef_construction: 150,
        ef_search: 150,
        storage: Storage::RaBitQ,
        rerank_candidates: 50,
        seed: Some(7),
        ..Default::default()
    };
    let index = HNSWIndex::build_parallel(base.clone(), config);

    let k = 10;
    let mut total_recall = 0.0f32;
    for q in &queries {
        let mut exact: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, i)
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

        let got: HashSet<usize> = index
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();

        total_recall += truth.intersection(&got).count() as f32 / k as f32;
    }
    let recall = total_recall / n_queries as f32;

    // Measured on this exact seed/config at the time this test was written: 100%. (These
    // are well-separated Gaussian-ish blobs at ef_search=150 for n=800 — an easy corpus,
    // deliberately: the point of this test is to catch a broken *metric*, and a
    // discriminating-power check confirms it does. With the traversal kernel sabotaged to
    // return a constant (carrying zero information), recall on this exact test collapsed
    // to 7% — so the floor below is not a rubber stamp, and 0.75 leaves a wide margin
    // below the real 100% for run-to-run noise without coming anywhere near the ~7% a
    // broken kernel produces. See `lesson_untested_public_options`: a floor equal to the
    // measurement is not a regression test, it is a coin flip against float
    // non-determinism.
    assert!(
        recall > 0.75,
        "Storage::RaBitQ recall@{k} on clustered data = {:.1}% (held-out queries, \
             brute-force ground truth) — below floor, traversal metric likely broken",
        recall * 100.0,
    );
}

/// `rerank_candidates: 0` must work: codes-only, the cold `full` array dropped entirely,
/// and the estimate itself used as the final ranking with no exact-distance correction.
/// Must not panic even though `full` stays empty for the whole life of the index.
// `HNSWIndex::build(.., Storage::RaBitQ + rerank_candidates: 0)` used to PANIC IN RELEASE:
//   range start index 24 out of range for slice of length 0   (hnsw.rs, get_embedding)
//
// Both halves of that config are things the docs actively recommend: `rerank_candidates: 0`
// is the README's "smallest index foxstash can build", and `BuildStrategy::Sequential` is
// the #[default]. Nobody hit it only because every caller in the tree reached for
// `build_parallel`, which builds the graph from the caller's f32 slice and never reads a
// vector back. The one test covering this config hard-coded `build_parallel` and left a
// comment explaining that `insert_node` "assumes `full` is populated" — the bug was
// documented and walked around instead of fixed.
//
// This test goes through the PUBLIC `build`, on the DEFAULT strategy, for both quantized
// modes, and checks the memory promise too: dropping the vectors is the entire point of
// `rerank_candidates: 0`, so "it stopped panicking because we kept them" is not a fix.
#[test]
fn zero_rerank_quantized_builds_on_the_default_strategy_and_still_drops_its_vectors() {
    let mut rng = StdRng::seed_from_u64(11);
    let dim = 24;
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..320)
        .map(|i| {
            let c = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    // BOTH strategies, explicitly. `Sequential` is where the panic lived — it is the only
    // builder that reads vectors back out of storage mid-build. It used to be the #[default],
    // which is what made this a default-config crash; it no longer is (see `BuildStrategy`),
    // but it is still a supported public option, so it still must not blow up. Naming it
    // explicitly rather than leaning on the default also keeps this test from quietly
    // becoming a no-op the next time the default moves.
    for storage in [Storage::SQ8, Storage::RaBitQ] {
        for build_strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
            let config = HNSWConfig {
                metric: DistanceMetric::L2,
                storage,
                rerank_candidates: 0,
                seed: Some(4),
                build_strategy,
                ..Default::default()
            };

            let index = HNSWIndex::build(base.clone(), config);

            assert!(
                index.full.is_empty(),
                "{storage:?}/{build_strategy:?}: rerank_candidates = 0 must still DROP the f32 \
                     vectors — retaining them would 'fix' the panic by silently ignoring the \
                     caller's memory request"
            );
            assert_eq!(
                index.rerank_candidates(),
                0,
                "{storage:?}/{build_strategy:?}: the caller's rerank_candidates must be \
                     restored after the build"
            );

            // And the graph must actually work — a build that produces a valid-but-empty
            // index would pass every assertion above.
            let mut hits = 0;
            for (i, q) in base.iter().enumerate().step_by(17) {
                let got = index.search(q, 5).expect("search must not panic");
                assert_eq!(got.len(), 5);
                if got.iter().any(|r| r.id == i.to_string()) {
                    hits += 1;
                }
            }
            assert!(
                hits > 0,
                "{storage:?}/{build_strategy:?}: index returns results but finds nothing — \
                     graph is broken"
            );
        }
    }
}

// `set_rerank_candidates` exists so the rerank pool can be swept at search time, the way
// `set_ef_search` sweeps `ef` — the legacy `RaBitQHNSWIndex::search_and_rerank(q, pool, k)`
// took the pool per call, and that was the one capability `Storage::RaBitQ` lacked.
//
// The interesting half is the REFUSAL. `rerank_candidates: 0` discards the f32 vectors, so
// raising the pool afterwards has nothing to rescore against. Accepting it would silently
// return the coarse ranking — a knob that reports success and does nothing, which is the
// exact bug shape this codebase has now shipped ten times. So it must be an `Err`, and the
// test asserts the error rather than just "doesn't panic".
#[test]
fn raising_the_rerank_pool_on_an_index_that_dropped_its_vectors_is_an_error() {
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 24;
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..320)
        .map(|i| {
            let c = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    let cfg = |rerank| HNSWConfig {
        metric: DistanceMetric::L2,
        storage: Storage::RaBitQ,
        rerank_candidates: rerank,
        seed: Some(9),
        ..Default::default()
    };

    // Built WITHOUT the f32 vectors: raising the pool must be refused, not silently ignored.
    let mut dropped = HNSWIndex::build(base.clone(), cfg(0));
    assert!(
        matches!(
            dropped.set_rerank_candidates(64),
            Err(crate::RagError::FullPrecisionDropped)
        ),
        "raising the rerank pool on a vectors-dropped index must be an error"
    );
    assert_eq!(
        dropped.rerank_candidates(),
        0,
        "the refused set must not take effect"
    );
    // Lowering to 0 is always fine — nothing to rescore against is what it already wants.
    assert!(dropped.set_rerank_candidates(0).is_ok());

    // Built WITH the f32 vectors: the pool is a live search-time dial.
    let mut kept = HNSWIndex::build(base, cfg(100));
    assert!(kept.set_rerank_candidates(64).is_ok());
    assert_eq!(kept.rerank_candidates(), 64);
    assert!(kept.set_rerank_candidates(0).is_ok());
    assert_eq!(kept.rerank_candidates(), 0);
    // ...and back up again, because this index kept what it needs to honor that.
    assert!(kept.set_rerank_candidates(200).is_ok());
    assert_eq!(kept.rerank_candidates(), 200);
}

#[test]
fn rabitq_zero_rerank_drops_full_precision_vectors_and_does_not_panic() {
    let mut rng = StdRng::seed_from_u64(55);
    let dim = 24;
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..320)
        .map(|i| {
            let c = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    let config = HNSWConfig {
        metric: DistanceMetric::L2,
        storage: Storage::RaBitQ,
        rerank_candidates: 0,
        seed: Some(3),
        ..Default::default()
    };
    // `build_parallel` explicitly: it quantizes via `push_node` without ever calling
    // `get_embedding` on quantized storage mid-build (unlike `insert_node`, which
    // `Sequential` uses and which — like `Storage::SQ8` — assumes `full` is populated
    // whenever it needs a candidate's embedding for neighbour selection).
    let index = HNSWIndex::build_parallel(base.clone(), config);

    assert!(
        index.full.is_empty(),
        "rerank_candidates = 0 must drop the full-precision side array"
    );

    for q in base.iter().step_by(37) {
        let results = index.search(q, 5).expect("search must not panic");
        assert_eq!(results.len(), 5);
    }
}

// ========================================================================
// Quantized storage must honor `config.metric` (it silently ignored it and always
// computed L2, which under the DEFAULT metric — Cosine — meant the whole walk ran under
// the wrong metric and `score_from_distance` scored a squared-L2 value as if it were a
// bounded cosine distance).
// ========================================================================

/// Directions with per-point norms scaled 0.5x-50x apart. Without varying norms, every
/// point in a cluster has roughly the same magnitude and cosine/L2 rank near-identically
/// — this fixture exists specifically so a metric mix-up *must* change the answer. See
/// `cosine_and_l2_pick_different_neighbors` for the two-point version of the same idea;
/// this is its recall-scale generalisation.
fn nonuniform_norm_clusters(
    seed: u64,
    dim: usize,
    n_clusters: usize,
    per_cluster: usize,
    n_queries: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let directions: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            crate::vector::ops::normalize(&mut v);
            v
        })
        .collect();

    let make = |n: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let dir = &directions[i % n_clusters];
                // Small angular jitter so points sharing a direction aren't identical.
                let jittered: Vec<f32> =
                    dir.iter().map(|x| x + rng.random::<f32>() * 0.05).collect();
                // Wildly different magnitude per point — the part that makes cosine and
                // L2 disagree.
                let scale = 0.5 + rng.random::<f32>() * 49.5;
                jittered.into_iter().map(|x| x * scale).collect()
            })
            .collect()
    };

    let base = make(n_clusters * per_cluster, &mut rng);
    let queries = make(n_queries, &mut rng);
    (base, queries)
}

/// Prove the fixture above is discriminating *before* trusting any test built on it — an
/// assertion that can't fail proves nothing (the self-retrieval trap, generalized).
#[test]
fn nonuniform_norm_fixture_discriminates_cosine_from_l2() {
    let (base, queries) = nonuniform_norm_clusters(11, 16, 10, 30, 20);

    let build = |metric: DistanceMetric| {
        HNSWIndex::build_parallel(
            base.clone(),
            HNSWConfig {
                metric,
                ef_construction: 150,
                ef_search: 150,
                seed: Some(1),
                ..Default::default()
            },
        )
    };
    let cosine_idx = build(DistanceMetric::Cosine);
    let l2_idx = build(DistanceMetric::L2);

    let disagreements = queries
        .iter()
        .filter(|q| {
            let c = cosine_idx.search(q, 1).unwrap()[0].id.clone();
            let l = l2_idx.search(q, 1).unwrap()[0].id.clone();
            c != l
        })
        .count();

    assert!(
        disagreements * 2 >= queries.len(),
        "fixture is not discriminating: cosine and L2 only disagreed on {disagreements}/{} \
             queries — a metric mix-up test built on this fixture could pass by accident",
        queries.len()
    );
}

/// The regression test for the actual bug: `Storage::SQ8` with `..Default::default()` —
/// metric deliberately NOT spelled out, so this exercises the default (Cosine) path the
/// way a real caller who forgets to set `metric` would. Recall is measured against
/// brute-force COSINE ground truth on held-out queries; the old, broken code would have
/// silently run the walk under L2 instead, which — on this fixture, where cosine and L2
/// disagree on most queries — would collapse recall against the cosine ground truth.
#[test]
fn sq8_default_metric_is_cosine_not_l2() {
    let (base, queries) = nonuniform_norm_clusters(23, 24, 12, 30, 40);

    let config = HNSWConfig {
        storage: Storage::SQ8,
        rerank_candidates: 50,
        ef_construction: 150,
        ef_search: 150,
        seed: Some(2),
        ..Default::default() // metric: Cosine, the default — not spelled out on purpose
    };
    assert_eq!(
        config.metric,
        DistanceMetric::Cosine,
        "test setup sanity check"
    );
    let index = HNSWIndex::build_parallel(base.clone(), config);

    let k = 10;
    let mut cosine_recall = 0.0f32;
    let mut l2_recall = 0.0f32;
    for q in &queries {
        let mut by_cosine: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
            .collect();
        by_cosine.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cosine_truth: HashSet<usize> = by_cosine.iter().take(k).map(|(_, i)| *i).collect();

        let mut by_l2: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (crate::vector::simd::l2_squared_distance_simd(v, q), i))
            .collect();
        by_l2.sort_by(|a, b| a.0.total_cmp(&b.0));
        let l2_truth: HashSet<usize> = by_l2.iter().take(k).map(|(_, i)| *i).collect();

        let got: HashSet<usize> = index
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();

        cosine_recall += cosine_truth.intersection(&got).count() as f32 / k as f32;
        l2_recall += l2_truth.intersection(&got).count() as f32 / k as f32;
    }
    cosine_recall /= queries.len() as f32;
    l2_recall /= queries.len() as f32;

    assert!(
        cosine_recall > 0.6,
        "Storage::SQ8 with default metric: recall@{k} against COSINE ground truth = \
             {:.1}% — the default metric is meant to be cosine",
        cosine_recall * 100.0
    );
    assert!(
        cosine_recall > l2_recall + 0.2,
        "Storage::SQ8 with default metric answers cosine ({:.1}% recall) no better than \
             L2 ({:.1}% recall) — this is the exact shape of the metric-ignoring bug",
        cosine_recall * 100.0,
        l2_recall * 100.0
    );
}

/// Same regression test, `Storage::RaBitQ`.
#[test]
fn rabitq_default_metric_is_cosine_not_l2() {
    let (base, queries) = nonuniform_norm_clusters(29, 24, 12, 30, 40);

    let config = HNSWConfig {
        storage: Storage::RaBitQ,
        rerank_candidates: 50,
        ef_construction: 150,
        ef_search: 150,
        seed: Some(4),
        ..Default::default() // metric: Cosine, the default — not spelled out on purpose
    };
    assert_eq!(
        config.metric,
        DistanceMetric::Cosine,
        "test setup sanity check"
    );
    let index = HNSWIndex::build_parallel(base.clone(), config);

    let k = 10;
    let mut cosine_recall = 0.0f32;
    let mut l2_recall = 0.0f32;
    for q in &queries {
        let mut by_cosine: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
            .collect();
        by_cosine.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cosine_truth: HashSet<usize> = by_cosine.iter().take(k).map(|(_, i)| *i).collect();

        let mut by_l2: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (crate::vector::simd::l2_squared_distance_simd(v, q), i))
            .collect();
        by_l2.sort_by(|a, b| a.0.total_cmp(&b.0));
        let l2_truth: HashSet<usize> = by_l2.iter().take(k).map(|(_, i)| *i).collect();

        let got: HashSet<usize> = index
            .search(q, k)
            .expect("search")
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();

        cosine_recall += cosine_truth.intersection(&got).count() as f32 / k as f32;
        l2_recall += l2_truth.intersection(&got).count() as f32 / k as f32;
    }
    cosine_recall /= queries.len() as f32;
    l2_recall /= queries.len() as f32;

    assert!(
        cosine_recall > 0.5,
        "Storage::RaBitQ with default metric: recall@{k} against COSINE ground truth = \
             {:.1}%",
        cosine_recall * 100.0
    );
    assert!(
        cosine_recall > l2_recall + 0.2,
        "Storage::RaBitQ with default metric answers cosine ({:.1}% recall) no better \
             than L2 ({:.1}% recall) — this is the exact shape of the metric-ignoring bug",
        cosine_recall * 100.0,
        l2_recall * 100.0
    );
}

/// The old code fed a squared-L2 value (unbounded, frequently large) into
/// `score_from_distance`'s `1.0 - dist` cosine formula, producing large negative scores.
/// Under a correctly metric-aware SQ8, distances are true (rescored, exact) cosine
/// distances in `[0, 2]`, so scores (`1 - dist`) must land in `[-1, 1]` and decrease
/// monotonically as the true angle widens.
#[test]
fn sq8_cosine_scores_are_bounded_and_monotonic() {
    // Same axis, increasingly different directions; magnitudes deliberately unequal so
    // an L2-in-disguise bug (which cares about magnitude) would rank these differently
    // than a genuine cosine metric (which does not).
    let query = vec![10.0, 0.0, 0.0, 0.0];
    let vectors = [
        ("same_dir", vec![500.0, 0.0, 0.0, 0.0]), // identical direction, huge magnitude
        ("close_dir", vec![8.0, 2.0, 0.0, 0.0]),
        ("far_dir", vec![2.0, 8.0, 0.0, 0.0]),
        ("opposite_dir", vec![-30.0, 0.0, 0.0, 0.0]),
    ];

    let mut index = HNSWIndex::new(
        4,
        HNSWConfig {
            storage: Storage::SQ8,
            rerank_candidates: 100,
            ..Default::default() // metric: Cosine
        },
    );
    index
        .train(&vectors.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>())
        .unwrap();
    for (id, v) in &vectors {
        index.add_embedding((*id).to_string(), v.clone()).unwrap();
    }

    let results = index.search(&query, vectors.len()).unwrap();
    assert_eq!(results.len(), vectors.len());
    for r in &results {
        assert!(
            (-1.0..=1.0).contains(&r.score),
            "SQ8 cosine score {} for {} outside [-1, 1] — the metric-ignoring bug fed a \
                 squared-L2 value into the cosine score formula",
            r.score,
            r.id
        );
    }
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores must be descending: {:?}",
            results
        );
    }
    assert_eq!(
        results[0].id, "same_dir",
        "identical direction must rank first under cosine"
    );
    assert_eq!(
        results.last().unwrap().id,
        "opposite_dir",
        "opposite direction must rank last under cosine"
    );
}

// ========================================================================
// Discriminating tests for options flagged VACUOUS/UNCOVERED in the public-option audit:
// each one had a test that set the field without any assertion able to tell whether it was
// actually read. Every test below states, in its doc comment, the specific sabotage it
// would catch ("if I hardcoded this to its default and deleted the config read, would this
// fail?"), per the standard the rest of this module already holds itself to.
//
// NOT COMPILED. Written and reasoned through by hand while a benchmark held the CPU; the
// team lead will compile and sabotage-verify these directly. Where a test's margin depends
// on empirical behavior I could not run (rather than being guaranteed by construction), the
// doc comment says so.
// ========================================================================

/// `ef_search` must bound how many candidates the layer-0 walk explores.
///
/// Sabotage this catches: hardcode `ef` in `search_inner` to a fixed value (e.g.
/// `k.max(100)`) instead of reading `self.config.ef_search`. `distance_calls()` would then
/// stay flat no matter what a caller sets `ef_search` to, because the real code never
/// explores more than the hardcoded constant.
#[test]
fn ef_search_controls_distance_calls() {
    let mut rng = StdRng::seed_from_u64(9001);
    let centers: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..32).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..2000)
        .map(|i| {
            let c = &centers[i % 20];
            c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
        })
        .collect();

    let mut index = HNSWIndex::build(embeddings.clone(), HNSWConfig::default().with_seed(7));
    let query = embeddings[0].clone();

    let calls_at = |index: &mut HNSWIndex, ef: usize| -> u64 {
        index.set_ef_search(ef);
        let mut searcher = index.searcher();
        searcher.search(&query, 10).unwrap();
        searcher.distance_calls()
    };

    let low = calls_at(&mut index, 10);
    let high = calls_at(&mut index, 800);

    assert!(
        high > low,
        "ef_search has no effect on work done: {high} distance calls at ef=800 vs {low} at \
             ef=10 — ef_search is being ignored"
    );
}

/// `ef_construction` must bound the candidate pool used while building each node's edges. A
/// starved pool at build time produces a measurably worse GRAPH.
///
/// Sabotage this catches: hardcode `ef_construction` in `insert_node` to a fixed value
/// instead of reading `self.config.ef_construction`. Both configs below would then build the
/// identical graph and score identical recall, regardless of which value a caller set.
///
/// # The first version of this test was wrong, and wrong in an instructive way
///
/// It held `ef_search: 300` at query time, with a comment claiming that "isolates the
/// build-time effect". It does the **opposite**. A large search-time `ef` explores most of
/// the corpus regardless of how the graph is wired, which *compensates for a bad graph* and
/// masks the very thing under test. On 320 vectors with `ef_search: 300` the search is
/// nearly exhaustive, so a graph built with `ef_construction: 1` still scored **98.7%** —
/// the test failed, and it deserved to.
///
/// To see graph quality you must make the search *depend* on it: a small `ef_search`, on a
/// corpus too large to sweep by brute force. Then a badly-linked graph has nowhere to hide.
/// The threshold below is unchanged from the original — the fixture was hardened rather than
/// the assertion weakened, which is the rule whenever a test like this comes back red.
#[test]
fn ef_construction_controls_graph_quality() {
    let mut rng = StdRng::seed_from_u64(3113);
    let centers: Vec<Vec<f32>> = (0..16)
        .map(|_| (0..24).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..800)
        .map(|i| {
            let c = &centers[i % 16];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..60)
        .map(|i| {
            let c = &centers[i % 16];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    let recall_for = |ef_construction: usize| -> f32 {
        let index = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                m: 8,
                m0: 16,
                ef_construction,
                // SMALL, deliberately. A generous `ef_search` papers over a badly-linked
                // graph by exploring everything anyway — which is how the first version of
                // this test scored 98.7% on a graph built with ef_construction = 1.
                ef_search: 12,
                seed: Some(11),
                build_strategy: BuildStrategy::Sequential,
                ..Default::default()
            },
        );
        let k = 10;
        let mut total = 0.0f32;
        for q in &queries {
            let mut exact: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            total += truth.intersection(&got).count() as f32 / k as f32;
        }
        total / queries.len() as f32
    };

    let starved = recall_for(1);
    let generous = recall_for(200);

    assert!(
        generous > starved + 0.1,
        "ef_construction has no measurable effect on graph quality: recall@10 = {:.3} at \
             ef_construction=1 vs {:.3} at ef_construction=200 — ef_construction is being \
             ignored at build time",
        starved,
        generous
    );
}

/// `use_heuristic` must select which neighbour-selection algorithm actually runs: Algorithm
/// 4's diversity heuristic (default) vs plain nearest-M. Whitebox test of `select_neighbors`
/// directly — the two algorithms are *proven* to disagree on this exact fixture by hand
/// below, so there is no fixture-sensitivity risk the way an end-to-end recall test has.
///
/// Fixture (2-D, `DistanceMetric::L2`, query at the origin):
///   A = (1.00, 0.0)  dist to query = 1.00
///   B = (1.05, 0.0)  dist to query = 1.05, dist to A  = 0.05
///   C = (0.00, 1.2)  dist to query = 1.20, dist to A  = 1.562
///
/// Nearest-2 by raw distance: {A, B} — 1.00 and 1.05 both beat 1.20.
/// Algorithm 4 with m=2: A is accepted first (always is). B is checked against A:
/// dist(B,A)=0.05 < dist(B,query)=1.05, so B is "behind" A and pruned. C is checked against
/// A: dist(C,A)=1.562 is NOT less than dist(C,query)=1.20, so C is accepted. Heuristic result:
/// {A, C}.
///
/// Sabotage this catches: hardcode `use_heuristic` to `true` (delete the `if
/// !self.config.use_heuristic` early return in `select_neighbors`) — the `false` config
/// below would then also return {A, C} instead of {A, B}.
#[test]
fn use_heuristic_selects_a_different_neighbor_set_than_simple() {
    let a = [1.0f32, 0.0];
    let b = [1.05f32, 0.0];
    let c = [0.0f32, 1.2];
    let query = [0.0f32, 0.0];

    let build_index = |use_heuristic: bool| -> HNSWIndex {
        let config = HNSWConfig {
            metric: DistanceMetric::L2,
            use_heuristic,
            extend_candidates: false,
            ..Default::default()
        };
        let mut index = HNSWIndex::new(2, config);
        index.push_node(&a); // id 0
        index.push_node(&b); // id 1
        index.push_node(&c); // id 2
        index
    };

    let heuristic_selected: HashSet<usize> = build_index(true)
        .select_neighbors(&[0, 1, 2], &query, 2, 0)
        .into_iter()
        .collect();
    let simple_selected: HashSet<usize> = build_index(false)
        .select_neighbors(&[0, 1, 2], &query, 2, 0)
        .into_iter()
        .collect();

    assert_eq!(
        heuristic_selected,
        HashSet::from([0, 2]),
        "Algorithm-4 heuristic should pick the diverse pair {{A, C}}, got \
             {heuristic_selected:?}"
    );
    assert_eq!(
        simple_selected,
        HashSet::from([0, 1]),
        "simple selection should pick the two nearest {{A, B}}, got {simple_selected:?}"
    );
    assert_ne!(
        heuristic_selected, simple_selected,
        "use_heuristic has no effect: both configs picked the same neighbours — \
             use_heuristic is being ignored"
    );
}

/// `extend_candidates` must broaden the pool `select_neighbors`'s heuristic prunes from —
/// pulling in each direct candidate's own layer-0 neighbours before scoring. Whitebox again:
/// D is the only member of `candidates`; a strictly-second point E is reachable *exclusively*
/// through D's layer-0 neighbour list, never passed to `select_neighbors` directly. Without
/// `extend_candidates`, `select_neighbors` cannot see E at all. With it, D's neighbour list
/// is walked and E enters the working pool before pruning.
///
/// `keep_pruned_connections: true` (the default) is held fixed in both configs so the size
/// difference below is attributable only to `extend_candidates`, not to whether pruned
/// candidates get backfilled.
///
/// Sabotage this catches: hardcode `extend_candidates` to `false` (delete the `if
/// self.config.extend_candidates` block in `select_neighbors`, or make it a no-op) — the
/// `true` config below would then also return only `{D}`, size 1, instead of `{D, E}`, size 2.
#[test]
fn extend_candidates_pulls_in_neighbors_of_candidates() {
    let d = [1.0f32, 0.0];
    let e = [2.0f32, 0.0];
    let query = [0.0f32, 0.0];

    let build_index = |extend_candidates: bool| -> HNSWIndex {
        let config = HNSWConfig {
            metric: DistanceMetric::L2,
            use_heuristic: true,
            extend_candidates,
            keep_pruned_connections: true,
            m0: 4,
            ..Default::default()
        };
        let mut index = HNSWIndex::new(2, config);
        index.push_node(&d); // id 0
        index.push_node(&e); // id 1
                             // D's only layer-0 neighbour is E. `candidates` passed to `select_neighbors` below
                             // is `[0]` (D) only — E is reachable exclusively by walking this link, which only
                             // happens when `extend_candidates` is set.
        index.l0_push(0, 1);
        index
    };

    let extended = build_index(true).select_neighbors(&[0], &query, 2, 0);
    let not_extended = build_index(false).select_neighbors(&[0], &query, 2, 0);

    assert_eq!(
        not_extended.len(),
        1,
        "without extend_candidates, only the directly-passed candidate D can be selected, \
             got {not_extended:?}"
    );
    assert_eq!(
        extended.len(),
        2,
        "extend_candidates has no effect: expected D's neighbour E to be pulled into the \
             pool and selected alongside D, got {extended:?} — extend_candidates is being \
             ignored"
    );
}

/// `HNSWConfig::m` must bound the number of edges kept per node at layers >= 1, independent
/// of `m0` (which bounds layer 0 only — see `keep_pruned_connections_controls_graph_density_
/// in_both_builders` for that one). `m0`, `ml`, `ef_construction` and `seed` are held
/// IDENTICAL between the two configs below; only `m` differs, so any difference in average
/// layer-1 degree is attributable to `m` alone. `ml` is fixed at an unusually high 0.8
/// (rather than the typical `1/ln(m)`) purely to get enough nodes above layer 0 to average
/// over; sharing both `seed` and `ml` means the two builds assign the *same* set of nodes to
/// layer >= 1, so the population being averaged over is identical too.
///
/// # This test was VACUOUS in its first form, and the way it failed is worth keeping
///
/// It originally compared the *average* layer-1 degree at `m=4` vs `m=32` and asserted the
/// wide one was 2x the narrow one. But `config.m` is read in TWO places during insertion:
/// the cap on the new node's own edge count, and the pruning of each existing neighbour's
/// edge list. Sabotaging *either one alone* left the other to spread the averages apart, so
/// the test still passed. It only failed if you broke **both**. That is an OR, not an AND —
/// and a real regression hardcodes ONE site. The test would have sailed straight past the
/// bug it was written to catch. (Verified: hardcoding each site in turn → still green.)
///
/// So don't measure a statistic; assert the **invariant**. With `m = 4`, no node above layer
/// 0 may hold more than 4 neighbours — *ever*. Either read site breaking that produces an
/// over-degree node, and there is nowhere for it to hide in an average.
///
/// Sabotage this now catches: hardcode EITHER `self.config.m` site in `insert_node` (the
/// select-neighbours cap, or `neighbor_m` in the prune step) to a constant.
///
/// The control comes first, as always: assert the cap actually BINDS on this fixture (some
/// node genuinely reaches `m` neighbours). If nothing ever reaches the cap, "no node exceeds
/// the cap" is trivially true and proves nothing — that is exactly how the `ml: 0.8` fixture
/// hid this, by promoting too few nodes for any degree limit to matter.
/// `use_heuristic` and `extend_candidates` must be honoured by **both** builders.
///
/// They were not. `par_select_heuristic` took only `(metric, sorted, points, m, keep_pruned)`
/// and the entire parallel build path mentioned these two options **zero times** — so
/// `use_heuristic: false` / `extend_candidates: true` were silently ignored on the DEFAULT
/// builder. The pre-existing tests missed it because they call `select_neighbors` directly:
/// the *sequential* path. **An option tested against one implementation of a strategy is not a
/// tested option.**
///
/// # This test needs a NOISE FLOOR, and the first version of it did not have one
///
/// My first attempt compared "graph with the option on" against "graph with it off" and
/// asserted they differed on >5% of nodes. It passed under sabotage. The reason is its own
/// bug (see `seed`): **the parallel builder is not reproducible even at a fixed seed** —
/// threads race to write neighbour lists, and two builds of the *identical* config differ on
/// ~15% of nodes. The test was measuring the race, not the option.
///
/// So the control comes first and it is not optional: build the SAME config twice, measure how
/// much it varies from thread scheduling alone, and only then require the option's effect to
/// exceed that floor by a clear margin. A difference smaller than the noise is not evidence.
/// `build_parallel` shuffles its insertion order. The ids it hands back must still be the
/// caller's ORIGINAL row indices.
///
/// This is load-bearing far outside this crate. The Python binding maps a `SearchResult` back
/// to a row of the `X` numpy handed us by parsing `r.id` as an integer — so if the shuffle
/// leaked, every recall number the binding reported would be scored against the wrong
/// ground-truth rows. It would not crash. It would not look wrong. It would be fiction, and we
/// would publish it. The binding asserted this in a doc comment, and a doc comment cannot fail.
///
/// It asserts the MAPPING, not the search. The first version of this test queried each row
/// with its own vector and demanded itself back at k=1 — and failed, on a correct index, at
/// the default `ef_search`, because an approximate index is allowed to miss. Which is to say
/// it was testing recall while claiming to test identity. The direct check below cannot be
/// confused by search quality: node `j` stores the vector of the row whose id it claims.
#[test]
fn build_parallel_returns_original_row_indices_despite_its_shuffle() {
    let n = 300;
    let base: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; 32];
            v[i % 32] = 1.0 + i as f32;
            v[(i * 7 + 3) % 32] = 0.5 + (i % 13) as f32;
            v
        })
        .collect();

    let ix = HNSWIndex::build(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            seed: Some(4),
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );
    assert_eq!(ix.len(), n, "the build dropped or duplicated rows");

    for j in 0..ix.len() {
        let claimed: usize = ix.ids[j]
            .parse()
            .expect("build_parallel labels every node with its original row index");
        assert_eq!(
            ix.get_embedding(j),
            base[claimed].as_slice(),
            "node {j} claims to be row {claimed}, but the vector it stores is not row \
                 {claimed}'s. build_parallel's insertion shuffle has leaked into the ids it \
                 returns, so every id this index reports is a permutation of the caller's rows."
        );
    }

    // End to end, through the public API -- but on the SEQUENTIAL builder.
    //
    // This half of the test was flaky and it was my own fault. It ran on the default parallel
    // builder and demanded, for each probe row, that the index return that exact row at k=1.
    // The parallel builder is not reproducible (see
    // `seed_gives_reproducible_builds_only_on_the_sequential_builder`), so whether any given
    // node lands well-connected varies run to run, and an approximate index is entitled to
    // miss one. It passed for a while and then failed on an unrelated commit, which is the
    // worst way for a test to spend its time.
    //
    // An exact assertion needs a deterministic build. The mapping claim itself is already
    // proven exhaustively above, for all n nodes, without going through search at all.
    let seq = HNSWIndex::build(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            seed: Some(4),
            ef_search: 300,
            build_strategy: BuildStrategy::Sequential,
            ..Default::default()
        },
    );
    for i in [0, 7, 42, 199, 292, n - 1] {
        assert_eq!(
            seq.search(&base[i], 1).unwrap()[0].id,
            i.to_string(),
            "row {i} queried with its own vector did not come back as itself"
        );
    }
}

/// The two builders must produce the SAME GRAPH, to within the parallel builder's thread noise.
///
/// This is the guard on the bug class, rather than on any one bug. Both builders now call one
/// [`select_neighbors_core`], so an option cannot be honoured by one and ignored by the other —
/// but they still adapt it to different graph storage, and an adapter can lie. This test fails
/// the moment those adapters disagree about anything: a dropped option, a wrong distance, a
/// different pruning rule.
///
/// It is the test that would have caught, in one shot, every builder bug in the 1.0 audit:
///
/// | bug | the gap it opened |
/// |---|---|
/// | parallel ignored `m`/`m0` | degree pinned at the capacity constant |
/// | parallel ignored `use_heuristic` | 16.00 vs 8.11 |
/// | parallel ignored `extend_candidates` | 7.99 vs 9.06 |
/// | parallel scored extended candidates against the wrong point | 10.58 vs 9.06 |
///
/// The last of those was introduced *while fixing* the third, and shipped for three commits
/// under a test that watched degree move and concluded the option worked. Degree moved. It
/// moved to the wrong number.
#[test]
fn both_builders_produce_the_same_graph() {
    let mut rng = StdRng::seed_from_u64(31337);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..600)
        .map(|i| {
            let c: &Vec<f32> = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
        })
        .collect();

    let avg_degree = |strategy: BuildStrategy, heuristic: bool, extend: bool| -> f64 {
        let reps = 3;
        (0..reps)
            .map(|r| {
                let ix = HNSWIndex::build(
                    base.clone(),
                    HNSWConfig {
                        metric: DistanceMetric::L2,
                        m: 8,
                        m0: 16,
                        ef_construction: 16,
                        seed: Some(21 + r),
                        use_heuristic: heuristic,
                        extend_candidates: extend,
                        keep_pruned_connections: false,
                        build_strategy: strategy,
                        ..Default::default()
                    },
                );
                let total: usize = (0..ix.len()).map(|i| ix.get_neighbors_l0(i).len()).sum();
                total as f64 / ix.len() as f64
            })
            .sum::<f64>()
            / reps as f64
    };

    for (heuristic, extend) in [(false, false), (true, false), (true, true)] {
        let par = avg_degree(BuildStrategy::Parallel, heuristic, extend);
        let seq = avg_degree(BuildStrategy::Sequential, heuristic, extend);
        assert!(
            (par - seq).abs() < 0.5,
            "use_heuristic={heuristic} extend_candidates={extend}: the parallel builder \
                 produced average degree {par:.2} and the sequential one {seq:.2}. They run the \
                 same algorithm on the same seed, so a gap this size means an adapter is dropping \
                 an option or computing a distance against the wrong point."
        );
    }
}

/// Degree equivalence (above) is a PROXY for graph quality; recall at fixed `ef_search` IS the
/// quality. The two builders once matched on layer-0 degree (64 vs 64) while the parallel graph
/// recalled ~15 points BELOW the sequential one — because `par_insert` truncated the candidate
/// beam to `m0` before the diversity heuristic, so the heuristic picked `m0` from `m0` and could
/// not bridge clusters. The degree test could never see it. This one measures the output.
///
/// If the parallel builder ever again feeds its heuristic a truncated pool (or otherwise builds
/// a worse graph), parallel recall drops below sequential and the parity assertion fails. The
/// floor assertion keeps it from passing vacuously by comparing two equally-broken indexes.
#[test]
fn both_builders_reach_similar_recall() {
    // Fixture chosen to EXPOSE the failure mode, not merely to run: UNIT-NORMALISED vectors
    // with gaussian noise, so the clusters sit close on the sphere and a query's true top-10
    // spans cluster boundaries — which the diversity heuristic must bridge. Queried at a modest
    // ef_search where a bridge-poor graph visibly loses recall. Verified by sabotage: truncating
    // the candidate pool to m0 before the heuristic drops parallel recall ~15 points here and
    // fails the parity assertion below. Well-separated clusters (uniform centers, tiny noise)
    // give BOTH builders 100% and hide the gap — the classic wrong fixture (benchmarking-traps /
    // fixture-per-failure-mode lessons).
    let mut rng = StdRng::seed_from_u64(0xEF54);
    let dim = 128;
    let norm = |v: &mut Vec<f32>| {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
    };
    // Box-Muller standard normal from the StdRng uniform stream.
    let mut gauss = |rng: &mut StdRng| -> f32 {
        let u1 = (rng.random::<f32>()).max(1e-7);
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    };
    let unit = |rng: &mut StdRng| -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
        norm(&mut v);
        v
    };
    let sample_from = |rng: &mut StdRng,
                       centers: &[Vec<f32>],
                       gauss: &mut dyn FnMut(&mut StdRng) -> f32|
     -> Vec<f32> {
        let c = &centers[rng.random::<u64>() as usize % centers.len()];
        let mut v: Vec<f32> = c.iter().map(|x| x + 0.05 * gauss(rng)).collect();
        norm(&mut v);
        v
    };
    // Base and queries are drawn from INDEPENDENT centre sets, so a query's true top-10 is not
    // trivially its own dense cluster — it requires real cross-cluster search, which is what
    // stresses the graph's long-range bridges. (Queries from the SAME centres as the base give
    // both builders ~100% and hide the gap.)
    let base_centers: Vec<Vec<f32>> = (0..100).map(|_| unit(&mut rng)).collect();
    let query_centers: Vec<Vec<f32>> = (0..100).map(|_| unit(&mut rng)).collect();
    let base: Vec<Vec<f32>> = (0..10_000)
        .map(|_| sample_from(&mut rng, &base_centers, &mut gauss))
        .collect();
    let queries: Vec<Vec<f32>> = (0..200)
        .map(|_| sample_from(&mut rng, &query_centers, &mut gauss))
        .collect();

    const K: usize = 10;
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut d: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    (
                        q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum::<f32>(),
                        j,
                    )
                })
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            d.iter().take(K).map(|(_, j)| *j).collect()
        })
        .collect();

    let recall_of = |strategy: BuildStrategy| -> f32 {
        let mut ix = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 32,
                m0: 64,
                ef_construction: 200,
                ef_search: 40,
                seed: Some(7),
                build_strategy: strategy,
                ..Default::default()
            },
        );
        ix.set_ef_search(40);
        let mut hit = 0.0f32;
        for (qi, q) in queries.iter().enumerate() {
            let got: std::collections::HashSet<usize> = ix
                .search(q, K)
                .unwrap()
                .iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            hit += truth[qi].iter().filter(|t| got.contains(t)).count() as f32 / K as f32;
        }
        hit / queries.len() as f32
    };

    let seq = recall_of(BuildStrategy::Sequential);
    let par = recall_of(BuildStrategy::Parallel);

    // Non-vacuous: both builders must actually work on this clustered data.
    assert!(
        seq > 0.65 && par > 0.55,
        "recall floor not met (seq={seq:.3}, par={par:.3}) — the test is comparing broken \
             indexes and would pass vacuously"
    );
    // The point of the test: the default (parallel) builder must not ship a materially worse
    // graph than the sequential one. 5 points covers the parallel builder's non-reproducibility.
    assert!(
        seq - par < 0.05,
        "parallel recall {par:.3} trails sequential {seq:.3} by {:.3} at equal ef_search — the \
             parallel builder is producing a worse graph (last time: it truncated the candidate \
             pool to m0 before the diversity heuristic).",
        seq - par
    );
}

#[test]
fn use_heuristic_and_extend_candidates_are_honoured_by_both_builders() {
    let mut rng = StdRng::seed_from_u64(31337);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..600)
        .map(|i| {
            let c: &Vec<f32> = &centers[i % 8];
            c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
        })
        .collect();

    const M0: usize = 16;
    // Averaged over builds, so that thread scheduling shows up as a small wobble in the mean
    // rather than as a signal. `keep_pruned_connections: false` is essential: the backfill
    // exists precisely to refill the slots the heuristic emptied, and would erase the effect.
    let avg_degree = |heuristic: bool, extend: bool, strategy: BuildStrategy| -> f64 {
        // 6, not 3: this averages away the parallel builder's thread-scheduling wobble, and a
        // CI runner with a different core count than the dev box wobbles differently. More
        // reps pull each mean closer to its true expectation, so the effect sizes below clear
        // their margins on every platform (this test was flaky on Windows/macOS at reps=3).
        let reps = 6;
        (0..reps)
            .map(|r| {
                let ix = HNSWIndex::build(
                    base.clone(),
                    HNSWConfig {
                        metric: DistanceMetric::L2,
                        m: 8,
                        m0: M0,
                        ef_construction: 16,
                        seed: Some(21 + r),
                        use_heuristic: heuristic,
                        extend_candidates: extend,
                        keep_pruned_connections: false,
                        build_strategy: strategy,
                        ..Default::default()
                    },
                );
                let total: usize = (0..ix.len()).map(|i| ix.get_neighbors_l0(i).len()).sum();
                total as f64 / ix.len() as f64
            })
            .sum::<f64>()
            / reps as f64
    };

    for strategy in [BuildStrategy::Parallel, BuildStrategy::Sequential] {
        let greedy = avg_degree(false, false, strategy);
        let heuristic = avg_degree(true, false, strategy);
        let extended = avg_degree(true, true, strategy);

        // The noise floor, measured on the same statistic the assertions use. The parallel
        // builder is not reproducible (see `seed_gives_reproducible_builds_only_on_the_...`),
        // so an effect must be shown to exceed the wobble, not merely to exist.
        //
        // A SINGLE resample diff is itself one draw from that wobble — on a differently-
        // scheduled runner it can land high and break a `noise * 10` guard calibrated on the
        // dev box (the Windows/macOS flake). Average several independent same-config pairs so
        // the floor is a stable statistic. Sequential is deterministic, so its floor is 0.
        let noise = {
            let pairs = 5;
            (0..pairs)
                .map(|_| {
                    (avg_degree(true, false, strategy) - avg_degree(true, false, strategy)).abs()
                })
                .sum::<f64>()
                / pairs as f64
        };

        // Greedy selection takes the m0 nearest candidates and fills every slot.
        assert!(
            (greedy - M0 as f64).abs() < 0.01,
            "{strategy:?}: use_heuristic=false must keep the m0 nearest candidates, filling \
                 every slot, but average degree was {greedy:.2} of a possible {M0}"
        );
        // The heuristic rejects any candidate that lies closer to an already-accepted
        // neighbour than to the query, which empties slots that greedy would have filled.
        // Measured ~8/16; require a margin far above `noise` (~0.05).
        assert!(
            heuristic < 0.75 * M0 as f64 && (greedy - heuristic) > noise * 10.0,
            "{strategy:?}: use_heuristic=true must prune candidates that hide behind an \
                 accepted neighbour, dropping degree below m0={M0}, but degree was \
                 {heuristic:.2} vs greedy's {greedy:.2} (build-to-build noise {noise:.3}) — the \
                 flag is not reaching this builder"
        );
        // Extending the candidate set with the neighbours-of-neighbours gives the heuristic
        // more diverse candidates to accept, so degree climbs back. Measured +12% (sequential)
        // to +33% (parallel).
        assert!(
            extended > heuristic * 1.05 && (extended - heuristic) > noise * 10.0,
            "{strategy:?}: extend_candidates=true must widen the candidate pool and let the \
                 heuristic accept more of it, but degree was {extended:.2} vs {heuristic:.2} \
                 without it (build-to-build noise {noise:.3}) — the flag is not reaching this \
                 builder"
        );
    }
}

/// `seed` promises reproducible builds. **It does not deliver them on the default builder.**
///
/// Measured: two `BuildStrategy::Parallel` builds at the same seed differ on ~15% of nodes;
/// `Sequential` is bit-identical. Threads race to write neighbour lists through the `RwLock`s,
/// and the seed fixes the level assignments and insertion order — not the interleaving.
///
/// The pre-existing `seed_makes_builds_reproducible_and_distinguishable` test pinned
/// `Sequential` — the path where the promise happens to hold — so it passed. Third instance of
/// the same pattern today, after `m`/`m0` and `use_heuristic`/`extend_candidates`.
///
/// This test pins the ACTUAL guarantee rather than the one the docs used to imply, so that the
/// day someone makes the parallel build deterministic, it fails and tells them to update the
/// contract. A test that encodes a lie is worse than no test; a test that encodes the truth,
/// including an unwelcome truth, is a spec.
#[test]
fn seed_gives_reproducible_builds_only_on_the_sequential_builder() {
    let base: Vec<Vec<f32>> = (0..600)
        .map(|i| {
            (0..16)
                .map(|d| (((i * 7 + d * 13) % 50) as f32) * 0.1 + (i % 8) as f32)
                .collect()
        })
        .collect();

    let graph_of = |strategy: BuildStrategy| -> Vec<Vec<u32>> {
        let ix = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 8,
                m0: 16,
                ef_construction: 100,
                seed: Some(21),
                build_strategy: strategy,
                ..Default::default()
            },
        );
        (0..ix.len())
            .map(|i| {
                let mut n: Vec<u32> = ix.get_neighbors_l0(i).to_vec();
                n.sort_unstable();
                n
            })
            .collect()
    };

    let seq_a = graph_of(BuildStrategy::Sequential);
    let seq_b = graph_of(BuildStrategy::Sequential);
    assert_eq!(
        seq_a, seq_b,
        "Sequential + a fixed seed must be bit-reproducible — that is the whole contract of \
             `seed`, and it is the only builder that honours it"
    );

    // And the unwelcome half of the truth, pinned so it cannot rot silently.
    let par_a = graph_of(BuildStrategy::Parallel);
    let par_b = graph_of(BuildStrategy::Parallel);
    let differing = par_a.iter().zip(&par_b).filter(|(x, y)| x != y).count();
    assert!(
        differing > 0,
        "the parallel builder has become reproducible under a fixed seed ({differing} nodes \
             differ). That is GOOD — but `HNSWConfig::seed`'s documentation says it is not, and \
             this test exists to catch that contract changing. Update the docs and this assertion."
    );
}

/// The sibling of `m_caps_upper_layer_degree_independent_of_m0`, and the test that should
/// have existed first.
///
/// That test pinned `BuildStrategy::Sequential`. The PARALLEL builder — now the default —
/// ignored `config.m` and `config.m0` **entirely**: it passed the hardcoded `M0_MAX`/`M_MAX`
/// constants everywhere the config values belonged, built a degree-64 graph no matter what
/// you asked for, and let the conversion clean up afterwards. Build cost was FLAT at ~61,500
/// distance computations per insert across m0 = 24/32/48/64 — it was doing the m0=64 build
/// every single time.
///
/// I wrote the Sequential test and then made the untested builder the default. **Testing an
/// option against ONE implementation of a strategy is not testing the option.**
///
/// # This test is on LAYER 1, and that is not an accident
///
/// The obvious version — assert layer-0 degree respects `m0` — is VACUOUS, and I wrote it
/// that way first and watched the sabotage pass. Layer 0 is stored in the flat node block,
/// whose stride is `node_stride(config.m0, ..)`: the block physically has room for exactly
/// `m0` neighbours, so the surplus is silently dropped at conversion no matter what the
/// builder did. The storage layout *launders* the bug. Layer >= 1 lives in `UpperNode`
/// (capacity `M_MAX = 32`) and is copied out untruncated — so that is the one place where
/// "the builder ignored `config.m`" is actually observable.
///
/// Sabotage this catches: revert `UpperNode::from_zero(z, m)` to `take(M_MAX)`, or
/// `search_upper(.., m)` to `M_MAX`.
#[test]
fn m_caps_upper_layer_degree_in_the_parallel_builder_too() {
    let mut rng = StdRng::seed_from_u64(4242);
    let centers: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..2000)
        .map(|i| {
            let c = &centers[i % 10];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    let build = |m: usize| {
        HNSWIndex::build(
            embeddings.clone(),
            HNSWConfig {
                m,
                m0: 64,
                ef_construction: 100,
                seed: Some(5),
                keep_pruned_connections: true, // saturate, so the cap actually binds
                build_strategy: BuildStrategy::Parallel, // THE DEFAULT. The untested path.
                ..Default::default()
            },
        )
    };

    // Peak degree over every node present at layer 1.
    let peak_l1 = |ix: &HNSWIndex| -> (usize, usize) {
        let mut peak = 0;
        let mut count = 0;
        for id in 0..ix.len() {
            if ix.connections[id].len() > 1 {
                peak = peak.max(ix.connections[id][1].len());
                count += 1;
            }
        }
        (peak, count)
    };

    let (narrow_peak, narrow_count) = peak_l1(&build(4));
    let (wide_peak, _) = peak_l1(&build(32));

    // CONTROL 1: layer 1 is actually populated. A cap cannot bind on an empty layer.
    assert!(
        narrow_count > 10,
        "fixture put only {narrow_count} nodes on layer 1 — the assertions below would be \
             vacuous"
    );
    // CONTROL 2: the cap BINDS. Something must reach m=4, or "nothing exceeds 4" is free.
    assert_eq!(
        narrow_peak, 4,
        "the m=4 cap never bound (peak layer-1 degree {narrow_peak}) — nothing is pressing \
             against the limit, so the invariant below proves nothing"
    );

    // THE INVARIANT: m = 4 means no layer-1 node may hold more than 4 edges.
    assert!(
        narrow_peak <= 4,
        "m = 4 but a layer-1 node holds {narrow_peak} neighbours — the parallel builder is \
             ignoring config.m (it used to hardcode M_MAX = 32)"
    );
    // ...and m = 32 must genuinely produce a wider graph.
    assert!(
        wide_peak > 4,
        "m = 32 produced a peak layer-1 degree of only {wide_peak}, no better than m=4 — \
             config.m is not reaching the parallel builder"
    );
}

#[test]
fn m_caps_upper_layer_degree_independent_of_m0() {
    let mut rng = StdRng::seed_from_u64(5150);
    let centers: Vec<Vec<f32>> = (0..12)
        .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..800)
        .map(|i| {
            let c = &centers[i % 12];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    // Max degree over every node present at `layer`, and how many nodes that was.
    let peak_degree_at = |index: &HNSWIndex, layer: usize| -> (usize, usize) {
        let mut peak = 0usize;
        let mut count = 0usize;
        for node_id in 0..index.len() {
            if layer < index.connections[node_id].len() {
                peak = peak.max(index.connections[node_id][layer].len());
                count += 1;
            }
        }
        (peak, count)
    };

    let build_with_m = |m: usize| -> HNSWIndex {
        HNSWIndex::build(
            embeddings.clone(),
            HNSWConfig {
                m,
                m0: 64,
                // ml: 3.0, NOT the 0.8 this test used to carry. `ml` sets how aggressively
                // nodes are promoted above layer 0; at 0.8 barely any were, so no node ever
                // accumulated enough neighbours for a degree cap of 4 — let alone 32 — to
                // bind. An option that is never placed under load cannot be measured.
                ml: 3.0,
                ef_construction: 150,
                seed: Some(99),
                build_strategy: BuildStrategy::Sequential,
                ..Default::default()
            },
        )
    };

    let narrow = build_with_m(4);
    let wide = build_with_m(32);

    let (narrow_peak, narrow_count) = peak_degree_at(&narrow, 1);
    let (wide_peak, wide_count) = peak_degree_at(&wide, 1);

    // CONTROL 1: the fixture actually populates layer 1.
    assert!(
        narrow_count > 20 && wide_count > 20,
        "fixture put almost nothing on layer 1 ({narrow_count} / {wide_count} nodes) — a \
             degree cap cannot bind on an empty layer, so the assertions below would be vacuous"
    );

    // CONTROL 2: the cap actually BINDS. Some node must reach exactly m=4 neighbours,
    // otherwise "no node exceeds 4" is true for free and tests nothing.
    assert_eq!(
        narrow_peak, 4,
        "the m=4 cap never bound (peak layer-1 degree was {narrow_peak}) — with nothing
             pressing against the limit, the over-degree assertion below proves nothing"
    );

    // THE INVARIANT: m=4 means *no* node above layer 0 may exceed 4 edges. Breaking either
    // read site lets some node through, and a peak — unlike a mean — cannot absorb it.
    assert!(
        narrow_peak <= 4,
        "m = 4 but a layer-1 node holds {narrow_peak} neighbours — config.m is not capping \
             upper-layer degree"
    );

    // ...and m=32 must genuinely permit a wider graph, or `m` is being clamped somewhere.
    assert!(
        wide_peak > 4,
        "m = 32 produced a peak layer-1 degree of only {wide_peak}, no better than m=4 — \
             config.m is being ignored during insertion"
    );
}

/// `HNSWConfig::seed` must control the RNG used for level generation (and everything else
/// stochastic in the builder) deterministically: the same seed on the same data must produce
/// a bit-identical graph, and two different seeds must produce a different one. Both halves
/// matter — see the two sabotage cases below, each caught by a different assertion.
///
/// `BuildStrategy::Sequential` is used deliberately: `Parallel` uses rayon, and this test
/// asks whether `config.seed` is read at all, not whether the parallel builder is internally
/// deterministic under concurrent scheduling — a separate question, out of scope here.
///
/// Sabotage this catches (two different bugs, one per assertion):
///  - "same seed -> same graph" fails if `config.seed` is ignored and the code always calls
///    `rand::random()` regardless of what the caller passed — two `Some(42)` builds would
///    then almost certainly diverge.
///  - "different seed -> different graph" fails if the code reads `config.seed` but maps it
///    through a broken or constant transform (e.g. hardcoding the RNG's internal seed to a
///    fixed value regardless of what `Some(N)` says) — `Some(1)` and `Some(2)` would then
///    collapse onto the same graph.
#[test]
fn seed_makes_builds_reproducible_and_distinguishable() {
    let mut rng = StdRng::seed_from_u64(2718);
    let centers: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..12).map(|_| rng.random::<f32>() * 10.0).collect())
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..400)
        .map(|i| {
            let c = &centers[i % 10];
            c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
        })
        .collect();

    let build_with_seed = |seed: u64| -> HNSWIndex {
        HNSWIndex::build(
            embeddings.clone(),
            HNSWConfig {
                seed: Some(seed),
                build_strategy: BuildStrategy::Sequential,
                ..Default::default()
            },
        )
    };

    #[allow(clippy::type_complexity)]
    let fingerprint = |index: &HNSWIndex| -> (Vec<u32>, Vec<Vec<Vec<u32>>>, Option<usize>, usize) {
        (
            index.nodes.clone(),
            index.connections.clone(),
            index.entry_point,
            index.max_layer,
        )
    };

    let a1 = build_with_seed(42);
    let a2 = build_with_seed(42);
    let b = build_with_seed(43);

    assert_eq!(
        fingerprint(&a1),
        fingerprint(&a2),
        "two builds with the same seed produced different graphs — seed is not being used \
             deterministically (or is being ignored in favor of a fresh random seed each time)"
    );
    assert_ne!(
        fingerprint(&a1),
        fingerprint(&b),
        "two builds with DIFFERENT seeds produced the identical graph — seed is being \
             ignored in favor of some fixed internal value"
    );
}

/// `rerank_candidates > 0` must measurably improve recall over `rerank_candidates: 0` on the
/// *same* built index (same seed, same embeddings) — reranking is supposed to correct the
/// coarse quantized ranking's mistakes, not merely "not panic". The existing
/// `rabitq_zero_rerank_drops_full_precision_vectors_and_does_not_panic` only covers the zero
/// case in isolation and cannot tell a working rerank from a disabled one.
///
/// This is exactly the shape of the historical `PQHNSWConfig::rerank_candidates` bug: gated
/// behind a second field that defaulted off, silently reranking nothing, with recall
/// unchanged (0.840 vs 0.840) between "on" and "off". Nothing at the `HNSWIndex` level
/// pairs the two the way this test does.
///
/// `Storage::RaBitQ` is used because its 1-bit estimate is the coarsest ranking in the
/// crate — and per this crate's own dimension-crossover findings, RaBitQ's coarseness cost
/// is *worse*, not better, at low dimension, which is why `dim: 16` is used deliberately
/// rather than a higher, easier dimension.
///
/// Empirical margin, not a hand-proof: this relies on the coarse RaBitQ ranking genuinely
/// misordering some held-out queries' top-10 on this fixture. It was reasoned through, not
/// run. If it turns out both configs score at or near 100% recall on this exact fixture
/// (i.e. the assertion fails because the values are equal or too close), that means the
/// fixture is too easy, not that the test is wrong — harden it (denser clusters, e.g. more
/// points per cluster) rather than weakening the assertion.
///
/// Sabotage this catches: make the `rerank_candidates > 0` branch in `search_inner` a no-op
/// (skip the rescore-and-resort) while still keeping the full-precision array allocated —
/// recall for `rerank_candidates: 50` would then equal `rerank_candidates: 0`'s recall
/// instead of exceeding it.
#[test]
fn rerank_candidates_nonzero_beats_zero_on_the_same_index() {
    let mut rng = StdRng::seed_from_u64(707);
    let dim = 16;
    let n_clusters = 10;
    let per_cluster = 60;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 20.0).collect())
        .collect();
    let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.9).collect()
        })
        .collect();
    let n_queries = 50;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|x| x + rng.random::<f32>() * 0.9).collect()
        })
        .collect();

    let recall_for = |rerank_candidates: usize| -> f32 {
        let config = HNSWConfig {
            metric: DistanceMetric::L2,
            m: 16,
            m0: 32,
            ef_construction: 150,
            ef_search: 150,
            storage: Storage::RaBitQ,
            rerank_candidates,
            seed: Some(21),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);

        let k = 10;
        let mut total = 0.0f32;
        for q in &queries {
            let mut exact: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i)
                })
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            total += truth.intersection(&got).count() as f32 / k as f32;
        }
        total / queries.len() as f32
    };

    let coarse_only = recall_for(0);
    let reranked = recall_for(50);

    assert!(
        reranked > coarse_only,
        "rerank_candidates=50 must beat rerank_candidates=0 on the same index — got \
             {reranked:.3} vs {coarse_only:.3}. Equal recall means the rerank pool is being \
             silently ignored, exactly the shape of the historical PQHNSWConfig bug."
    );
}
