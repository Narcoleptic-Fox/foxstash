//! Hybrid search: merge vector and keyword results.

use foxstash_core::SearchResult;
use std::collections::HashMap;

/// How to combine vector and keyword scores.
#[derive(Debug, Clone, Default)]
pub enum MergeStrategy {
    /// Reciprocal Rank Fusion — rank-based, no normalization needed.
    #[default]
    Rrf,
    /// Min-max normalize each system's scores to \[0,1\], then weighted sum.
    WeightedSum,
}

/// Configuration for hybrid search merging.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    vector_weight: f32,
    keyword_weight: f32,
    merge_strategy: MergeStrategy,
    /// RRF smoothing constant (only used with `MergeStrategy::Rrf`).
    rrf_k: f32,
}

impl HybridConfig {
    /// Vector similarity weight.
    pub fn vector_weight(&self) -> f32 {
        self.vector_weight
    }

    /// Keyword (BM25) weight.
    pub fn keyword_weight(&self) -> f32 {
        self.keyword_weight
    }

    /// Merge strategy in use.
    pub fn merge_strategy(&self) -> &MergeStrategy {
        &self.merge_strategy
    }

    /// RRF smoothing constant.
    pub fn rrf_k(&self) -> f32 {
        self.rrf_k
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            keyword_weight: 0.3,
            merge_strategy: MergeStrategy::Rrf,
            rrf_k: 60.0,
        }
    }
}

impl HybridConfig {
    pub fn with_weights(mut self, vector: f32, keyword: f32) -> Self {
        assert!(
            vector.is_finite() && vector >= 0.0,
            "vector_weight must be finite and non-negative, got {vector}"
        );
        assert!(
            keyword.is_finite() && keyword >= 0.0,
            "keyword_weight must be finite and non-negative, got {keyword}"
        );
        self.vector_weight = vector;
        self.keyword_weight = keyword;
        self
    }

    pub fn with_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    pub fn with_rrf_k(mut self, rrf_k: f32) -> Self {
        assert!(
            rrf_k.is_finite() && rrf_k >= 0.0,
            "rrf_k must be finite and non-negative, got {rrf_k}"
        );
        self.rrf_k = rrf_k;
        self
    }
}

/// Merge vector and keyword search results.
///
/// `doc_lookup` resolves internal positions (from BM25) to `SearchResult`.
/// This avoids coupling hybrid.rs to `CollectionInner`.
pub fn merge_results(
    vector_results: &[SearchResult],
    keyword_results: &[(usize, f32)],
    doc_lookup: &dyn Fn(usize) -> Option<SearchResult>,
    k: usize,
    config: &HybridConfig,
) -> Vec<SearchResult> {
    match config.merge_strategy {
        MergeStrategy::Rrf => merge_rrf(vector_results, keyword_results, doc_lookup, k, config),
        MergeStrategy::WeightedSum => {
            merge_weighted_sum(vector_results, keyword_results, doc_lookup, k, config)
        }
    }
}

fn merge_rrf(
    vector_results: &[SearchResult],
    keyword_results: &[(usize, f32)],
    doc_lookup: &dyn Fn(usize) -> Option<SearchResult>,
    k: usize,
    config: &HybridConfig,
) -> Vec<SearchResult> {
    // Accumulate RRF scores by document ID.
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut results_by_id: HashMap<String, SearchResult> = HashMap::new();

    // Vector results (already ranked).
    for (rank, r) in vector_results.iter().enumerate() {
        let rrf_score = config.vector_weight / (config.rrf_k + rank as f32 + 1.0);
        *scores.entry(r.id.clone()).or_default() += rrf_score;
        results_by_id
            .entry(r.id.clone())
            .or_insert_with(|| r.clone());
    }

    // Keyword results (already ranked by BM25 score).
    for (rank, (pos, _bm25_score)) in keyword_results.iter().enumerate() {
        if let Some(r) = doc_lookup(*pos) {
            let rrf_score = config.keyword_weight / (config.rrf_k + rank as f32 + 1.0);
            *scores.entry(r.id.clone()).or_default() += rrf_score;
            results_by_id.entry(r.id.clone()).or_insert(r);
        }
    }

    collect_top_k(scores, results_by_id, k)
}

fn merge_weighted_sum(
    vector_results: &[SearchResult],
    keyword_results: &[(usize, f32)],
    doc_lookup: &dyn Fn(usize) -> Option<SearchResult>,
    k: usize,
    config: &HybridConfig,
) -> Vec<SearchResult> {
    // Min-max normalize vector scores.
    let v_scores: Vec<f32> = vector_results.iter().map(|r| r.score).collect();
    let v_norm = min_max_normalize(&v_scores);

    // Min-max normalize keyword scores.
    let kw_scores: Vec<f32> = keyword_results.iter().map(|r| r.1).collect();
    let kw_norm = min_max_normalize(&kw_scores);

    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut results_by_id: HashMap<String, SearchResult> = HashMap::new();

    for (i, r) in vector_results.iter().enumerate() {
        let norm_score = config.vector_weight * v_norm[i];
        *scores.entry(r.id.clone()).or_default() += norm_score;
        results_by_id
            .entry(r.id.clone())
            .or_insert_with(|| r.clone());
    }

    for (i, (pos, _)) in keyword_results.iter().enumerate() {
        if let Some(r) = doc_lookup(*pos) {
            let norm_score = config.keyword_weight * kw_norm[i];
            *scores.entry(r.id.clone()).or_default() += norm_score;
            results_by_id.entry(r.id.clone()).or_insert(r);
        }
    }

    collect_top_k(scores, results_by_id, k)
}

fn min_max_normalize(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range <= f32::EPSILON {
        vec![1.0; values.len()]
    } else {
        values.iter().map(|v| (v - min) / range).collect()
    }
}

fn collect_top_k(
    scores: HashMap<String, f32>,
    results_by_id: HashMap<String, SearchResult>,
    k: usize,
) -> Vec<SearchResult> {
    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);

    ranked
        .into_iter()
        .filter_map(|(id, score)| {
            results_by_id.get(&id).map(|r| SearchResult {
                id: r.id.clone(),
                content: r.content.clone(),
                score,
                metadata: r.metadata.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sr(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.into(),
            content: format!("content-{id}"),
            score,
            metadata: None,
        }
    }

    fn make_lookup(docs: Vec<(usize, SearchResult)>) -> impl Fn(usize) -> Option<SearchResult> {
        let map: HashMap<usize, SearchResult> = docs.into_iter().collect();
        move |pos| map.get(&pos).cloned()
    }

    #[test]
    fn rrf_basic_merge() {
        let vector = vec![sr("a", 0.9), sr("b", 0.8)];
        let keyword = vec![(10, 5.0), (20, 3.0)]; // pos 10 -> "c", pos 20 -> "a"
        let lookup = make_lookup(vec![(10, sr("c", 5.0)), (20, sr("a", 3.0))]);

        let config = HybridConfig::default();
        let results = merge_results(&vector, &keyword, &lookup, 10, &config);

        // "a" appears in both → should be boosted.
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn rrf_overlap_boosts_ranking() {
        let vector = vec![sr("a", 0.9), sr("b", 0.8), sr("c", 0.7)];
        let keyword = vec![(0, 5.0), (1, 3.0)]; // pos 0 -> "c", pos 1 -> "b"
        let lookup = make_lookup(vec![(0, sr("c", 5.0)), (1, sr("b", 3.0))]);

        let config = HybridConfig::default();
        let results = merge_results(&vector, &keyword, &lookup, 10, &config);

        // "b" and "c" appear in both lists → should be ranked above "a" (vector-only).
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let a_pos = ids.iter().position(|id| *id == "a").unwrap();
        let b_pos = ids.iter().position(|id| *id == "b").unwrap();
        assert!(b_pos < a_pos);
    }

    #[test]
    fn rrf_disjoint_lists() {
        let vector = vec![sr("a", 0.9)];
        let keyword = vec![(0, 5.0)];
        let lookup = make_lookup(vec![(0, sr("b", 5.0))]);

        let config = HybridConfig::default();
        let results = merge_results(&vector, &keyword, &lookup, 10, &config);

        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn empty_inputs() {
        let config = HybridConfig::default();
        let lookup = make_lookup(vec![]);

        // Both empty.
        let results = merge_results(&[], &[], &lookup, 10, &config);
        assert!(results.is_empty());

        // Vector only.
        let vector = vec![sr("a", 0.9)];
        let results = merge_results(&vector, &[], &lookup, 10, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");

        // Keyword only.
        let keyword = vec![(0, 5.0)];
        let lookup = make_lookup(vec![(0, sr("b", 5.0))]);
        let results = merge_results(&[], &keyword, &lookup, 10, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn weighted_sum_basic() {
        let vector = vec![sr("a", 0.9), sr("b", 0.5)];
        let keyword = vec![(0, 5.0), (1, 3.0)];
        let lookup = make_lookup(vec![(0, sr("b", 5.0)), (1, sr("a", 3.0))]);

        let config = HybridConfig::default().with_strategy(MergeStrategy::WeightedSum);
        let results = merge_results(&vector, &keyword, &lookup, 10, &config);

        // Both a and b appear.
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn single_result_normalization() {
        // Single result normalizes to 1.0.
        let vector = vec![sr("a", 0.42)];
        let config = HybridConfig::default().with_strategy(MergeStrategy::WeightedSum);
        let lookup = make_lookup(vec![]);

        let results = merge_results(&vector, &[], &lookup, 10, &config);
        assert_eq!(results.len(), 1);
        // Single value normalizes to 1.0, so score = vector_weight * 1.0 = 0.7.
        assert!((results[0].score - config.vector_weight).abs() < 0.001);
    }

    #[test]
    fn config_defaults() {
        let config = HybridConfig::default();
        assert!((config.vector_weight - 0.7).abs() < f32::EPSILON);
        assert!((config.keyword_weight - 0.3).abs() < f32::EPSILON);
        assert!((config.rrf_k - 60.0).abs() < f32::EPSILON);
        assert!(matches!(config.merge_strategy, MergeStrategy::Rrf));
    }

    #[test]
    fn config_builder() {
        let config = HybridConfig::default()
            .with_weights(0.5, 0.5)
            .with_strategy(MergeStrategy::WeightedSum)
            .with_rrf_k(30.0);

        assert!((config.vector_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.keyword_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.rrf_k - 30.0).abs() < f32::EPSILON);
        assert!(matches!(config.merge_strategy, MergeStrategy::WeightedSum));
    }

    // ── Edge-case tests ────────────────────────────────────────────

    #[test]
    fn rrf_with_zero_rrf_k() {
        // rrf_k=0 means score = weight / (0 + rank + 1). Should produce finite scores.
        let vector = vec![sr("a", 0.9), sr("b", 0.8)];
        let keyword = vec![(10, 5.0)];
        let lookup = make_lookup(vec![(10, sr("c", 5.0))]);

        let config = HybridConfig::default().with_rrf_k(0.0);
        let results = merge_results(&vector, &keyword, &lookup, 10, &config);

        assert!(!results.is_empty());
        for r in &results {
            assert!(
                r.score.is_finite(),
                "score should be finite, got {}",
                r.score
            );
        }
    }

    #[test]
    fn weighted_sum_identical_scores() {
        // All identical scores → range ≈ 0 → normalize to 1.0.
        let vector = vec![sr("a", 0.5), sr("b", 0.5), sr("c", 0.5)];
        let config = HybridConfig::default().with_strategy(MergeStrategy::WeightedSum);
        let lookup = make_lookup(vec![]);

        let results = merge_results(&vector, &[], &lookup, 10, &config);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.score.is_finite(), "score should be finite");
        }
    }

    #[test]
    fn weighted_sum_single_element() {
        // Single element normalizes to 1.0, score = vector_weight * 1.0.
        let vector = vec![sr("a", 0.42)];
        let config = HybridConfig::default().with_strategy(MergeStrategy::WeightedSum);
        let lookup = make_lookup(vec![]);

        let results = merge_results(&vector, &[], &lookup, 10, &config);
        assert_eq!(results.len(), 1);
        assert!((results[0].score - config.vector_weight).abs() < 0.001);
    }

    #[test]
    #[should_panic(expected = "vector_weight must be finite and non-negative")]
    fn config_validation_rejects_negative_vector_weight() {
        HybridConfig::default().with_weights(-1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "vector_weight must be finite and non-negative")]
    fn config_validation_rejects_nan_weight() {
        HybridConfig::default().with_weights(f32::NAN, 0.5);
    }

    #[test]
    #[should_panic(expected = "rrf_k must be finite and non-negative")]
    fn config_validation_rejects_negative_rrf_k() {
        HybridConfig::default().with_rrf_k(-10.0);
    }
}
