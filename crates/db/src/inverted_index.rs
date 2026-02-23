//! BM25-scored inverted index for keyword search.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

/// BM25 tuning parameters.
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Term frequency saturation.
    pub k1: f32,
    /// Document length normalization.
    pub b: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A single posting: document position + term frequency.
#[derive(Debug, Clone)]
struct Posting {
    doc_id: usize,
    tf: u32,
}

/// Per-document metadata for length normalization.
#[derive(Debug, Clone)]
struct DocInfo {
    token_count: u32,
}

/// Inverted index with BM25 scoring.
///
/// Uses `usize` document IDs (positions from `IdMap`) that align with the
/// `documents` vec in `CollectionInner`.
pub struct InvertedIndex {
    term_index: HashMap<String, Vec<Posting>>,
    doc_info: HashMap<usize, DocInfo>,
    total_tokens: u64,
    doc_count: usize,
    config: BM25Config,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self::with_config(BM25Config::default())
    }

    pub fn with_config(config: BM25Config) -> Self {
        Self {
            term_index: HashMap::new(),
            doc_info: HashMap::new(),
            total_tokens: 0,
            doc_count: 0,
            config,
        }
    }

    /// Index a document's tokens.
    pub fn add(&mut self, doc_id: usize, tokens: &[String]) {
        // Count term frequencies.
        let mut tf_map: HashMap<&str, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token.as_str()).or_default() += 1;
        }

        // Insert postings.
        for (term, tf) in tf_map {
            self.term_index
                .entry(term.to_string())
                .or_default()
                .push(Posting { doc_id, tf });
        }

        // Store doc info.
        let token_count = tokens.len() as u32;
        self.doc_info.insert(doc_id, DocInfo { token_count });
        self.total_tokens += token_count as u64;
        self.doc_count += 1;
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, doc_id: usize) {
        if let Some(info) = self.doc_info.remove(&doc_id) {
            self.total_tokens -= info.token_count as u64;
            self.doc_count -= 1;

            // Remove postings referencing this doc.
            self.term_index.retain(|_, postings| {
                postings.retain(|p| p.doc_id != doc_id);
                !postings.is_empty()
            });
        }
    }

    /// Search for documents matching query tokens, returning top-k by BM25 score.
    pub fn search(&self, query_tokens: &[String], k: usize) -> Vec<(usize, f32)> {
        if self.doc_count == 0 || query_tokens.is_empty() {
            return Vec::new();
        }

        let avgdl = self.total_tokens as f32 / self.doc_count as f32;
        let n = self.doc_count as f32;

        // Accumulate BM25 scores per document.
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in query_tokens {
            let Some(postings) = self.term_index.get(term.as_str()) else {
                continue;
            };

            let df = postings.len() as f32;
            // Smoothed IDF: ln((N - df + 0.5) / (df + 0.5) + 1) — never negative.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for posting in postings {
                let dl = self
                    .doc_info
                    .get(&posting.doc_id)
                    .map(|d| d.token_count as f32)
                    .unwrap_or(0.0);

                let tf = posting.tf as f32;
                let tf_component =
                    (tf * (self.config.k1 + 1.0))
                        / (tf + self.config.k1 * (1.0 - self.config.b + self.config.b * dl / avgdl));

                *scores.entry(posting.doc_id).or_default() += idf * tf_component;
            }
        }

        // Top-k via min-heap.
        let mut heap: BinaryHeap<Reverse<OrdF32Entry>> = BinaryHeap::new();

        for (doc_id, score) in scores {
            let entry = Reverse(OrdF32Entry { score, doc_id });
            if heap.len() < k {
                heap.push(entry);
            } else if let Some(min) = heap.peek() {
                if score > min.0.score {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        let mut results: Vec<(usize, f32)> = heap.into_iter().map(|Reverse(e)| (e.doc_id, e.score)).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn clear(&mut self) {
        self.term_index.clear();
        self.doc_info.clear();
        self.total_tokens = 0;
        self.doc_count = 0;
    }

    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for f32 that implements Ord (for BinaryHeap).
#[derive(Clone, Copy)]
struct OrdF32Entry {
    score: f32,
    doc_id: usize,
}

impl PartialEq for OrdF32Entry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for OrdF32Entry {}

impl PartialOrd for OrdF32Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.doc_id.cmp(&other.doc_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn basic_add_and_search() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("gateway service running"));
        idx.add(1, &tokens("database connection pool"));
        idx.add(2, &tokens("gateway timeout error"));

        let results = idx.search(&tokens("gateway"), 10);
        assert_eq!(results.len(), 2);
        // Both docs with "gateway" returned.
        let ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[test]
    fn tf_boost() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("rust rust rust programming"));
        idx.add(1, &tokens("rust programming language"));

        let results = idx.search(&tokens("rust"), 10);
        assert_eq!(results.len(), 2);
        // Doc 0 has higher TF for "rust" → higher score.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn idf_boost() {
        let mut idx = InvertedIndex::new();
        // "common" appears in all 3 docs, "rare" only in doc 2.
        idx.add(0, &tokens("common word here"));
        idx.add(1, &tokens("common word there"));
        idx.add(2, &tokens("common rare special"));

        let results = idx.search(&tokens("rare"), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);

        // Searching for "common" should return all 3.
        let results = idx.search(&tokens("common"), 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn doc_length_normalization() {
        let mut idx = InvertedIndex::new();
        // Short doc with "target" should score higher than long doc with "target".
        idx.add(0, &tokens("target"));
        idx.add(1, &tokens("target word word word word word word word word word"));

        let results = idx.search(&tokens("target"), 10);
        assert_eq!(results.len(), 2);
        // Shorter doc should rank first due to length normalization.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn top_k_truncation() {
        let mut idx = InvertedIndex::new();
        for i in 0..20 {
            idx.add(i, &tokens("shared term"));
        }

        let results = idx.search(&tokens("shared"), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn empty_index() {
        let idx = InvertedIndex::new();
        let results = idx.search(&tokens("anything"), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn no_matching_terms() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("alpha beta gamma"));

        let results = idx.search(&tokens("delta epsilon"), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn remove_doc() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("gateway service"));
        idx.add(1, &tokens("database service"));

        idx.remove(0);
        assert_eq!(idx.len(), 1);

        let results = idx.search(&tokens("gateway"), 10);
        assert!(results.is_empty());

        let results = idx.search(&tokens("service"), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn clear_index() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("hello world"));
        idx.add(1, &tokens("foo bar"));

        idx.clear();
        assert!(idx.is_empty());
        assert!(idx.search(&tokens("hello"), 10).is_empty());
    }

    #[test]
    fn rebuild_from_scratch() {
        let mut idx = InvertedIndex::new();
        idx.add(0, &tokens("old data here"));
        idx.add(1, &tokens("more old data"));

        // Simulate rebuild: clear then re-add.
        idx.clear();
        idx.add(0, &tokens("new data here"));
        idx.add(1, &tokens("fresh content"));

        let results = idx.search(&tokens("new"), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);

        let results = idx.search(&tokens("old"), 10);
        assert!(results.is_empty());
    }
}
