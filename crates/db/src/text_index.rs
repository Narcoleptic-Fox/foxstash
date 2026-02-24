//! Trait for text-based (keyword) indexes.

/// Common interface for text/keyword indexes.
///
/// Mirrors [`foxstash_core::index::VectorIndex`] but for token-level search
/// (BM25, TF-IDF, etc.) rather than dense vector similarity.
///
/// Object-safe — works with `Box<dyn TextIndex>`.
pub trait TextIndex {
    /// Index a document's tokens.
    fn add(&mut self, doc_id: usize, tokens: &[String]);

    /// Remove a document from the index.
    fn remove(&mut self, doc_id: usize);

    /// Search for documents matching query tokens, returning top-k by score.
    fn search(&self, query_tokens: &[String], k: usize) -> Vec<(usize, f32)>;

    /// Remove all documents from the index.
    fn clear(&mut self);

    /// Return the number of indexed documents.
    fn len(&self) -> usize;

    /// Return true if the index contains no documents.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverted_index::InvertedIndex;

    fn tokens(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn text_index_object_safety() {
        let mut idx: Box<dyn TextIndex> = Box::new(InvertedIndex::new());

        assert!(idx.is_empty());

        idx.add(0, &tokens("hello world"));
        idx.add(1, &tokens("hello rust"));
        assert_eq!(idx.len(), 2);

        let results = idx.search(&tokens("hello"), 10);
        assert_eq!(results.len(), 2);

        idx.remove(0);
        assert_eq!(idx.len(), 1);

        let results = idx.search(&tokens("world"), 10);
        assert!(results.is_empty());

        idx.clear();
        assert!(idx.is_empty());
    }
}
