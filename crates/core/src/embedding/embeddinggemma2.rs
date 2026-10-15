//! EmbeddingGemma 2 prompt formatting and Matryoshka truncation (issue #51).
//!
//! EmbeddingGemma 2 is prefix-conditioned: the input text must carry a task prompt, and the query
//! and document sides use different ones. foxstash does not run the model — callers embed through
//! their own sidecar (e.g. a llama.cpp process) and insert the precomputed vectors via
//! [`crate::index`] / `foxstash-db` — but it owns these helpers so the prompt contract and the
//! Matryoshka math live in one verified place instead of being re-derived by every caller.
//!
//! Prompts and truncation dimensions are taken from the model card (`google/embeddinggemma-2`,
//! 2026-10-06) and are pure string/vector operations with no model dependency.
//!
//! ```
//! use foxstash_core::embedding::embeddinggemma2::{document_prompt, query_prompt, Task};
//! assert_eq!(query_prompt(Task::SearchQuery, "borrow checker"),
//!            "task: search result | query: borrow checker");
//! assert_eq!(document_prompt(None, "ownership is enforced at compile time"),
//!            "title: none | text: ownership is enforced at compile time");
//! ```

use crate::{RagError, Result};

/// The retrieval/task type, which selects the query-side prompt. The asymmetric tasks share one
/// document-side format ([`document_prompt`]); the symmetric tasks ([`Task::is_asymmetric`] is
/// false) embed both sides with the query prompt and have no separate document side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Web / document search.
    SearchQuery,
    /// Question answering.
    QuestionAnswering,
    /// Fact checking.
    FactChecking,
    /// Code search.
    CodeRetrieval,
    /// Text classification (symmetric).
    Classification,
    /// Clustering (symmetric).
    Clustering,
    /// Measuring similarity (symmetric).
    SentenceSimilarity,
}

impl Task {
    /// The query task-instruction prefix the card specifies, including the trailing `query: `.
    /// [`query_prompt`] appends the text to this.
    pub fn query_prefix(self) -> &'static str {
        match self {
            Task::SearchQuery => "task: search result | query: ",
            Task::QuestionAnswering => "task: question answering | query: ",
            Task::FactChecking => "task: fact checking | query: ",
            Task::CodeRetrieval => "task: code retrieval | query: ",
            Task::Classification => "task: classification | query: ",
            Task::Clustering => "task: clustering | query: ",
            Task::SentenceSimilarity => "task: sentence similarity | query: ",
        }
    }

    /// Whether the task has a distinct document side. The asymmetric retrieval tasks
    /// (search / QA / fact-checking / code) embed documents with [`document_prompt`]; the
    /// symmetric tasks embed the single input with [`query_prompt`].
    pub fn is_asymmetric(self) -> bool {
        matches!(
            self,
            Task::SearchQuery | Task::QuestionAnswering | Task::FactChecking | Task::CodeRetrieval
        )
    }
}

/// Format a query for EmbeddingGemma 2: the task prefix followed by the query text.
pub fn query_prompt(task: Task, query: &str) -> String {
    format!("{}{}", task.query_prefix(), query)
}

/// Format a document for the asymmetric retrieval tasks: `title: {title} | text: {content}`.
///
/// Pass `None` for `title` when the document has no title (the card's `title: none`). The model was
/// trained with a title field, so passing a real title when one exists improves retrieval.
pub fn document_prompt(title: Option<&str>, content: &str) -> String {
    format!("title: {} | text: {}", title.unwrap_or("none"), content)
}

/// Output dimensions EmbeddingGemma 2 supports via Matryoshka Representation Learning: the native
/// 768, and the truncations 512 / 256 / 128. The card reports minimal quality loss down to 256;
/// 128 is best for text-only workloads.
pub const MATRYOSHKA_DIMS: [usize; 4] = [128, 256, 512, 768];

/// Truncate a native 768-d EmbeddingGemma 2 embedding to a supported Matryoshka dimension and
/// L2-renormalize (the card's documented procedure: take the leading `dim` components, then
/// renormalize).
///
/// # Errors
/// - [`RagError::InvalidInput`] if `dim` is not one of [`MATRYOSHKA_DIMS`], or is longer than the
///   input embedding.
pub fn truncate_matryoshka(embedding: &[f32], dim: usize) -> Result<Vec<f32>> {
    if !MATRYOSHKA_DIMS.contains(&dim) {
        return Err(RagError::InvalidInput(format!(
            "EmbeddingGemma 2 supports Matryoshka dims {MATRYOSHKA_DIMS:?}, not {dim}"
        )));
    }
    if dim > embedding.len() {
        return Err(RagError::InvalidInput(format!(
            "cannot truncate a {}-d embedding to {dim}",
            embedding.len()
        )));
    }
    let head = &embedding[..dim];
    let norm = head.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        // A zero (or near-zero) head cannot be renormalized; return it rather than dividing by ~0.
        return Ok(head.to_vec());
    }
    Ok(head.iter().map(|&x| x / norm).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_prompts_match_the_card() {
        assert_eq!(
            query_prompt(Task::SearchQuery, "how does Arc work"),
            "task: search result | query: how does Arc work"
        );
        assert_eq!(
            query_prompt(Task::CodeRetrieval, "fn main"),
            "task: code retrieval | query: fn main"
        );
        assert_eq!(
            query_prompt(Task::QuestionAnswering, "what is a lifetime"),
            "task: question answering | query: what is a lifetime"
        );
    }

    #[test]
    fn document_prompt_handles_title_presence() {
        assert_eq!(document_prompt(None, "body"), "title: none | text: body");
        assert_eq!(
            document_prompt(Some("Arc"), "atomic refcount"),
            "title: Arc | text: atomic refcount"
        );
    }

    #[test]
    fn asymmetric_flag_is_right() {
        assert!(Task::SearchQuery.is_asymmetric());
        assert!(Task::CodeRetrieval.is_asymmetric());
        assert!(!Task::Clustering.is_asymmetric());
        assert!(!Task::SentenceSimilarity.is_asymmetric());
    }

    #[test]
    fn matryoshka_truncates_and_renormalizes() {
        // A ramp vector; truncating to 256 must keep the first 256 and come back unit-norm.
        let v: Vec<f32> = (0..768).map(|i| (i as f32 + 1.0) / 768.0).collect();
        let t = truncate_matryoshka(&v, 256).unwrap();
        assert_eq!(t.len(), 256);
        let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        // Direction of the head is preserved (proportional to the original leading components).
        let ratio = t[10] / v[10];
        assert!((t[20] / v[20] - ratio).abs() < 1e-5);
    }

    #[test]
    fn matryoshka_rejects_unsupported_dims() {
        let v = vec![0.5f32; 768];
        assert!(truncate_matryoshka(&v, 384).is_err());
        assert!(truncate_matryoshka(&v, 1024).is_err());
        assert!(truncate_matryoshka(&v, 768).is_ok());
    }

    #[test]
    fn matryoshka_zero_head_does_not_divide_by_zero() {
        let mut v = vec![0.0f32; 768];
        v[300] = 1.0; // nonzero only outside the 256 head
        let t = truncate_matryoshka(&v, 256).unwrap();
        assert_eq!(t.len(), 256);
        assert!(t.iter().all(|&x| x == 0.0));
    }
}
