//! Text tokenization for keyword search.

use std::collections::HashSet;

/// Tokenizes text into terms for indexing and querying.
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Lowercase + split on non-alphanumeric + filter short tokens + stop words.
pub struct SimpleTokenizer {
    min_token_len: usize,
    stop_words: HashSet<String>,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let stop_words: HashSet<String> = [
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has",
            "have", "he", "her", "his", "if", "in", "into", "is", "it", "its", "no", "not", "of",
            "on", "or", "she", "so", "than", "that", "the", "their", "them", "then", "there",
            "these", "they", "this", "to", "was", "we", "were", "what", "when", "which", "who",
            "will", "with", "you",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        Self {
            min_token_len: 2,
            stop_words,
        }
    }

    pub fn without_stop_words() -> Self {
        Self {
            min_token_len: 2,
            stop_words: HashSet::new(),
        }
    }

    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        Self {
            min_token_len: 2,
            stop_words,
        }
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= self.min_token_len)
            .filter(|t| !self.stop_words.contains(*t))
            .map(String::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenization() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn punctuation_splitting() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("gateway-service: running (pid=1234)");
        assert_eq!(tokens, vec!["gateway", "service", "running", "pid", "1234"]);
    }

    #[test]
    fn stop_word_filtering() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("the quick brown fox is in the garden");
        assert_eq!(tokens, vec!["quick", "brown", "fox", "garden"]);
    }

    #[test]
    fn empty_input() {
        let t = SimpleTokenizer::new();
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn short_token_filtering() {
        let t = SimpleTokenizer::new();
        // "i" -> len 1 -> filtered, "x" -> len 1 -> filtered, "a" -> stop word, "go" -> passes, "developer" -> passes
        let tokens = t.tokenize("I x a go developer");
        assert_eq!(tokens, vec!["go", "developer"]);
    }

    #[test]
    fn unicode_chars() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("café résumé naïve");
        assert_eq!(tokens, vec!["café", "résumé", "naïve"]);
    }

    #[test]
    fn without_stop_words_variant() {
        let t = SimpleTokenizer::without_stop_words();
        let tokens = t.tokenize("the quick fox");
        assert_eq!(tokens, vec!["the", "quick", "fox"]);
    }
}
