//! ONNX Runtime-based embedding generation
//!
//! This module provides high-performance embedding generation using ONNX Runtime.
//! It supports models like MiniLM-L6-v2 which produce 384-dimensional embeddings.
//!
//! # Example
//!
//! ```no_run
//! use foxstash_core::embedding::OnnxEmbedder;
//!
//! let embedder = OnnxEmbedder::new(
//!     "models/model.onnx",
//!     "models/tokenizer.json"
//! )?;
//!
//! let embedding = embedder.embed("Hello, world!")?;
//! assert_eq!(embedding.len(), 384);
//! # Ok::<(), foxstash_core::RagError>(())
//! ```

use crate::{RagError, Result};
use ndarray::{Array1, Array2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use tokenizers::Tokenizer;

/// ONNX Runtime-based embedder for text-to-vector conversion
///
/// This embedder loads an ONNX model (e.g., MiniLM-L6-v2) and its corresponding
/// tokenizer to generate normalized embeddings from text input.
///
/// # Features
///
/// - Single and batch embedding generation
/// - Mean pooling over sequence dimension
/// - L2 normalization (unit vectors)
/// - Efficient batch processing with padding
#[derive(Debug)]
pub struct OnnxEmbedder {
    /// ONNX Runtime session
    session: Session,
    /// Tokenizer for text preprocessing
    tokenizer: Tokenizer,
    /// Embedding dimension (e.g., 384 for MiniLM-L6-v2)
    embedding_dim: usize,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer JSON file
    ///
    /// # Errors
    ///
    /// Returns `RagError::EmbeddingError` if:
    /// - Model file cannot be loaded
    /// - Tokenizer file cannot be loaded
    /// - ONNX Runtime initialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use foxstash_core::embedding::OnnxEmbedder;
    ///
    /// let embedder = OnnxEmbedder::new(
    ///     "models/model.onnx",
    ///     "models/tokenizer.json"
    /// )?;
    /// # Ok::<(), foxstash_core::RagError>(())
    /// ```
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        // Load the ONNX model
        let session = Session::builder()
            .map_err(|e| RagError::EmbeddingError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RagError::EmbeddingError(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| RagError::EmbeddingError(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| {
                RagError::EmbeddingError(format!(
                    "Failed to load model from {:?}: {}",
                    model_path.as_ref(),
                    e
                ))
            })?;

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
            RagError::EmbeddingError(format!(
                "Failed to load tokenizer from {:?}: {}",
                tokenizer_path.as_ref(),
                e
            ))
        })?;

        // For MiniLM-L6-v2, the embedding dimension is 384
        let embedding_dim = 384;

        Ok(Self {
            session,
            tokenizer,
            embedding_dim,
        })
    }

    /// Generate an embedding for a single text
    ///
    /// The resulting embedding is L2-normalized (unit vector) and has
    /// the dimension specified by `embedding_dim()`.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to embed
    ///
    /// # Returns
    ///
    /// A normalized embedding vector of length `embedding_dim()`
    ///
    /// # Errors
    ///
    /// Returns `RagError::EmbeddingError` if:
    /// - Tokenization fails
    /// - ONNX inference fails
    /// - Output processing fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use foxstash_core::embedding::OnnxEmbedder;
    /// # let embedder = OnnxEmbedder::new("models/model.onnx", "models/tokenizer.json")?;
    /// let embedding = embedder.embed("Machine learning is fascinating")?;
    /// assert_eq!(embedding.len(), 384);
    ///
    /// // Verify normalization (should be close to 1.0)
    /// let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    /// assert!((norm - 1.0).abs() < 1e-5);
    /// # Ok::<(), foxstash_core::RagError>(())
    /// ```
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| RagError::EmbeddingError("No embedding generated".to_string()))
    }

    /// Generate embeddings for multiple texts in batch
    ///
    /// This is more efficient than calling `embed()` multiple times as it
    /// processes all texts in a single inference pass.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text references to embed
    ///
    /// # Returns
    ///
    /// A vector of normalized embedding vectors, one per input text
    ///
    /// # Errors
    ///
    /// Returns `RagError::EmbeddingError` if:
    /// - Input is empty
    /// - Tokenization fails
    /// - ONNX inference fails
    /// - Output processing fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use foxstash_core::embedding::OnnxEmbedder;
    /// # let mut embedder = OnnxEmbedder::new("models/model.onnx", "models/tokenizer.json")?;
    /// let texts = vec![
    ///     "First document",
    ///     "Second document",
    ///     "Third document",
    /// ];
    /// let embeddings = embedder.embed_batch(&texts)?;
    /// assert_eq!(embeddings.len(), 3);
    /// assert_eq!(embeddings[0].len(), 384);
    /// # Ok::<(), foxstash_core::RagError>(())
    /// ```
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(RagError::EmbeddingError(
                "Cannot embed empty batch".to_string(),
            ));
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RagError::EmbeddingError(format!("Tokenization failed: {}", e)))?;

        // Get the maximum sequence length and pad all sequences
        let max_length = encodings
            .iter()
            .map(|enc| enc.len())
            .max()
            .unwrap_or(0);

        if max_length == 0 {
            return Err(RagError::EmbeddingError(
                "All texts tokenized to empty sequences".to_string(),
            ));
        }

        let batch_size = texts.len();

        // Prepare input tensors: input_ids and attention_mask
        let mut input_ids = Vec::with_capacity(batch_size * max_length);
        let mut attention_mask = Vec::with_capacity(batch_size * max_length);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            // Add tokens
            input_ids.extend_from_slice(ids);
            attention_mask.extend_from_slice(mask);

            // Pad to max_length
            let padding_length = max_length - ids.len();
            if padding_length > 0 {
                input_ids.extend(std::iter::repeat(0).take(padding_length));
                attention_mask.extend(std::iter::repeat(0).take(padding_length));
            }
        }

        // Convert to i64 for ONNX
        let input_ids: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

        // Create input arrays
        let input_ids_array = Array2::from_shape_vec((batch_size, max_length), input_ids)
            .map_err(|e| RagError::EmbeddingError(format!("Failed to create input array: {}", e)))?;

        let attention_mask_array =
            Array2::from_shape_vec((batch_size, max_length), attention_mask).map_err(|e| {
                RagError::EmbeddingError(format!("Failed to create attention mask array: {}", e))
            })?;

        // Create ONNX tensors from ndarray
        // Note: ort v2.0 requires passing shape and data separately for i64 arrays
        let input_ids_vec = input_ids_array.as_slice().unwrap().to_vec();
        let attention_mask_vec = attention_mask_array.as_slice().unwrap().to_vec();

        let input_ids_tensor = Value::from_array(([batch_size, max_length], input_ids_vec))
            .map_err(|e| RagError::EmbeddingError(format!("Failed to create input tensor: {}", e)))?;
        let attention_mask_tensor = Value::from_array(([batch_size, max_length], attention_mask_vec))
            .map_err(|e| RagError::EmbeddingError(format!("Failed to create attention mask tensor: {}", e)))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
            .map_err(|e| RagError::EmbeddingError(format!("ONNX inference failed: {}", e)))?;

        // Extract the output tensor (last_hidden_state)
        // Shape: (batch_size, sequence_length, hidden_size)
        let output_tensor = &outputs["last_hidden_state"];

        let (output_shape, output_data) = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::EmbeddingError(format!("Failed to extract output tensor: {}", e)))?;

        if output_shape.len() != 3 {
            return Err(RagError::EmbeddingError(format!(
                "Unexpected output shape: {:?}",
                output_shape
            )));
        }

        // Extract dimensions for processing
        let seq_len = output_shape[1] as usize;
        let hidden_size = output_shape[2] as usize;

        // Perform mean pooling for each sample in the batch
        let mut embeddings = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Extract hidden states for this sample from the flat array
            let start_idx = i * seq_len;
            let end_idx = start_idx + seq_len;
            let sample_data = &output_data[(start_idx * hidden_size)..(end_idx * hidden_size)];

            let hidden_states = Array2::from_shape_vec((seq_len, hidden_size), sample_data.to_vec())
                .map_err(|e| RagError::EmbeddingError(format!("Failed to reshape sample data: {}", e)))?;

            // Get the attention mask for this sample
            let mask_start = i * max_length;
            let mask_end = mask_start + max_length;
            let sample_mask = &attention_mask_array.as_slice().unwrap()[mask_start..mask_end];

            // Perform mean pooling
            let embedding = mean_pooling(&hidden_states, sample_mask)?;

            // L2 normalize
            let normalized = l2_normalize(&embedding);

            embeddings.push(normalized);
        }

        Ok(embeddings)
    }

    /// Get the embedding dimension
    ///
    /// For MiniLM-L6-v2, this returns 384.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use foxstash_core::embedding::OnnxEmbedder;
    /// # let embedder = OnnxEmbedder::new("models/model.onnx", "models/tokenizer.json")?;
    /// assert_eq!(embedder.embedding_dim(), 384);
    /// # Ok::<(), foxstash_core::RagError>(())
    /// ```
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Perform mean pooling over the sequence dimension
///
/// This function computes the average of token embeddings, weighted by the
/// attention mask to exclude padding tokens.
///
/// # Arguments
///
/// * `hidden_states` - Token embeddings of shape (sequence_length, hidden_size)
/// * `attention_mask` - Binary mask indicating valid tokens (1) vs padding (0)
///
/// # Returns
///
/// A single embedding vector of length hidden_size
fn mean_pooling(hidden_states: &Array2<f32>, attention_mask: &[i64]) -> Result<Vec<f32>> {
    let (seq_len, hidden_size) = hidden_states.dim();

    if seq_len != attention_mask.len() {
        return Err(RagError::EmbeddingError(format!(
            "Sequence length mismatch: hidden_states={}, attention_mask={}",
            seq_len,
            attention_mask.len()
        )));
    }

    // Sum embeddings weighted by attention mask
    let mut summed = Array1::<f32>::zeros(hidden_size);
    let mut mask_sum = 0.0f32;

    for (i, &mask_value) in attention_mask.iter().enumerate() {
        if mask_value > 0 {
            let token_embedding = hidden_states.row(i);
            summed = summed + &token_embedding;
            mask_sum += 1.0;
        }
    }

    if mask_sum == 0.0 {
        return Err(RagError::EmbeddingError(
            "All tokens masked in attention mask".to_string(),
        ));
    }

    // Compute mean
    let mean = summed / mask_sum;

    Ok(mean.to_vec())
}

/// L2 normalize a vector (convert to unit vector)
///
/// # Arguments
///
/// * `vector` - Input vector to normalize
///
/// # Returns
///
/// A normalized vector with L2 norm = 1.0
///
/// # Note
///
/// If the input vector has zero norm, it is returned unchanged.
fn l2_normalize(vector: &[f32]) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm < 1e-12 {
        // Avoid division by zero
        return vector.to_vec();
    }

    vector.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test single text embedding
    ///
    /// Note: This test is marked as `#[ignore]` because it requires model files.
    /// To run it, provide the model files and use: `cargo test -- --ignored`
    #[test]
    #[ignore]
    fn test_single_embedding() {
        let mut embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json",
        )
        .expect("Failed to create embedder");

        let text = "This is a test sentence.";
        let embedding = embedder.embed(text).expect("Failed to generate embedding");

        // Check dimension
        assert_eq!(
            embedding.len(),
            384,
            "Embedding dimension should be 384 for MiniLM-L6-v2"
        );

        // Check normalization (L2 norm should be 1.0)
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Embedding should be normalized (L2 norm = 1.0), got {}",
            norm
        );

        // Check that embedding is not all zeros
        let sum: f32 = embedding.iter().map(|&x| x.abs()).sum();
        assert!(sum > 0.0, "Embedding should not be all zeros");
    }

    /// Test batch embedding
    ///
    /// Note: This test is marked as `#[ignore]` because it requires model files.
    #[test]
    #[ignore]
    fn test_batch_embedding() {
        let mut embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json",
        )
        .expect("Failed to create embedder");

        let texts = vec![
            "First test sentence.",
            "Second test sentence with more words.",
            "Third sentence.",
        ];

        let embeddings = embedder
            .embed_batch(&texts)
            .expect("Failed to generate batch embeddings");

        // Check batch size
        assert_eq!(embeddings.len(), 3, "Should generate 3 embeddings");

        // Check each embedding
        for (i, embedding) in embeddings.iter().enumerate() {
            // Check dimension
            assert_eq!(
                embedding.len(),
                384,
                "Embedding {} dimension should be 384",
                i
            );

            // Check normalization
            let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Embedding {} should be normalized, got norm {}",
                i,
                norm
            );

            // Check not all zeros
            let sum: f32 = embedding.iter().map(|&x| x.abs()).sum();
            assert!(sum > 0.0, "Embedding {} should not be all zeros", i);
        }

        // Check that embeddings are different from each other
        let similarity_0_1: f32 = embeddings[0]
            .iter()
            .zip(&embeddings[1])
            .map(|(a, b)| a * b)
            .sum();
        let similarity_0_2: f32 = embeddings[0]
            .iter()
            .zip(&embeddings[2])
            .map(|(a, b)| a * b)
            .sum();

        // Embeddings should be similar but not identical
        assert!(
            similarity_0_1 < 0.99,
            "Different texts should produce different embeddings"
        );
        assert!(
            similarity_0_2 < 0.99,
            "Different texts should produce different embeddings"
        );
    }

    /// Test empty input handling
    ///
    /// Note: This test is marked as `#[ignore]` because it requires model files.
    #[test]
    #[ignore]
    fn test_empty_input() {
        let mut embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json",
        )
        .expect("Failed to create embedder");

        // Test empty batch
        let result = embedder.embed_batch(&[]);
        assert!(
            result.is_err(),
            "Empty batch should return an error"
        );

        // Test empty string
        let result = embedder.embed("");
        // This may or may not error depending on tokenizer behavior
        // If it succeeds, the embedding should still be valid
        if let Ok(embedding) = result {
            assert_eq!(embedding.len(), 384);
            let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    /// Test embedding dimension accessor
    #[test]
    #[ignore]
    fn test_embedding_dim() {
        let mut embedder = OnnxEmbedder::new(
            "models/model.onnx",
            "models/tokenizer.json",
        )
        .expect("Failed to create embedder");

        assert_eq!(
            embedder.embedding_dim(),
            384,
            "MiniLM-L6-v2 should have 384 dimensions"
        );
    }

    #[test]
    fn test_l2_normalize() {
        let vector = vec![3.0, 4.0];
        let normalized = l2_normalize(&vector);

        // 3-4-5 triangle: norm = 5.0
        assert_eq!(normalized.len(), 2);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Check unit norm
        let norm: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let vector = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&vector);

        // Zero vector should remain unchanged
        assert_eq!(normalized, vector);
    }

    #[test]
    fn test_mean_pooling() {
        // Create a simple 2x3 hidden state matrix
        let hidden_states = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();

        // All tokens are valid
        let attention_mask = vec![1, 1, 1];
        let result = mean_pooling(&hidden_states, &attention_mask).unwrap();

        // Mean should be [3.0, 4.0]
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);

        // Only first two tokens are valid
        let attention_mask = vec![1, 1, 0];
        let result = mean_pooling(&hidden_states, &attention_mask).unwrap();

        // Mean should be [2.0, 3.0] (average of first two rows)
        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pooling_all_masked() {
        let hidden_states = Array2::from_shape_vec(
            (2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let attention_mask = vec![0, 0];
        let result = mean_pooling(&hidden_states, &attention_mask);

        assert!(result.is_err(), "All masked tokens should return an error");
    }
}
