//! Example: Using the ONNX Embedder
//!
//! This example demonstrates how to use the OnnxEmbedder to generate embeddings.
//!
//! To run this example:
//! 1. Download the MiniLM-L6-v2 ONNX model and tokenizer
//! 2. Place them in a `models/` directory
//! 3. Run: `cargo run --example embedding_example --features onnx`

use nexus_rag_core::embedding::OnnxEmbedder;
use nexus_rag_core::Result;

fn main() -> Result<()> {
    println!("ONNX Embedder Example\n");

    // Create embedder with MiniLM-L6-v2 model
    let embedder = OnnxEmbedder::new(
        "models/model.onnx",
        "models/tokenizer.json",
    )?;

    println!("Embedder initialized successfully");
    println!("Embedding dimension: {}\n", embedder.embedding_dim());

    // Example 1: Single text embedding
    println!("Example 1: Single text embedding");
    let text = "The quick brown fox jumps over the lazy dog";
    let embedding = embedder.embed(text)?;

    println!("Text: {}", text);
    println!("Embedding length: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5]);

    // Verify normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("L2 norm: {:.6} (should be ~1.0)\n", norm);

    // Example 2: Batch embedding
    println!("Example 2: Batch embedding");
    let texts = vec![
        "Artificial intelligence is transforming technology",
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks with multiple layers",
    ];

    let embeddings = embedder.embed_batch(&texts)?;
    println!("Generated {} embeddings", embeddings.len());

    // Compute similarities between texts
    println!("\nSimilarity matrix:");
    for (i, text_i) in texts.iter().enumerate() {
        for (j, text_j) in texts.iter().enumerate() {
            if i <= j {
                let similarity: f32 = embeddings[i]
                    .iter()
                    .zip(&embeddings[j])
                    .map(|(a, b)| a * b)
                    .sum();
                println!("  Text {} <-> Text {}: {:.4}", i, j, similarity);
            }
        }
    }

    // Example 3: Semantic search
    println!("\nExample 3: Semantic search");
    let query = "neural networks and AI";
    let query_embedding = embedder.embed(query)?;

    println!("Query: {}", query);
    println!("\nRanked results:");

    let mut results: Vec<_> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let similarity: f32 = query_embedding
                .iter()
                .zip(&embeddings[i])
                .map(|(a, b)| a * b)
                .sum();
            (text, similarity)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (text, score)) in results.iter().enumerate() {
        println!("  {}. [Score: {:.4}] {}", i + 1, score, text);
    }

    Ok(())
}
