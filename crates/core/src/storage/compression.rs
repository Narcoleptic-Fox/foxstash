//! Compression codecs for efficient storage
//!
//! This module provides multiple compression algorithms with automatic fallback
//! and detailed statistics. Compressed data includes a 4-byte header for codec
//! detection and version information.
//!
//! # Supported Codecs
//!
//! - **LZ4**: Fast compression with good ratios (feature: `lz4`)
//! - **Zstd**: Better compression ratios, slightly slower (feature: `zstd`)
//! - **Gzip**: Always available fallback via flate2
//! - **None**: No compression (passthrough)
//!
//! # Examples
//!
//! ```
//! use foxstash_core::storage::compression::{compress, decompress, best_codec};
//!
//! let data = b"Hello, World! This is some test data to compress.".repeat(100);
//!
//! // Compress with best available codec
//! let (compressed, stats) = compress(&data).unwrap();
//! println!("Compressed {} bytes to {} bytes ({:.2}x ratio) using {:?}",
//!          stats.original_size, stats.compressed_size, stats.ratio, stats.codec);
//!
//! // Decompress (codec detected automatically from header)
//! let decompressed = decompress(&compressed).unwrap();
//! assert_eq!(data.as_slice(), decompressed.as_slice());
//! ```

use std::io::{self, Write};
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Compression codec identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    /// No compression (passthrough)
    None,
    /// Gzip compression (always available)
    Gzip,
    /// LZ4 compression (requires `lz4` feature)
    #[cfg(feature = "lz4")]
    Lz4,
    /// Zstd compression (requires `zstd` feature)
    #[cfg(feature = "zstd")]
    Zstd,
}

impl Codec {
    /// Get the codec ID for the header
    fn id(&self) -> u8 {
        match self {
            Codec::None => 0,
            Codec::Gzip => 1,
            #[cfg(feature = "lz4")]
            Codec::Lz4 => 2,
            #[cfg(feature = "zstd")]
            Codec::Zstd => 3,
        }
    }

    /// Parse codec from ID
    fn from_id(id: u8) -> Result<Self, CompressionError> {
        match id {
            0 => Ok(Codec::None),
            1 => Ok(Codec::Gzip),
            #[cfg(feature = "lz4")]
            2 => Ok(Codec::Lz4),
            #[cfg(not(feature = "lz4"))]
            2 => Err(CompressionError::UnsupportedCodec("LZ4 feature not enabled".to_string())),
            #[cfg(feature = "zstd")]
            3 => Ok(Codec::Zstd),
            #[cfg(not(feature = "zstd"))]
            3 => Err(CompressionError::UnsupportedCodec("Zstd feature not enabled".to_string())),
            _ => Err(CompressionError::InvalidHeader(format!("Unknown codec ID: {}", id))),
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Codec::None => "None",
            Codec::Gzip => "Gzip",
            #[cfg(feature = "lz4")]
            Codec::Lz4 => "LZ4",
            #[cfg(feature = "zstd")]
            Codec::Zstd => "Zstd",
        }
    }
}

/// Statistics about a compression operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Original uncompressed size in bytes
    pub original_size: usize,
    /// Compressed size in bytes (including header)
    pub compressed_size: usize,
    /// Compression ratio (original / compressed)
    pub ratio: f64,
    /// Codec used for compression
    pub codec: Codec,
    /// Compression duration in milliseconds
    pub duration_ms: f64,
}

impl CompressionStats {
    /// Calculate space saved in bytes
    pub fn space_saved(&self) -> i64 {
        self.original_size as i64 - self.compressed_size as i64
    }

    /// Calculate space saved as percentage
    pub fn space_saved_percent(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            (self.space_saved() as f64 / self.original_size as f64) * 100.0
        }
    }

    /// Compression throughput in MB/s
    pub fn throughput_mbps(&self) -> f64 {
        if self.duration_ms == 0.0 {
            0.0
        } else {
            (self.original_size as f64 / 1_000_000.0) / (self.duration_ms / 1000.0)
        }
    }
}

/// Error types for compression operations
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
}

/// Magic header: [codec_id, version, reserved, reserved]
const HEADER_SIZE: usize = 4;
const VERSION: u8 = 1;

/// Create compression header
fn create_header(codec: Codec) -> [u8; HEADER_SIZE] {
    [codec.id(), VERSION, 0, 0]
}

/// Parse compression header
fn parse_header(data: &[u8]) -> Result<(Codec, usize), CompressionError> {
    if data.len() < HEADER_SIZE {
        return Err(CompressionError::InvalidHeader(
            format!("Data too small: {} bytes", data.len())
        ));
    }

    let codec = Codec::from_id(data[0])?;
    let version = data[1];

    if version != VERSION {
        return Err(CompressionError::InvalidHeader(
            format!("Unsupported version: {}", version)
        ));
    }

    Ok((codec, HEADER_SIZE))
}

/// Get the best available codec based on enabled features
///
/// Priority order: LZ4 > Zstd > Gzip
///
/// # Examples
///
/// ```
/// use foxstash_core::storage::compression::best_codec;
///
/// let codec = best_codec();
/// println!("Best available codec: {:?}", codec);
/// ```
pub fn best_codec() -> Codec {
    #[cfg(feature = "lz4")]
    {
        return Codec::Lz4;
    }

    #[cfg(all(feature = "zstd", not(feature = "lz4")))]
    {
        return Codec::Zstd;
    }

    #[cfg(not(any(feature = "lz4", feature = "zstd")))]
    {
        Codec::Gzip
    }
}

/// Compress data using the best available codec
///
/// This function automatically selects the best compression codec based on
/// enabled features and returns both the compressed data and statistics.
///
/// # Examples
///
/// ```
/// use foxstash_core::storage::compression::compress;
///
/// let data = b"Hello, World!".repeat(100);
/// let (compressed, stats) = compress(&data).unwrap();
/// assert!(stats.compressed_size < stats.original_size);
/// ```
pub fn compress(data: &[u8]) -> Result<(Vec<u8>, CompressionStats), CompressionError> {
    compress_with(data, best_codec())
}

/// Compress data using a specific codec
///
/// # Examples
///
/// ```
/// use foxstash_core::storage::compression::{compress_with, Codec};
///
/// let data = b"Hello, World!".repeat(100);
/// let (compressed, stats) = compress_with(&data, Codec::Gzip).unwrap();
/// assert_eq!(stats.codec, Codec::Gzip);
/// ```
pub fn compress_with(data: &[u8], codec: Codec) -> Result<(Vec<u8>, CompressionStats), CompressionError> {
    let start = Instant::now();
    let original_size = data.len();

    let header = create_header(codec);
    let mut compressed = Vec::with_capacity(HEADER_SIZE + data.len());
    compressed.extend_from_slice(&header);

    match codec {
        Codec::None => {
            compressed.extend_from_slice(data);
        }
        Codec::Gzip => {
            compress_gzip(data, &mut compressed)?;
        }
        #[cfg(feature = "lz4")]
        Codec::Lz4 => {
            compress_lz4(data, &mut compressed)?;
        }
        #[cfg(feature = "zstd")]
        Codec::Zstd => {
            compress_zstd(data, &mut compressed)?;
        }
    }

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let compressed_size = compressed.len();
    let ratio = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    let stats = CompressionStats {
        original_size,
        compressed_size,
        ratio,
        codec,
        duration_ms,
    };

    Ok((compressed, stats))
}

/// Decompress data (codec detected automatically from header)
///
/// # Examples
///
/// ```
/// use foxstash_core::storage::compression::{compress, decompress};
///
/// let original = b"Hello, World!".repeat(100);
/// let (compressed, _) = compress(&original).unwrap();
/// let decompressed = decompress(&compressed).unwrap();
/// assert_eq!(original.as_slice(), decompressed.as_slice());
/// ```
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let (codec, offset) = parse_header(data)?;
    let compressed_data = &data[offset..];

    match codec {
        Codec::None => Ok(compressed_data.to_vec()),
        Codec::Gzip => decompress_gzip(compressed_data),
        #[cfg(feature = "lz4")]
        Codec::Lz4 => decompress_lz4(compressed_data),
        #[cfg(feature = "zstd")]
        Codec::Zstd => decompress_zstd(compressed_data),
    }
}

// ============================================================================
// Codec Implementations
// ============================================================================

/// Compress using Gzip (always available)
fn compress_gzip(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let mut encoder = GzEncoder::new(output, Compression::default());
    encoder.write_all(data)?;
    encoder.finish()?;
    Ok(())
}

/// Decompress using Gzip
fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(data);
    let mut result = Vec::new();
    decoder.read_to_end(&mut result)
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
    Ok(result)
}

/// Compress using LZ4
///
/// Stores the original size as a 4-byte little-endian integer before the compressed data
/// to enable proper decompression.
#[cfg(feature = "lz4")]
fn compress_lz4(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    // Store the original size (4 bytes) so we can decompress correctly
    let original_size = data.len() as u32;
    output.extend_from_slice(&original_size.to_le_bytes());

    let compressed = lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(9)), false)
        .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
    output.extend_from_slice(&compressed);
    Ok(())
}

/// Decompress using LZ4
///
/// Reads the original size from the first 4 bytes, then decompresses the data.
#[cfg(feature = "lz4")]
fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    if data.len() < 4 {
        return Err(CompressionError::DecompressionFailed(
            "LZ4 data too small: missing size header".to_string()
        ));
    }

    // Read original size from first 4 bytes
    let size_bytes = [data[0], data[1], data[2], data[3]];
    let original_size = u32::from_le_bytes(size_bytes) as usize;

    // Decompress the remaining data
    let compressed_data = &data[4..];
    lz4::block::decompress(compressed_data, Some(original_size as i32))
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
}

/// Compress using Zstd
#[cfg(feature = "zstd")]
fn compress_zstd(data: &[u8], output: &mut Vec<u8>) -> Result<(), CompressionError> {
    let compressed = zstd::encode_all(data, 3)
        .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
    output.extend_from_slice(&compressed);
    Ok(())
}

/// Decompress using Zstd
#[cfg(feature = "zstd")]
fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    zstd::decode_all(data)
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate test data with varying characteristics
    fn generate_test_data(size: usize, compressibility: Compressibility) -> Vec<u8> {
        match compressibility {
            Compressibility::Random => {
                // Random data (incompressible)
                (0..size).map(|i| (i * 7 + 13) as u8).collect()
            }
            Compressibility::Repeated => {
                // Highly compressible repeated pattern
                b"Hello, World! ".repeat(size / 14 + 1)[..size].to_vec()
            }
            Compressibility::Structured => {
                // JSON-like structured data
                let json = r#"{"id": "doc-001", "content": "This is a test document", "score": 0.95}"#;
                json.repeat(size / json.len() + 1).as_bytes()[..size].to_vec()
            }
        }
    }

    enum Compressibility {
        Random,
        Repeated,
        Structured,
    }

    #[test]
    fn test_codec_id_roundtrip() {
        // Test all codec IDs can be converted back and forth
        for &id in &[0u8, 1u8] {
            let codec = Codec::from_id(id).unwrap();
            assert_eq!(codec.id(), id);
        }

        #[cfg(feature = "lz4")]
        {
            let codec = Codec::from_id(2).unwrap();
            assert_eq!(codec.id(), 2);
        }

        #[cfg(feature = "zstd")]
        {
            let codec = Codec::from_id(3).unwrap();
            assert_eq!(codec.id(), 3);
        }
    }

    #[test]
    fn test_invalid_codec_id() {
        let result = Codec::from_id(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_best_codec_available() {
        let codec = best_codec();
        // Should always return a valid codec
        assert!(!codec.name().is_empty());

        #[cfg(feature = "lz4")]
        assert_eq!(codec, Codec::Lz4);

        #[cfg(all(feature = "zstd", not(feature = "lz4")))]
        assert_eq!(codec, Codec::Zstd);

        #[cfg(not(any(feature = "lz4", feature = "zstd")))]
        assert_eq!(codec, Codec::Gzip);
    }

    #[test]
    fn test_compression_none() {
        let data = b"Hello, World!";
        let (compressed, stats) = compress_with(data, Codec::None).unwrap();

        assert_eq!(stats.codec, Codec::None);
        assert_eq!(stats.original_size, data.len());
        assert_eq!(stats.compressed_size, data.len() + HEADER_SIZE);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_compression_gzip() {
        let data = generate_test_data(1000, Compressibility::Repeated);
        let (compressed, stats) = compress_with(&data, Codec::Gzip).unwrap();

        assert_eq!(stats.codec, Codec::Gzip);
        assert_eq!(stats.original_size, data.len());
        assert!(stats.compressed_size < data.len(), "Gzip should compress repeated data");
        assert!(stats.ratio > 1.0);
        assert!(stats.duration_ms >= 0.0);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn test_compression_lz4() {
        let data = generate_test_data(1000, Compressibility::Repeated);
        let (compressed, stats) = compress_with(&data, Codec::Lz4).unwrap();

        assert_eq!(stats.codec, Codec::Lz4);
        assert_eq!(stats.original_size, data.len());
        assert!(stats.compressed_size < data.len(), "LZ4 should compress repeated data");
        assert!(stats.ratio > 1.0);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_compression_zstd() {
        let data = generate_test_data(1000, Compressibility::Repeated);
        let (compressed, stats) = compress_with(&data, Codec::Zstd).unwrap();

        assert_eq!(stats.codec, Codec::Zstd);
        assert_eq!(stats.original_size, data.len());
        assert!(stats.compressed_size < data.len(), "Zstd should compress repeated data");
        assert!(stats.ratio > 1.0);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_roundtrip_best_codec() {
        let data = generate_test_data(5000, Compressibility::Structured);
        let (compressed, stats) = compress(&data).unwrap();

        println!("Best codec: {:?}, ratio: {:.2}x", stats.codec, stats.ratio);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_header_format() {
        let data = b"test data";
        let (compressed, _) = compress_with(data, Codec::Gzip).unwrap();

        // Check header is present and valid
        assert!(compressed.len() >= HEADER_SIZE);
        assert_eq!(compressed[0], Codec::Gzip.id());
        assert_eq!(compressed[1], VERSION);
    }

    #[test]
    fn test_invalid_header_too_small() {
        let data = &[1, 2]; // Too small
        let result = decompress(data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CompressionError::InvalidHeader(_)));
    }

    #[test]
    fn test_invalid_header_wrong_version() {
        let mut data = vec![1, 99, 0, 0]; // Wrong version
        data.extend_from_slice(b"some data");
        let result = decompress(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_stats() {
        let data = generate_test_data(10000, Compressibility::Repeated);
        let (_, stats) = compress(&data).unwrap();

        assert_eq!(stats.original_size, 10000);
        assert!(stats.compressed_size > 0);
        assert!(stats.ratio > 0.0);
        assert!(stats.duration_ms >= 0.0);

        // Test derived stats
        let saved = stats.space_saved();
        let saved_percent = stats.space_saved_percent();
        let throughput = stats.throughput_mbps();

        assert!(saved > 0, "Should save space on repeated data");
        assert!(saved_percent > 0.0 && saved_percent < 100.0);
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let (compressed, stats) = compress(data).unwrap();
        assert_eq!(stats.original_size, 0);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_large_data() {
        // Test with 1MB of data
        let data = generate_test_data(1_000_000, Compressibility::Structured);
        let (compressed, stats) = compress(&data).unwrap();

        println!("Large data compression: {:.2} MB -> {:.2} MB ({:.2}x, {:.2} MB/s)",
                 stats.original_size as f64 / 1_000_000.0,
                 stats.compressed_size as f64 / 1_000_000.0,
                 stats.ratio,
                 stats.throughput_mbps());

        assert!(stats.ratio > 1.0, "Should compress structured data");

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data.len(), decompressed.len());
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_random_data_incompressible() {
        let data = generate_test_data(1000, Compressibility::Random);
        let (compressed, stats) = compress(&data).unwrap();

        // Random data should not compress well
        // Note: Our pseudo-random data is still somewhat compressible
        // True random data would have ratio < 1.0 (expansion)
        println!("Random data ratio: {:.2}x", stats.ratio);
        assert!(stats.ratio < 10.0, "Random data should not compress as well as structured data");

        // Verify roundtrip works
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_embedding_vectors() {
        // Simulate embedding vectors (f32 arrays)
        let embeddings: Vec<f32> = (0..384).map(|i| (i as f32) * 0.001).collect();
        let data: Vec<u8> = embeddings.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let (compressed, stats) = compress(&data).unwrap();

        println!("Embedding compression: {} -> {} bytes ({:.2}x)",
                 stats.original_size, stats.compressed_size, stats.ratio);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Verify we can reconstruct embeddings
        let reconstructed: Vec<f32> = decompressed.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(embeddings, reconstructed);
    }

    #[test]
    fn test_codec_names() {
        assert_eq!(Codec::None.name(), "None");
        assert_eq!(Codec::Gzip.name(), "Gzip");

        #[cfg(feature = "lz4")]
        assert_eq!(Codec::Lz4.name(), "LZ4");

        #[cfg(feature = "zstd")]
        assert_eq!(Codec::Zstd.name(), "Zstd");
    }

    // ========================================================================
    // Benchmarks (run with --release for meaningful results)
    // ========================================================================

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored bench_
    fn bench_compression_comparison() {
        let test_cases = vec![
            ("Small Random", 1_000, Compressibility::Random),
            ("Small Repeated", 1_000, Compressibility::Repeated),
            ("Small Structured", 1_000, Compressibility::Structured),
            ("Large Random", 100_000, Compressibility::Random),
            ("Large Repeated", 100_000, Compressibility::Repeated),
            ("Large Structured", 100_000, Compressibility::Structured),
        ];

        println!("\n{:<20} {:<10} {:<12} {:<12} {:<10} {:<12}",
                 "Test Case", "Codec", "Original", "Compressed", "Ratio", "Speed (MB/s)");
        println!("{}", "=".repeat(85));

        for (name, size, comp_type) in test_cases {
            let data = generate_test_data(size, comp_type);

            // Test all available codecs
            let codecs = vec![
                Codec::None,
                Codec::Gzip,
                #[cfg(feature = "lz4")]
                Codec::Lz4,
                #[cfg(feature = "zstd")]
                Codec::Zstd,
            ];

            for codec in codecs {
                let (_, stats) = compress_with(&data, codec).unwrap();
                println!("{:<20} {:<10} {:<12} {:<12} {:<10.2} {:<12.2}",
                         name,
                         codec.name(),
                         stats.original_size,
                         stats.compressed_size,
                         stats.ratio,
                         stats.throughput_mbps());
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_decompression_speed() {
        let data = generate_test_data(1_000_000, Compressibility::Structured);
        let (compressed, comp_stats) = compress(&data).unwrap();

        println!("\nDecompression benchmark:");
        println!("Codec: {:?}, Compressed size: {} bytes", comp_stats.codec, compressed.len());

        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = decompress(&compressed).unwrap();
        }

        let elapsed = start.elapsed().as_secs_f64();
        let throughput = (data.len() * iterations) as f64 / 1_000_000.0 / elapsed;

        println!("Decompression throughput: {:.2} MB/s", throughput);
    }
}
