# Storage Benchmarks

Comprehensive benchmarking suite for the Nexus Local RAG storage layer.

## Overview

The storage benchmarks (`benches/storage.rs`) provide extensive performance testing for:
- Compression codecs (None, Gzip, LZ4, Zstd)
- Serialization (Documents, FlatIndex, HNSWIndex)
- File storage operations
- Batch operations
- Realistic workloads

## Benchmark Categories

### 1. Compression Codec Benchmarks

Tests each codec on different data types:
- **Random data**: Incompressible baseline
- **JSON documents**: Structured text
- **Embedding vectors**: Float arrays
- **Repeated patterns**: Highly compressible

Measures:
- Compression time and throughput (MB/s)
- Decompression time and throughput (MB/s)
- Compression ratios
- Size comparisons

### 2. Compression Size Benchmarks

Tests how compression performance scales with data size:
- 1 KB, 10 KB, 100 KB, 1 MB test sizes
- Throughput measurements at each size
- Helps identify optimal chunk sizes

### 3. Serialization Benchmarks

Tests bincode serialization performance:
- **Documents**: Small (100B), Medium (1KB), Large (10KB)
- **FlatIndex**: 100, 1K, 10K documents
- **HNSWIndex**: 100, 1K, 10K documents

Measures both serialization and deserialization times.

### 4. File Storage Benchmarks

Tests actual file I/O operations:
- Save/load documents with different codecs
- Save/load FlatIndex with different sizes
- Save/load HNSWIndex with different sizes
- Uses temporary directories for isolation

### 5. Batch Operations Benchmarks

Tests bulk save/load performance:
- 10, 100, 1000 document batches
- Simulates bulk ingestion scenarios
- Measures per-document overhead

### 6. Realistic Workload Benchmarks

End-to-end scenarios:
- **Ingestion Pipeline**: Add 100 docs to index and persist
- **Cold Start**: Load 1000-doc index and search
- **Incremental Update**: Load, add 10 docs, save
- **HNSW Persistence**: Save and load 1000-doc HNSW index

## Running Benchmarks

### Run all storage benchmarks
```bash
cargo bench -p nexus-rag-benches storage
```

### Run specific benchmark groups
```bash
# Compression codecs
cargo bench -p nexus-rag-benches compression_codecs

# Compression at different sizes
cargo bench -p nexus-rag-benches compression_sizes

# Serialization
cargo bench -p nexus-rag-benches serialization

# File storage operations
cargo bench -p nexus-rag-benches file_storage

# Batch operations
cargo bench -p nexus-rag-benches batch_operations

# Realistic workloads
cargo bench -p nexus-rag-benches realistic_workloads
```

### Run specific benchmarks
```bash
# Benchmark specific codec and data type
cargo bench -p nexus-rag-benches "gzip_compress/json"
cargo bench -p nexus-rag-benches "lz4_decompress/embeddings"

# Benchmark specific size
cargo bench -p nexus-rag-benches "serialize_document/medium"
cargo bench -p nexus-rag-benches "save_flat_index/1000"
```

## Performance Targets

### Compression Throughput
- **LZ4**: 500+ MB/s (fast, moderate ratio)
- **Zstd**: 100-300 MB/s (best ratio, good speed)
- **Gzip**: 10-50 MB/s (good ratio, slower)

### Serialization
- Small documents (<1KB): <100μs
- FlatIndex (1K docs): <10ms
- HNSWIndex (1K docs): <50ms

### File I/O
- Save document: <5ms
- Load document: <2ms
- Save index (1K docs): <100ms
- Load index (1K docs): <50ms

## Current Status

**Note**: The benchmarks currently use mock implementations because the actual storage layer is being implemented by other agents. Once the core library's storage module is complete, the benchmarks will automatically test the real implementations.

### Mock vs Real Implementation

The benchmarks are structured to work with both:
- **Mock implementations** (current): Use bincode directly for compression/decompression
- **Real implementations** (future): Will use actual `FileStorage` and compression codecs

To switch from mock to real implementations, update the benchmark code to:
1. Import real `FileStorage` and `Codec` from `nexus_rag_core::storage`
2. Replace mock compress/decompress calls with real codec methods
3. Replace direct file I/O with `FileStorage` API calls

## Understanding Results

### Criterion Output

Criterion provides detailed statistics:
```
compression_codecs/gzip_compress/json
                        time:   [2.3ms 2.4ms 2.5ms]
                        thrpt:  [400.0 MB/s 416.7 MB/s 434.8 MB/s]
```

Key metrics:
- **time**: Mean, lower bound, upper bound (95% confidence)
- **thrpt**: Throughput in operations/sec or MB/s
- **change**: Performance change vs previous run

### Compression Ratios

During benchmarks, compression ratios are printed to stderr:
```
gzip compression ratio for json: 23.45% (original: 1048576 bytes, compressed: 245760 bytes)
```

Lower percentage = better compression (smaller file).

### Serialized Sizes

Benchmark output includes serialized sizes:
```
FlatIndex (1000 docs) serialized size: 1572864 bytes (1.50 MB)
HNSWIndex (1000 docs) serialized size: 2359296 bytes (2.25 MB)
```

## Interpreting Results

### Good Performance Indicators
- LZ4 compress/decompress: >500 MB/s
- Zstd compress: >100 MB/s, decompress: >300 MB/s
- Document serialization: <100μs
- Index load times: <100ms for 1K docs

### Red Flags
- Compression slower than 10 MB/s for any codec
- Serialization taking >1ms for small documents
- Index load times >500ms for 1K docs
- Compression ratios >100% (expansion) on text data

## Optimization Guide

### If Compression Is Slow
1. Check codec selection - LZ4 is fastest
2. Verify release mode (`--release`)
3. Consider chunk sizes
4. Profile with `cargo flamegraph`

### If Serialization Is Slow
1. Check for large metadata fields
2. Consider using `bincode::config()` with size limits
3. Profile struct layout (padding)

### If File I/O Is Slow
1. Check disk type (SSD vs HDD)
2. Measure with different filesystems
3. Consider write buffer sizes
4. Check for fsync overhead

### If Batch Operations Are Slow
1. Consider parallel writes
2. Check file handle overhead
3. Batch fsync operations
4. Use memory-mapped files

## Data Generation

The benchmarks use deterministic data generation:
- **Random bytes**: Cryptographic RNG with fixed seed
- **JSON documents**: Structured with realistic metadata
- **Embeddings**: Random f32 vectors in [-1, 1]
- **Repeated patterns**: Text repetition

All generators use fixed seeds for reproducibility.

## Integration with Real Storage

Once the storage module is complete, update benchmarks:

```rust
// Replace mock codec
use nexus_rag_core::storage::compression::Codec;

// Replace mock file operations
use nexus_rag_core::storage::FileStorage;

// In benchmarks, replace:
let compressed = codec.compress(&data).unwrap();

// With:
let (compressed, stats) = compression::compress_with(&data, codec).unwrap();
```

## Contributing

When adding new benchmarks:
1. Follow existing naming conventions
2. Use deterministic data generation
3. Include both small and large dataset tests
4. Add performance targets in comments
5. Measure both time and throughput
6. Use `BatchSize::SmallInput` for setup-heavy benchmarks

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Benchmark Design Patterns](https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_with_inputs.html)
- [Understanding Benchmark Results](https://bheisler.github.io/criterion.rs/book/user_guide/command_line_output.html)
