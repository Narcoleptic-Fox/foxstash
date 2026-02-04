# Nexus Local RAG Benchmarks

Comprehensive benchmarking suite for the Nexus Local RAG project using Criterion.rs.

## Benchmark Categories

### 1. Vector Operations (`vector_operations`)
Benchmarks core vector operations used in similarity search:
- `cosine_similarity_384d` - Cosine similarity for 384-dim vectors
- `cosine_similarity_128d` - Cosine similarity for 128-dim vectors
- `l2_distance_384d` - Euclidean distance for 384-dim vectors
- `dot_product_384d` - Dot product for 384-dim vectors
- `normalize_384d` - In-place normalization for 384-dim vectors
- `normalize_128d` - In-place normalization for 128-dim vectors

**Performance Target**: <1µs for 384-dimensional vectors

### 2. Flat Index (`flat_index`)
Benchmarks exact search using brute-force index:
- `add/{100,1000,10000}` - Index construction with varying document counts
- `add_batch/{100,1000,10000}` - Batch insertion performance
- `search_k5/{100,1000,10000}` - Search for 5 nearest neighbors
- `search_k10/{100,1000,10000}` - Search for 10 nearest neighbors

**Performance Target**: <1ms search for 1,000 documents

### 3. HNSW Index (`hnsw_index`)
Benchmarks approximate nearest neighbor search:
- `add/{100,1000,10000}` - Index construction with varying document counts
- `search_k5/{100,1000,10000}` - Search for 5 nearest neighbors
- `search_k10/{100,1000,10000}` - Search for 10 nearest neighbors
- `search_k20/{100,1000,10000}` - Search for 20 nearest neighbors

**Performance Target**: <10ms search for 10,000 documents

### 4. Embedding Generation (Not Yet Implemented)
Future benchmarks for ONNX-based embedding generation:
- Single text embedding (short, medium, long texts)
- Batch embedding (varying batch sizes: 1, 5, 10, 20, 50)
- Cache performance (hit, miss, uncached)

**Performance Target**: <30ms per embedding

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench -p nexus-rag-benches
```

### Run Specific Benchmark Group
```bash
# Vector operations only
cargo bench -p nexus-rag-benches --bench embedding vector_operations

# Flat index only
cargo bench -p nexus-rag-benches --bench embedding flat_index

# HNSW index only
cargo bench -p nexus-rag-benches --bench embedding hnsw_index
```

### Run Specific Benchmark
```bash
# Run only cosine similarity benchmark
cargo bench -p nexus-rag-benches --bench embedding 'cosine_similarity_384d$'

# Run only HNSW search with 10k documents
cargo bench -p nexus-rag-benches --bench embedding 'hnsw_index/search.*10000'
```

### Compare Performance Over Time
Criterion automatically saves baseline measurements. To compare:
```bash
# Save current performance as baseline
cargo bench -p nexus-rag-benches -- --save-baseline before

# Make changes to code...

# Compare against baseline
cargo bench -p nexus-rag-benches -- --baseline before
```

## Viewing Results

### Terminal Output
Benchmarks display results directly in the terminal with statistics:
- Mean execution time
- Standard deviation
- Outlier detection

### HTML Reports
Detailed reports are generated in `target/criterion/`:
```bash
# Open the main report
open target/criterion/report/index.html
```

### Example Output
```
vector_operations/cosine_similarity_384d
                        time:   [425.15 ns 427.20 ns 429.42 ns]
                        change: [-2.3421% -1.1234% +0.5678%] (p = 0.12 > 0.05)
                        No change in performance detected.

flat_index/search_k5/1000
                        time:   [847.32 µs 850.15 µs 853.45 µs]

hnsw_index/search_k5/10000
                        time:   [8.2451 ms 8.3012 ms 8.3589 ms]
```

## Performance Characteristics

### Vector Operations
- All operations show sub-microsecond performance for 384-dim vectors
- Operations are SIMD-optimized where possible
- Linear scaling with vector dimension

### Flat Index
- O(n) search complexity - linear with document count
- Exact results guaranteed
- Best for small datasets (<10,000 documents)

### HNSW Index
- O(log n) search complexity - logarithmic with document count
- Approximate results (high recall)
- Best for large datasets (>10,000 documents)
- Construction is slower than Flat but search is much faster

## Implementation Details

### Helper Functions
- `generate_random_embedding(dim, seed)` - Creates deterministic random vectors
- `create_test_document(id, embedding)` - Creates test documents
- `create_test_documents(count, dim, seed)` - Batch document creation

### Black Box Usage
All benchmarks use `black_box()` to prevent compiler optimizations from skewing results.

### Batch Size Configuration
Criterion is configured with appropriate sample sizes:
- Vector ops: 100 samples (fast operations)
- Index ops: 20-100 samples (slower operations with large datasets)

## Future Work

When the ONNX embedding module is implemented:
1. Uncomment embedding benchmark functions
2. Add the `onnx` feature flag
3. Update the criterion groups to include embedding benchmarks
4. Run with: `cargo bench -p nexus-rag-benches --features onnx`

## Troubleshooting

### Benchmarks Take Too Long
Reduce sample size or iterations:
```bash
cargo bench -p nexus-rag-benches -- --sample-size 10
```

### Inconsistent Results
- Close background applications
- Run benchmarks multiple times
- Check CPU frequency scaling settings
- Use `--save-baseline` for comparisons

### Memory Issues
For large document count benchmarks, ensure sufficient RAM:
- 10,000 docs × 384 dims × 4 bytes ≈ 15MB per index
- Plus overhead for graph structures in HNSW

## Contributing

When adding new benchmarks:
1. Follow the existing naming convention
2. Use appropriate sample sizes
3. Document performance targets
4. Add to the appropriate benchmark group
5. Update this README
