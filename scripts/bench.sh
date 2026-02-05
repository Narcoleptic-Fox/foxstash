#!/bin/bash
# Foxstash Benchmark Suite
# Run all benchmarks including Python library comparisons

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.bench-venv"

cd "$PROJECT_DIR"

echo "=== Foxstash Benchmark Suite ==="
echo ""

# Setup Python venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q numpy hnswlib faiss-cpu
    echo "Python environment ready."
    echo ""
fi

# Run Rust benchmarks
echo "=== Rust Benchmarks ==="
echo ""

echo "--- Quick Comparison (Foxstash vs instant-distance) ---"
cargo run -p foxstash-benches --example quick_comparison --release
echo ""

echo "--- Build Strategy Comparison ---"
cargo run -p foxstash-benches --example compare_strategies --release
echo ""

# Run Python benchmarks
echo "=== Python Library Comparison ==="
echo ""

"$VENV_DIR/bin/python" << 'PYTHON_BENCH'
import numpy as np
import time

NUM_VECTORS = 100_000
DIM = 128
NUM_QUERIES = 1000
K = 10

print(f"Dataset: {NUM_VECTORS:,} vectors, {DIM}d, {NUM_QUERIES:,} queries, top-{K}")
print()

np.random.seed(42)
base = np.random.randn(NUM_VECTORS, DIM).astype('float32')
base /= np.linalg.norm(base, axis=1, keepdims=True)

np.random.seed(123)
queries = np.random.randn(NUM_QUERIES, DIM).astype('float32')
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# Ground truth
print("Computing ground truth...")
ground_truth = []
for q in queries[:100]:
    dists = np.linalg.norm(base - q, axis=1)
    ground_truth.append(np.argsort(dists)[:K])

def measure_recall(results, truth):
    recall = 0
    for i in range(len(truth)):
        recall += len(set(results[i]) & set(truth[i])) / K
    return recall / len(truth)

# hnswlib
import hnswlib
idx = hnswlib.Index(space='l2', dim=DIM)
idx.init_index(max_elements=NUM_VECTORS, ef_construction=100, M=32)

start = time.time()
idx.add_items(base)
build_time = time.time() - start

idx.set_ef(64)
start = time.time()
for q in queries:
    idx.knn_query(q, k=K)
search_time = time.time() - start
qps = NUM_QUERIES / search_time

results, _ = idx.knn_query(queries[:100], k=K)
recall = measure_recall(results, ground_truth)

print(f"hnswlib:     Build {build_time:>6.2f}s | Search {qps:>5.0f} QPS | Recall@{K}: {recall*100:.1f}%")

# faiss
import faiss
idx = faiss.IndexHNSWFlat(DIM, 32)
idx.hnsw.efConstruction = 100

start = time.time()
idx.add(base)
build_time = time.time() - start

idx.hnsw.efSearch = 64
start = time.time()
for q in queries:
    idx.search(q.reshape(1, -1), K)
search_time = time.time() - start
qps = NUM_QUERIES / search_time

_, results = idx.search(queries[:100], K)
recall = measure_recall(results, ground_truth)

print(f"faiss-hnsw:  Build {build_time:>6.2f}s | Search {qps:>5.0f} QPS | Recall@{K}: {recall*100:.1f}%")
PYTHON_BENCH

echo ""
echo "=== Benchmarks Complete ==="
