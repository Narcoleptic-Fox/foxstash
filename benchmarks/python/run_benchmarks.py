#!/usr/bin/env python3
"""
ANN Benchmark Suite - Compare FAISS, hnswlib, Annoy, and Foxstash.

Measures:
- Recall@10, Recall@100
- Queries per second (QPS)
- Index build time
- Peak memory usage

Usage:
    python run_benchmarks.py --dataset sift10k   # Quick test
    python run_benchmarks.py --dataset sift1m    # Full benchmark
    python run_benchmarks.py --library faiss     # Single library
"""

import os
import sys
import json
import time
import gc
import tracemalloc
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from tqdm import tqdm
import psutil


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    library: str
    algorithm: str
    dataset: str
    index_size: int
    build_time_sec: float
    peak_memory_mb: float
    recall_at_10: float
    recall_at_100: float
    qps: float  # Queries per second
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    ef_search_values: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 500])
    n_trees_values: List[int] = field(default_factory=lambda: [10, 50, 100])
    nprobe_values: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    k_values: List[int] = field(default_factory=lambda: [10, 100])
    num_queries: int = 1000
    warmup_queries: int = 100


def load_dataset(data_dir: Path, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from numpy files."""
    dataset_dir = data_dir / name
    
    base = np.load(dataset_dir / "base.npy")
    query = np.load(dataset_dir / "query.npy")
    groundtruth = np.load(dataset_dir / "groundtruth.npy")
    
    return base, query, groundtruth


def compute_recall(retrieved: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    n_queries = len(retrieved)
    recall_sum = 0.0
    
    for i in range(n_queries):
        retrieved_set = set(retrieved[i][:k])
        gt_set = set(groundtruth[i][:k])
        recall_sum += len(retrieved_set & gt_set) / k
    
    return recall_sum / n_queries


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    gc.collect()
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, peak / (1024 * 1024)  # Convert to MB


# =============================================================================
# FAISS Benchmarks
# =============================================================================

def benchmark_faiss_flat(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                        config: BenchmarkConfig, dataset_name: str) -> List[BenchmarkResult]:
    """Benchmark FAISS Flat (brute force) index."""
    import faiss
    
    results = []
    dim = base.shape[1]
    
    print("\n  FAISS Flat (brute force)...")
    
    # Build index
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    
    index = faiss.IndexFlatL2(dim)
    index.add(base.astype(np.float32))
    
    build_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    # Warm up
    for _ in range(config.warmup_queries):
        index.search(query[:1], 100)
    
    # Benchmark search
    for k in config.k_values:
        query_subset = query[:config.num_queries]
        
        start = time.perf_counter()
        distances, indices = index.search(query_subset.astype(np.float32), k)
        search_time = time.perf_counter() - start
        
        qps = config.num_queries / search_time
        recall = compute_recall(indices, groundtruth[:config.num_queries], k)
        
        results.append(BenchmarkResult(
            library="faiss",
            algorithm="flat",
            dataset=dataset_name,
            index_size=len(base),
            build_time_sec=build_time,
            peak_memory_mb=peak_memory_mb,
            recall_at_10=recall if k >= 10 else 0.0,
            recall_at_100=recall if k >= 100 else 0.0,
            qps=qps,
            parameters={"k": k},
            notes="Brute force baseline (perfect recall)"
        ))
    
    del index
    gc.collect()
    
    return results


def benchmark_faiss_hnsw(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                        config: BenchmarkConfig, dataset_name: str) -> List[BenchmarkResult]:
    """Benchmark FAISS HNSW index."""
    import faiss
    
    results = []
    dim = base.shape[1]
    
    print("\n  FAISS HNSW...")
    
    # HNSW parameters
    M = 32  # Number of connections per layer
    efConstruction = 200
    
    # Build index
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.add(base.astype(np.float32))
    
    build_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print(f"    Build time: {build_time:.2f}s, Memory: {peak_memory_mb:.1f}MB")
    
    # Warm up
    index.hnsw.efSearch = 100
    for _ in range(config.warmup_queries):
        index.search(query[:1], 100)
    
    # Benchmark with different ef_search values
    for ef_search in config.ef_search_values:
        index.hnsw.efSearch = ef_search
        
        query_subset = query[:config.num_queries]
        
        start = time.perf_counter()
        distances, indices = index.search(query_subset.astype(np.float32), 100)
        search_time = time.perf_counter() - start
        
        qps = config.num_queries / search_time
        recall_10 = compute_recall(indices, groundtruth[:config.num_queries], 10)
        recall_100 = compute_recall(indices, groundtruth[:config.num_queries], 100)
        
        results.append(BenchmarkResult(
            library="faiss",
            algorithm="hnsw",
            dataset=dataset_name,
            index_size=len(base),
            build_time_sec=build_time,
            peak_memory_mb=peak_memory_mb,
            recall_at_10=recall_10,
            recall_at_100=recall_100,
            qps=qps,
            parameters={"M": M, "efConstruction": efConstruction, "efSearch": ef_search}
        ))
        
        print(f"    ef_search={ef_search}: R@10={recall_10:.4f}, R@100={recall_100:.4f}, QPS={qps:.0f}")
    
    del index
    gc.collect()
    
    return results


def benchmark_faiss_ivf(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                       config: BenchmarkConfig, dataset_name: str) -> List[BenchmarkResult]:
    """Benchmark FAISS IVF index."""
    import faiss
    
    results = []
    dim = base.shape[1]
    n_base = len(base)
    
    print("\n  FAISS IVF...")
    
    # IVF parameters - scale clusters with data size
    nlist = min(int(np.sqrt(n_base)), 1000)
    
    # Build index
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    
    # Train on subset
    train_size = min(n_base, 100_000)
    index.train(base[:train_size].astype(np.float32))
    index.add(base.astype(np.float32))
    
    build_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print(f"    Build time: {build_time:.2f}s, Memory: {peak_memory_mb:.1f}MB")
    
    # Warm up
    index.nprobe = 10
    for _ in range(config.warmup_queries):
        index.search(query[:1], 100)
    
    # Benchmark with different nprobe values
    for nprobe in config.nprobe_values:
        if nprobe > nlist:
            continue
            
        index.nprobe = nprobe
        
        query_subset = query[:config.num_queries]
        
        start = time.perf_counter()
        distances, indices = index.search(query_subset.astype(np.float32), 100)
        search_time = time.perf_counter() - start
        
        qps = config.num_queries / search_time
        recall_10 = compute_recall(indices, groundtruth[:config.num_queries], 10)
        recall_100 = compute_recall(indices, groundtruth[:config.num_queries], 100)
        
        results.append(BenchmarkResult(
            library="faiss",
            algorithm="ivf",
            dataset=dataset_name,
            index_size=len(base),
            build_time_sec=build_time,
            peak_memory_mb=peak_memory_mb,
            recall_at_10=recall_10,
            recall_at_100=recall_100,
            qps=qps,
            parameters={"nlist": nlist, "nprobe": nprobe}
        ))
        
        print(f"    nprobe={nprobe}: R@10={recall_10:.4f}, R@100={recall_100:.4f}, QPS={qps:.0f}")
    
    del index
    gc.collect()
    
    return results


# =============================================================================
# hnswlib Benchmarks
# =============================================================================

def benchmark_hnswlib(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                     config: BenchmarkConfig, dataset_name: str) -> List[BenchmarkResult]:
    """Benchmark hnswlib."""
    import hnswlib
    
    results = []
    dim = base.shape[1]
    n_base = len(base)
    
    print("\n  hnswlib...")
    
    # HNSW parameters
    M = 32
    ef_construction = 200
    
    # Build index
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n_base, ef_construction=ef_construction, M=M)
    index.add_items(base.astype(np.float32), np.arange(n_base))
    
    build_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print(f"    Build time: {build_time:.2f}s, Memory: {peak_memory_mb:.1f}MB")
    
    # Warm up
    index.set_ef(100)
    for _ in range(config.warmup_queries):
        index.knn_query(query[:1], k=100)
    
    # Benchmark with different ef values
    for ef in config.ef_search_values:
        index.set_ef(ef)
        
        query_subset = query[:config.num_queries]
        
        start = time.perf_counter()
        indices, distances = index.knn_query(query_subset.astype(np.float32), k=100)
        search_time = time.perf_counter() - start
        
        qps = config.num_queries / search_time
        recall_10 = compute_recall(indices, groundtruth[:config.num_queries], 10)
        recall_100 = compute_recall(indices, groundtruth[:config.num_queries], 100)
        
        results.append(BenchmarkResult(
            library="hnswlib",
            algorithm="hnsw",
            dataset=dataset_name,
            index_size=len(base),
            build_time_sec=build_time,
            peak_memory_mb=peak_memory_mb,
            recall_at_10=recall_10,
            recall_at_100=recall_100,
            qps=qps,
            parameters={"M": M, "ef_construction": ef_construction, "ef": ef}
        ))
        
        print(f"    ef={ef}: R@10={recall_10:.4f}, R@100={recall_100:.4f}, QPS={qps:.0f}")
    
    del index
    gc.collect()
    
    return results


# =============================================================================
# Annoy Benchmarks
# =============================================================================

def benchmark_annoy(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                   config: BenchmarkConfig, dataset_name: str) -> List[BenchmarkResult]:
    """Benchmark Annoy."""
    from annoy import AnnoyIndex
    
    results = []
    dim = base.shape[1]
    n_base = len(base)
    
    print("\n  Annoy...")
    
    # Test different number of trees
    for n_trees in config.n_trees_values:
        print(f"\n    n_trees={n_trees}")
        
        # Build index
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        index = AnnoyIndex(dim, 'euclidean')
        for i, vec in enumerate(base):
            index.add_item(i, vec)
        index.build(n_trees)
        
        build_time = time.perf_counter() - start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_memory / (1024 * 1024)
        
        print(f"      Build time: {build_time:.2f}s, Memory: {peak_memory_mb:.1f}MB")
        
        # Warm up
        for _ in range(config.warmup_queries):
            index.get_nns_by_vector(query[0], 100, search_k=-1)
        
        # Benchmark with different search_k values
        search_k_values = [n_trees * 100, n_trees * 500, n_trees * 1000, -1]
        
        for search_k in search_k_values:
            query_subset = query[:config.num_queries]
            
            start = time.perf_counter()
            indices = np.array([
                index.get_nns_by_vector(q, 100, search_k=search_k)
                for q in query_subset
            ])
            search_time = time.perf_counter() - start
            
            qps = config.num_queries / search_time
            recall_10 = compute_recall(indices, groundtruth[:config.num_queries], 10)
            recall_100 = compute_recall(indices, groundtruth[:config.num_queries], 100)
            
            results.append(BenchmarkResult(
                library="annoy",
                algorithm="annoy",
                dataset=dataset_name,
                index_size=len(base),
                build_time_sec=build_time,
                peak_memory_mb=peak_memory_mb,
                recall_at_10=recall_10,
                recall_at_100=recall_100,
                qps=qps,
                parameters={"n_trees": n_trees, "search_k": search_k}
            ))
            
            sk_str = str(search_k) if search_k > 0 else "all"
            print(f"      search_k={sk_str}: R@10={recall_10:.4f}, R@100={recall_100:.4f}, QPS={qps:.0f}")
        
        del index
        gc.collect()
    
    return results


# =============================================================================
# Foxstash Benchmarks (via subprocess)
# =============================================================================

def benchmark_foxstash(base: np.ndarray, query: np.ndarray, groundtruth: np.ndarray,
                      config: BenchmarkConfig, dataset_name: str,
                      foxstash_path: Optional[Path] = None) -> List[BenchmarkResult]:
    """Benchmark Foxstash HNSW (requires native binary or Python bindings)."""
    results = []
    
    print("\n  Foxstash HNSW...")
    print("    (Foxstash results come from Rust benchmarks - see Part 2)")
    
    # Placeholder for when we have Python bindings
    # For now, we document that Foxstash benchmarks run separately in Rust
    
    # We can still add synthetic results based on Rust benchmark output
    # This will be filled in when running the full suite
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmarks(data_dir: Path, dataset_name: str, 
                   libraries: List[str] = None,
                   output_dir: Optional[Path] = None) -> List[BenchmarkResult]:
    """Run complete benchmark suite."""
    
    if libraries is None:
        libraries = ["faiss", "hnswlib", "annoy"]
    
    print(f"\n{'='*60}")
    print(f"ANN Benchmark Suite")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Libraries: {', '.join(libraries)}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    base, query, groundtruth = load_dataset(data_dir, dataset_name)
    print(f"  Base: {base.shape}")
    print(f"  Query: {query.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    config = BenchmarkConfig()
    results = []
    
    # Run benchmarks for each library
    if "faiss" in libraries:
        print(f"\n{'='*60}")
        print("FAISS Benchmarks")
        print(f"{'='*60}")
        
        # Flat (baseline)
        results.extend(benchmark_faiss_flat(base, query, groundtruth, config, dataset_name))
        
        # HNSW
        results.extend(benchmark_faiss_hnsw(base, query, groundtruth, config, dataset_name))
        
        # IVF
        results.extend(benchmark_faiss_ivf(base, query, groundtruth, config, dataset_name))
    
    if "hnswlib" in libraries:
        print(f"\n{'='*60}")
        print("hnswlib Benchmarks")
        print(f"{'='*60}")
        results.extend(benchmark_hnswlib(base, query, groundtruth, config, dataset_name))
    
    if "annoy" in libraries:
        print(f"\n{'='*60}")
        print("Annoy Benchmarks")
        print(f"{'='*60}")
        results.extend(benchmark_annoy(base, query, groundtruth, config, dataset_name))
    
    if "foxstash" in libraries:
        print(f"\n{'='*60}")
        print("Foxstash Benchmarks")
        print(f"{'='*60}")
        results.extend(benchmark_foxstash(base, query, groundtruth, config, dataset_name))
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"results_{dataset_name}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of results."""
    from tabulate import tabulate
    
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # Group by library/algorithm
    table_data = []
    for r in results:
        table_data.append([
            f"{r.library}/{r.algorithm}",
            r.index_size,
            f"{r.build_time_sec:.2f}s",
            f"{r.peak_memory_mb:.1f}MB",
            f"{r.recall_at_10:.4f}",
            f"{r.recall_at_100:.4f}",
            f"{r.qps:.0f}",
            str(r.parameters)[:30]
        ])
    
    headers = ["Library/Algo", "Size", "Build", "Memory", "R@10", "R@100", "QPS", "Params"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ANN benchmarks")
    parser.add_argument("--data-dir", type=str, default="../data",
                       help="Directory containing datasets")
    parser.add_argument("--output-dir", type=str, default="../results",
                       help="Directory for output files")
    parser.add_argument("--dataset", type=str, default="sift10k",
                       choices=["sift10k", "sift1m"],
                       help="Dataset to benchmark")
    parser.add_argument("--library", type=str, nargs="+",
                       choices=["faiss", "hnswlib", "annoy", "foxstash", "all"],
                       default=["all"],
                       help="Libraries to benchmark")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Expand "all" to all libraries
    if "all" in args.library:
        libraries = ["faiss", "hnswlib", "annoy"]
    else:
        libraries = args.library
    
    # Check dataset exists
    dataset_dir = data_dir / args.dataset
    if not (dataset_dir / "base.npy").exists():
        print(f"Dataset not found: {dataset_dir}")
        print("Run download_datasets.py first:")
        print(f"  python download_datasets.py --dataset {args.dataset}")
        sys.exit(1)
    
    # Run benchmarks
    results = run_benchmarks(data_dir, args.dataset, libraries, output_dir)
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Benchmarks complete!")


if __name__ == "__main__":
    main()
