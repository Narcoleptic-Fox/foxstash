#!/usr/bin/env python3
"""Quick benchmark comparing FAISS, hnswlib, and Annoy."""

import time
import numpy as np
import psutil
import gc

def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def compute_recall(results, groundtruth, k):
    """Compute recall@k."""
    recalls = []
    for i, (res, gt) in enumerate(zip(results, groundtruth)):
        gt_set = set(gt[:k])
        res_set = set(res[:k])
        recalls.append(len(gt_set & res_set) / k)
    return np.mean(recalls)

def benchmark_faiss(base, query, gt, ef_search=64):
    import faiss
    gc.collect()
    mem_before = get_memory_mb()
    
    # Build HNSW index
    dim = base.shape[1]
    M = 16
    
    t0 = time.time()
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 200
    index.add(base)
    build_time = time.time() - t0
    
    mem_after = get_memory_mb()
    
    # Search
    index.hnsw.efSearch = ef_search
    t0 = time.time()
    D, I = index.search(query, 100)
    search_time = time.time() - t0
    
    recall_10 = compute_recall(I, gt, 10)
    recall_100 = compute_recall(I, gt, 100)
    qps = len(query) / search_time
    
    return {
        'library': 'FAISS',
        'algorithm': f'HNSW (M={M}, ef={ef_search})',
        'build_time': build_time,
        'memory_mb': mem_after - mem_before,
        'recall@10': recall_10,
        'recall@100': recall_100,
        'qps': qps
    }

def benchmark_hnswlib(base, query, gt, ef_search=64):
    import hnswlib
    gc.collect()
    mem_before = get_memory_mb()
    
    dim = base.shape[1]
    M = 16
    
    t0 = time.time()
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=len(base), ef_construction=200, M=M)
    index.add_items(base)
    build_time = time.time() - t0
    
    mem_after = get_memory_mb()
    
    # Search
    index.set_ef(ef_search)
    t0 = time.time()
    labels, distances = index.knn_query(query, k=100)
    search_time = time.time() - t0
    
    recall_10 = compute_recall(labels, gt, 10)
    recall_100 = compute_recall(labels, gt, 100)
    qps = len(query) / search_time
    
    return {
        'library': 'hnswlib',
        'algorithm': f'HNSW (M={M}, ef={ef_search})',
        'build_time': build_time,
        'memory_mb': mem_after - mem_before,
        'recall@10': recall_10,
        'recall@100': recall_100,
        'qps': qps
    }

def benchmark_annoy(base, query, gt, n_trees=10):
    from annoy import AnnoyIndex
    gc.collect()
    mem_before = get_memory_mb()
    
    dim = base.shape[1]
    
    t0 = time.time()
    index = AnnoyIndex(dim, 'euclidean')
    for i, vec in enumerate(base):
        index.add_item(i, vec)
    index.build(n_trees)
    build_time = time.time() - t0
    
    mem_after = get_memory_mb()
    
    # Search
    t0 = time.time()
    results = []
    for q in query:
        results.append(index.get_nns_by_vector(q, 100))
    search_time = time.time() - t0
    
    results = np.array(results)
    recall_10 = compute_recall(results, gt, 10)
    recall_100 = compute_recall(results, gt, 100)
    qps = len(query) / search_time
    
    return {
        'library': 'Annoy',
        'algorithm': f'Trees (n={n_trees})',
        'build_time': build_time,
        'memory_mb': mem_after - mem_before,
        'recall@10': recall_10,
        'recall@100': recall_100,
        'qps': qps
    }

def main():
    import sys
    
    # Load data
    data_dir = '../data/sift100k/sift1m'  # Use 100K synthetic
    print(f"Loading data from {data_dir}...")
    
    base = np.load(f'{data_dir}/base.npy')
    query = np.load(f'{data_dir}/query.npy')
    gt = np.load(f'{data_dir}/groundtruth.npy')
    
    print(f"Dataset: {len(base):,} vectors, {base.shape[1]}d")
    print(f"Queries: {len(query):,}")
    print()
    
    results = []
    
    # FAISS
    print("Benchmarking FAISS HNSW...")
    for ef in [32, 64, 128]:
        r = benchmark_faiss(base, query, gt, ef_search=ef)
        results.append(r)
        print(f"  ef={ef}: recall@10={r['recall@10']:.3f}, recall@100={r['recall@100']:.3f}, QPS={r['qps']:.0f}")
    
    # hnswlib
    print("\nBenchmarking hnswlib...")
    for ef in [32, 64, 128]:
        r = benchmark_hnswlib(base, query, gt, ef_search=ef)
        results.append(r)
        print(f"  ef={ef}: recall@10={r['recall@10']:.3f}, recall@100={r['recall@100']:.3f}, QPS={r['qps']:.0f}")
    
    # Annoy
    print("\nBenchmarking Annoy...")
    for trees in [10, 50, 100]:
        r = benchmark_annoy(base, query, gt, n_trees=trees)
        results.append(r)
        print(f"  trees={trees}: recall@10={r['recall@10']:.3f}, recall@100={r['recall@100']:.3f}, QPS={r['qps']:.0f}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Library':<15} {'Algorithm':<25} {'Build(s)':<10} {'Mem(MB)':<10} {'R@10':<8} {'R@100':<8} {'QPS':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['library']:<15} {r['algorithm']:<25} {r['build_time']:<10.2f} {r['memory_mb']:<10.1f} {r['recall@10']:<8.3f} {r['recall@100']:<8.3f} {r['qps']:<10.0f}")
    
    # Save results
    import json
    with open('../data/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ../data/benchmark_results.json")

if __name__ == '__main__':
    main()
