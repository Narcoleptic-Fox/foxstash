#!/usr/bin/env python3
"""Recall/QPS curves for the libraries foxstash is measured against.

Usage:  ./venv/bin/python competitors.py [dataset]     # default sift10k

Every number here is **single-threaded**. hnswlib's `knn_query` defaults to
`num_threads=-1` — every core — and timing that against a single-threaded Rust
loop is how this project once published "foxstash is ~11x slower than hnswlib".
It was a 16-core number racing a 1-core number. `set_num_threads(1)` below is
not a detail; it is the whole point.

Recall is reported against the dataset's shipped exact-L2 ground truth, and an
exact brute-force control row is printed first. If the control is not ~100%, the
harness is broken and nothing else in the output means anything.
"""
import sys
import time

import faiss
import hnswlib
import numpy as np

K = 10
M = 32
EF_CONSTRUCTION = 200
EF_SWEEP = [10, 20, 50, 100, 200, 500]
CONTROL_N = 200


def load(name):
    d = f"../data/{name}"
    base = np.load(f"{d}/base.npy").astype(np.float32)
    query = np.load(f"{d}/query.npy").astype(np.float32)
    truth = np.load(f"{d}/groundtruth.npy")
    return base, query, truth


def recall_at(got, truth, k=K):
    """got: (nq, k) retrieved base-indices. truth: (nq, >=k) exact neighbours."""
    return np.mean([
        len(set(got[i][:k]) & set(truth[i][:k])) / k for i in range(len(got))
    ])


def control(base, query, truth):
    """Brute force. Must score ~100% or the ground truth does not match the base."""
    q = query[:CONTROL_N]
    d = (base * base).sum(1)[None, :] - 2.0 * (q @ base.T) + (q * q).sum(1)[:, None]
    got = np.argsort(d, axis=1)[:, :K]
    return recall_at(got, truth[:CONTROL_N])


def bench_hnswlib(base, query, truth):
    n, dim = base.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)
    idx.set_num_threads(-1)                       # build: use every core, like foxstash
    t = time.perf_counter()
    idx.add_items(base, np.arange(n))
    build = time.perf_counter() - t

    idx.set_num_threads(1)                        # SEARCH: single-threaded. The trap.
    rows = []
    for ef in EF_SWEEP:
        idx.set_ef(ef)
        idx.knn_query(query[:64], k=K)            # warm
        t = time.perf_counter()
        got, _ = idx.knn_query(query, k=K)
        qps = len(query) / (time.perf_counter() - t)
        rows.append((ef, recall_at(got, truth) * 100, qps))
    return build, rows


def bench_faiss_hnsw(base, query, truth):
    n, dim = base.shape
    faiss.omp_set_num_threads(8)
    idx = faiss.IndexHNSWFlat(dim, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    t = time.perf_counter()
    idx.add(base)
    build = time.perf_counter() - t

    faiss.omp_set_num_threads(1)                  # SEARCH: single-threaded
    rows = []
    for ef in EF_SWEEP:
        idx.hnsw.efSearch = ef
        idx.search(query[:64], K)                 # warm
        t = time.perf_counter()
        _, got = idx.search(query, K)
        qps = len(query) / (time.perf_counter() - t)
        rows.append((ef, recall_at(got, truth) * 100, qps))
    return build, rows


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sift10k"
    base, query, truth = load(name)
    print(f"{name} — {base.shape[0]} base x {base.shape[1]}d, {len(query)} queries")
    print(f"single-threaded search, k={K}, M={M}, ef_construction={EF_CONSTRUCTION}\n")

    c = control(base, query, truth)
    print(f"exact control (brute force, {CONTROL_N} queries): {c*100:.2f}%  "
          f"{'PASS' if c > 0.99 else '*** FAIL — output below is void ***'}")
    if c <= 0.99:
        sys.exit(1)

    for label, fn in [("hnswlib", bench_hnswlib), ("faiss-HNSW", bench_faiss_hnsw)]:
        build, rows = fn(base, query, truth)
        print(f"\n=== {label} ===  build: {build:.1f}s")
        print(f"{'ef':>6} {'recall@10':>11} {'QPS':>10}")
        print("-" * 30)
        for ef, r, qps in rows:
            print(f"{ef:>6} {r:>10.2f}% {qps:>10.0f}")


if __name__ == "__main__":
    main()
