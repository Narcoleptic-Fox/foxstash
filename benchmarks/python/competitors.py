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
M_SWEEP = [16, 32]
EF_CONSTRUCTION = 200
EF_SWEEP = [10, 20, 50, 100, 200, 500]
CONTROL_N = 200

# Sweep M on EVERY library, not just ours.
#
# Lowering the graph degree improves foxstash's recall-per-distance-computation. It very
# likely does the same for hnswlib and faiss. Tuning our own M while pinning theirs at a
# default would manufacture a win — which is precisely the class of error that produced this
# repo's earlier false headlines. The honest comparison is each library's own Pareto
# frontier: for every library, the best QPS it can reach at a given recall, over all M.


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
    """Brute force. Must score ~100% or the ground truth does not match the base.

    Chunked, and argpartition rather than argsort. Sorting every row in full is O(n log n)
    over a million elements just to read off the top 10 — on GIST1M (1M x 960d) that burned
    ~18 minutes of single-core time and materialised a 1.6 GB distance matrix before the
    benchmark could even start. argpartition is O(n) and chunking keeps the matrix small.
    (`fetch-data.sh` already did it this way; this file never got the fix.)
    """
    q = query[:CONTROL_N]
    bn = (base * base).sum(1)
    out = np.empty((len(q), K), dtype=np.int64)
    for i in range(0, len(q), 50):
        qc = q[i:i + 50]
        d = bn[None, :] - 2.0 * (qc @ base.T)      # |q|^2 is constant per row; omit it
        top = np.argpartition(d, K, axis=1)[:, :K]
        rows = np.arange(top.shape[0])[:, None]
        out[i:i + 50] = top[rows, np.argsort(d[rows, top], axis=1)]
    return recall_at(out, truth[:CONTROL_N])


def bench_hnswlib(base, query, truth, m):
    n, dim = base.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=m)
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


def bench_faiss_hnsw(base, query, truth, m):
    n, dim = base.shape
    faiss.omp_set_num_threads(8)
    idx = faiss.IndexHNSWFlat(dim, m)
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
    print(f"single-threaded search, k={K}, ef_construction={EF_CONSTRUCTION}, M swept over {M_SWEEP}\n")

    c = control(base, query, truth)
    print(f"exact control (brute force, {CONTROL_N} queries): {c*100:.2f}%  "
          f"{'PASS' if c > 0.99 else '*** FAIL — output below is void ***'}")
    if c <= 0.99:
        sys.exit(1)

    for label, fn in [("hnswlib", bench_hnswlib), ("faiss-HNSW", bench_faiss_hnsw)]:
        for m in M_SWEEP:
            build, rows = fn(base, query, truth, m)
            print(f"\n=== {label}  M={m} ===  build: {build:.1f}s")
            print(f"{'ef':>6} {'recall@10':>11} {'QPS':>10}")
            print("-" * 30)
            for ef, r, qps in rows:
                print(f"{ef:>6} {r:>10.2f}% {qps:>10.0f}")


if __name__ == "__main__":
    main()
