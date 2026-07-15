#!/usr/bin/env python3
"""faiss's distance computations per query, at matched recall — the reference for
`cargo run --release -p foxstash-benches --example work_per_query`.

faiss exposes the counter foxstash now mirrors: `faiss.cvar.hnsw_stats.ndis`, the number
of distance computations performed. hnswlib's Python bindings do not expose theirs, which
is why faiss is the reference here — we sit within ~3% of it, so it is an equally good one.

The point of this number: QPS alone cannot tell you *why* one implementation is slower.
Either it does more work (higher dist/query — a worse graph, or a search that stops too
late) or it does the same work more slowly (same dist/query, worse ns/dist — a worse inner
loop or worse latency hiding). Those have completely different fixes.

Run on an IDLE machine, single-threaded. Usage: ./venv/bin/python work_per_query.py [dataset]
"""
import sys
import time

import faiss
import numpy as np

K = 10
M = 32
EF_CONSTRUCTION = 200
EF_SWEEP = [10, 20, 50, 100, 200, 500]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sift1m"
    d = f"../data/{name}"
    base = np.load(f"{d}/base.npy").astype(np.float32)
    query = np.load(f"{d}/query.npy").astype(np.float32)
    truth = np.load(f"{d}/groundtruth.npy")

    n, dim = base.shape
    print(f"{name} — {n} base x {dim}d, {len(query)} queries — faiss IndexHNSWFlat, "
          f"single-threaded, k={K}, M={M}\n")

    faiss.omp_set_num_threads(8)
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(base)

    faiss.omp_set_num_threads(1)
    print(f"{'ef':>6} {'recall@10':>11} {'QPS':>10} {'dist/query':>12} {'ns/dist':>10}")
    print("-" * 54)

    for ef in EF_SWEEP:
        index.hnsw.efSearch = ef
        index.search(query[:64], K)                       # warm

        faiss.cvar.hnsw_stats.reset()
        t = time.perf_counter()
        _, got = index.search(query, K)
        elapsed = time.perf_counter() - t

        ndis = faiss.cvar.hnsw_stats.ndis
        nq = len(query)
        recall = np.mean([len(set(got[i][:K]) & set(truth[i][:K])) / K for i in range(nq)])
        qps = nq / elapsed
        per_query = ndis / nq
        ns_per_dist = elapsed * 1e9 / ndis if ndis else float("nan")

        print(f"{ef:>6} {recall*100:>10.2f}% {qps:>10.0f} {per_query:>12.0f} {ns_per_dist:>10.1f}")


if __name__ == "__main__":
    main()
