#!/usr/bin/env bash
# Fetch and build the ANN benchmark corpora in benchmarks/data/.
#
# Datasets are NOT committed. They are large, and — more importantly — a corpus
# sitting in git is a corpus nobody ever re-verifies. The repository shipped a
# `sift1m/` directory containing a 10,000-vector base for months. Benchmarking
# "SIFT1M" against it would have produced a completely plausible number off an
# index 100x smaller than its name claimed.
#
# So: rebuild from the canonical source, and verify before use. This script is
# idempotent — it skips any dataset already present and valid.
#
# Builds:
#   sift10k    10,000 base x 128d,     1,000 queries   (TEXMEX siftsmall, authors' GT)
#   sift100k   100,000 base x 128d,   10,000 queries   (prefix of sift1m, GT recomputed)
#   sift1m     1,000,000 base x 128d, 10,000 queries   (TEXMEX ANN_SIFT1M, authors' GT)
#   gist1m     1,000,000 base x 960d,  1,000 queries   (TEXMEX ANN_GIST1M, authors' GT)
#
# gist1m exists to settle a question SIFT cannot answer. A node block is
# `header(m0) + vector`, so which half dominates is a pure function of `dim`. At
# SIFT's 128d the adjacency dominates and 1-bit codes (Storage::RaBitQ) save
# almost nothing while wrecking the metric -- they lose ~12x. At 960d the vector
# dominates and the arithmetic inverts. Nobody runs RAG on 128-d vectors; MiniLM
# is 384d and OpenAI's embeddings are 1536d. Concluding anything about
# quantization from SIFT alone would be a dataset-generalization error.
#
# Usage: benchmarks/fetch-data.sh            (sift only -- gist is a 3.6 GB download)
#        benchmarks/fetch-data.sh --with-gist
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data
cd data

SIFT1M_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFTSMALL_URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
GIST1M_URL="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"

WITH_GIST=0
[ "${1:-}" = "--with-gist" ] && WITH_GIST=1

need() { [ ! -f "$1/base.npy" ]; }

if need sift1m || need sift100k; then
  if [ ! -d sift ]; then
    echo "==> downloading ANN_SIFT1M (~168 MB)"
    curl -# -o sift.tar.gz "$SIFT1M_URL"
    tar xzf sift.tar.gz
    rm -f sift.tar.gz
  fi
fi

if [ "$WITH_GIST" = 1 ] && need gist1m; then
  if [ ! -d gist ]; then
    echo "==> downloading ANN_GIST1M (~3.6 GB — this takes a while)"
    curl -# -o gist.tar.gz "$GIST1M_URL"
    tar xzf gist.tar.gz
    rm -f gist.tar.gz
  fi
fi

if need sift10k; then
  if [ ! -d siftsmall ]; then
    echo "==> downloading ANN_SIFT10K"
    curl -# -o siftsmall.tar.gz "$SIFTSMALL_URL"
    tar xzf siftsmall.tar.gz
    rm -f siftsmall.tar.gz
  fi
fi

echo "==> building .npy corpora"
python3 - <<'PY'
import numpy as np, os

def read_vecs(path, dtype=np.float32):
    """TEXMEX .fvecs/.ivecs: each row is <int32 dim><dim values>."""
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].view(dtype).copy()

def write(name, base, query, gt):
    os.makedirs(name, exist_ok=True)
    np.save(f"{name}/base.npy", base.astype(np.float32))
    np.save(f"{name}/query.npy", query.astype(np.float32))
    np.save(f"{name}/groundtruth.npy", gt.astype(np.int32))
    print(f"    {name}: {base.shape[0]} base x {base.shape[1]}d, {query.shape[0]} queries")

def exact_gt(base, query, k=100, chunk=500):
    """Exact-L2 top-k. Needed for any subset: the authors' GT indexes the FULL base."""
    bn = (base * base).sum(1)
    out = np.empty((query.shape[0], k), dtype=np.int32)
    for i in range(0, query.shape[0], chunk):
        q = query[i:i+chunk]
        d = bn[None, :] - 2.0 * (q @ base.T)      # |q|^2 is constant per row; omit
        top = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.arange(top.shape[0])[:, None]
        out[i:i+chunk] = top[rows, np.argsort(d[rows, top], axis=1)]
    return out

if not os.path.exists("sift10k/base.npy"):
    b = read_vecs("siftsmall/siftsmall_base.fvecs")
    q = read_vecs("siftsmall/siftsmall_query.fvecs")
    g = read_vecs("siftsmall/siftsmall_groundtruth.ivecs", np.int32)
    write("sift10k", b, q, g)

if not os.path.exists("sift1m/base.npy") or not os.path.exists("sift100k/base.npy"):
    base = read_vecs("sift/sift_base.fvecs")
    query = read_vecs("sift/sift_query.fvecs")
    gt = read_vecs("sift/sift_groundtruth.ivecs", np.int32)

    if not os.path.exists("sift1m/base.npy"):
        write("sift1m", base, query, gt)

    if not os.path.exists("sift100k/base.npy"):
        # A true prefix of the 1M base. The shipped GT indexes all 1,000,000
        # vectors, so it is meaningless for a subset — recompute it.
        print("    computing exact ground truth for the 100k subset...")
        sub = base[:100_000]
        write("sift100k", sub, query, exact_gt(sub, query))

if os.path.isdir("gist") and not os.path.exists("gist1m/base.npy"):
    write("gist1m",
          read_vecs("gist/gist_base.fvecs"),
          read_vecs("gist/gist_query.fvecs"),
          read_vecs("gist/gist_groundtruth.ivecs", np.int32))
PY

echo "==> verifying (exact-L2 control — a corpus that fails this is unusable)"
python3 - <<'PY'
import numpy as np, os, sys

EXPECT = {"sift10k": (10_000, 1_000), "sift100k": (100_000, 10_000), "sift1m": (1_000_000, 10_000),
          "gist1m": (1_000_000, 1_000)}
ok = True
for name, (nb, nq) in EXPECT.items():
    if not os.path.exists(f"{name}/base.npy"):
        continue   # gist is opt-in; absence is not failure
    b = np.load(f"{name}/base.npy", mmap_mode="r")
    q = np.load(f"{name}/query.npy", mmap_mode="r")
    g = np.load(f"{name}/groundtruth.npy", mmap_mode="r")

    if (b.shape[0], q.shape[0]) != (nb, nq):
        print(f"    {name}: FAIL — expected {nb} base / {nq} queries, "
              f"got {b.shape[0]} / {q.shape[0]}")
        ok = False
        continue
    if g.max() >= b.shape[0]:
        print(f"    {name}: FAIL — ground truth indexes vector {g.max()} of a "
              f"{b.shape[0]}-vector base")
        ok = False
        continue

    # Brute force a sample of queries and score the shipped GT against it.
    n = 300
    B = np.asarray(b, dtype=np.float32)
    Q = np.asarray(q[:n], dtype=np.float32)
    d = (B * B).sum(1)[None, :] - 2.0 * (Q @ B.T) + (Q * Q).sum(1)[:, None]
    top = np.argsort(d, axis=1)[:, :10]
    hit = np.mean([len(set(top[i]) & set(g[i][:10])) / 10 for i in range(n)])
    # <100% at 1M is float32 tie-breaking among equidistant points, not corruption.
    good = hit > 0.999
    ok &= good
    print(f"    {name}: {'PASS' if good else 'FAIL'} — {b.shape[0]:>7d} base, "
          f"exact-L2 control recall@10 = {hit*100:.2f}%")

sys.exit(0 if ok else 1)
PY

rm -rf sift siftsmall
echo "==> done"
