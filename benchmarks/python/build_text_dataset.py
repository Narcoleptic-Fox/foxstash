#!/usr/bin/env python3
"""Turn the extracted text corpora into a `benchmarks/data/` dataset of real embeddings.

# Why this exists

Every dataset in `benchmarks/data/` so far is SIFT or GIST — image descriptors.
Neither is a neural embedding, and this project's own findings say that matters
more than anything else about a dataset:

    nomic-768 scores RaBitQ 0.53 while distilroberta-768 scores 0.9993 — same
    dimension, same metric, only the embedder differs.

So accuracy is a property of the *distribution*, and dimension is only a cost.
SIFT and GIST can tell you about speed and about the graph; they cannot tell you
whether a quantizer will survive contact with the embeddings a user actually has.
This builds a third distribution from text this machine already holds.

# What it builds

    benchmarks/data/rustdocs45k/{base,query,groundtruth}.npy

45,000 base + 1,000 query vectors at 768 dim, from EmbeddingGemma-300M over two
different kinds of text: Rust API *prose* scraped from rustdoc HTML
(`extract_rustdoc_corpus.py`) and doc comments from library *source*
(`extract_crate_corpus.py`). Two sources rather than one because a single-source
corpus is a single mode, and a unimodal corpus flatters recall in the same
direction uniform noise does.

The mix is roughly 13% prose / 87% code, and that is a ceiling rather than a
choice: only 12,655 unique docblocks exist across std/core/alloc at all, and
6,680 survive the near-duplicate filter. The exact per-source counts are written
to `manifest.json` — the split is measured, not assumed, because "half and half"
was the intention and 13/87 is what the corpus actually contains.

# Queries are held-out documents, and both sides use the document prefix

The query set is 1,000 documents excluded from the base, and they are embedded
with the same `search_document:` prefix as the base rather than EmbeddingGemma's
`search_query:` prefix.

That is intentional. Recall@k here measures whether the *index* returns what
exact search would have returned — a pure geometry question about the graph and
the quantizer. Mixing prefixes would rotate the query into a different region of
the embedding space and fold "is the asymmetric prefix any good" into a number
that is supposed to isolate the index. Held-out documents also avoid the
self-retrieval trap: a query that is already in the index scores well on a
provably broken metric, which is how a quantized index once measured 100%.

# Ground truth is computed here, not trusted

Exact top-100 by cosine over L2-normalized vectors, as one matmul. The vectors
are normalized by `llama-embedding` already, so cosine, dot and L2 all induce
the same ordering — but it is asserted rather than assumed, because a silently
unnormalized base would make the ground truth wrong without making it look wrong.

# Usage

    python3 benchmarks/python/extract_rustdoc_corpus.py -o /tmp/rustdoc.jsonl
    python3 benchmarks/python/extract_crate_corpus.py --limit 40000 -o /tmp/crates.jsonl
    python3 benchmarks/python/build_text_dataset.py /tmp/rustdoc.jsonl /tmp/crates.jsonl

Embedding is CPU-only and takes ~45 minutes for 46k documents. The ROCm build of
llama.cpp on this machine dies with `SCALE: invalid device function` on both the
dGPU and the iGPU, so it is a stale build missing kernels, not a device-split
problem — do not re-probe it without rebuilding the HIP backend first.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

# Held-out query count and neighbour depth. 100 matches the depth SIFT/GIST ship,
# so recall@k is comparable against them for any k <= 100.
N_QUERIES = 1_000
TRUTH_K = 100
DIM = 768

LLAMA_EMBEDDING = pathlib.Path.home() / "src/llama.cpp/build/bin/llama-embedding"
MODEL = pathlib.Path.home() / "models/embeddinggemma-300m/embeddinggemma-300M-F32.gguf"

# EmbeddingGemma is prefix-conditioned; the prefix is part of the input contract,
# not decoration. See the module docstring for why both sides use the same one.
DOC_PREFIX = "search_document: "


def embed_shard(batch: list[str], threads: int) -> np.ndarray | None:
    """One llama-embedding invocation. Returns None if the output came back bad.

    Bad output is expected occasionally — see `embed` for why — so this reports
    failure rather than raising, and the caller retries.
    """
    # Newline-delimited: llama-embedding treats each line as one sequence, so any
    # embedded newline would silently split one document into several and shift
    # every subsequent row. Collapse whitespace rather than trusting it.
    payload = "\n".join(" ".join(t.split()) for t in batch) + "\n"
    proc = subprocess.run(
        [
            str(LLAMA_EMBEDDING),
            "-m", str(MODEL),
            "-f", "/dev/stdin",
            "--pooling", "mean",
            "--embd-output-format", "array",
            "-c", "2048",
            "-b", "8192",
            "-ub", "2048",
            "-t", str(threads),
            "--no-warmup",
            # llama.cpp processes escape sequences in the prompt BY DEFAULT, so a
            # literal backslash-n inside a Rust doc example (`println!("a\n")`)
            # becomes a real newline and splits that document into two. It fails
            # silently: every subsequent row shifts by one, so the vectors stay
            # well-formed while no longer matching their documents. 100 lines came
            # back as 106 embeddings before this flag.
            "--no-escape",
        ],
        input=payload,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return None
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if len(rows) != len(batch):
        return None
    arr = np.asarray(rows, dtype=np.float32)
    return arr if arr.ndim == 2 and arr.shape[1] == DIM else None


def embed(texts: list[str], shard: int, threads: int, retries: int = 3) -> np.ndarray:
    """Embed `texts`, one line per text, in verified shards.

    # Why shards are small, and why every one is checked

    `llama-embedding` emits its JSON through llama.cpp's `LOG`, which writes into
    a **bounded ring buffer**. `--embd-output-format array` logs once per float,
    so a large shard produces millions of entries, overruns the ring, and silently
    drops some — the process still exits 0, having written a truncated array.

    It is not deterministic. The same 2,000-document shard was cut off at 7.3 MB
    on one run and 5.1 MB on the next. That is the dangerous shape of failure:
    had the output been merely *shorter* rather than un-parseable, it would have
    produced a perfectly well-formed matrix of the wrong vectors.

    So shards are small enough to stay well inside the ring, and each one is
    verified for row count and width before it is accepted. A shard that fails
    any check is re-run rather than trusted.
    """
    out = np.empty((len(texts), DIM), dtype=np.float32)
    written = 0
    for start in range(0, len(texts), shard):
        batch = texts[start : start + shard]
        for attempt in range(1, retries + 1):
            arr = embed_shard(batch, threads)
            if arr is not None:
                break
            print(f"  shard at {start} came back bad, retry {attempt}/{retries}", flush=True)
        else:
            raise SystemExit(
                f"shard at {start} failed {retries} times. Lower --shard: the output "
                f"is being truncated by llama.cpp's log ring buffer."
            )
        out[written : written + len(arr)] = arr
        written += len(arr)
        print(f"  embedded {written}/{len(texts)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpora", nargs="+", help="JSONL files with a `content` field")
    ap.add_argument("--name", default="rustdocs45k")
    ap.add_argument("--base", type=int, default=45_000)
    ap.add_argument("--out-root", default="benchmarks/data")
    # 500 is comfortably inside the log ring buffer; 2,000 truncates. See `embed`.
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    if not LLAMA_EMBEDDING.is_file() or not MODEL.is_file():
        print(f"need {LLAMA_EMBEDDING} and {MODEL}", file=sys.stderr)
        return 1

    # Merge, then deduplicate across corpora. The two extractors dedupe within
    # themselves but not against each other, and a crate vendoring a doc comment
    # that also appears in rustdoc would put the same vector in base and query —
    # which would show up as suspiciously perfect recall on exactly those rows.
    seen: set[str] = set()
    docs: list[dict] = []
    for path in args.corpora:
        n_before = len(docs)
        for line in open(path, encoding="utf8"):
            rec = json.loads(line)
            key = " ".join(rec["content"].split()).lower()
            if key in seen:
                continue
            seen.add(key)
            rec["corpus"] = pathlib.Path(path).stem
            docs.append(rec)
        print(f"{path}: +{len(docs) - n_before} unique")

    need = args.base + N_QUERIES
    if len(docs) < need:
        print(
            f"have {len(docs)} unique documents, need {need}. Raise --limit on the "
            f"extractors or lower --base.",
            file=sys.stderr,
        )
        return 1

    # Deterministic shuffle so base/query membership is reproducible, and so the
    # two corpora interleave instead of the base being all prose and the queries
    # all code.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(docs))[:need]
    picked = [docs[i] for i in order]
    base_docs, query_docs = picked[: args.base], picked[args.base :]

    print(f"\nembedding {len(base_docs)} base + {len(query_docs)} query documents")
    base = embed([DOC_PREFIX + d["content"] for d in base_docs], args.shard, args.threads)
    query = embed([DOC_PREFIX + d["content"] for d in query_docs], args.shard, args.threads)

    # Assert normalization rather than assuming it. If llama-embedding ever stops
    # L2-normalizing, cosine and L2 stop agreeing and the ground truth below would
    # be wrong while still looking perfectly well-formed.
    for label, arr in (("base", base), ("query", query)):
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(
                f"{label} vectors are not unit norm (min {norms.min():.4f}, "
                f"max {norms.max():.4f}) — cosine and L2 no longer agree, so the "
                f"ground truth below would be computed under a different metric "
                f"than the index uses.",
                file=sys.stderr,
            )
            return 1

    # Exact top-100 by cosine. One matmul: 1000x768 @ 768x45000.
    print("computing exact ground truth")
    sims = query @ base.T
    truth = np.argpartition(-sims, TRUTH_K, axis=1)[:, :TRUTH_K]
    rows = np.arange(len(query))[:, None]
    truth = truth[rows, np.argsort(-sims[rows, truth], axis=1)].astype(np.int32)

    out_dir = pathlib.Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "base.npy", base)
    np.save(out_dir / "query.npy", query)
    np.save(out_dir / "groundtruth.npy", truth)

    # The text that produced each vector, in base order then query order, so row i of
    # base.npy is line i here.
    #
    # Kept because the vectors alone cannot exercise the text half of the system. BM25
    # and hybrid search have only ever been benchmarked against synthetic content like
    # "document number 7 about topic 42", which gives an inverted index no real
    # vocabulary, no term-frequency skew and no shared stems — so hybrid ranking was
    # effectively unmeasured. Re-deriving this later would mean re-running the whole
    # embedding pass to guarantee the same rows.
    with open(out_dir / "documents.jsonl", "w", encoding="utf8") as fh:
        for split, group in (("base", base_docs), ("query", query_docs)):
            for rec in group:
                fh.write(
                    json.dumps(
                        {
                            "split": split,
                            "id": rec["id"],
                            "content": rec["content"],
                            "corpus": rec["corpus"],
                            "crate": rec.get("crate"),
                            "license": rec.get("license"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Provenance travels with the data. Which crates and licences went in is not
    # recoverable from a .npy, and a corpus whose origin nobody can reconstruct is
    # a corpus nobody can re-verify.
    meta = {
        "name": args.name,
        "base": len(base_docs),
        "queries": len(query_docs),
        "dim": int(base.shape[1]),
        "embedder": MODEL.name,
        "prefix": DOC_PREFIX,
        "metric": "cosine (unit-norm; L2 and dot induce the same order)",
        "truth_k": TRUTH_K,
        "seed": args.seed,
        "sources": args.corpora,
        # Measured, not assumed — see the module docstring on the 13/87 split.
        "source_mix": {
            c: sum(1 for d in picked if d["corpus"] == c)
            for c in sorted({d["corpus"] for d in picked})
        },
        "licenses": sorted({d["license"] for d in picked if "license" in d}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    nn1 = sims[np.arange(len(query)), truth[:, 0]]
    print(
        f"\nwrote {out_dir}\n"
        f"  base {base.shape}, query {query.shape}, truth {truth.shape}\n"
        f"  nearest-neighbour cosine: mean {nn1.mean():.3f}, "
        f"min {nn1.min():.3f}, max {nn1.max():.3f}"
    )
    print(
        f"\nAdd to MANIFEST in crates/benches/src/lib.rs:\n"
        f'    DatasetSpec {{ name: "{args.name}", base: {len(base_docs)}, '
        f"queries: {len(query_docs)}, dim: {base.shape[1]} }},"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
