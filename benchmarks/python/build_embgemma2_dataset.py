#!/usr/bin/env python3
"""Re-embed the rustdocs45k documents with EmbeddingGemma 2, for the #51 storage sweep.

Reads the EXISTING `benchmarks/data/rustdocs45k/documents.jsonl` (the exact 45,000 base +
1,000 query documents that rustdocs45k was built from) and re-embeds them with EmbeddingGemma 2,
so only the embedder changes between rustdocs45k (original EmbeddingGemma) and this set. That makes
the storage-mode comparison a clean v1-vs-v2 of the same text.

Methodology matches build_text_dataset.py deliberately: BOTH base and queries use the DOCUMENT
prefix. This benchmark isolates the index/quantizer geometry (does the index return what exact
search would?), not the asymmetric query/doc prefix quality, so a symmetric prefix is correct and
keeps the number comparable to rustdocs45k. The EmbeddingGemma 2 v2 prefix format is
`title: none | text: {content}`.

The embedding traps (log-ring-buffer truncation -> small verified shards; `--no-escape` so Rust
doc examples do not split; unit-norm assertion so cosine ground truth stays valid) are copied from
build_text_dataset.py verbatim, because they were each a real bug.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

TRUTH_K = 100
DIM = 768
LLAMA_EMBEDDING = pathlib.Path.home() / "src/llama.cpp-foxstash/build/bin/llama-embedding"
MODEL = pathlib.Path.home() / "models/embeddinggemma-2-Q8_0.gguf"
# EmbeddingGemma 2 document prompt (model card). `title: none` when there is no title.
DOC_PREFIX = "title: none | text: "
SRC = pathlib.Path("benchmarks/data/rustdocs45k/documents.jsonl")
OUT = pathlib.Path("benchmarks/data/embgemma2-rustdocs45k")


def embed_shard(batch: list[str], threads: int) -> np.ndarray | None:
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
    out = np.empty((len(texts), DIM), dtype=np.float32)
    written = 0
    for start in range(0, len(texts), shard):
        batch = texts[start : start + shard]
        for attempt in range(1, retries + 1):
            arr = embed_shard(batch, threads)
            if arr is not None:
                break
            print(f"  shard at {start} bad, retry {attempt}/{retries}", flush=True)
        else:
            raise SystemExit(f"shard at {start} failed {retries}x; lower --shard")
        out[written : written + len(arr)] = arr
        written += len(arr)
        print(f"  embedded {written}/{len(texts)}", flush=True)
    return out


def main() -> int:
    if not LLAMA_EMBEDDING.is_file() or not MODEL.is_file():
        print(f"need {LLAMA_EMBEDDING} and {MODEL}", file=sys.stderr)
        return 1

    base_docs, query_docs = [], []
    for line in open(SRC, encoding="utf8"):
        rec = json.loads(line)
        (base_docs if rec["split"] == "base" else query_docs).append(rec)
    print(f"{len(base_docs)} base + {len(query_docs)} query documents from {SRC}")

    base = embed([DOC_PREFIX + d["content"] for d in base_docs], 500, 8)
    query = embed([DOC_PREFIX + d["content"] for d in query_docs], 500, 8)

    for label, arr in (("base", base), ("query", query)):
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"{label} not unit norm ({norms.min():.4f}..{norms.max():.4f})", file=sys.stderr)
            return 1

    print("computing exact ground truth")
    sims = query @ base.T
    truth = np.argpartition(-sims, TRUTH_K, axis=1)[:, :TRUTH_K]
    rows = np.arange(len(query))[:, None]
    truth = truth[rows, np.argsort(-sims[rows, truth], axis=1)].astype(np.int32)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "base.npy", base)
    np.save(OUT / "query.npy", query)
    np.save(OUT / "groundtruth.npy", truth)
    meta = {
        "name": "embgemma2-rustdocs45k",
        "base": len(base_docs),
        "queries": len(query_docs),
        "dim": DIM,
        "embedder": MODEL.name,
        "prefix": DOC_PREFIX,
        "metric": "cosine (unit-norm)",
        "truth_k": TRUTH_K,
        "note": "same documents as rustdocs45k, re-embedded with EmbeddingGemma 2 (v1-vs-v2).",
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    nn1 = sims[np.arange(len(query)), truth[:, 0]]
    print(f"wrote {OUT}: base {base.shape}, query {query.shape}, truth {truth.shape}")
    print(f"  nearest-neighbour cosine: mean {nn1.mean():.3f}, min {nn1.min():.3f}")
    print(f'\nAdd to MANIFEST: DatasetSpec {{ name: "embgemma2-rustdocs45k", base: {len(base_docs)}, queries: {len(query_docs)}, dim: {DIM} }},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
