#!/usr/bin/env python3
"""Produce EmbeddingGemma 2 Matryoshka-truncated datasets from the 768-d set (issue #51).

EmbeddingGemma 2 embeddings are Matryoshka: the leading `dim` components, L2-renormalized, are the
`dim`-d embedding (card dims 512 / 256 / 128). So the truncated datasets are built by slicing the
already-embedded 768-d `embgemma2-rustdocs45k` vectors, not by re-embedding. Ground truth is
recomputed per dim (truncation changes the exact neighbours), matching how `build_text_dataset.py`
derives it: exact top-100 by cosine over unit-norm vectors.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

SRC = pathlib.Path("benchmarks/data/embgemma2-rustdocs45k")
DIMS = [512, 256, 128]
TRUTH_K = 100


def normalize(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (a / n).astype(np.float32)


def main() -> int:
    base = np.load(SRC / "base.npy")
    query = np.load(SRC / "query.npy")
    for dim in DIMS:
        b = normalize(base[:, :dim])
        q = normalize(query[:, :dim])
        sims = q @ b.T
        truth = np.argpartition(-sims, TRUTH_K, axis=1)[:, :TRUTH_K]
        rows = np.arange(len(q))[:, None]
        truth = truth[rows, np.argsort(-sims[rows, truth], axis=1)].astype(np.int32)

        out = pathlib.Path(f"benchmarks/data/embgemma2-rustdocs45k-{dim}d")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "base.npy", b)
        np.save(out / "query.npy", q)
        np.save(out / "groundtruth.npy", truth)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "name": f"embgemma2-rustdocs45k-{dim}d",
                    "base": int(b.shape[0]),
                    "queries": int(q.shape[0]),
                    "dim": dim,
                    "embedder": "embeddinggemma-2-Q8_0.gguf",
                    "note": f"embgemma2-rustdocs45k truncated to {dim}d (Matryoshka) and renormalized.",
                    "metric": "cosine (unit-norm)",
                    "truth_k": TRUTH_K,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {out}: base {b.shape}, query {q.shape}")
        print(f'  MANIFEST: Spec {{ name: "embgemma2-rustdocs45k-{dim}d", base: 45_000, queries: 1_000, dim: {dim} }},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
