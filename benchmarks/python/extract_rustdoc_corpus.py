#!/usr/bin/env python3
"""Extract a real-text corpus from locally installed Rust API documentation.

# Why a real corpus

Every foxstash harness written so far generated vectors from an LCG — uniform
noise. That is the one distribution this project has a written rule against:

    "Never benchmark ANN recall on uniform-random vectors — they have no cluster
     structure and mask graph-connectivity bugs (every ANN scores ~60% on random
     data). Use clustered/real embeddings."

Uniform noise makes self-retrieval trivially easy, so "200/200 self-retrieval"
measured almost nothing. It also makes the text side meaningless: synthetic
content like "document number 7 about topic 42" gives BM25 no real vocabulary,
so hybrid search was never exercised either.

It matters most for the quantizers. This project's own findings record that
accuracy is *distribution-specific* — nomic-768 scores RaBitQ 0.53 while
distilroberta-768 scores 0.9993 at the same dimension and metric. A quantizer
evaluated on noise tells you nothing about a quantizer on embeddings.

# What this produces

One JSONL record per rustdoc item (a struct, function, trait or keyword
description), not per page: pages run to hundreds of kilobytes, while an item is
the unit a user would actually retrieve.

    {"id": "...", "content": "...", "source": "std/vec/struct.Vec.html"}

About **12,655 unique** items exist across std/core/alloc, yielding ~6,700 after
the module cap and near-duplicate filter below. (A first estimate said 500k: that
counted docblocks across a 400-page sample without deduplicating, and rustdoc
repeats the same trait and impl prose across dozens of pages. Count, do not
extrapolate.)

Two filters keep the distribution honest, because the point of a real corpus is
defeated by a *differently* unrealistic one:

- **per-module cap** — `core::arch` alone is a third of the raw corpus
- **near-duplicate filter** — SIMD intrinsic docs differ only in a type name

# Usage

    python3 benchmarks/python/extract_rustdoc_corpus.py --limit 50000 -o corpus.jsonl

The docs come from `rustup component add rust-docs` and live under the active
toolchain. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

# rustdoc wraps each item's prose in <div class="docblock">.
DOCBLOCK = re.compile(r'<div class="docblock">(.*?)</div>', re.S)
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"<(script|style).*?</\1>", re.S)
WS = re.compile(r"\s+")

# Too short and it carries no signal; too long and one item dominates a batch.
MIN_CHARS = 80
MAX_CHARS = 2000


def toolchain_doc_root() -> pathlib.Path | None:
    """Locate the active toolchain's HTML docs."""
    try:
        sysroot = subprocess.check_output(
            ["rustc", "--print", "sysroot"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    root = pathlib.Path(sysroot) / "share" / "doc" / "rust" / "html"
    return root if root.is_dir() else None


def module_of(rel: str) -> str:
    """Coarse module key: `std/sync/atomic/struct.X.html` -> `std/sync`."""
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def text_of(html_fragment: str) -> str:
    stripped = TAG.sub(" ", SCRIPT_STYLE.sub("", html_fragment))
    return WS.sub(" ", stripped).strip()


def extract(path: pathlib.Path, rel: str) -> list[dict]:
    try:
        html = path.read_text(encoding="utf8", errors="ignore")
    except OSError:
        return []
    out = []
    for block in DOCBLOCK.findall(html):
        content = text_of(block)
        if not (MIN_CHARS <= len(content) <= MAX_CHARS):
            continue
        # Content-addressed id: stable across runs, and dedupes the same prose
        # appearing on several pages (rustdoc repeats trait docs on impls).
        ident = hashlib.sha256(content.encode("utf8")).hexdigest()[:16]
        out.append({"id": ident, "content": content, "source": rel})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="rustdoc_corpus.jsonl")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="approximate document cap; 0 means everything",
    )
    ap.add_argument(
        "--max-module-share",
        type=float,
        default=0.08,
        help="max fraction of the corpus any one module may contribute",
    )
    ap.add_argument(
        "--max-similar",
        type=int,
        default=2,
        help="max documents sharing an 80-char opening (near-duplicate filter)",
    )
    ap.add_argument(
        "--crates",
        default="std,core,alloc",
        help="comma-separated doc subtrees to walk",
    )
    args = ap.parse_args()

    root = toolchain_doc_root()
    if root is None:
        print(
            "no local Rust HTML docs found — run `rustup component add rust-docs`",
            file=sys.stderr,
        )
        return 1

    files: list[pathlib.Path] = []
    for crate in args.crates.split(","):
        files.extend(sorted((root / crate.strip()).rglob("*.html")))
    if not files:
        print(f"no HTML under {root} for {args.crates}", file=sys.stderr)
        return 1

    # Walk everything, then subsample the RESULT. A first version estimated
    # items-per-page from a 400-page probe and strided the file list — the probe
    # was measured on rustdoc's alphabetical head, which is unusually doc-dense,
    # so it overshot by 12x and produced a corpus a fraction of the requested
    # size. Counting is cheap; guessing was not.
    seen: set[str] = set()
    by_module: dict[str, list[dict]] = {}
    for path in files:
        rel = str(path.relative_to(root))
        for rec in extract(path, rel):
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            by_module.setdefault(module_of(rel), []).append(rec)

    total = sum(len(v) for v in by_module.values())

    # Cap any single module's share. `core::arch` alone is ~40% of the corpus and
    # is SIMD intrinsics — thousands of near-identical sentences ("Computes the
    # absolute value of packed..."). Letting it dominate would replace one
    # unrealistic distribution (uniform noise) with another (near-duplicates),
    # and near-duplicates make recall look better than it is.
    # Cap against what actually EXISTS, not what was asked for. Computing it from
    # `--limit` alone meant a 50k request produced a 4000-doc cap over a 12.6k
    # corpus — a cap that never bound, leaving core::arch at 33% and a 58%
    # near-duplicate rate.
    target = min(args.limit, total) if args.limit else total
    cap = max(50, int(target * args.max_module_share))
    selected: list[dict] = []
    for mod in sorted(by_module):
        docs = by_module[mod]
        if len(docs) > cap:
            stride = len(docs) / cap
            docs = [docs[int(i * stride)] for i in range(cap)]
        selected.extend(docs)

    # Drop near-duplicates. rustdoc's SIMD intrinsics are thousands of sentences
    # differing only in a type name ("Computes the absolute value of packed i8 /
    # i16 / i32..."). They are real text, but a corpus made of them measures
    # retrieval against near-identical neighbours, which flatters recall in the
    # same way uniform noise flatters self-retrieval — a different unrealistic
    # distribution, not a fix for the first one.
    kept_prefix: dict[str, int] = {}
    deduped = []
    for rec in selected:
        key = " ".join(rec["content"][:80].lower().split())
        n = kept_prefix.get(key, 0)
        if n >= args.max_similar:
            continue
        kept_prefix[key] = n + 1
        deduped.append(rec)
    near_dupes_dropped = len(selected) - len(deduped)
    selected = deduped

    # Even subsample across modules, preserving topical spread.
    if args.limit and len(selected) > args.limit:
        stride = len(selected) / args.limit
        selected = [selected[int(i * stride)] for i in range(args.limit)]

    with open(args.out, "w", encoding="utf8") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"{len(selected)} documents -> {args.out} "
        f"({total} available across {len(by_module)} modules, "
        f"per-module cap {cap}, {near_dupes_dropped} near-duplicates dropped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
