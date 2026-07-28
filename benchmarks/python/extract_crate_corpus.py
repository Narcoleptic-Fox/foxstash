#!/usr/bin/env python3
"""Extract a real-code corpus from permissively-licensed crates in the cargo registry.

# Why the cargo registry

A corpus needs to be *real* (see `extract_rustdoc_corpus.py` for why uniform-noise
vectors measure nothing) and it needs to be **safe to build tooling around in a
public repo**. Private source code satisfies the first and fails the second: a
`corpus.jsonl` containing private code looks exactly like test data, and
committing one to a public repository is a silent leak.

`~/.cargo/registry/src` solves both. It is already on disk (no download), it is
published open-source code, and every crate declares its licence in `Cargo.toml`
so the filter is machine-checkable rather than assumed.

# What it produces

One JSONL record per doc-commented item, carrying its provenance:

    {"id": ..., "content": ..., "crate": "serde-1.0.229",
     "license": "MIT OR Apache-2.0", "source": "src/de/mod.rs"}

Recording the licence per document is deliberate. The corpus is not meant to be
redistributed, but if it ever is, attribution is a field lookup rather than an
archaeology exercise.

# Filters, and why each exists

- **permissive licences only** — anything not clearly MIT/Apache/BSD/ISC/Unicode
  or missing is skipped, rather than assumed safe.
- **per-crate cap** — `windows-sys` and friends are enormous and formulaic; one
  crate dominating reproduces the `core::arch` problem that made the rustdoc
  corpus 58% near-duplicates.
- **near-duplicate filter** — generated bindings differ by a type name.

Pairs with `extract_rustdoc_corpus.py`: that one is API *prose*, this is *code*.
Together they give a heterogeneous distribution, which is a harder and more
honest test than either half.

# Usage

    python3 benchmarks/python/extract_crate_corpus.py --limit 40000 -o /tmp/crates.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys

# Licences whose text may be extracted and indexed locally without further
# thought. Deliberately a strict allow-list: an unrecognised or absent licence
# is skipped, never assumed.
PERMISSIVE = re.compile(
    r"\b(MIT|Apache-2\.0|BSD-2-Clause|BSD-3-Clause|ISC|Unicode-3\.0|Zlib|0BSD)\b"
)
# Anything with a copyleft or unknown term is excluded even if it also lists a
# permissive option, because parsing SPDX properly is out of scope here.
EXCLUDE = re.compile(r"\b(GPL|AGPL|LGPL|MPL|CDDL|EUPL|SSPL|proprietary)\b", re.I)

DOC_COMMENT = re.compile(r"^\s*(///|//!)\s?(.*)$")
MIN_CHARS = 100
MAX_CHARS = 2000


def crate_license(crate_dir: pathlib.Path) -> str | None:
    manifest = crate_dir / "Cargo.toml"
    try:
        text = manifest.read_text(encoding="utf8", errors="ignore")
    except OSError:
        return None
    m = re.search(r'^\s*license\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        return None
    lic = m.group(1)
    if EXCLUDE.search(lic) or not PERMISSIVE.search(lic):
        return None
    return lic


def doc_blocks(path: pathlib.Path) -> list[str]:
    """Contiguous runs of `///` / `//!` comments, as one document each.

    A run is the unit a human wrote as a single explanation, which makes it the
    right retrieval unit — unlike a fixed-size chunk, which cuts sentences in
    half and blurs the embedding.
    """
    try:
        lines = path.read_text(encoding="utf8", errors="ignore").splitlines()
    except OSError:
        return []
    out, buf = [], []
    for line in lines:
        m = DOC_COMMENT.match(line)
        if m:
            buf.append(m.group(2))
        elif buf:
            text = " ".join(buf).strip()
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                out.append(text)
            buf = []
    if buf:
        text = " ".join(buf).strip()
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            out.append(text)
    return out


def registry_roots() -> list[pathlib.Path]:
    try:
        home = subprocess.check_output(
            ["cargo", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        _ = home
    except (OSError, subprocess.CalledProcessError):
        pass
    base = pathlib.Path.home() / ".cargo" / "registry" / "src"
    return [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="crate_corpus.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--max-crate-share",
        type=float,
        default=0.05,
        help="max fraction of the corpus any one crate may contribute",
    )
    ap.add_argument(
        "--max-similar",
        type=int,
        default=2,
        help="max documents sharing an 80-char opening",
    )
    args = ap.parse_args()

    roots = registry_roots()
    if not roots:
        print("no cargo registry sources found under ~/.cargo/registry/src", file=sys.stderr)
        return 1

    by_crate: dict[str, list[dict]] = {}
    seen: set[str] = set()
    skipped_licence = 0
    for root in roots:
        for crate_dir in sorted(root.iterdir()):
            if not crate_dir.is_dir():
                continue
            lic = crate_license(crate_dir)
            if lic is None:
                skipped_licence += 1
                continue
            for rs in crate_dir.rglob("*.rs"):
                if any(p in {"target", "tests", "benches", "examples"} for p in rs.parts):
                    continue
                rel = str(rs.relative_to(crate_dir))
                for content in doc_blocks(rs):
                    ident = hashlib.sha256(content.encode("utf8")).hexdigest()[:16]
                    if ident in seen:
                        continue
                    seen.add(ident)
                    by_crate.setdefault(crate_dir.name, []).append(
                        {
                            "id": ident,
                            "content": content,
                            "crate": crate_dir.name,
                            "license": lic,
                            "source": rel,
                        }
                    )

    total = sum(len(v) for v in by_crate.values())
    if total == 0:
        print("no documents extracted", file=sys.stderr)
        return 1

    target = min(args.limit, total) if args.limit else total
    cap = max(20, int(target * args.max_crate_share))
    selected: list[dict] = []
    for name in sorted(by_crate):
        docs = by_crate[name]
        if len(docs) > cap:
            stride = len(docs) / cap
            docs = [docs[int(i * stride)] for i in range(cap)]
        selected.extend(docs)

    kept: dict[str, int] = collections.Counter()
    deduped = []
    for rec in selected:
        key = " ".join(rec["content"][:80].lower().split())
        if kept[key] >= args.max_similar:
            continue
        kept[key] += 1
        deduped.append(rec)
    dropped = len(selected) - len(deduped)
    selected = deduped

    if args.limit and len(selected) > args.limit:
        stride = len(selected) / args.limit
        selected = [selected[int(i * stride)] for i in range(args.limit)]

    with open(args.out, "w", encoding="utf8") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    licences = collections.Counter(r["license"] for r in selected)
    print(
        f"{len(selected)} documents -> {args.out}\n"
        f"  {total} available from {len(by_crate)} permissively-licensed crates "
        f"({skipped_licence} crates skipped: non-permissive or no licence field)\n"
        f"  per-crate cap {cap}, {dropped} near-duplicates dropped\n"
        f"  licences: {dict(licences.most_common(4))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
