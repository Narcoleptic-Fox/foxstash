#!/usr/bin/env python3
"""Which config field is actually read on which code path? Derived from source, never hand-written.

Every builder bug in the 1.0 audit has one signature: a field with ZERO reads anywhere in an entry
point's call subtree, while its docs promise it applies there. `config.m`/`config.m0` (the parallel
builder used the array-capacity constants instead), then `use_heuristic` and `extend_candidates`
(the parallel builder never received them at all). Each is an empty cell below.

A diagram cannot fail, so it rots. This is the same picture, generated, so it can.

Two rules this tool must obey, both learned by getting them wrong:

  * Prose is not enforcement. `config.use_heuristic` written in a doc comment is not a read of
    `config.use_heuristic`. Comments and string literals are stripped before anything is counted --
    otherwise the tool certifies the bug it exists to find, since the docs describing an option sit
    right next to the code ignoring it.

  * A name is not a function. `search` is defined three times over (Searcher, HNSWIndex, and the
    VectorIndex impl). Keying by name kept only the last, silently collapsing the whole `search`
    column to empty. Resolving a call to one definition needs type inference we do not have, so a
    call is treated as reaching every definition of that name. That over-approximates reachability,
    which can only ever *hide* a missing read -- so a cell reported empty is trustworthy, and a
    cell reported `read` is merely probable. Guarded by expected_failures.txt, which pins the four
    cells this tool must still report empty at commit 9aa7def.

Usage: config_matrix.py [path/to/hnsw.rs]
"""
import re
import sys
import pathlib

DEFAULT = pathlib.Path(__file__).parent.parent / "crates/core/src/index/hnsw.rs"
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT).read_text()

FIELDS = re.findall(
    r"^\s+pub (\w+):", re.search(r"pub struct HNSWConfig \{(.*?)\n\}", src, re.S).group(1), re.M
)


def strip(line: str) -> str:
    """Remove comments and literals: only executable text may count as a read."""
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)  # string literals
    line = re.sub(r"'(?:[^'\\]|\\.)'", "''", line)  # char literals, not lifetimes
    return re.sub(r"//.*", "", line)  # line comments and doc comments


raw = src.split("\n")
end = next((i for i, l in enumerate(raw) if l.strip() in ("#[cfg(test)]", "mod tests {")), len(raw))
code = [strip(l) for l in raw[:end]]  # a field only tests read is a field nothing honours

# fn name -> [(start, end), ...], by brace depth over comment-free lines.
# `pending` handles multi-line signatures -- `fn search_inner(` puts its `{` several lines down.
# Requiring `{` on the fn line made those functions invisible, which is what made `ef_search` look
# dead: the only reader is `search_inner`.
fns: dict[str, list] = {}
stack: list = []
pending = None
for i, l in enumerate(code):
    m = re.match(r"\s*(?:pub(?:\([\w:]+\))? )?(?:const |unsafe |async )*fn (\w+)", l)
    if m:
        pending = (m.group(1), i)
    if pending and "{" in l:
        name, start = pending
        pending = None
        stack.append((name, start, l.count("{") - l.count("}")))
        continue
    if stack:
        name, start, depth = stack[-1]
        depth += l.count("{") - l.count("}")
        if depth <= 0:
            fns.setdefault(name, []).append((start, i))
            stack.pop()
        else:
            stack[-1] = (name, start, depth)


def body(fn: str) -> str:
    return "\n".join("\n".join(code[s : e + 1]) for s, e in fns[fn])


reads = {f: {x for x in FIELDS if re.search(rf"config\.{x}\b", body(f))} for f in fns}
calls = {f: {c for c in fns if c != f and re.search(rf"\b{c}\s*\(", body(f))} for f in fns}


def subtree(fn, seen=None):
    seen = set() if seen is None else seen
    if fn in seen or fn not in fns:
        return set()
    seen.add(fn)
    out = set(reads[fn])
    for c in calls[fn]:
        out |= subtree(c, seen)
    return out


# The public entry points. `insert` is deliberately NOT here: it resolves to `Visited::insert`, a
# bitset method. The per-node insertion path is `insert_node`, reached through `add`.
ENTRIES = {
    "build": "build",
    "build:Sequential": "build_sequential",
    "build:Parallel": "build_parallel",
    "add": "add",
    "search": "search",
}
absent = [e for e, fn in ENTRIES.items() if fn not in fns]
if absent:
    sys.exit(f"entry point(s) not found, so the parser is broken, not the code: {absent}")

reach = {e: subtree(fn) for e, fn in ENTRIES.items()}
w = max(len(f) for f in FIELDS) + 2
print(f"{'config field':<{w}}" + "".join(f"{e:>18}" for e in ENTRIES))
print("-" * (w + 18 * len(ENTRIES)))
for f in FIELDS:
    print(f"{f:<{w}}" + "".join(f"{'read' if f in reach[e] else '.':>18}" for e in ENTRIES))

dead = [f for f in FIELDS if not any(f in reach[e] for e in ENTRIES)]
if dead:
    print("\nDEAD: no entry point reads these -- a public option nothing honours:", ", ".join(dead))
    sys.exit(1)
print("\nOK: every public HNSWConfig field is read on at least one entry path.")
