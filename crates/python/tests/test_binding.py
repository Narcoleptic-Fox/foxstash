"""End-to-end checks on the PyO3 binding, against brute-force ground truth computed here in numpy.

These test THE BINDING -- that numpy arrays cross the FFI boundary intact, that ids map back to the
caller's rows, that the dials are connected, that errors surface as exceptions. They are not a
re-validation of index quality; that lives in `benchmarks/` on real corpora, and the difference
turns out to matter a great deal (see `CLUSTER_SEP`).

Every recall assertion is scored against ground truth computed in this file from the same X. A
binding that silently mis-mapped row indices would still return k plausible integers per query and
would look perfectly healthy -- brute-force scoring is the only thing that tells them apart.

Run: maturin develop --release && pytest crates/python/tests -q
"""

import numpy as np
import pytest

import foxstash

DIM = 256
N = 5_000
NQ = 100
K = 10

# How far apart the synthetic clusters sit, relative to their own spread.
#
# THIS NUMBER DECIDES WHETHER RaBitQ CAN BE TESTED AT ALL, and finding that out cost a false alarm.
# RaBitQ stores one bit per dimension: the sign of each rotated, centred coordinate. When clusters
# are tight and far apart, that bit vector encodes WHICH CLUSTER a point is in and nothing about
# where it sits inside it -- while recall@10 is a purely within-cluster ranking problem. So the
# code is blind to exactly the question being asked, and recall collapses:
#
#     separation      f32      sq8   rabitq
#            6.0    99.9%    99.9%    27.5%
#            1.5   100.0%   100.0%    64.8%
#           0.75   100.0%   100.0%    79.8%
#
# On real 960-d GIST under cosine, the same code gets 97.98% (benchmarks/RESULTS.md). Real
# embeddings have local structure that a mixture of Gaussian blobs does not. The 27.5% is a fact
# about this fixture, not about the quantizer -- and it is the same trap as "never benchmark ANN
# recall on uniform-random vectors", one level down: the fixture that stresses the GRAPH is not the
# fixture that stresses the QUANTIZER.
CLUSTER_SEP = 0.75


def corpus(dim: int = DIM, sep: float = CLUSTER_SEP, seed: int = 0):
    """Returns (base, queries). Queries are HELD OUT of the same generator, never a fresh one.

    Drawing queries from a *different* set of cluster centers makes every query an outlier whose
    "nearest neighbours" are ill-conditioned peripheral points, and drags even exact f32 storage
    down to 83% -- which reads exactly like a broken index. It is not. It is a broken fixture.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(16, dim)).astype(np.float32) * sep
    both = centers[rng.integers(0, 16, N + NQ)] + rng.normal(size=(N + NQ, dim)).astype(np.float32)
    both = np.ascontiguousarray(both, dtype=np.float32)
    return both[:N], both[N:]


def brute_force(x, queries, metric, k=K):
    if metric == "cosine":
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        scores = -(qn @ xn.T)
    else:
        scores = ((queries[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    return np.argsort(scores, axis=1)[:, :k]


def recall_at_k(got, want):
    return float(np.mean([len(set(g) & set(w)) / len(w) for g, w in zip(got, want)]))


def build(x, metric, storage, rerank=0, ef=200, dim=DIM):
    ix = foxstash.Foxstash(
        metric=metric, dim=dim, m=16, ef_construction=200, storage=storage,
        rerank_candidates=rerank,
    )
    ix.fit(x)
    ix.set_query_arguments(ef)
    return ix


# floor is per storage mode, because a single floor would either be unreachable for RaBitQ on this
# fixture or so low it would pass on a broken f32 path.
@pytest.mark.parametrize(
    "metric,storage,rerank,floor",
    [
        ("euclidean", "f32", 0, 0.95),
        ("cosine", "f32", 0, 0.95),
        ("euclidean", "sq8", 0, 0.90),   # codes only, vectors dropped: a real, shipped config
        ("cosine", "sq8", 50, 0.95),
        ("cosine", "rabitq", 50, 0.70),  # see CLUSTER_SEP: 98% on real GIST, capped here
    ],
)
def test_recall_against_brute_force(metric, storage, rerank, floor):
    x, queries = corpus()
    ix = build(x, metric, storage, rerank)

    got = np.array([ix.query(q, K) for q in queries])
    assert got.shape == (NQ, K), f"binding returned {got.shape}, expected {(NQ, K)}"
    assert got.min() >= 0 and got.max() < N, "returned ids outside the corpus"

    recall = recall_at_k(got, brute_force(x, queries, metric))
    assert recall >= floor, (
        f"{metric}/{storage} rerank={rerank}: recall@{K} = {recall:.1%}, floor {floor:.0%}. "
        f"Scored against brute force computed in numpy from the same X."
    )


def test_ids_are_rows_of_x_and_not_a_permutation():
    """Query each row with its own vector and demand itself back.

    `build_parallel` shuffles insertion order internally. If that leaked into the ids, every recall
    number above would be scored against the wrong ground-truth rows -- silently, no crash, nothing
    looking wrong. Also pinned in Rust by
    `build_parallel_returns_original_row_indices_despite_its_shuffle`.
    """
    x, _ = corpus()
    ix = build(x, "euclidean", "f32", ef=400)
    probes = np.arange(0, N, 53)
    hits = np.array([ix.query(x[i], 1)[0] for i in probes])
    assert np.array_equal(hits, probes), (
        f"{(hits != probes).sum()} of {len(probes)} rows did not return themselves at k=1 -- "
        f"build_parallel's insertion shuffle has leaked into the ids."
    )


def test_ef_search_actually_moves_recall():
    """A dial nobody checks is a dial that might not be connected.

    The 1.0 audit was almost entirely options that went into the config struct and quietly nowhere.
    `ef_search` is the one read on no path but `search`, so nothing else would catch it.
    """
    x, queries = corpus()
    want = brute_force(x, queries, "euclidean")
    ix = build(x, "euclidean", "f32")

    ix.set_query_arguments(5)
    low = recall_at_k(np.array([ix.query(q, K) for q in queries]), want)
    ix.set_query_arguments(400)
    high = recall_at_k(np.array([ix.query(q, K) for q in queries]), want)

    assert high > low + 0.05, (
        f"ef=5 gave {low:.1%} and ef=400 gave {high:.1%}. ef_search is the recall/QPS dial; if an "
        f"80x change does not move recall, it is not wired to anything."
    )


def test_dimension_mismatch_is_rejected():
    x, _ = corpus()
    ix = foxstash.Foxstash(
        metric="cosine", dim=DIM + 1, m=16, ef_construction=200, storage="f32",
        rerank_candidates=0,
    )
    with pytest.raises(ValueError, match="dim"):
        ix.fit(x)


def test_unsupported_metric_fails_loudly():
    """VIBE's vocabulary includes "ip", "normalized" and "hamming". We implement none of them.

    Quietly falling back to L2 for a metric we do not support would produce a full set of
    confident, meaningless recall numbers -- which is worse than crashing.
    """
    with pytest.raises(ValueError, match="unsupported metric"):
        foxstash.Foxstash(
            metric="hamming", dim=DIM, m=16, ef_construction=200, storage="f32",
            rerank_candidates=0,
        )


def test_query_before_fit_errors_instead_of_panicking():
    ix = foxstash.Foxstash(
        metric="cosine", dim=DIM, m=16, ef_construction=200, storage="f32", rerank_candidates=0
    )
    with pytest.raises(ValueError, match="call fit"):
        ix.query(np.zeros(DIM, dtype=np.float32), K)


def test_rerank_on_a_codes_only_index_is_refused():
    """`rerank_candidates: 0` drops the full-precision vectors. You cannot rerank against nothing.

    The index used to PANIC in release on this config. It now refuses, and the refusal has to cross
    the FFI boundary as an exception rather than aborting the interpreter.
    """
    x, _ = corpus()
    ix = build(x, "euclidean", "sq8", rerank=0)
    with pytest.raises(ValueError):
        ix.set_query_arguments(100, 50)
