import numpy as np

from ..base.module import BaseANN


class FoxstashHNSW(BaseANN):
    """VIBE wrapper around the `foxstash` PyO3 extension (crates/python in the foxstash repo).

    Named `FoxstashHNSW` rather than `Foxstash` to keep this harness-side class distinct from
    `foxstash.Foxstash`, the Rust extension type it wraps.
    """

    # VIBE labels a dataset of L2-normalized vectors "normalized"; foxstash's binding only knows
    # "euclidean" and "cosine". On unit-norm vectors the three candidate metrics are rank-identical
    # — cosine distance (1 - a·b), squared L2 (2 - 2a·b) and negative inner product are all monotone
    # in a·b — so the nearest-neighbour ORDER (and thus recall against the dataset's ground truth) is
    # the same whichever we pick. We map to "cosine": it is the metric these embedding datasets are
    # built for, and it is what foxstash's RaBitQ/SQ8 codebooks are tuned against at these dims.
    # ("ip" on UN-normalized vectors and "hamming" are genuinely different orderings with no foxstash
    # equivalent — left unmapped so they fail loudly rather than silently scoring the wrong metric.)
    _METRIC_MAP = {"normalized": "cosine", "cosine": "cosine", "euclidean": "euclidean"}

    def __init__(self, metric, dim, m, ef_construction, storage, rerank_candidates):
        import foxstash

        foxstash_metric = self._METRIC_MAP.get(metric, metric)
        self.metric = foxstash_metric
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.storage = storage
        self.rerank_candidates = rerank_candidates
        self._index = foxstash.Foxstash(foxstash_metric, dim, m, ef_construction, storage, rerank_candidates)

    def fit(self, X):
        # PyReadonlyArray2<f32> on the Rust side requires an exact dtype/contiguity match;
        # VIBE's datasets are typically already float32, but this is cheap insurance against
        # a float64 or non-contiguous slice getting through and failing inside the extension.
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._index.fit(X)

    def set_query_arguments(self, ef, rerank=None):
        self._index.set_query_arguments(ef, rerank)
        self.ef_query = ef

    def query(self, v, n):
        v = np.ascontiguousarray(v, dtype=np.float32)
        return self._index.query(v, n)

    def __str__(self):
        return str(self._index)
