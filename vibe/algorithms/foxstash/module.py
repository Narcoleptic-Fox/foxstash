import numpy as np

from ..base.module import BaseANN


class FoxstashHNSW(BaseANN):
    """VIBE wrapper around the `foxstash` PyO3 extension (crates/python in the foxstash repo).

    Named `FoxstashHNSW` rather than `Foxstash` to keep this harness-side class distinct from
    `foxstash.Foxstash`, the Rust extension type it wraps.
    """

    def __init__(self, metric, dim, m, ef_construction, storage, rerank_candidates):
        import foxstash

        self.metric = metric
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.storage = storage
        self.rerank_candidates = rerank_candidates
        self._index = foxstash.Foxstash(metric, dim, m, ef_construction, storage, rerank_candidates)

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
