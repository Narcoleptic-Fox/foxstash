//! PyO3 bindings for [`foxstash_core::index::hnsw::HNSWIndex`], built for ANN benchmark
//! harnesses (VIBE, and anything else shaped like ann-benchmarks) rather than as a general
//! embedding client — see `vibe/algorithms/foxstash/` for the harness-side wrapper.
//!
//! Built and tested against PyO3 0.29. Note `Python::detach` (formerly `allow_threads`) and
//! `Python::attach` (formerly `with_gil`) — renamed upstream once Python 3.13 free-threading made
//! "the GIL" the wrong noun for what is an attach/detach of this thread from the interpreter.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use foxstash_core::index::hnsw::{
    BuildStrategy, DistanceMetric, FilterMask, HNSWConfig, HNSWIndex, Storage,
};

/// VIBE's metric vocabulary is `"euclidean" | "cosine" | "ip" | "normalized" | "hamming"`
/// (see `vibe/distance.py` upstream — NOT ann-benchmarks' old `"angular"`). foxstash only
/// implements two of those; the rest fail loudly rather than silently picking one, which is
/// the whole point of making this a `Result` instead of a default.
fn parse_metric(s: &str) -> PyResult<DistanceMetric> {
    match s {
        "euclidean" => Ok(DistanceMetric::L2),
        "cosine" => Ok(DistanceMetric::Cosine),
        other => Err(PyValueError::new_err(format!(
            "foxstash: unsupported metric {other:?}. foxstash implements \"euclidean\" (-> L2) \
             and \"cosine\" (-> Cosine) only; VIBE's \"ip\", \"normalized\" and \"hamming\" have \
             no foxstash equivalent."
        ))),
    }
}

/// Parse a storage string into `(mode, bits)`. `bits` is meaningful only for the multi-bit
/// modes: `"turboquant"` defaults to 2 total bits, `"turboquant3"` requests 3; `"turborabit"`
/// defaults to 3 total bits, `"turborabit4"` requests 4. Encoding the bit budget in the string
/// keeps the positional constructor (and VIBE's caller) unchanged.
fn parse_storage(s: &str) -> PyResult<(Storage, usize)> {
    match s {
        "f32" => Ok((Storage::F32, 0)),
        "sq8" => Ok((Storage::SQ8, 0)),
        "rabitq" => Ok((Storage::RaBitQ, 0)),
        other if other.starts_with("turboquant") => {
            let bits = other["turboquant".len()..].parse::<usize>().unwrap_or(2);
            if !(1..=4).contains(&bits) {
                return Err(PyValueError::new_err(
                    "foxstash: turboquant bit budget must be in 1..=4 (e.g. \"turboquant2\") — \
                     the packed MSE kernel dequantizes through an 8-entry LUT.",
                ));
            }
            Ok((Storage::TurboQuant, bits))
        }
        other if other.starts_with("turborabit") => {
            let bits = other["turborabit".len()..].parse::<usize>().unwrap_or(3);
            if !(1..=4).contains(&bits) {
                return Err(PyValueError::new_err(
                    "foxstash: turborabit bit budget must be in 1..=4 (e.g. \"turborabit3\") — \
                     codes are nibble-packed, and b=4 already reaches F32 recall.",
                ));
            }
            Ok((Storage::TurboRabit, bits))
        }
        // Warren: turborabit's 4-bit walk + a two-level 8+8 residual rerank, no retained f32.
        // The bit budget is turborabit's, so "warren4" style suffixes route the same way.
        other if other.starts_with("warren") => {
            let bits = other["warren".len()..].parse::<usize>().unwrap_or(4);
            if !(1..=4).contains(&bits) {
                return Err(PyValueError::new_err(
                    "foxstash: warren bit budget must be in 1..=4 (e.g. \"warren4\") — it is \
                     turborabit's walk code, which is nibble-packed.",
                ));
            }
            Ok((Storage::Warren, bits))
        }
        other => Err(PyValueError::new_err(format!(
            "foxstash: unsupported storage {other:?}. Expected \"f32\", \"sq8\", \"rabitq\", \
             \"turboquant[N]\", \"turborabit[N]\", or \"warren[N]\"."
        ))),
    }
}

/// Wraps one [`HNSWIndex`]. `index` is `None` until [`Foxstash::fit`] has run — every other
/// method errors out with a clear message rather than panicking on the unwrap.
#[pyclass(module = "foxstash")]
struct Foxstash {
    index: Option<HNSWIndex>,
    metric: DistanceMetric,
    metric_arg: String,
    dim: usize,
    m: usize,
    ef_construction: usize,
    storage: Storage,
    storage_arg: String,
    turbo_bits: usize,
    rerank_candidates: usize,
    ef_query: usize,
}

impl Foxstash {
    fn built(&self) -> PyResult<&HNSWIndex> {
        self.index
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("foxstash: call fit() before query()"))
    }

    fn built_mut(&mut self) -> PyResult<&mut HNSWIndex> {
        self.index.as_mut().ok_or_else(|| {
            PyValueError::new_err("foxstash: call fit() before set_query_arguments()")
        })
    }

    /// The full `HNSWConfig` this instance's constructor args imply — what `fit()` builds.
    fn target_config(&self) -> HNSWConfig {
        HNSWConfig {
            metric: self.metric,
            m: self.m,
            m0: self.m * 2,
            ef_construction: self.ef_construction,
            ef_search: self.ef_construction,
            ml: 1.0 / (self.m as f32).ln(),
            use_heuristic: true,
            extend_candidates: false,
            keep_pruned_connections: true,
            build_strategy: BuildStrategy::Parallel,
            seed: None,
            storage: self.storage,
            rerank_candidates: self.rerank_candidates,
            // `self.turbo_bits` holds whichever budget the storage string encoded; route it
            // to the field its storage mode reads and leave the other at its default.
            turbo_bits: if self.storage == Storage::TurboQuant {
                self.turbo_bits
            } else {
                HNSWConfig::default().turbo_bits
            },
            // Warren's walk IS turborabit's, so it reads the same bit budget.
            rabit_bits: if matches!(self.storage, Storage::TurboRabit | Storage::Warren) {
                self.turbo_bits
            } else {
                HNSWConfig::default().rabit_bits
            },
            // Reflect the shipped default: the BFS locality relabel is a real query-QPS win, so
            // VIBE should measure foxstash as it actually ships. The build-cache (fit_cached)
            // reorders the F32 build once; requantize carries that order into every storage arm.
            reorder_for_locality: HNSWConfig::default().reorder_for_locality,
        }
    }

    /// Same graph knobs, F32 storage — the shared build every storage arm requantizes from.
    /// The quantizer fields are pinned to their defaults so the cached snapshot's config is
    /// identical no matter which storage arm happened to build it first.
    fn f32_config(&self) -> HNSWConfig {
        HNSWConfig {
            storage: Storage::F32,
            rerank_candidates: 0,
            turbo_bits: HNSWConfig::default().turbo_bits,
            rabit_bits: HNSWConfig::default().rabit_bits,
            ..self.target_config()
        }
    }

    fn copy_rows(&self, x: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vec<f32>>> {
        let view = x.as_array();
        let dim = view.shape()[1];
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "foxstash: fit() received {dim}-d vectors but this index was constructed with dim={}",
                self.dim
            )));
        }
        Ok((0..view.shape()[0]).map(|i| view.row(i).to_vec()).collect())
    }
}

#[pymethods]
impl Foxstash {
    #[new]
    fn new(
        metric: &str,
        dim: usize,
        m: usize,
        ef_construction: usize,
        storage: &str,
        rerank_candidates: usize,
    ) -> PyResult<Self> {
        let metric_enum = parse_metric(metric)?;
        let (storage_enum, turbo_bits) = parse_storage(storage)?;
        Ok(Self {
            index: None,
            metric: metric_enum,
            metric_arg: metric.to_string(),
            dim,
            m,
            ef_construction,
            storage: storage_enum,
            storage_arg: storage.to_string(),
            turbo_bits,
            rerank_candidates,
            ef_query: ef_construction,
        })
    }

    /// Build the index from an `(n, dim)` float32 array.
    ///
    /// `X` is copied into the `Vec<Vec<f32>>` `build_parallel` wants, and that copy is
    /// dropped the moment this function returns — nothing keeps `X` (or a clone of it) alive
    /// past `fit()`. VIBE measures index size as whole-process RSS (psutil) right after
    /// `fit()` returns; retaining the training array would inflate that number for free.
    /// (`HNSWConfig::rerank_candidates > 0` under a quantized `storage` still keeps its own
    /// f32 copy inside the index for reranking — that is a real, documented, benchmarked part
    /// of the index's footprint, not this array surviving by accident.)
    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let embeddings = self.copy_rows(&x)?;
        let config = self.target_config();

        // `embeddings` is plain owned Rust data (no Py<T>/Bound<T> inside it), so it's safe
        // to move into a GIL-released closure. VIBE times query latency single-threaded, but
        // there's no reason to hold the GIL hostage for the whole build either.
        //
        // `detach`, not `allow_threads`: PyO3 renamed it (along with `with_gil` -> `attach`) when
        // Python 3.13 free-threading made "the GIL" the wrong noun for what is really an
        // attach/detach of this thread from the interpreter.
        let index = py.detach(move || HNSWIndex::build_parallel(embeddings, config));
        self.index = Some(index);
        Ok(())
    }

    /// [`Foxstash::fit`] with a build-once cache: the expensive part of `fit` is HNSW graph
    /// construction, and the graph is storage-independent (built in exact f32, quantized
    /// after) — so one F32 build can serve every storage arm of a benchmark sweep.
    ///
    /// If `cache_path` exists it is loaded as an F32 snapshot and requantized into this
    /// instance's storage; otherwise the index is built in F32 from `x`, snapshotted to
    /// `cache_path` (write-to-temp + rename, so a crash mid-write never publishes a torn
    /// file), then requantized. Graph-shaping knobs (metric/m/ef_construction) must match
    /// the cached build — the caller keys `cache_path` on them, and `requantize` re-checks
    /// and errors loudly rather than trusting the key.
    ///
    /// Caveats, deliberate:
    /// - The snapshot is a same-version cache (`snapshot_from_file` rejects other foxstash
    ///   versions), so a stale cache dir after a rebuild fails loudly instead of silently
    ///   benchmarking old code. Delete the dir and rerun.
    /// - On a cache hit `x` is never copied; on a miss (and on every requantize) the F32
    ///   source index is dropped before this returns, but the allocator may not hand that
    ///   memory back to the OS — an RSS-based index-size metric can read high on cached
    ///   runs. Cached runs are for recall/QPS iteration; take size numbers from a cold run.
    #[pyo3(signature = (x, cache_path))]
    fn fit_cached(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        cache_path: &str,
    ) -> PyResult<()> {
        let path = std::path::PathBuf::from(cache_path);
        let target = self.target_config();
        let f32_config = self.f32_config();
        let target_is_the_cached_build = target == f32_config;

        let src = if path.exists() {
            let loaded = py
                .detach(|| HNSWIndex::snapshot_from_file(&path))
                .map_err(|e| PyValueError::new_err(format!("foxstash: cache load failed: {e}")))?;
            if loaded.embedding_dim() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "foxstash: cache at {cache_path} holds {}-d vectors, this index wants {}-d \
                     — the cache key must include the dataset",
                    loaded.embedding_dim(),
                    self.dim
                )));
            }
            loaded
        } else {
            let embeddings = self.copy_rows(&x)?;
            py.detach(move || -> foxstash_core::Result<HNSWIndex> {
                let idx = HNSWIndex::build_parallel(embeddings, f32_config);
                // Publish atomically: a concurrent process either sees the whole snapshot or
                // none of it. PID-suffixed temp so two concurrent misses don't clobber each
                // other's half-written file (they'll race the rename; last one wins, both
                // renames are of complete files).
                let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
                idx.snapshot_to_file(&tmp)?;
                std::fs::rename(&tmp, &path)?;
                Ok(idx)
            })
            .map_err(|e| PyValueError::new_err(format!("foxstash: cache build failed: {e}")))?
        };

        let index = if target_is_the_cached_build {
            src // the cached build *is* the target; requantizing would only copy it
        } else {
            py.detach(move || src.requantize(target))
                .map_err(|e| PyValueError::new_err(format!("foxstash: requantize failed: {e}")))?
        };
        self.index = Some(index);
        Ok(())
    }

    /// `ef`: search-time candidate pool (`HNSWIndex::set_ef_search`), the recall/QPS dial.
    /// `rerank`: exact-rerank pool size for quantized storage (`HNSWIndex::
    /// set_rerank_candidates`); `None` leaves it unchanged. Only meaningful under `storage
    /// != "f32"`, and only settable at all if this index was built with `rerank_candidates >
    /// 0` — raising it on an index built with `rerank_candidates: 0` (which drops the
    /// full-precision vectors entirely) has nothing left to rerank against and fails with
    /// `RagError::FullPrecisionDropped`, propagated below rather than swallowed.
    #[pyo3(signature = (ef, rerank=None))]
    fn set_query_arguments(&mut self, ef: usize, rerank: Option<usize>) -> PyResult<()> {
        // Scoped, so the mutable borrow of `self.index` ends before the bookkeeping fields are
        // written. The fields are only mirrored for `__str__`, so they are updated only once the
        // index has actually accepted the values -- if `set_rerank_candidates` rejects `r`,
        // `__str__` must not claim we applied it.
        {
            let index = self.built_mut()?;
            index.set_ef_search(ef);
            if let Some(r) = rerank {
                index
                    .set_rerank_candidates(r)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        self.ef_query = ef;
        if let Some(r) = rerank {
            self.rerank_candidates = r;
        }
        Ok(())
    }

    /// Single-query search, returning the `n` nearest neighbours' row indices into the `X` passed
    /// to `fit()`.
    ///
    /// That mapping survives `build_parallel`'s internal insertion-order shuffle. This used to be
    /// asserted here, in a comment, which is worth nothing: had the shuffle leaked, every recall
    /// number this binding reports would have been scored against the wrong ground-truth rows —
    /// silently, with no crash and nothing looking wrong. It is now pinned in core by
    /// `build_parallel_returns_original_row_indices_despite_its_shuffle`, which fails if it does.
    fn query<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let index = self.built()?;
        let query: Vec<f32> = v.as_slice()?.to_vec();

        let results = py
            .detach(move || index.search(&query, n))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let ids: Vec<i64> = results
            .iter()
            .map(|r| {
                r.id.parse::<i64>()
                    .expect("foxstash result ids are always the row index build_parallel assigned")
            })
            .collect();

        Ok(ids.into_pyarray(py))
    }

    /// Build a reusable [`Filter`] admitting exactly the rows in `allowed` (indices into the `X`
    /// passed to `fit()`). Building it scans every node once — do it ONCE per predicate and reuse
    /// the returned object across `query_filtered` calls; rebuilding per query would erase the
    /// graph's sub-linear advantage (the whole reason filtered search lives in the graph and not a
    /// post-filter). The row→node-slot mapping (`build_parallel` shuffles internally) is handled by
    /// matching on the row-index id core assigns, so the caller always speaks in fit()-row terms.
    fn make_filter(&self, allowed: PyReadonlyArray1<'_, i64>) -> PyResult<Filter> {
        let index = self.built()?;
        let set: std::collections::HashSet<i64> = allowed.as_slice()?.iter().copied().collect();
        let mask = index.filter_mask(|id, _content, _meta| {
            set.contains(
                &id.parse::<i64>()
                    .expect("foxstash result ids are always the row index build_parallel assigned"),
            )
        });
        Ok(Filter { mask })
    }

    /// Like [`Foxstash::query`], but returns only rows admitted by `filter` — up to `n` of them.
    ///
    /// The graph is walked in full (excluded nodes are traversed for connectivity); only the result
    /// set is restricted, so there is no over-fetch and no separate post-filter. Cost scales with
    /// `filter`'s selectivity — see [`HNSWIndex::search_filtered`].
    fn query_filtered<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        filter: &Filter,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let index = self.built()?;
        let query: Vec<f32> = v.as_slice()?.to_vec();

        let results = py
            .detach(move || index.search_filtered(&query, n, &filter.mask))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let ids: Vec<i64> = results
            .iter()
            .map(|r| {
                r.id.parse::<i64>()
                    .expect("foxstash result ids are always the row index build_parallel assigned")
            })
            .collect();

        Ok(ids.into_pyarray(py))
    }

    fn __str__(&self) -> String {
        format!(
            "Foxstash(metric={}, storage={}, M={}, efConstruction={}, efSearch={}, rerank={})",
            self.metric_arg,
            self.storage_arg,
            self.m,
            self.ef_construction,
            self.ef_query,
            self.rerank_candidates
        )
    }
}

/// A prebuilt allow-list over the rows passed to `fit()`, produced by [`Foxstash::make_filter`]
/// and consumed by [`Foxstash::query_filtered`]. Opaque to Python beyond `allowed_count`; hold it
/// and reuse it across queries that share the same predicate.
#[pyclass(module = "foxstash")]
struct Filter {
    mask: FilterMask,
}

#[pymethods]
impl Filter {
    /// How many rows this filter admits. A `query_filtered` cannot return more than this many,
    /// and a very small count means the walk may explore most of the graph to collect them.
    fn allowed_count(&self) -> usize {
        self.mask.allowed_count()
    }

    fn __str__(&self) -> String {
        format!("Filter(allowed={})", self.mask.allowed_count())
    }
}

#[pymodule]
fn foxstash(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Foxstash>()?;
    m.add_class::<Filter>()?;
    Ok(())
}
