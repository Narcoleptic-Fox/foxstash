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

use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};

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
        other => Err(PyValueError::new_err(format!(
            "foxstash: unsupported storage {other:?}. Expected \"f32\", \"sq8\", \"rabitq\", \
             \"turboquant[N]\", or \"turborabit[N]\"."
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
        self.index
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("foxstash: call fit() before set_query_arguments()"))
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
        let view = x.as_array();
        let n = view.shape()[0];
        let dim = view.shape()[1];
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "foxstash: fit() received {dim}-d vectors but this index was constructed with dim={}",
                self.dim
            )));
        }

        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| view.row(i).to_vec()).collect();

        let config = HNSWConfig {
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
            rabit_bits: if self.storage == Storage::TurboRabit {
                self.turbo_bits
            } else {
                HNSWConfig::default().rabit_bits
            },
        };

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

    fn __str__(&self) -> String {
        format!(
            "Foxstash(metric={}, storage={}, M={}, efConstruction={}, efSearch={}, rerank={})",
            self.metric_arg, self.storage_arg, self.m, self.ef_construction, self.ef_query, self.rerank_candidates
        )
    }
}

#[pymodule]
fn foxstash(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Foxstash>()?;
    Ok(())
}
