//! Shared helpers for foxstash benchmarks.
//!
//! The headline export is [`sift`], which loads the real ANN datasets in
//! `benchmarks/data/` and scores against the ground truth shipped with them.
//!
//! Use real data. Synthetic vectors have no cluster structure, so *every* ANN
//! implementation collapses to ~60% recall on them regardless of quality — which
//! both flatters a broken index and hides real graph-connectivity bugs. Foxstash
//! published "beats gold standards on recall" for a full release on the strength
//! of a synthetic run in which hnswlib scored 40.3%; on real SIFT10K hnswlib
//! scores 99.99%.

pub mod sift {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    /// The shape every known dataset is required to have.
    struct Spec {
        name: &'static str,
        base: usize,
        queries: usize,
        dim: usize,
    }

    /// Known datasets and the shape each one must have on disk.
    ///
    /// [`Dataset::load`] refuses to return a dataset that does not match its entry
    /// here. That check is not paranoia: `benchmarks/data/sift1m/` really did contain
    /// a **10,000**-vector base. Benchmarking "SIFT1M" against it would have produced
    /// an entirely plausible — and entirely meaningless — number, with nothing in the
    /// output to suggest the index was 100x smaller than its label claimed.
    ///
    /// A dataset that fails to load is a good day. A dataset that loads as the wrong
    /// thing is how you ship a false benchmark.
    const MANIFEST: &[Spec] = &[
        Spec {
            name: "sift10k",
            base: 10_000,
            queries: 100,
            dim: 128,
        },
        Spec {
            name: "sift100k",
            base: 100_000,
            queries: 10_000,
            dim: 128,
        },
        Spec {
            name: "sift1m",
            base: 1_000_000,
            queries: 10_000,
            dim: 128,
        },
        // 960-d. SIFT's 128 dimensions cannot answer whether a quantized storage mode pays
        // off, because a node block is `header(m0) + vector` and which half dominates is a
        // pure function of `dim`. Real embeddings are 384-1536d, not 128.
        Spec {
            name: "gist1m",
            base: 1_000_000,
            queries: 1_000,
            dim: 960,
        },
        // The first dataset here that is an actual neural embedding. SIFT and GIST are
        // image descriptors, and this project's own quantizer findings say that is the
        // distinction that matters: nomic-768 scores RaBitQ 0.53 while distilroberta-768
        // scores 0.9993 — same dimension, same metric, only the embedder differs. So
        // accuracy is a property of the distribution and dimension is only a cost, which
        // means no number measured on SIFT or GIST predicts what a quantizer does to a
        // user's embeddings.
        //
        // EmbeddingGemma-300M over permissively-licensed Rust documentation and library
        // doc comments. Built by `benchmarks/python/build_text_dataset.py`; `manifest.json`
        // beside the vectors records the embedder, prefix, source mix and licences.
        Spec {
            name: "rustdocs45k",
            base: 45_000,
            queries: 1_000,
            dim: 768,
        },
    ];

    /// A loaded ANN benchmark dataset with its shipped ground truth.
    pub struct Dataset {
        pub name: String,
        /// Corpus vectors to index.
        pub base: Vec<Vec<f32>>,
        /// Query vectors.
        pub queries: Vec<Vec<f32>>,
        /// `truth[q]` lists the exact nearest neighbours of `queries[q]`, nearest
        /// first, as indices into `base`.
        ///
        /// For the SIFT and GIST sets this is exact L2, computed by the dataset
        /// authors. For `rustdocs45k` it is exact cosine, computed locally by
        /// `build_text_dataset.py` — which is the same ranking, because those vectors
        /// are unit-norm and the builder fails rather than proceeding if they are not.
        /// Harnesses may therefore score every dataset here under [`DistanceMetric::L2`]
        /// without silently changing what "correct" means.
        ///
        /// [`DistanceMetric::L2`]: foxstash_core::index::hnsw::DistanceMetric::L2
        pub truth: Vec<Vec<usize>>,
    }

    impl Dataset {
        pub fn dim(&self) -> usize {
            self.base[0].len()
        }

        /// Load `<root>/<name>/{base,query,groundtruth}.npy`, then verify it is
        /// actually the dataset it claims to be (see [`MANIFEST`]).
        pub fn load(root: impl AsRef<Path>, name: &str) -> std::io::Result<Self> {
            use std::io::{Error, ErrorKind};
            let bad = |m: String| Error::new(ErrorKind::InvalidData, m);

            let spec = MANIFEST.iter().find(|s| s.name == name).ok_or_else(|| {
                let known: Vec<_> = MANIFEST.iter().map(|s| s.name).collect();
                bad(format!(
                    "unknown dataset {name:?}; known: {known:?}. \
                     Add it to MANIFEST with its expected shape — datasets are not \
                     loaded on trust."
                ))
            })?;

            let dir: PathBuf = root.as_ref().join(name);
            let ds = Self {
                name: name.to_string(),
                base: load_f32(&dir.join("base.npy"))?,
                queries: load_f32(&dir.join("query.npy"))?,
                truth: load_i32(&dir.join("groundtruth.npy"))?
                    .into_iter()
                    .map(|row| row.into_iter().map(|i| i as usize).collect())
                    .collect(),
            };

            // Does it match the label on the tin?
            let got = (ds.base.len(), ds.queries.len(), ds.base[0].len());
            let want = (spec.base, spec.queries, spec.dim);
            if got != want {
                return Err(bad(format!(
                    "{name} is not {name}: expected {} base x {}d with {} queries, \
                     found {} base x {}d with {} queries. The directory is mislabelled \
                     or the download is truncated — fix the data, do not adjust MANIFEST \
                     to match it.",
                    want.0, want.2, want.1, got.0, got.2, got.1
                )));
            }

            // Internal consistency: ground truth must index into the base we loaded,
            // and cover every query.
            if ds.truth.len() != ds.queries.len() {
                return Err(bad(format!(
                    "{name}: {} queries but {} ground-truth rows",
                    ds.queries.len(),
                    ds.truth.len()
                )));
            }
            if let Some(&i) = ds.truth.iter().flatten().max() {
                if i >= ds.base.len() {
                    return Err(bad(format!(
                        "{name}: ground truth references base index {i}, but the base has \
                         only {} vectors. The ground truth belongs to a different (larger) \
                         base than the one on disk.",
                        ds.base.len()
                    )));
                }
            }
            if ds.queries.iter().any(|q| q.len() != spec.dim) {
                return Err(bad(format!("{name}: queries are not all {}d", spec.dim)));
            }

            Ok(ds)
        }

        /// Recall@k of `search`, which must return the retrieved base-indices for a query.
        pub fn recall_at(&self, k: usize, search: impl Fn(&[f32]) -> Vec<usize>) -> f32 {
            let mut total = 0.0;
            for (qi, q) in self.queries.iter().enumerate() {
                let gt: HashSet<usize> = self.truth[qi].iter().take(k).copied().collect();
                let got: HashSet<usize> = search(q).into_iter().take(k).collect();
                total += gt.intersection(&got).count() as f32 / gt.len().max(1) as f32;
            }
            total / self.queries.len() as f32
        }

        /// Exact-search control. **Every** recall table must include this row: if it is
        /// not ~1.0, the loader or the distance metric is wrong and no other row in the
        /// table means anything.
        pub fn exact_control(&self, k: usize) -> f32 {
            self.exact_control_sampled(k, self.queries.len())
        }

        /// [`Self::exact_control`] over the first `n` queries.
        ///
        /// The full control is O(queries x base) and takes minutes on SIFT1M, which is
        /// exactly the excuse one uses to skip it. Sampling a few hundred queries costs
        /// a second and catches every failure the full run would: a bad loader, a metric
        /// mismatch, ground truth belonging to a different corpus.
        pub fn exact_control_sampled(&self, k: usize, n: usize) -> f32 {
            use rayon::prelude::*;
            let n = n.min(self.queries.len());
            let total: f32 = self.queries[..n]
                .par_iter()
                .enumerate()
                .map(|(qi, q)| {
                    let mut d: Vec<(f32, usize)> = self
                        .base
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (l2_sq(q, v), i))
                        .collect();
                    d.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
                    d.truncate(k);
                    let got: HashSet<usize> = d.into_iter().map(|(_, i)| i).collect();
                    let gt: HashSet<usize> = self.truth[qi].iter().take(k).copied().collect();
                    gt.intersection(&got).count() as f32 / gt.len().max(1) as f32
                })
                .sum();
            total / n as f32
        }

        /// How hard is this dataset, really? Mean ratio `d(100th NN) / d(kth NN)`.
        ///
        /// Recall@k is **not comparable across datasets**, and this number is why. On
        /// SIFT10K the 100th neighbour sits only 4.7% further from the query than the
        /// 10th — the true top-10 is buried in a shell of ~90 near-equidistant vectors,
        /// and separating 10th from 11th is nearly impossible for an approximate method.
        /// On SIFT100K the same ratio is 13.5%, and *every* index scores far better.
        ///
        /// A ratio near 1.0 means a punishing dataset. Quoting a recall number without
        /// it invites the reader to compare figures that measure different problems.
        pub fn separation(&self, k: usize, n: usize) -> f32 {
            use rayon::prelude::*;
            let n = n.min(self.queries.len());
            let total: f32 = self.queries[..n]
                .par_iter()
                .map(|q| {
                    let mut d: Vec<f32> = self.base.iter().map(|v| l2_sq(q, v)).collect();
                    let hundredth = 99.min(d.len() - 1);
                    d.sort_unstable_by(|a, b| a.total_cmp(b));
                    let dk = d[(k - 1).min(d.len() - 1)].sqrt().max(f32::EPSILON);
                    d[hundredth].sqrt() / dk
                })
                .sum();
            total / n as f32
        }

        /// Brute-force **cosine** ground truth, computed here rather than shipped.
        ///
        /// SIFT's own ground truth is exact L2. Foxstash's [`HNSWIndex`] ranks by
        /// `1 - cosine_similarity`, and SIFT vector magnitudes vary by 1.4x, so cosine
        /// and L2 order the neighbours differently. Scoring a cosine index against the
        /// L2 answer key measures the metric gap, not the index — it reads ~55% for a
        /// perfectly healthy graph. Use this to ask the fair question: does the graph
        /// find the true *cosine* neighbours?
        ///
        /// [`HNSWIndex`]: foxstash_core::index::HNSWIndex
        pub fn cosine_truth(&self, k: usize) -> Vec<Vec<usize>> {
            self.queries
                .iter()
                .map(|q| {
                    let mut d: Vec<(f32, usize)> = self
                        .base
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (cosine_dist(q, v), i))
                        .collect();
                    d.sort_by(|a, b| a.0.total_cmp(&b.0));
                    d.into_iter().take(k).map(|(_, i)| i).collect()
                })
                .collect()
        }

        /// Recall@k of `search` against an explicit ground truth (see [`Self::cosine_truth`]).
        pub fn recall_against(
            &self,
            truth: &[Vec<usize>],
            k: usize,
            search: impl Fn(&[f32]) -> Vec<usize>,
        ) -> f32 {
            let mut total = 0.0;
            for (qi, q) in self.queries.iter().enumerate() {
                let gt: HashSet<usize> = truth[qi].iter().take(k).copied().collect();
                let got: HashSet<usize> = search(q).into_iter().take(k).collect();
                total += gt.intersection(&got).count() as f32 / gt.len().max(1) as f32;
            }
            total / self.queries.len() as f32
        }
    }

    pub fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
        1.0 - dot / denom
    }

    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Minimal `.npy` reader: v1/v2 header, little-endian, C-order, 2-D only.
    fn npy_parse(bytes: &[u8], want_descr: &str) -> std::io::Result<(usize, usize, Vec<u8>)> {
        use std::io::{Error, ErrorKind};
        let bad = |m: String| Error::new(ErrorKind::InvalidData, m);

        if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
            return Err(bad("not a .npy file".into()));
        }
        let (header_len, data_start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
            v => return Err(bad(format!("unsupported .npy version {v}"))),
        };

        let header = std::str::from_utf8(&bytes[data_start..data_start + header_len])
            .map_err(|e| bad(e.to_string()))?;
        if !header.contains(want_descr) {
            return Err(bad(format!(
                "expected dtype {want_descr}, header: {header}"
            )));
        }
        if !header.contains("'fortran_order': False") {
            return Err(bad("fortran-order arrays unsupported".into()));
        }

        let shape = header
            .split("'shape':")
            .nth(1)
            .and_then(|s| s.split('(').nth(1))
            .and_then(|s| s.split(')').next())
            .ok_or_else(|| bad("no shape in header".into()))?;
        let dims: Vec<usize> = shape
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        if dims.len() != 2 {
            return Err(bad(format!("expected a 2-D array, got shape ({shape})")));
        }

        Ok((dims[0], dims[1], bytes[data_start + header_len..].to_vec()))
    }

    fn load_f32(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
        let (rows, cols, payload) = npy_parse(&std::fs::read(path)?, "<f4")?;
        Ok((0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let o = (r * cols + c) * 4;
                        f32::from_le_bytes([
                            payload[o],
                            payload[o + 1],
                            payload[o + 2],
                            payload[o + 3],
                        ])
                    })
                    .collect()
            })
            .collect())
    }

    fn load_i32(path: &Path) -> std::io::Result<Vec<Vec<i32>>> {
        let (rows, cols, payload) = npy_parse(&std::fs::read(path)?, "<i4")?;
        Ok((0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let o = (r * cols + c) * 4;
                        i32::from_le_bytes([
                            payload[o],
                            payload[o + 1],
                            payload[o + 2],
                            payload[o + 3],
                        ])
                    })
                    .collect()
            })
            .collect())
    }
}
