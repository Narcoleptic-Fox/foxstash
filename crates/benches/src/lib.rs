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

    /// A loaded ANN benchmark dataset with its shipped ground truth.
    pub struct Dataset {
        pub name: String,
        /// Corpus vectors to index.
        pub base: Vec<Vec<f32>>,
        /// Query vectors.
        pub queries: Vec<Vec<f32>>,
        /// `truth[q]` lists the exact nearest neighbours of `queries[q]`, nearest
        /// first, as indices into `base`. Computed by exact L2 by the dataset authors.
        pub truth: Vec<Vec<usize>>,
    }

    impl Dataset {
        pub fn dim(&self) -> usize {
            self.base[0].len()
        }

        /// Load `<root>/<name>/{base,query,groundtruth}.npy`.
        pub fn load(root: impl AsRef<Path>, name: &str) -> std::io::Result<Self> {
            let dir: PathBuf = root.as_ref().join(name);
            Ok(Self {
                name: name.to_string(),
                base: load_f32(&dir.join("base.npy"))?,
                queries: load_f32(&dir.join("query.npy"))?,
                truth: load_i32(&dir.join("groundtruth.npy"))?
                    .into_iter()
                    .map(|row| row.into_iter().map(|i| i as usize).collect())
                    .collect(),
            })
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
            self.recall_at(k, |q| {
                let mut d: Vec<(f32, usize)> = self
                    .base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (l2_sq(q, v), i))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.into_iter().take(k).map(|(_, i)| i).collect()
            })
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
