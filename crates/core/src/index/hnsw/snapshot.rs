//! On-disk snapshot representation and its format version.
//!
//! Split out of `hnsw/mod.rs` as pure code motion.

use super::*;

/// Bump when the meaning of any [`HNSWSnapshot`] field changes. Belt-and-braces on top of the
/// crate-version check: two builds of the *same* crate version can still disagree if a field is
/// reinterpreted on a dev branch.
///
/// v2: `HNSWConfig` gained `reorder_for_locality` (serialized in the snapshot's `config`) without a
/// crate-version bump, so a v1 snapshot's bincode layout no longer matches — it must be rejected
/// cleanly here rather than deserialized into garbage (bincode read a stray byte as a bool and
/// panicked with `InvalidBoolEncoding`).
///
/// v3: `HNSWSnapshot` gained `mouse_res` + `mouse_rc` (for `Storage::Mouse`) mid-struct, shifting
/// the positional bincode layout — a v2 snapshot must be rejected, not misparsed.
///
/// v4: `HNSWSnapshot` gained `vectors_external` (for the shared-f32 rerank path, issue #58). An
/// index built with externalized vectors serializes an empty `full` and this flag set; a standalone
/// index serializes `full` as before with the flag clear. A v3 snapshot lacks the field, so its
/// bincode layout no longer matches and it must be rejected rather than misparsed.
pub(super) const SNAPSHOT_FORMAT_VERSION: u32 = 4;

/// The verbatim on-disk image behind [`HNSWIndex::snapshot_to_file`]. Every field is a direct
/// clone of the corresponding `HNSWIndex` field except the two version stamps; `stride`, `hdr`
/// and `level_rng` are derived from `config` on load rather than stored.
///
/// Same-version cache format only — see `snapshot_to_file` for why this is not the portable path.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct HNSWSnapshot {
    pub(super) format_version: u32,
    pub(super) crate_version: String,
    pub(super) embedding_dim: usize,
    pub(super) config: HNSWConfig,
    pub(super) nodes: Vec<u32>,
    pub(super) connections: Vec<Vec<Vec<u32>>>,
    pub(super) q_min: Vec<f32>,
    pub(super) q_scale: Vec<f32>,
    pub(super) full: Vec<f32>,
    pub(super) rabitq: Option<crate::vector::rabitq::RaBitQuantizer>,
    pub(super) turboquant: Option<crate::vector::turboquant::TurboQuantizer>,
    pub(super) turborabit: Option<crate::vector::turborabit::TurboRabitQuantizer>,
    pub(super) mouse_res: Vec<u8>,
    pub(super) mouse_rc: Vec<f32>,
    pub(super) ids: Vec<String>,
    pub(super) contents: Vec<String>,
    /// `serde_json::Value` cannot ride bincode (its `Deserialize` is self-describing —
    /// `deserialize_any` — which a non-self-describing format rejects at runtime, not compile
    /// time). Stored as JSON text and re-parsed on load.
    pub(super) metadata: Vec<Option<String>>,
    pub(super) entry_point: Option<usize>,
    pub(super) max_layer: usize,
    /// When true, `full` is empty by design and the vectors are supplied externally at search time
    /// (issue #58). A loader must refuse to search such an index without a [`VectorSource`].
    pub(super) vectors_external: bool,
}
