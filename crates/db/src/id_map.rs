//! ID-to-index mapping with tombstone tracking.
//!
//! Maps string document IDs to their positional index in the HNSW graph,
//! and tracks deleted IDs as tombstones until compaction.

use std::collections::{HashMap, HashSet};

/// Bidirectional mapping between document IDs and HNSW node indices,
/// with tombstone tracking for soft deletion.
#[derive(Debug, Clone)]
pub struct IdMap {
    /// ID → positional index in the HNSW graph.
    id_to_pos: HashMap<String, usize>,
    /// Positional index → ID (for reverse lookup during search).
    pos_to_id: HashMap<usize, String>,
    /// Deleted IDs awaiting compaction.
    tombstones: HashSet<String>,
    /// Next position to assign.
    next_pos: usize,
}

impl IdMap {
    pub fn new() -> Self {
        Self {
            id_to_pos: HashMap::new(),
            pos_to_id: HashMap::new(),
            tombstones: HashSet::new(),
            next_pos: 0,
        }
    }

    /// Insert an ID and return its assigned position.
    ///
    /// If the ID was previously tombstoned, the tombstone is removed
    /// and a new position is assigned. If the ID already occupies an
    /// older position, that stale reverse mapping is removed to prevent
    /// `id_at(old_pos)` from returning the ID after a re-insert.
    pub fn insert(&mut self, id: String) -> usize {
        self.tombstones.remove(&id);
        // Remove stale reverse mapping if ID already exists at a different position.
        if let Some(&old_pos) = self.id_to_pos.get(&id) {
            self.pos_to_id.remove(&old_pos);
        }
        let pos = self.next_pos;
        self.id_to_pos.insert(id.clone(), pos);
        self.pos_to_id.insert(pos, id);
        self.next_pos += 1;
        pos
    }

    /// Mark an ID as deleted (tombstoned). Returns `true` if the ID existed.
    pub fn remove(&mut self, id: &str) -> bool {
        if self.id_to_pos.contains_key(id) {
            self.tombstones.insert(id.to_string());
            true
        } else {
            false
        }
    }

    /// Look up the position for an ID. Returns `None` if missing or tombstoned.
    pub fn get(&self, id: &str) -> Option<usize> {
        if self.tombstones.contains(id) {
            return None;
        }
        self.id_to_pos.get(id).copied()
    }

    /// Look up the ID at a given position (regardless of tombstone status).
    pub fn id_at(&self, pos: usize) -> Option<&str> {
        self.pos_to_id.get(&pos).map(|s| s.as_str())
    }

    /// Check if an ID is tombstoned.
    pub fn is_tombstoned(&self, id: &str) -> bool {
        self.tombstones.contains(id)
    }

    /// Check if an ID is live (present and not tombstoned).
    pub fn is_live(&self, id: &str) -> bool {
        self.id_to_pos.contains_key(id) && !self.tombstones.contains(id)
    }

    /// Check if a position's document is tombstoned.
    pub fn is_pos_tombstoned(&self, pos: usize) -> bool {
        self.pos_to_id
            .get(&pos)
            .map(|id| self.tombstones.contains(id))
            .unwrap_or(false)
    }

    /// Number of live (non-tombstoned) documents.
    pub fn live_count(&self) -> usize {
        self.id_to_pos.len() - self.tombstones.len()
    }

    /// Total positions ever assigned (including tombstoned).
    pub fn total_count(&self) -> usize {
        self.id_to_pos.len()
    }

    /// Number of tombstoned documents.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Iterate over all live (non-tombstoned) IDs.
    pub fn live_ids(&self) -> impl Iterator<Item = &str> {
        self.id_to_pos
            .keys()
            .filter(|id| !self.tombstones.contains(id.as_str()))
            .map(|s| s.as_str())
    }

    /// Clear all mappings and tombstones.
    pub fn clear(&mut self) {
        self.id_to_pos.clear();
        self.pos_to_id.clear();
        self.tombstones.clear();
        self.next_pos = 0;
    }

    /// Rebuild the map from a fresh set of IDs (used after compaction).
    /// Positions are reassigned sequentially starting from 0.
    pub fn rebuild(&mut self, ids: impl IntoIterator<Item = String>) {
        self.clear();
        for id in ids {
            self.insert(id);
        }
    }
}

impl Default for IdMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = IdMap::new();
        let pos = map.insert("doc1".into());
        assert_eq!(pos, 0);
        assert_eq!(map.get("doc1"), Some(0));
        assert_eq!(map.id_at(0), Some("doc1"));
    }

    #[test]
    fn sequential_positions() {
        let mut map = IdMap::new();
        assert_eq!(map.insert("a".into()), 0);
        assert_eq!(map.insert("b".into()), 1);
        assert_eq!(map.insert("c".into()), 2);
        assert_eq!(map.total_count(), 3);
    }

    #[test]
    fn remove_tombstones() {
        let mut map = IdMap::new();
        map.insert("doc1".into());
        map.insert("doc2".into());

        assert!(map.remove("doc1"));
        assert!(map.is_tombstoned("doc1"));
        assert_eq!(map.get("doc1"), None);
        assert_eq!(map.live_count(), 1);
        assert_eq!(map.tombstone_count(), 1);
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut map = IdMap::new();
        assert!(!map.remove("nope"));
    }

    #[test]
    fn is_pos_tombstoned() {
        let mut map = IdMap::new();
        map.insert("doc1".into());
        map.insert("doc2".into());
        map.remove("doc1");

        assert!(map.is_pos_tombstoned(0));
        assert!(!map.is_pos_tombstoned(1));
    }

    #[test]
    fn reinsert_cleans_old_pos_to_id() {
        let mut map = IdMap::new();
        let old_pos = map.insert("x".into());
        assert_eq!(map.id_at(old_pos), Some("x"));

        map.remove("x");
        let new_pos = map.insert("x".into());
        assert_ne!(old_pos, new_pos);
        assert_eq!(map.id_at(new_pos), Some("x"));
        assert_eq!(map.id_at(old_pos), None); // stale entry must be gone
        assert_eq!(map.get("x"), Some(new_pos));
    }

    #[test]
    fn reinsert_after_tombstone() {
        let mut map = IdMap::new();
        map.insert("doc1".into());
        map.remove("doc1");
        assert!(map.is_tombstoned("doc1"));

        let new_pos = map.insert("doc1".into());
        assert!(!map.is_tombstoned("doc1"));
        assert_eq!(map.get("doc1"), Some(new_pos));
    }

    #[test]
    fn live_ids_excludes_tombstones() {
        let mut map = IdMap::new();
        map.insert("a".into());
        map.insert("b".into());
        map.insert("c".into());
        map.remove("b");

        let mut live: Vec<&str> = map.live_ids().collect();
        live.sort();
        assert_eq!(live, vec!["a", "c"]);
    }

    #[test]
    fn clear_resets_everything() {
        let mut map = IdMap::new();
        map.insert("a".into());
        map.insert("b".into());
        map.remove("a");
        map.clear();

        assert_eq!(map.live_count(), 0);
        assert_eq!(map.total_count(), 0);
        assert_eq!(map.tombstone_count(), 0);
    }

    #[test]
    fn rebuild_reassigns_positions() {
        let mut map = IdMap::new();
        map.insert("old1".into());
        map.insert("old2".into());

        map.rebuild(vec!["new_a".into(), "new_b".into(), "new_c".into()]);

        assert_eq!(map.get("new_a"), Some(0));
        assert_eq!(map.get("new_b"), Some(1));
        assert_eq!(map.get("new_c"), Some(2));
        assert_eq!(map.get("old1"), None);
        assert_eq!(map.live_count(), 3);
    }
}
