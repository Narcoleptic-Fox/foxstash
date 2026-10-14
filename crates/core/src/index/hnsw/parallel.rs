//! Internal data structures for the parallel HNSW builder.
//!
//! Split out of `hnsw/mod.rs` as pure code motion: these types are used only by
//! `build_parallel` and its helpers, and kept the module at 8,800 lines where the
//! search and build paths were impossible to review separately.

use super::*;

pub(super) const M0_MAX: usize = 64;
/// Maximum connections per node in upper layers (M)
pub(super) const M_MAX: usize = 32;
/// Invalid point ID marker
pub(super) const INVALID: u32 = u32::MAX;

/// Point ID wrapper (u32 for memory efficiency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PointId(pub(super) u32);

impl PointId {
    pub(super) fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub(super) fn is_valid(self) -> bool {
        self.0 != INVALID
    }
}

/// Layer 0 node with M*2 fixed connections
#[derive(Clone)]
pub(super) struct ZeroNode {
    /// Fixed array of neighbor IDs (INVALID = empty slot)
    pub(super) nearest: [PointId; M0_MAX],
}

impl Default for ZeroNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M0_MAX],
        }
    }
}

impl ZeroNode {
    /// Count of valid neighbors
    pub(super) fn count(&self) -> usize {
        self.nearest.iter().take_while(|p| p.is_valid()).count()
    }

    /// Iterate over valid neighbors
    pub(super) fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Upper layer node with M fixed connections
#[derive(Clone)]
pub(super) struct UpperNode {
    pub(super) nearest: [PointId; M_MAX],
}

impl Default for UpperNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M_MAX],
        }
    }
}

impl UpperNode {
    /// Create from ZeroNode, truncating to M neighbors
    pub(super) fn from_zero(zero: &ZeroNode, m: usize) -> Self {
        let mut node = Self::default();
        for (i, &pid) in zero.nearest.iter().take(m.min(M_MAX)).enumerate() {
            node.nearest[i] = pid;
        }
        node
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Visited bitmap with generation counter (O(1) clear)
pub(super) struct Visited {
    pub(super) store: Vec<u8>,
    pub(super) generation: u8,
}

impl Visited {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    pub(super) fn clear(&mut self) {
        if self.generation == 255 {
            self.store.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    pub(super) fn insert(&mut self, pid: PointId) -> bool {
        let idx = pid.as_usize();
        if self.store[idx] == self.generation {
            false
        } else {
            self.store[idx] = self.generation;
            true
        }
    }

    pub(super) fn reserve(&mut self, capacity: usize) {
        if self.store.len() < capacity {
            self.store.resize(capacity, 0);
        }
    }
}

/// Candidate for search (distance + point ID)
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Candidate {
    pub(super) distance: f32,
    pub(super) pid: PointId,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by distance, then by pid for stability
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.pid.cmp(&other.pid))
    }
}

/// Search state for parallel construction
pub(super) struct Search {
    /// Metric used for every distance computed during this search.
    pub(super) metric: DistanceMetric,
    /// Candidates to explore (min-heap by distance)
    pub(super) candidates: BinaryHeap<Reverse<Candidate>>,
    /// Best results found (sorted by distance)
    pub(super) nearest: Vec<Candidate>,
    /// Visited nodes
    pub(super) visited: Visited,
    /// Current ef value
    pub(super) ef: usize,
}

impl Search {
    pub(super) fn new(capacity: usize, metric: DistanceMetric) -> Self {
        Self {
            metric,
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            visited: Visited::new(capacity),
            ef: 1,
        }
    }

    pub(super) fn reset(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.visited.clear();
    }

    pub(super) fn push(&mut self, pid: PointId, point: &[f32], points: &[Vec<f32>]) {
        let distance = HNSWIndex::parallel_distance(self.metric, point, &points[pid.as_usize()]);
        let candidate = Candidate { distance, pid };
        self.candidates.push(Reverse(candidate));
        self.nearest.push(candidate);
        self.visited.insert(pid);
    }

    /// After searching a layer, prepare for the next layer down
    pub(super) fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in &self.nearest {
            self.candidates.push(Reverse(candidate));
        }
        self.visited.clear();
        for c in &self.nearest {
            self.visited.insert(c.pid);
        }
    }

    /// Search within a layer (generic over layer type)
    pub(super) fn search_zero(
        &mut self,
        point: &[f32],
        layer: &[RwLock<ZeroNode>],
        points: &[Vec<f32>],
        num: usize,
    ) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Explore neighbors
            let node = layer[candidate.pid.as_usize()].read();
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance = HNSWIndex::parallel_distance(
                        self.metric,
                        point,
                        &points[neighbor_pid.as_usize()],
                    );
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    // Add to candidates if potentially useful
                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));

                        // Insert into nearest (sorted)
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    pub(super) fn search_upper(
        &mut self,
        point: &[f32],
        layer: &[UpperNode],
        points: &[Vec<f32>],
        num: usize,
    ) {
        if layer.is_empty() {
            return;
        }

        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Safety: skip if candidate is beyond current layer snapshot
            if candidate.pid.as_usize() >= layer.len() {
                continue;
            }

            let node = &layer[candidate.pid.as_usize()];
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance = HNSWIndex::parallel_distance(
                        self.metric,
                        point,
                        &points[neighbor_pid.as_usize()],
                    );
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    /// Get best candidates (sorted by distance)
    pub(super) fn select_simple(&self) -> &[Candidate] {
        &self.nearest
    }
}

/// Pool of search states for thread-local reuse
pub(super) struct SearchPool {
    pub(super) pool: Mutex<Vec<Search>>,
    pub(super) capacity: usize,
    pub(super) metric: DistanceMetric,
    /// Whether to backfill a node's neighbour list to `m` with candidates the diversity
    /// heuristic rejected. See `par_select_heuristic` — this used to be ignored entirely.
    pub(super) keep_pruned: bool,
}

impl SearchPool {
    pub(super) fn new(capacity: usize, metric: DistanceMetric, keep_pruned: bool) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            capacity,
            keep_pruned,
            metric,
        }
    }

    pub(super) fn pop(&self) -> Search {
        self.pool
            .lock()
            .pop()
            .unwrap_or_else(|| Search::new(self.capacity, self.metric))
    }

    pub(super) fn push(&self, mut search: Search) {
        search.reset();
        self.pool.lock().push(search);
    }
}

/// Layer ID wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LayerId(pub(super) usize);

impl LayerId {
    pub(super) fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Iterate from this layer down to 0
    pub(super) fn descend(self) -> impl Iterator<Item = LayerId> {
        (0..=self.0).rev().map(LayerId)
    }
}
