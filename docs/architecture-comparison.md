# HNSW Build Architecture: Foxstash vs instant-distance

## Data Structures

### instant-distance
```
┌─────────────────────────────────────────────────────────────┐
│                     Hnsw<P>                                  │
├─────────────────────────────────────────────────────────────┤
│ points: Vec<P>              ← All embeddings                 │
│ zero: Vec<ZeroNode>         ← Layer 0 (M*2 neighbors)       │
│ layers: Vec<Vec<UpperNode>> ← Layers 1+ (M neighbors each)  │
│ ef_search: usize                                            │
└─────────────────────────────────────────────────────────────┘

ZeroNode {
    nearest: [PointId; M * 2]  ← Fixed array, 64 neighbors
}

UpperNode {
    nearest: [PointId; M]      ← Fixed array, 32 neighbors
}
```

### Foxstash
```
┌─────────────────────────────────────────────────────────────┐
│                    HNSWIndex                                 │
├─────────────────────────────────────────────────────────────┤
│ embeddings: Vec<f32>        ← Flat SoA (n * dim)            │
│ connections: Vec<Vec<HashSet<usize>>>                       │
│   └─ Per node: Vec of HashSets per layer                    │
│ ids: Vec<String>                                            │
│ contents: Vec<String>                                       │
│ metadata: Vec<Option<String>>                               │
│ entry_point: Option<usize>                                  │
│ max_layer: usize                                            │
│ config: HNSWConfig                                          │
└─────────────────────────────────────────────────────────────┘
```

## Build Process Comparison

### instant-distance Build Flow
```
┌──────────────────────────────────────────────────────────────────────┐
│                        BUILD PROCESS                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. PRE-CALCULATE LAYER SIZES                                        │
│     ┌────────────────────────────────────────┐                       │
│     │ sizes = []                              │                       │
│     │ num = points.len()                      │                       │
│     │ while (num * ml) >= M:                  │                       │
│     │     next = num * ml                     │                       │
│     │     sizes.push((num - next, num))       │                       │
│     │     num = next                          │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  2. SHUFFLE & REORDER POINTS                                         │
│     ┌────────────────────────────────────────┐                       │
│     │ Random shuffle → stable sort           │                       │
│     │ Points reordered by insertion order    │                       │
│     │ Output map: original_idx → new_idx     │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  3. CREATE RANGES FOR EACH LAYER                                     │
│     ┌────────────────────────────────────────┐                       │
│     │ Layer 4 (top):  range 0..3    (seq)    │ ← Few nodes          │
│     │ Layer 3:        range 3..12   (par)    │                       │
│     │ Layer 2:        range 12..50  (par)    │                       │
│     │ Layer 1:        range 50..200 (par)    │                       │
│     │ Layer 0:        range 200..N  (par)    │ ← Most nodes         │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  4. PROCESS LAYER BY LAYER (top → bottom)                            │
│     ┌────────────────────────────────────────────────────────────┐   │
│     │                                                             │   │
│     │  for (layer, range) in ranges:                             │   │
│     │      if layer == top:                                      │   │
│     │          range.into_iter().for_each(insert)    ← SEQUENTIAL│   │
│     │      else:                                                 │   │
│     │          range.into_par_iter().for_each(insert) ← PARALLEL │   │
│     │                                                             │   │
│     │      // CRITICAL: Copy zero layer to upper layer           │   │
│     │      if layer > 0:                                         │   │
│     │          layers[layer-1] = zero[..end]                     │   │
│     │              .par_iter()                                   │   │
│     │              .map(|z| UpperNode::from_zero(z))             │   │
│     │              .collect()                                    │   │
│     │                                                             │   │
│     └────────────────────────────────────────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘

Key Insight: After processing each layer, copy zero-layer state UP
             This ensures upper layers have consistent, complete data
```

### Foxstash Build Flow (Current)
```
┌──────────────────────────────────────────────────────────────────────┐
│                        BUILD PROCESS                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. PRE-GENERATE ALL LEVELS                                          │
│     ┌────────────────────────────────────────┐                       │
│     │ for each node:                          │                       │
│     │     level = floor(-ln(rand) * ml)       │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  2. SORT BY LEVEL DESCENDING                                         │
│     ┌────────────────────────────────────────┐                       │
│     │ sorted_indices.sort_by(level desc)      │                       │
│     │ entry_point = sorted_indices[0]         │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  3. CREATE RwLock<ParallelNode> FOR EACH NODE                        │
│     ┌────────────────────────────────────────┐                       │
│     │ nodes: Vec<RwLock<ParallelNode>>        │                       │
│     │   ParallelNode {                        │                       │
│     │     embedding: Vec<f32>,                │                       │
│     │     connections: Vec<HashSet<usize>>,   │ ← Dynamic sets       │
│     │   }                                     │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
│  4. INSERT NODES SEQUENTIALLY                                        │
│     ┌────────────────────────────────────────────────────────────┐   │
│     │                                                             │   │
│     │  for node_id in sorted_indices[1..]:                       │   │
│     │      // Greedy descent from entry_point                    │   │
│     │      current = entry_point                                 │   │
│     │      for layer in (node_level+1..=max_level).rev():       │   │
│     │          current = search_single(current, layer)          │   │
│     │                                                             │   │
│     │      // Insert into all layers from node_level down        │   │
│     │      for layer in (0..=node_level).rev():                 │   │
│     │          neighbors = search_layer(current, ef, layer)     │   │
│     │          selected = neighbors.take(M)                     │   │
│     │          add_bidirectional_links(node, selected, layer)   │   │
│     │          prune_if_needed(neighbors)                       │   │
│     │                                                             │   │
│     └────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  5. CONVERT TO SOA LAYOUT                                            │
│     ┌────────────────────────────────────────┐                       │
│     │ embeddings = flatten all node.embedding│                       │
│     │ connections = collect all connections  │                       │
│     └────────────────────────────────────────┘                       │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Why instant-distance Parallel Works

```
┌────────────────────────────────────────────────────────────────────┐
│                  instant-distance's Secret Sauce                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LAYER COPYING STRATEGY                                         │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ When inserting at layer L:                               │    │
│     │   - All nodes for layers > L are ALREADY INSERTED       │    │
│     │   - Upper layers are SNAPSHOTS of zero layer state      │    │
│     │   - Reads from upper layers are CONSISTENT              │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  2. FIXED-SIZE ARRAYS                                              │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ ZeroNode { nearest: [PointId; 64] }  ← No reallocation  │    │
│     │ UpperNode { nearest: [PointId; 32] } ← Cache-friendly   │    │
│     │                                                          │    │
│     │ vs Foxstash:                                             │    │
│     │ HashSet<usize> ← Dynamic, slower, more memory           │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  3. SEARCH POOL (THREAD-LOCAL REUSE)                               │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ SearchPool {                                             │    │
│     │     pool: Mutex<Vec<(Search, Search)>>                  │    │
│     │ }                                                        │    │
│     │                                                          │    │
│     │ - Pre-allocate search state per thread                  │    │
│     │ - Avoid allocation during parallel insertion            │    │
│     │ - Pop from pool, use, push back                         │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  4. VISITED BITMAP (NOT HashSet)                                   │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Visited {                                                │    │
│     │     store: Vec<u8>,  ← Bit per node                     │    │
│     │     generation: u8,  ← Increment instead of clear       │    │
│     │ }                                                        │    │
│     │                                                          │    │
│     │ O(1) check, O(1) insert, O(1) "clear" via generation   │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Why Foxstash Parallel Fails at Scale

```
┌────────────────────────────────────────────────────────────────────┐
│                     Race Condition Problem                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  When inserting node A and B simultaneously at layer 0:            │
│                                                                     │
│  Thread 1 (Node A)           Thread 2 (Node B)                     │
│  ─────────────────           ─────────────────                     │
│  1. Read neighbors of C      1. Read neighbors of C                │
│  2. Decide to connect A→C    2. Decide to connect B→C              │
│  3. Add A to C.neighbors     3. Add B to C.neighbors               │
│  4. C.neighbors > M, prune   4. C.neighbors > M, prune             │
│       ↓                           ↓                                │
│  A or B might get pruned!    Race to see final state!              │
│                                                                     │
│  Result: Inconsistent graph, poor connectivity, low recall         │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  instant-distance avoids this because:                             │
│  - Upper layers are READ-ONLY during lower layer processing        │
│  - Zero layer uses RwLock but with consistent snapshots            │
│  - Layer copying creates stable references for parallel reads      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Recommended Fix for Foxstash

```
┌────────────────────────────────────────────────────────────────────┐
│                      Adoption Strategy                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option A: Full instant-distance approach                          │
│  ─────────────────────────────────────────                         │
│  1. Use fixed-size arrays instead of HashSet                       │
│  2. Separate ZeroNode and UpperNode types                          │
│  3. Process by layer ranges, not by node level                     │
│  4. Copy zero-layer state up after each layer                      │
│  5. Implement SearchPool for thread-local state reuse              │
│  6. Use Visited bitmap instead of HashSet                          │
│                                                                     │
│  Option B: Hybrid approach (less invasive)                         │
│  ─────────────────────────────────────────                         │
│  1. Keep current data structures                                   │
│  2. Process in batches with sync points                            │
│  3. After each batch, ensure all pruning is complete               │
│  4. Use read-mostly pattern: batch reads, then batch writes        │
│                                                                     │
│  Option C: Keep sequential, optimize elsewhere                     │
│  ─────────────────────────────────────────────                     │
│  1. Sequential insertion preserves correctness                     │
│  2. Parallel distance calculations (already using SIMD)            │
│  3. Parallel final conversion to SoA                               │
│  4. Pre-allocate all memory upfront                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Performance Summary

| Metric | Foxstash (10k) | Foxstash (100k) | instant-distance |
|--------|----------------|-----------------|------------------|
| Build | 5.2s | 86.8s | 72.4s |
| Search QPS | 2,447 | 806 | 580 |
| Recall@10 | 97.5% | 58% | ~95% |

**Key Finding:** Foxstash search is faster (SIMD), but parallel build breaks recall at scale.
The instant-distance layer-copying approach is the key to safe parallelization.
