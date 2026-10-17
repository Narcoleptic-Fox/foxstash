// FFI functions intentionally dereference raw pointers without being marked
// `unsafe` — they are C-ABI entry points with internal validation and panic
// guards, following the common pattern for Rust-to-C library exports.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! Native FFI bindings for iOS, Android, and Desktop platforms
//!
//! Provides a C-compatible API for embedding the RAG system in native applications.
//!
//! # Safety
//!
//! This module exposes raw C-compatible functions that bypass Rust's memory safety
//! guarantees. All functions perform extensive validation and use panic guards to
//! prevent undefined behavior.
//!
//! # Architecture
//!
//! - **RagHandle**: Opaque handle wrapping either FlatIndex or HNSWIndex
//! - **Thread-local errors**: Safe error reporting across FFI boundary
//! - **Atomic writes**: Persistence with FileStorage
//! - **Memory management**: Proper CString/Vec lifecycle management
//!
//! # Example (C)
//!
//! ```c
//! #include "foxstash_rag.h"
//!
//! int main() {
//!     // Create index
//!     RagHandle* rag = rag_create(384, 1);  // 384-dim, HNSW
//!
//!     // Add document
//!     float embedding[384];
//!     for (int i = 0; i < 384; i++) embedding[i] = 0.1f;
//!
//!     rag_add_document(rag, "doc1", "Hello world", embedding, 384);
//!
//!     // Search
//!     SearchResult* results;
//!     size_t count;
//!     rag_search(rag, embedding, 384, 5, &results, &count);
//!
//!     // Use results...
//!     for (size_t i = 0; i < count; i++) {
//!         printf("Result: %s (%.2f)\n", results[i].id, results[i].score);
//!     }
//!
//!     // Cleanup
//!     rag_free_results(results, count);
//!     rag_destroy(rag);
//!
//!     return 0;
//! }
//! ```

use foxstash_core::index::{FlatIndex, HNSWIndex};
use foxstash_core::storage::file::{FileStorage, FlatIndexWrapper, HNSWIndexWrapper};
use foxstash_core::{Document, SearchResult as CoreSearchResult};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::{Mutex, OnceLock, RwLock};

// =============================================================================
// Thread-local Error Storage
// =============================================================================

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn results_allocations() -> &'static Mutex<HashMap<usize, usize>> {
    static RESULTS: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    RESULTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set the last error message for this thread
fn set_last_error(err: String) {
    LAST_ERROR.with(|e| {
        // Ensure error string is valid C string
        let err_cstring = CString::new(err)
            .unwrap_or_else(|_| CString::new("Error message contains null bytes").unwrap());
        *e.borrow_mut() = Some(err_cstring);
    });
}

/// Clear the last error for this thread
fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// =============================================================================
// Core Types
// =============================================================================

/// Trait for unified index operations
trait RagIndex: Send + Sync {
    fn add(&mut self, document: Document) -> Result<(), String>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<CoreSearchResult>, String>;
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn save(&self, storage: &FileStorage, name: &str) -> Result<(), String>;
    /// Fit the quantization codebook from a sample, so a quantized index can accept `add`.
    /// A no-op precondition for `Storage::F32`; unsupported for a flat index.
    fn train(&mut self, _sample: &[Vec<f32>]) -> Result<(), String> {
        Err("train() is only supported for an HNSW index".to_string())
    }
}

impl RagIndex for FlatIndex {
    fn add(&mut self, document: Document) -> Result<(), String> {
        self.add(document).map_err(|e| e.to_string())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<CoreSearchResult>, String> {
        self.search(query, k).map_err(|e| e.to_string())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn save(&self, storage: &FileStorage, name: &str) -> Result<(), String> {
        let wrapper = FlatIndexWrapper::from_index(self);
        storage
            .save_flat_index(name, &wrapper)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

impl RagIndex for HNSWIndex {
    fn add(&mut self, document: Document) -> Result<(), String> {
        self.add(document).map_err(|e| e.to_string())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<CoreSearchResult>, String> {
        self.search(query, k).map_err(|e| e.to_string())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn save(&self, storage: &FileStorage, name: &str) -> Result<(), String> {
        let wrapper = HNSWIndexWrapper::from_index(self);
        storage
            .save_hnsw_index(name, &wrapper)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn train(&mut self, sample: &[Vec<f32>]) -> Result<(), String> {
        self.train(sample).map_err(|e| e.to_string())
    }
}

/// Opaque handle to RAG index.
///
/// # Threading Model
///
/// `RagHandle` is `Send + Sync` and designed for concurrent access from multiple
/// threads via the C FFI. Internal synchronization uses:
///
/// - `RwLock<Box<dyn RagIndex>>` for the index (read-shared, write-exclusive)
/// - `Mutex<Option<FileStorage>>` for persistence storage
///
/// # Lock Ordering
///
/// When both locks are needed (e.g., `rag_save`), always acquire the **storage
/// `Mutex` before** the **index `RwLock`**. Never acquire the index lock and then
/// attempt to acquire the storage lock -- doing so risks deadlock.
pub struct RagHandle {
    index: RwLock<Box<dyn RagIndex>>,
    storage: Mutex<Option<FileStorage>>,
    embedding_dim: usize,
}

/// C-compatible search result
#[repr(C)]
pub struct SearchResult {
    pub id: *mut c_char,
    pub content: *mut c_char,
    pub score: f32,
}

// =============================================================================
// Configuration (issue #61)
// =============================================================================

/// Distance metric codes for [`FoxstashConfig::metric`]. Stable integers, not the Rust enum.
pub const FOXSTASH_METRIC_COSINE: u32 = 0;
pub const FOXSTASH_METRIC_L2: u32 = 1;

/// Storage mode codes for [`FoxstashConfig::storage`]. Stable integers, not the Rust enum.
pub const FOXSTASH_STORAGE_F32: u32 = 0;
pub const FOXSTASH_STORAGE_SQ8: u32 = 1;
pub const FOXSTASH_STORAGE_RABITQ: u32 = 2;
pub const FOXSTASH_STORAGE_TURBORABIT: u32 = 3;
pub const FOXSTASH_STORAGE_MOUSE: u32 = 4;

/// C-compatible view of the tuning knobs on `HNSWConfig`, so a native caller can select a metric,
/// a quantized storage mode, reranking, graph degree and so on — none of which the bare
/// `rag_create` exposes (issue #61). Build one with [`rag_default_config`], tweak the fields, and
/// pass it to [`rag_create_with_config`]. Booleans are `i32` (0 = false), `seed` is `i64`
/// (negative = unseeded), and `ml <= 0` means "derive the HNSW default `1/ln(m)`".
#[repr(C)]
pub struct FoxstashConfig {
    pub metric: u32,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f32,
    pub use_heuristic: i32,
    pub extend_candidates: i32,
    pub keep_pruned_connections: i32,
    pub seed: i64,
    pub storage: u32,
    pub rerank_candidates: usize,
    pub rabit_bits: usize,
    pub reorder_for_locality: i32,
}

fn metric_from_code(code: u32) -> Result<foxstash_core::index::DistanceMetric, String> {
    use foxstash_core::index::DistanceMetric;
    match code {
        FOXSTASH_METRIC_COSINE => Ok(DistanceMetric::Cosine),
        FOXSTASH_METRIC_L2 => Ok(DistanceMetric::L2),
        other => Err(format!(
            "unknown metric code {other} (expected 0=cosine, 1=l2)"
        )),
    }
}

fn storage_from_code(code: u32) -> Result<foxstash_core::index::Storage, String> {
    use foxstash_core::index::Storage;
    match code {
        FOXSTASH_STORAGE_F32 => Ok(Storage::F32),
        FOXSTASH_STORAGE_SQ8 => Ok(Storage::SQ8),
        FOXSTASH_STORAGE_RABITQ => Ok(Storage::RaBitQ),
        FOXSTASH_STORAGE_TURBORABIT => Ok(Storage::TurboRabit),
        FOXSTASH_STORAGE_MOUSE => Ok(Storage::Mouse),
        other => Err(format!(
            "unknown storage code {other} (expected 0=f32, 1=sq8, 2=rabitq, 3=turborabit, 4=mouse)"
        )),
    }
}

impl FoxstashConfig {
    fn to_hnsw(&self) -> Result<foxstash_core::index::HNSWConfig, String> {
        let defaults = foxstash_core::index::HNSWConfig::default();
        let m = if self.m == 0 { defaults.m } else { self.m };
        Ok(foxstash_core::index::HNSWConfig {
            metric: metric_from_code(self.metric)?,
            m,
            m0: if self.m0 == 0 { m * 2 } else { self.m0 },
            ef_construction: if self.ef_construction == 0 {
                defaults.ef_construction
            } else {
                self.ef_construction
            },
            ef_search: if self.ef_search == 0 {
                defaults.ef_search
            } else {
                self.ef_search
            },
            ml: if self.ml > 0.0 {
                self.ml
            } else {
                1.0 / (m as f32).ln()
            },
            use_heuristic: self.use_heuristic != 0,
            extend_candidates: self.extend_candidates != 0,
            keep_pruned_connections: self.keep_pruned_connections != 0,
            build_strategy: defaults.build_strategy,
            seed: if self.seed < 0 {
                None
            } else {
                Some(self.seed as u64)
            },
            storage: storage_from_code(self.storage)?,
            rerank_candidates: self.rerank_candidates,
            rabit_bits: if self.rabit_bits == 0 {
                defaults.rabit_bits
            } else {
                self.rabit_bits
            },
            reorder_for_locality: self.reorder_for_locality != 0,
        })
    }
}

/// Fill `out` with the default `HNSWConfig` values, expressed as a [`FoxstashConfig`]. A caller
/// tweaks the fields it cares about and passes the struct to [`rag_create_with_config`]; the
/// fields it leaves alone stay at the library defaults. Returns 0, or -1 if `out` is NULL.
///
/// # Safety
/// `out` must be a valid, writable pointer to a `FoxstashConfig`.
#[no_mangle]
pub extern "C" fn rag_default_config(out: *mut FoxstashConfig) -> i32 {
    if out.is_null() {
        set_last_error("Output config pointer is NULL".to_string());
        return -1;
    }
    let d = foxstash_core::index::HNSWConfig::default();
    let cfg = FoxstashConfig {
        metric: FOXSTASH_METRIC_COSINE,
        m: d.m,
        m0: d.m0,
        ef_construction: d.ef_construction,
        ef_search: d.ef_search,
        ml: d.ml,
        use_heuristic: d.use_heuristic as i32,
        extend_candidates: d.extend_candidates as i32,
        keep_pruned_connections: d.keep_pruned_connections as i32,
        seed: -1,
        storage: FOXSTASH_STORAGE_F32,
        rerank_candidates: d.rerank_candidates,
        rabit_bits: d.rabit_bits,
        reorder_for_locality: d.reorder_for_locality as i32,
    };
    unsafe { *out = cfg };
    0
}

// =============================================================================
// Creation and Destruction
// =============================================================================

/// Create a new RAG index.
///
/// # Arguments
/// * `embedding_dim` - Dimensionality of embeddings (e.g., 384)
/// * `use_hnsw` - 0 for Flat index (exact), 1 for HNSW (approximate)
///
/// # Returns
/// Pointer to RagHandle, or NULL on failure. Check `rag_last_error()` for error details.
///
/// # Safety
/// Must call `rag_destroy()` when done to free memory.
///
/// # Example (C)
/// ```c
/// RagHandle* rag = rag_create(384, 1);  // 384-dim, HNSW
/// if (rag == NULL) {
///     fprintf(stderr, "Failed to create RAG: %s\n", rag_last_error());
///     return -1;
/// }
/// // Use rag...
/// rag_destroy(rag);
/// ```
#[no_mangle]
pub extern "C" fn rag_create(embedding_dim: usize, use_hnsw: i32) -> *mut RagHandle {
    clear_last_error();

    // Catch any panics
    let result = panic::catch_unwind(|| {
        // Validate input
        if embedding_dim == 0 {
            set_last_error("Embedding dimension must be greater than 0".to_string());
            return ptr::null_mut();
        }

        if embedding_dim > 100000 {
            set_last_error("Embedding dimension too large (max 100000)".to_string());
            return ptr::null_mut();
        }

        // Create index based on type
        let index: Box<dyn RagIndex> = if use_hnsw != 0 {
            Box::new(HNSWIndex::with_defaults(embedding_dim))
        } else {
            Box::new(FlatIndex::new(embedding_dim))
        };

        let handle = RagHandle {
            index: RwLock::new(index),
            storage: Mutex::new(None),
            embedding_dim,
        };

        Box::into_raw(Box::new(handle))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("Panic occurred during index creation".to_string());
            ptr::null_mut()
        }
    }
}

/// Create an HNSW index with an explicit [`FoxstashConfig`] (issue #61), exposing the metric,
/// storage mode, reranking, graph degree and other knobs that [`rag_create`] leaves at their
/// defaults. Returns a handle, or NULL on failure (check `rag_last_error`).
///
/// A **quantized** storage mode (`sq8`, `rabitq`, `turborabit`, `mouse`) needs a fitted codebook
/// before it will accept documents: call [`rag_train`] with a representative sample first, then
/// [`rag_add_document`]. Under `f32` (the default) no training is needed. `mouse` additionally
/// cannot be grown incrementally at all (it keeps no exact vectors) and so is not usable through
/// this incremental ABI; it is accepted here only to round-trip the enum.
///
/// # Arguments
/// * `embedding_dim` - Dimensionality of embeddings.
/// * `config` - Pointer to a [`FoxstashConfig`] (see [`rag_default_config`]).
///
/// # Safety
/// `config` must be a valid pointer to a `FoxstashConfig` for the duration of the call.
#[no_mangle]
pub extern "C" fn rag_create_with_config(
    embedding_dim: usize,
    config: *const FoxstashConfig,
) -> *mut RagHandle {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if embedding_dim == 0 {
            set_last_error("Embedding dimension must be greater than 0".to_string());
            return ptr::null_mut();
        }
        if embedding_dim > 100000 {
            set_last_error("Embedding dimension too large (max 100000)".to_string());
            return ptr::null_mut();
        }
        if config.is_null() {
            set_last_error("Config pointer is NULL".to_string());
            return ptr::null_mut();
        }

        let hnsw_config = match unsafe { &*config }.to_hnsw() {
            Ok(c) => c,
            Err(e) => {
                set_last_error(format!("Invalid config: {e}"));
                return ptr::null_mut();
            }
        };

        let index: Box<dyn RagIndex> = Box::new(HNSWIndex::new(embedding_dim, hnsw_config));
        let handle = RagHandle {
            index: RwLock::new(index),
            storage: Mutex::new(None),
            embedding_dim,
        };
        Box::into_raw(Box::new(handle))
    }));

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("Panic occurred during index creation".to_string());
            ptr::null_mut()
        }
    }
}

/// Fit the quantization codebook of an index created with a quantized [`FoxstashConfig`], from a
/// representative sample of vectors, so it can accept documents via [`rag_add_document`] (issue
/// #61). A no-op-friendly call under `f32` (returns success). `sample` is a flat row-major array
/// of `n_vectors * embedding_dim` floats.
///
/// Returns 0 on success, -1 on failure (check `rag_last_error`).
///
/// # Safety
/// `handle` must be a valid `RagHandle`, and `sample` must point to at least
/// `n_vectors * embedding_dim` `f32` values.
#[no_mangle]
pub extern "C" fn rag_train(
    handle: *mut RagHandle,
    sample: *const f32,
    n_vectors: usize,
    embedding_dim: usize,
) -> i32 {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            set_last_error("Handle is NULL".to_string());
            return -1;
        }
        if sample.is_null() {
            set_last_error("Sample pointer is NULL".to_string());
            return -1;
        }
        let handle = unsafe { &*handle };
        if embedding_dim != handle.embedding_dim {
            set_last_error(format!(
                "Sample dimension mismatch: expected {}, got {}",
                handle.embedding_dim, embedding_dim
            ));
            return -1;
        }
        if n_vectors == 0 {
            set_last_error("Sample must contain at least one vector".to_string());
            return -1;
        }

        let flat = unsafe { std::slice::from_raw_parts(sample, n_vectors * embedding_dim) };
        let vectors: Vec<Vec<f32>> = flat
            .chunks_exact(embedding_dim)
            .map(|row| row.to_vec())
            .collect();

        let mut index = match handle.index.write() {
            Ok(g) => g,
            Err(_) => {
                set_last_error("Index lock poisoned".to_string());
                return -1;
            }
        };
        match index.train(&vectors) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(format!("train failed: {e}"));
                -1
            }
        }
    }));

    result.unwrap_or_else(|_| {
        set_last_error("Panic occurred during train".to_string());
        -1
    })
}

/// Destroy a RAG index and free all associated memory.
///
/// # Arguments
/// * `handle` - RAG handle to destroy
///
/// # Safety
/// After calling this function, the handle is invalid and must not be used.
/// Passing NULL is safe (no-op).
///
/// **Caller must ensure no other threads are using this handle when calling
/// `rag_destroy`.** Calling `rag_destroy` while another thread holds a read or
/// write lock on the handle is undefined behavior (the `Box::from_raw` reclaims
/// memory that may still be accessed by the other thread).
///
/// # Example (C)
/// ```c
/// RagHandle* rag = rag_create(384, 1);
/// // Use rag...
/// rag_destroy(rag);  // Handle is now invalid
/// ```
#[no_mangle]
pub extern "C" fn rag_destroy(handle: *mut RagHandle) {
    if handle.is_null() {
        return;
    }

    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            // Convert raw pointer back to Box and drop it
            let _handle = Box::from_raw(handle);
        }
    }));
}

// =============================================================================
// Document Operations
// =============================================================================

/// Add a document to the index.
///
/// # Arguments
/// * `handle` - RAG handle
/// * `id` - Unique document ID (null-terminated C string)
/// * `content` - Document content (null-terminated C string)
/// * `embedding` - Embedding vector (array of floats)
/// * `embedding_len` - Length of embedding vector
///
/// # Returns
/// * 0 on success
/// * -1 on error (check `rag_last_error()`)
///
/// # Safety
/// * `handle` must be a valid pointer from `rag_create()`
/// * `id` and `content` must be valid null-terminated C strings
/// * `embedding` must point to at least `embedding_len` floats
///
/// # Example (C)
/// ```c
/// float embedding[384];
/// for (int i = 0; i < 384; i++) {
///     embedding[i] = 0.1f;
/// }
///
/// int ret = rag_add_document(rag, "doc1", "Hello world", embedding, 384);
/// if (ret != 0) {
///     fprintf(stderr, "Error: %s\n", rag_last_error());
/// }
/// ```
#[no_mangle]
pub extern "C" fn rag_add_document(
    handle: *mut RagHandle,
    id: *const c_char,
    content: *const c_char,
    embedding: *const f32,
    embedding_len: usize,
) -> i32 {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // Null checks
        if handle.is_null() {
            set_last_error("Handle is NULL".to_string());
            return -1;
        }

        if id.is_null() {
            set_last_error("ID is NULL".to_string());
            return -1;
        }

        if content.is_null() {
            set_last_error("Content is NULL".to_string());
            return -1;
        }

        if embedding.is_null() {
            set_last_error("Embedding is NULL".to_string());
            return -1;
        }

        unsafe {
            let handle = &*handle;

            // Validate embedding dimension
            if embedding_len != handle.embedding_dim {
                set_last_error(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    handle.embedding_dim, embedding_len
                ));
                return -1;
            }

            // Convert C strings to Rust strings
            let id_str = match CStr::from_ptr(id).to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_last_error(format!("Invalid ID string: {}", e));
                    return -1;
                }
            };

            let content_str = match CStr::from_ptr(content).to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_last_error(format!("Invalid content string: {}", e));
                    return -1;
                }
            };

            // Copy embedding vector
            let embedding_vec = std::slice::from_raw_parts(embedding, embedding_len).to_vec();

            // Create document
            let document = Document {
                id: id_str,
                content: content_str,
                embedding: embedding_vec,
                metadata: None,
            };

            // Add to index
            let mut index = match handle.index.write() {
                Ok(lock) => lock,
                Err(_) => {
                    set_last_error("Index lock poisoned".to_string());
                    return -1;
                }
            };

            match index.add(document) {
                Ok(_) => 0,
                Err(e) => {
                    set_last_error(format!("Failed to add document: {}", e));
                    -1
                }
            }
        }
    }));

    match result {
        Ok(ret) => ret,
        Err(_) => {
            set_last_error("Panic occurred during add_document".to_string());
            -1
        }
    }
}

// =============================================================================
// Search Operations
// =============================================================================

/// Search for k nearest neighbors.
///
/// # Arguments
/// * `handle` - RAG handle
/// * `query` - Query embedding vector (array of floats)
/// * `query_len` - Length of query vector
/// * `k` - Number of results to return
/// * `results_out` - Output pointer for results array
/// * `count_out` - Output pointer for result count
///
/// # Returns
/// * 0 on success (results written to `results_out` and `count_out`)
/// * -1 on error (check `rag_last_error()`)
///
/// # Safety
/// * `handle` must be a valid pointer from `rag_create()`
/// * `query` must point to at least `query_len` floats
/// * `results_out` and `count_out` must be valid pointers
/// * Results must be freed with `rag_free_results()` when done
///
/// # Example (C)
/// ```c
/// float query[384];
/// // Fill query...
///
/// SearchResult* results;
/// size_t count;
///
/// if (rag_search(rag, query, 384, 5, &results, &count) == 0) {
///     for (size_t i = 0; i < count; i++) {
///         printf("%s: %.2f\n", results[i].id, results[i].score);
///     }
///     rag_free_results(results, count);
/// } else {
///     fprintf(stderr, "Search error: %s\n", rag_last_error());
/// }
/// ```
#[no_mangle]
pub extern "C" fn rag_search(
    handle: *const RagHandle,
    query: *const f32,
    query_len: usize,
    k: usize,
    results_out: *mut *mut SearchResult,
    count_out: *mut usize,
) -> i32 {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // Null checks
        if handle.is_null() {
            set_last_error("Handle is NULL".to_string());
            return -1;
        }

        if query.is_null() {
            set_last_error("Query is NULL".to_string());
            return -1;
        }

        if results_out.is_null() {
            set_last_error("Results output pointer is NULL".to_string());
            return -1;
        }

        if count_out.is_null() {
            set_last_error("Count output pointer is NULL".to_string());
            return -1;
        }

        unsafe {
            *results_out = ptr::null_mut();
            *count_out = 0;

            let handle = &*handle;

            // Validate query dimension
            if query_len != handle.embedding_dim {
                set_last_error(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    handle.embedding_dim, query_len
                ));
                return -1;
            }

            // Convert query to slice
            let query_vec = std::slice::from_raw_parts(query, query_len);

            // Perform search
            let index = match handle.index.read() {
                Ok(lock) => lock,
                Err(_) => {
                    set_last_error("Index lock poisoned".to_string());
                    return -1;
                }
            };

            let results = match index.search(query_vec, k) {
                Ok(r) => r,
                Err(e) => {
                    set_last_error(format!("Search failed: {}", e));
                    return -1;
                }
            };

            // Convert results to C format
            let c_results: Vec<SearchResult> = results
                .into_iter()
                .filter_map(|r| {
                    let id_cstring = CString::new(r.id).ok()?;
                    let content_cstring = CString::new(r.content).ok()?;

                    Some(SearchResult {
                        id: id_cstring.into_raw(),
                        content: content_cstring.into_raw(),
                        score: r.score,
                    })
                })
                .collect();

            let count = c_results.len();

            // Allocate results array
            let results_ptr = if count > 0 {
                let boxed = c_results.into_boxed_slice();
                Box::into_raw(boxed) as *mut SearchResult
            } else {
                ptr::null_mut()
            };

            if !results_ptr.is_null() {
                match results_allocations().lock() {
                    Ok(mut allocations) => {
                        allocations.insert(results_ptr as usize, count);
                    }
                    Err(_) => {
                        set_last_error("Results allocation tracker lock poisoned".to_string());
                        let boxed_ptr = std::ptr::slice_from_raw_parts_mut(results_ptr, count);
                        let _ = Box::from_raw(boxed_ptr);
                        return -1;
                    }
                }
            }

            *results_out = results_ptr;
            *count_out = count;

            0
        }
    }));

    match result {
        Ok(ret) => ret,
        Err(_) => {
            set_last_error("Panic occurred during search".to_string());
            -1
        }
    }
}

/// Free search results allocated by `rag_search()`.
///
/// # Arguments
/// * `results` - Results array to free
/// * `count` - Number of results in array (ignored; kept for ABI compatibility)
///
/// # Safety
/// * `results` must be from `rag_search()`
/// * Must be called exactly once per results array
/// * Passing NULL is safe (no-op)
///
/// # Example (C)
/// ```c
/// SearchResult* results;
/// size_t count;
///
/// rag_search(rag, query, 384, 5, &results, &count);
/// // Use results...
/// rag_free_results(results, count);
/// ```
#[no_mangle]
pub extern "C" fn rag_free_results(results: *mut SearchResult, _count: usize) {
    if results.is_null() {
        return;
    }

    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        let actual_count = match results_allocations().lock() {
            Ok(mut allocations) => match allocations.remove(&(results as usize)) {
                Some(saved) => saved,
                None => return,
            },
            Err(_) => return,
        };

        unsafe {
            // Reconstruct the boxed slice
            let boxed_ptr = std::ptr::slice_from_raw_parts_mut(results, actual_count);
            let results_slice = Box::from_raw(boxed_ptr);

            // Free each result's strings
            for result in results_slice.iter() {
                if !result.id.is_null() {
                    let _ = CString::from_raw(result.id);
                }
                if !result.content.is_null() {
                    let _ = CString::from_raw(result.content);
                }
            }
        }
    }));
}

// =============================================================================
// Persistence Operations
// =============================================================================

/// Save index to disk.
///
/// # Arguments
/// * `handle` - RAG handle
/// * `path` - Directory path for storage (null-terminated C string)
///
/// # Returns
/// * 0 on success
/// * -1 on error (check `rag_last_error()`)
///
/// # Safety
/// * `handle` must be a valid pointer from `rag_create()`
/// * `path` must be a valid null-terminated C string
///
/// # Example (C)
/// ```c
/// if (rag_save(rag, "/tmp/my_rag") != 0) {
///     fprintf(stderr, "Save failed: %s\n", rag_last_error());
/// }
/// ```
#[no_mangle]
pub extern "C" fn rag_save(handle: *mut RagHandle, path: *const c_char) -> i32 {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            set_last_error("Handle is NULL".to_string());
            return -1;
        }

        if path.is_null() {
            set_last_error("Path is NULL".to_string());
            return -1;
        }

        unsafe {
            let handle = &*handle;

            // Convert path to Rust string
            let path_str = match CStr::from_ptr(path).to_str() {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("Invalid path string: {}", e));
                    return -1;
                }
            };

            // Create or get storage
            let mut storage_guard = match handle.storage.lock() {
                Ok(lock) => lock,
                Err(_) => {
                    set_last_error("Storage lock poisoned".to_string());
                    return -1;
                }
            };

            if storage_guard.is_none() {
                // Create new storage
                match FileStorage::new(path_str) {
                    Ok(s) => *storage_guard = Some(s),
                    Err(e) => {
                        set_last_error(format!("Failed to create storage: {}", e));
                        return -1;
                    }
                };
            }

            let storage = match storage_guard.as_ref() {
                Some(s) => s,
                None => {
                    set_last_error("Storage not initialized".to_string());
                    return -1;
                }
            };

            // Save index
            let index = match handle.index.read() {
                Ok(lock) => lock,
                Err(_) => {
                    set_last_error("Index lock poisoned".to_string());
                    return -1;
                }
            };

            match index.save(storage, "index") {
                Ok(_) => 0,
                Err(e) => {
                    set_last_error(format!("Failed to save index: {}", e));
                    -1
                }
            }
        }
    }));

    match result {
        Ok(ret) => ret,
        Err(_) => {
            set_last_error("Panic occurred during save".to_string());
            -1
        }
    }
}

/// Load index from disk.
///
/// # Arguments
/// * `path` - Directory path for storage (null-terminated C string)
/// * `use_hnsw` - 0 for Flat index, 1 for HNSW index
///
/// # Returns
/// * Pointer to RagHandle on success
/// * NULL on error (check `rag_last_error()`)
///
/// # Safety
/// * `path` must be a valid null-terminated C string
/// * Must call `rag_destroy()` when done
///
/// # Example (C)
/// ```c
/// RagHandle* rag = rag_load("/tmp/my_rag", 1);
/// if (rag == NULL) {
///     fprintf(stderr, "Load failed: %s\n", rag_last_error());
///     return -1;
/// }
/// // Use rag...
/// rag_destroy(rag);
/// ```
#[no_mangle]
pub extern "C" fn rag_load(path: *const c_char, use_hnsw: i32) -> *mut RagHandle {
    clear_last_error();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            set_last_error("Path is NULL".to_string());
            return ptr::null_mut();
        }

        unsafe {
            // Convert path to Rust string
            let path_str = match CStr::from_ptr(path).to_str() {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("Invalid path string: {}", e));
                    return ptr::null_mut();
                }
            };

            // Create storage
            let storage = match FileStorage::new(path_str) {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("Failed to create storage: {}", e));
                    return ptr::null_mut();
                }
            };

            // Load index based on type
            if use_hnsw != 0 {
                // Load HNSW index
                match storage.load_hnsw_index("index") {
                    Ok(wrapper) => match wrapper.to_index() {
                        Ok(index) => {
                            let embedding_dim = index.embedding_dim();
                            let handle = RagHandle {
                                index: RwLock::new(Box::new(index)),
                                storage: Mutex::new(Some(storage)),
                                embedding_dim,
                            };
                            Box::into_raw(Box::new(handle))
                        }
                        Err(e) => {
                            set_last_error(format!("Failed to restore HNSW index: {}", e));
                            ptr::null_mut()
                        }
                    },
                    Err(e) => {
                        set_last_error(format!("Failed to load HNSW index: {}", e));
                        ptr::null_mut()
                    }
                }
            } else {
                // Load Flat index
                match storage.load_flat_index("index") {
                    Ok(wrapper) => match wrapper.to_index() {
                        Ok(index) => {
                            let embedding_dim = index.embedding_dim();
                            let handle = RagHandle {
                                index: RwLock::new(Box::new(index)),
                                storage: Mutex::new(Some(storage)),
                                embedding_dim,
                            };
                            Box::into_raw(Box::new(handle))
                        }
                        Err(e) => {
                            set_last_error(format!("Failed to restore Flat index: {}", e));
                            ptr::null_mut()
                        }
                    },
                    Err(e) => {
                        set_last_error(format!("Failed to load Flat index: {}", e));
                        ptr::null_mut()
                    }
                }
            }
        }
    }));

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("Panic occurred during load".to_string());
            ptr::null_mut()
        }
    }
}

// =============================================================================
// Utility Operations
// =============================================================================

/// Get the number of documents in the index.
///
/// # Arguments
/// * `handle` - RAG handle
///
/// # Returns
/// Number of documents, or 0 if handle is NULL
///
/// # Safety
/// * `handle` must be a valid pointer from `rag_create()`
///
/// # Example (C)
/// ```c
/// size_t count = rag_count(rag);
/// printf("Index contains %zu documents\n", count);
/// ```
#[no_mangle]
pub extern "C" fn rag_count(handle: *const RagHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        let handle = &*handle;
        let index = match handle.index.read() {
            Ok(lock) => lock,
            Err(_) => {
                set_last_error("Index lock poisoned".to_string());
                return 0;
            }
        };
        index.len()
    }));

    result.unwrap_or(0)
}

/// Clear all documents from the index.
///
/// # Arguments
/// * `handle` - RAG handle
///
/// # Returns
/// * 0 on success
/// * -1 on error
///
/// # Safety
/// * `handle` must be a valid pointer from `rag_create()`
///
/// # Example (C)
/// ```c
/// rag_clear(rag);
/// printf("Index cleared\n");
/// ```
#[no_mangle]
pub extern "C" fn rag_clear(handle: *mut RagHandle) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("Handle is NULL".to_string());
        return -1;
    }

    let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        let handle = &*handle;
        let mut index = match handle.index.write() {
            Ok(lock) => lock,
            Err(_) => {
                set_last_error("Index lock poisoned".to_string());
                return -1;
            }
        };
        index.clear();
        0
    }));

    match result {
        Ok(ret) => ret,
        Err(_) => {
            set_last_error("Panic occurred during clear".to_string());
            -1
        }
    }
}

// =============================================================================
// Error Handling
// =============================================================================

/// Get the last error message for this thread.
///
/// # Returns
/// * Pointer to error message string, or NULL if no error
///
/// # Safety
/// * Return value is valid until next FFI call on this thread
/// * Do not free the returned pointer
/// * Thread-safe: each thread has its own error
///
/// # Example (C)
/// ```c
/// if (rag_add_document(rag, ...) != 0) {
///     const char* error = rag_last_error();
///     if (error != NULL) {
///         fprintf(stderr, "Error: %s\n", error);
///     }
/// }
/// ```
#[no_mangle]
pub extern "C" fn rag_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(err) => err.as_ptr(),
        None => ptr::null(),
    })
}

/// Free an error string returned by `rag_last_error()`.
///
/// # Arguments
/// * `error` - Error string to free
///
/// # Safety
/// * This function is deprecated - error strings are managed internally
/// * Calling this is safe but does nothing
/// * Do not call CString::from_raw on error strings
///
/// # Note
/// Error strings are automatically managed and freed on the next FFI call
/// or when the thread exits. This function exists for API compatibility
/// but is a no-op.
#[no_mangle]
pub extern "C" fn rag_free_error(_error: *const c_char) {
    // No-op: errors are managed by thread-local storage
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn create_test_embedding(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32) / (dim as f32)).collect()
    }

    #[test]
    fn test_create_destroy_flat() {
        let handle = rag_create(384, 0);
        assert!(!handle.is_null());

        let count = rag_count(handle);
        assert_eq!(count, 0);

        rag_destroy(handle);
    }

    #[test]
    fn test_create_destroy_hnsw() {
        let handle = rag_create(384, 1);
        assert!(!handle.is_null());

        let count = rag_count(handle);
        assert_eq!(count, 0);

        rag_destroy(handle);
    }

    #[test]
    fn test_create_invalid_dimension() {
        let handle = rag_create(0, 0);
        assert!(handle.is_null());

        let error = rag_last_error();
        assert!(!error.is_null());
    }

    #[test]
    fn test_add_document() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test document").unwrap();
        let embedding = [0.1f32, 0.2, 0.3];

        let ret = rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );

        assert_eq!(ret, 0);
        assert_eq!(rag_count(handle), 1);

        rag_destroy(handle);
    }

    #[test]
    fn test_add_document_dimension_mismatch() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test").unwrap();
        let embedding = [0.1f32, 0.2]; // Wrong dimension

        let ret = rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );

        assert_eq!(ret, -1);

        let error = rag_last_error();
        assert!(!error.is_null());

        rag_destroy(handle);
    }

    #[test]
    fn test_search() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        // Add documents
        let docs = vec![
            ("doc1", vec![1.0f32, 0.0, 0.0]),
            ("doc2", vec![0.0f32, 1.0, 0.0]),
            ("doc3", vec![0.0f32, 0.0, 1.0]),
        ];

        for (id, embedding) in &docs {
            let id_c = CString::new(*id).unwrap();
            let content_c = CString::new(format!("Content for {}", id)).unwrap();

            let ret = rag_add_document(
                handle,
                id_c.as_ptr(),
                content_c.as_ptr(),
                embedding.as_ptr(),
                embedding.len(),
            );
            assert_eq!(ret, 0);
        }

        // Search
        let query = [1.0f32, 0.0, 0.0];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;

        let ret = rag_search(
            handle,
            query.as_ptr(),
            query.len(),
            2,
            &mut results,
            &mut count,
        );

        assert_eq!(ret, 0);
        assert_eq!(count, 2);
        assert!(!results.is_null());

        unsafe {
            let results_slice = std::slice::from_raw_parts(results, count);

            // First result should be doc1 with high score
            let first = &results_slice[0];
            assert!(!first.id.is_null());
            let first_id = CStr::from_ptr(first.id).to_str().unwrap();
            assert_eq!(first_id, "doc1");
            assert!(first.score > 0.9);
        }

        rag_free_results(results, count);
        rag_destroy(handle);
    }

    #[test]
    fn test_search_empty_index() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let query = [1.0f32, 0.0, 0.0];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;

        let ret = rag_search(
            handle,
            query.as_ptr(),
            query.len(),
            5,
            &mut results,
            &mut count,
        );

        assert_eq!(ret, 0);
        assert_eq!(count, 0);

        rag_destroy(handle);
    }

    #[test]
    fn test_search_error_initializes_outputs() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let query = [1.0f32, 0.0]; // wrong dimension
        let mut results: *mut SearchResult = std::ptr::dangling_mut::<SearchResult>();
        let mut count: usize = usize::MAX;

        let ret = rag_search(
            handle,
            query.as_ptr(),
            query.len(),
            5,
            &mut results,
            &mut count,
        );

        assert_eq!(ret, -1);
        assert!(results.is_null());
        assert_eq!(count, 0);

        rag_destroy(handle);
    }

    /// Verify `rag_search` accepts `*const RagHandle` (shared/read-only access).
    #[test]
    fn test_search_accepts_const_handle() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        // Add a document via the mutable handle
        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test document").unwrap();
        let embedding = [1.0f32, 0.0, 0.0];

        let ret = rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );
        assert_eq!(ret, 0);

        // Cast to *const to prove search works through a shared pointer
        let const_handle: *const RagHandle = handle;

        let query = [1.0f32, 0.0, 0.0];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;

        let ret = rag_search(
            const_handle,
            query.as_ptr(),
            query.len(),
            1,
            &mut results,
            &mut count,
        );

        assert_eq!(ret, 0);
        assert_eq!(count, 1);
        assert!(!results.is_null());

        unsafe {
            let results_slice = std::slice::from_raw_parts(results, count);
            let first_id = CStr::from_ptr(results_slice[0].id).to_str().unwrap();
            assert_eq!(first_id, "doc1");
        }

        rag_free_results(results, count);
        rag_destroy(handle);
    }

    #[test]
    fn test_clear() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        // Add document
        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test").unwrap();
        let embedding = [0.1f32, 0.2, 0.3];

        rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );

        assert_eq!(rag_count(handle), 1);

        // Clear
        let ret = rag_clear(handle);
        assert_eq!(ret, 0);
        assert_eq!(rag_count(handle), 0);

        rag_destroy(handle);
    }

    #[test]
    fn test_null_safety() {
        // All functions should handle NULL gracefully
        assert!(rag_create(0, 0).is_null());

        rag_destroy(ptr::null_mut()); // Should not crash

        assert_eq!(rag_count(ptr::null()), 0);

        assert_eq!(rag_clear(ptr::null_mut()), -1);

        let query = [1.0f32];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;

        assert_eq!(
            rag_search(
                ptr::null(),
                query.as_ptr(),
                query.len(),
                5,
                &mut results,
                &mut count,
            ),
            -1
        );

        rag_free_results(ptr::null_mut(), 0); // Should not crash
    }

    #[test]
    fn test_free_results_ignores_incorrect_count() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test document").unwrap();
        let embedding = [1.0f32, 0.0, 0.0];
        assert_eq!(
            rag_add_document(
                handle,
                id.as_ptr(),
                content.as_ptr(),
                embedding.as_ptr(),
                embedding.len(),
            ),
            0
        );

        let query = [1.0f32, 0.0, 0.0];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;
        assert_eq!(
            rag_search(
                handle,
                query.as_ptr(),
                query.len(),
                1,
                &mut results,
                &mut count,
            ),
            0
        );
        assert_eq!(count, 1);
        assert!(!results.is_null());

        // Wrong count should not cause UB or crash.
        rag_free_results(results, count + 10);

        rag_destroy(handle);
    }

    #[test]
    fn test_persistence_flat() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let path_c = CString::new(path).unwrap();

        // Create and save
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test").unwrap();
        let embedding = [0.1f32, 0.2, 0.3];

        rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );

        let ret = rag_save(handle, path_c.as_ptr());
        assert_eq!(ret, 0);

        rag_destroy(handle);

        // Load
        let loaded = rag_load(path_c.as_ptr(), 0);
        assert!(!loaded.is_null());
        assert_eq!(rag_count(loaded), 1);

        rag_destroy(loaded);
    }

    #[test]
    fn test_persistence_hnsw() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let path_c = CString::new(path).unwrap();

        // Create and save
        let handle = rag_create(3, 1);
        assert!(!handle.is_null());

        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test").unwrap();
        let embedding = [0.1f32, 0.2, 0.3];

        rag_add_document(
            handle,
            id.as_ptr(),
            content.as_ptr(),
            embedding.as_ptr(),
            embedding.len(),
        );

        let ret = rag_save(handle, path_c.as_ptr());
        assert_eq!(ret, 0);

        rag_destroy(handle);

        // Load
        let loaded = rag_load(path_c.as_ptr(), 1);
        assert!(!loaded.is_null());
        assert_eq!(rag_count(loaded), 1);

        rag_destroy(loaded);
    }

    #[test]
    fn test_error_messages() {
        // Create with invalid dimension
        let handle = rag_create(0, 0);
        assert!(handle.is_null());

        let error = rag_last_error();
        assert!(!error.is_null());

        unsafe {
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert!(error_str.contains("dimension"));
        }
    }

    #[test]
    fn test_multiple_documents() {
        let handle = rag_create(128, 0);
        assert!(!handle.is_null());

        // Add 100 documents
        for i in 0..100 {
            let id = CString::new(format!("doc{}", i)).unwrap();
            let content = CString::new(format!("Content {}", i)).unwrap();
            let embedding = create_test_embedding(128);

            let ret = rag_add_document(
                handle,
                id.as_ptr(),
                content.as_ptr(),
                embedding.as_ptr(),
                embedding.len(),
            );
            assert_eq!(ret, 0);
        }

        assert_eq!(rag_count(handle), 100);

        rag_destroy(handle);
    }

    #[test]
    fn test_rag_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RagHandle>();
    }

    // ─── Issue #61: config surface ───────────────────────────────────────────────────────────

    fn default_config() -> FoxstashConfig {
        let mut cfg = std::mem::MaybeUninit::<FoxstashConfig>::uninit();
        assert_eq!(rag_default_config(cfg.as_mut_ptr()), 0);
        // SAFETY: rag_default_config writes every field on success.
        unsafe { cfg.assume_init() }
    }

    #[test]
    fn test_create_with_config_f32_defaults_work() {
        let cfg = default_config();
        assert_eq!(cfg.storage, FOXSTASH_STORAGE_F32);
        assert_eq!(cfg.metric, FOXSTASH_METRIC_COSINE);

        let handle = rag_create_with_config(3, &cfg);
        assert!(!handle.is_null());
        let id = CString::new("d0").unwrap();
        let content = CString::new("c").unwrap();
        let emb = [1.0f32, 0.0, 0.0];
        assert_eq!(
            rag_add_document(handle, id.as_ptr(), content.as_ptr(), emb.as_ptr(), 3),
            0,
            "f32 index should accept a document without training"
        );
        assert_eq!(rag_count(handle), 1);
        rag_destroy(handle);
    }

    #[test]
    fn test_create_with_config_quantized_requires_train_then_works() {
        // A quantized storage mode reaches the index via the config (bare rag_create cannot do
        // this — it only builds a default f32 index).
        let mut cfg = default_config();
        cfg.storage = FOXSTASH_STORAGE_SQ8;
        cfg.rerank_candidates = 20;

        let dim = 8usize;
        let handle = rag_create_with_config(dim, &cfg);
        assert!(!handle.is_null());

        let vector = |i: usize| -> Vec<f32> {
            let mut s = (i as u64).wrapping_mul(6_364_136_223_846_793_005) + 1;
            (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5
                })
                .collect()
        };
        let corpus: Vec<Vec<f32>> = (0..40).map(vector).collect();

        // Adding before training a quantized index must fail with a clear error, not silently.
        let id = CString::new("early").unwrap();
        let content = CString::new("c").unwrap();
        assert_eq!(
            rag_add_document(
                handle,
                id.as_ptr(),
                content.as_ptr(),
                corpus[0].as_ptr(),
                dim
            ),
            -1,
            "a quantized index should refuse add() before train()"
        );

        // Train the codebook from the sample, then the same add succeeds.
        let flat: Vec<f32> = corpus.iter().flatten().copied().collect();
        assert_eq!(
            rag_train(handle, flat.as_ptr(), corpus.len(), dim),
            0,
            "train should succeed on a valid sample"
        );
        for (i, v) in corpus.iter().enumerate() {
            let id = CString::new(format!("d{i}")).unwrap();
            assert_eq!(
                rag_add_document(handle, id.as_ptr(), content.as_ptr(), v.as_ptr(), dim),
                0,
                "add should succeed after train"
            );
        }
        assert_eq!(rag_count(handle), 40);

        // Self-retrieval: the configured SQ8 + rerank index returns the query's own id.
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;
        assert_eq!(
            rag_search(handle, corpus[7].as_ptr(), dim, 5, &mut results, &mut count),
            0
        );
        let hits: Vec<String> = unsafe {
            std::slice::from_raw_parts(results, count)
                .iter()
                .map(|r| CStr::from_ptr(r.id).to_str().unwrap().to_string())
                .collect()
        };
        assert!(
            hits.iter().any(|h| h == "d7"),
            "d7 should be in its own top-5, got {hits:?}"
        );
        rag_free_results(results, count);
        rag_destroy(handle);
    }

    #[test]
    fn test_create_with_config_rejects_unknown_codes() {
        let mut cfg = default_config();
        cfg.storage = 99;
        assert!(
            rag_create_with_config(3, &cfg).is_null(),
            "an unknown storage code must be rejected"
        );
        assert!(!rag_last_error().is_null());

        let mut cfg = default_config();
        cfg.metric = 99;
        assert!(
            rag_create_with_config(3, &cfg).is_null(),
            "an unknown metric code must be rejected"
        );
        assert!(!rag_last_error().is_null());
    }

    #[test]
    fn test_create_with_config_null_rejected() {
        assert!(rag_create_with_config(3, ptr::null()).is_null());
        assert!(rag_default_config(ptr::null_mut()) == -1);
    }
}
