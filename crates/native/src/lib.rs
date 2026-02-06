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
//! #include "nexus_rag.h"
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
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;

// =============================================================================
// Thread-local Error Storage
// =============================================================================

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
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
trait RagIndex: Send {
    fn add(&mut self, document: Document) -> Result<(), String>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<CoreSearchResult>, String>;
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn save(&self, storage: &FileStorage, name: &str) -> Result<(), String>;
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
}

/// Opaque handle to RAG index
pub struct RagHandle {
    index: Box<dyn RagIndex>,
    storage: Option<FileStorage>,
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
            index,
            storage: None,
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

/// Destroy a RAG index and free all associated memory.
///
/// # Arguments
/// * `handle` - RAG handle to destroy
///
/// # Safety
/// After calling this function, the handle is invalid and must not be used.
/// Passing NULL is safe (no-op).
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
            let handle = &mut *handle;

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
            match handle.index.add(document) {
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
    handle: *mut RagHandle,
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
            let results = match handle.index.search(query_vec, k) {
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
/// * `count` - Number of results in array
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
pub extern "C" fn rag_free_results(results: *mut SearchResult, count: usize) {
    if results.is_null() || count == 0 {
        return;
    }

    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            // Reconstruct the boxed slice
            let results_slice = Vec::from_raw_parts(results, count, count);

            // Free each result's strings
            for result in results_slice {
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
            let handle = &mut *handle;

            // Convert path to Rust string
            let path_str = match CStr::from_ptr(path).to_str() {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("Invalid path string: {}", e));
                    return -1;
                }
            };

            // Create or get storage
            let storage = match &handle.storage {
                Some(s) => s,
                None => {
                    // Create new storage
                    match FileStorage::new(path_str) {
                        Ok(s) => {
                            handle.storage = Some(s);
                            handle.storage.as_ref().unwrap()
                        }
                        Err(e) => {
                            set_last_error(format!("Failed to create storage: {}", e));
                            return -1;
                        }
                    }
                }
            };

            // Save index
            match handle.index.save(storage, "index") {
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
                                index: Box::new(index),
                                storage: Some(storage),
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
                                index: Box::new(index),
                                storage: Some(storage),
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
        handle.index.len()
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
        let handle = &mut *handle;
        handle.index.clear();
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
        let embedding = vec![0.1f32, 0.2, 0.3];

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
        let embedding = vec![0.1f32, 0.2]; // Wrong dimension

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
        let query = vec![1.0f32, 0.0, 0.0];
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

        let query = vec![1.0f32, 0.0, 0.0];
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
    fn test_clear() {
        let handle = rag_create(3, 0);
        assert!(!handle.is_null());

        // Add document
        let id = CString::new("doc1").unwrap();
        let content = CString::new("Test").unwrap();
        let embedding = vec![0.1f32, 0.2, 0.3];

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

        let query = vec![1.0f32];
        let mut results: *mut SearchResult = ptr::null_mut();
        let mut count: usize = 0;

        assert_eq!(
            rag_search(
                ptr::null_mut(),
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
        let embedding = vec![0.1f32, 0.2, 0.3];

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
        let embedding = vec![0.1f32, 0.2, 0.3];

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
}
