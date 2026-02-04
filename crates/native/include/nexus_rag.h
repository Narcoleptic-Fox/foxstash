/**
 * Nexus RAG Native FFI - C/C++ Header
 *
 * Cross-platform native bindings for the Nexus Local RAG system.
 * Provides high-performance vector search with HNSW and Flat indices.
 *
 * Supported Platforms: iOS, Android, Linux, macOS, Windows
 *
 * Example Usage:
 *
 * ```c
 * #include "nexus_rag.h"
 *
 * // Create index
 * RagHandle* rag = rag_create(384, 1);  // 384-dim, HNSW
 * if (!rag) {
 *     fprintf(stderr, "Error: %s\n", rag_last_error());
 *     return 1;
 * }
 *
 * // Add document
 * float embedding[384];
 * for (int i = 0; i < 384; i++) {
 *     embedding[i] = 0.1f * i;
 * }
 * rag_add_document(rag, "doc1", "Hello world", embedding, 384);
 *
 * // Search
 * SearchResult* results;
 * size_t count;
 * if (rag_search(rag, embedding, 384, 5, &results, &count) == 0) {
 *     for (size_t i = 0; i < count; i++) {
 *         printf("%s: %.2f\n", results[i].id, results[i].score);
 *     }
 *     rag_free_results(results, count);
 * }
 *
 * // Cleanup
 * rag_destroy(rag);
 * ```
 */

#ifndef NEXUS_RAG_H
#define NEXUS_RAG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * Type Definitions
 * ============================================================================= */

/**
 * Opaque handle to a RAG index.
 *
 * Created by rag_create() or rag_load().
 * Must be freed with rag_destroy().
 */
typedef struct RagHandle RagHandle;

/**
 * Search result from rag_search().
 *
 * Contains document ID, content, and similarity score.
 * All string fields are owned by the result and freed by rag_free_results().
 */
typedef struct {
    char *id;       /**< Document ID (null-terminated) */
    char *content;  /**< Document content (null-terminated) */
    float score;    /**< Similarity score (cosine similarity, range: -1.0 to 1.0) */
} SearchResult;

/* =============================================================================
 * Index Creation and Destruction
 * ============================================================================= */

/**
 * Create a new RAG index.
 *
 * Creates either a Flat index (exact search) or HNSW index (approximate search)
 * with the specified embedding dimensionality.
 *
 * @param embedding_dim Dimensionality of embeddings (e.g., 384 for MiniLM)
 *                      Must be > 0 and <= 100000
 * @param use_hnsw 0 for Flat index (exact), 1 for HNSW (approximate)
 *
 * @return Pointer to RagHandle on success, NULL on failure
 *         On failure, call rag_last_error() for error details
 *
 * @note Must call rag_destroy() when done to free memory
 * @note Thread-safe: can be called from multiple threads
 *
 * @see rag_destroy
 * @see rag_last_error
 */
RagHandle* rag_create(size_t embedding_dim, int32_t use_hnsw);

/**
 * Destroy a RAG index and free all associated memory.
 *
 * Frees all memory associated with the index, including documents and
 * internal data structures.
 *
 * @param handle RAG handle to destroy (from rag_create() or rag_load())
 *               Can be NULL (no-op)
 *
 * @note After calling this function, the handle is invalid
 * @note Safe to call with NULL handle
 * @note Thread-safe if handle is not shared across threads
 *
 * @see rag_create
 */
void rag_destroy(RagHandle* handle);

/* =============================================================================
 * Document Operations
 * ============================================================================= */

/**
 * Add a document to the index.
 *
 * Adds a document with its embedding vector to the index. Documents with
 * duplicate IDs will replace existing documents.
 *
 * @param handle RAG handle (must not be NULL)
 * @param id Unique document ID (null-terminated C string, must not be NULL)
 * @param content Document content (null-terminated C string, must not be NULL)
 * @param embedding Embedding vector (array of floats, must not be NULL)
 * @param embedding_len Length of embedding vector (must match index dimension)
 *
 * @return 0 on success, -1 on error
 *         On error, call rag_last_error() for details
 *
 * @note Embedding dimension must match the dimension specified in rag_create()
 * @note Thread-safe: requires external synchronization if handle is shared
 *
 * @see rag_create
 * @see rag_search
 */
int32_t rag_add_document(
    RagHandle* handle,
    const char* id,
    const char* content,
    const float* embedding,
    size_t embedding_len
);

/* =============================================================================
 * Search Operations
 * ============================================================================= */

/**
 * Search for k nearest neighbors.
 *
 * Finds the k most similar documents to the query embedding using cosine
 * similarity. Results are sorted by descending score (most similar first).
 *
 * @param handle RAG handle (must not be NULL)
 * @param query Query embedding vector (array of floats, must not be NULL)
 * @param query_len Length of query vector (must match index dimension)
 * @param k Number of results to return (may return fewer if index has < k documents)
 * @param results_out Output pointer for results array (must not be NULL)
 *                    Will be set to allocated array or NULL if no results
 * @param count_out Output pointer for result count (must not be NULL)
 *                  Will be set to number of results (0 if empty index)
 *
 * @return 0 on success, -1 on error
 *         On error, call rag_last_error() for details
 *
 * @note Results must be freed with rag_free_results() when done
 * @note Query dimension must match the dimension specified in rag_create()
 * @note Thread-safe: multiple threads can search simultaneously
 *
 * @see rag_free_results
 * @see rag_add_document
 */
int32_t rag_search(
    RagHandle* handle,
    const float* query,
    size_t query_len,
    size_t k,
    SearchResult** results_out,
    size_t* count_out
);

/**
 * Free search results allocated by rag_search().
 *
 * Frees all memory associated with search results, including document IDs
 * and content strings.
 *
 * @param results Results array from rag_search() (can be NULL)
 * @param count Number of results in array
 *
 * @note Must be called exactly once per results array
 * @note Safe to call with NULL results
 * @note After calling this, all pointers in results are invalid
 *
 * @see rag_search
 */
void rag_free_results(SearchResult* results, size_t count);

/* =============================================================================
 * Persistence Operations
 * ============================================================================= */

/**
 * Save index to disk.
 *
 * Saves the index to the specified directory path with compression.
 * Creates the directory if it doesn't exist.
 *
 * @param handle RAG handle (must not be NULL)
 * @param path Directory path for storage (null-terminated C string, must not be NULL)
 *             Example: "/var/data/rag_index"
 *
 * @return 0 on success, -1 on error
 *         On error, call rag_last_error() for details
 *
 * @note Directory will be created if it doesn't exist
 * @note Existing index at path will be overwritten
 * @note Thread-safe: requires external synchronization if handle is shared
 *
 * @see rag_load
 */
int32_t rag_save(RagHandle* handle, const char* path);

/**
 * Load index from disk.
 *
 * Loads a previously saved index from the specified directory path.
 *
 * @param path Directory path for storage (null-terminated C string, must not be NULL)
 *             Must be a directory created by rag_save()
 * @param use_hnsw 0 for Flat index, 1 for HNSW index
 *                 Must match the index type that was saved
 *
 * @return Pointer to RagHandle on success, NULL on failure
 *         On failure, call rag_last_error() for error details
 *
 * @note Must call rag_destroy() when done to free memory
 * @note Index type (use_hnsw) must match the saved index
 * @note Thread-safe: can be called from multiple threads
 *
 * @see rag_save
 * @see rag_destroy
 */
RagHandle* rag_load(const char* path, int32_t use_hnsw);

/* =============================================================================
 * Utility Operations
 * ============================================================================= */

/**
 * Get the number of documents in the index.
 *
 * Returns the count of documents currently stored in the index.
 *
 * @param handle RAG handle (can be NULL, returns 0)
 *
 * @return Number of documents in index, 0 if handle is NULL
 *
 * @note Thread-safe: can be called while other operations are in progress
 *
 * @see rag_add_document
 * @see rag_clear
 */
size_t rag_count(const RagHandle* handle);

/**
 * Clear all documents from the index.
 *
 * Removes all documents from the index, resetting it to empty state.
 * Index configuration and dimension are preserved.
 *
 * @param handle RAG handle (must not be NULL)
 *
 * @return 0 on success, -1 on error
 *         On error, call rag_last_error() for details
 *
 * @note Does not free memory allocated for the index itself
 * @note Thread-safe: requires external synchronization if handle is shared
 *
 * @see rag_count
 */
int32_t rag_clear(RagHandle* handle);

/* =============================================================================
 * Error Handling
 * ============================================================================= */

/**
 * Get the last error message for this thread.
 *
 * Returns a pointer to the error message for the last failed operation
 * on this thread. Error messages are thread-local.
 *
 * @return Pointer to error message string (null-terminated), or NULL if no error
 *
 * @note Return value is valid until next FFI call on this thread
 * @note Do not free the returned pointer
 * @note Thread-safe: each thread has its own error message
 * @note Error is cleared on next successful operation
 *
 * @see rag_free_error
 */
const char* rag_last_error(void);

/**
 * Free an error string returned by rag_last_error().
 *
 * @param error Error string to free
 *
 * @deprecated This function is a no-op. Error strings are managed internally.
 * @note Safe to call but does nothing
 * @note Error strings are automatically freed on next FFI call
 */
void rag_free_error(const char* error);

/* =============================================================================
 * Version Information
 * ============================================================================= */

/**
 * Library version: 1.0.0
 *
 * API is stable for 1.x releases.
 */
#define NEXUS_RAG_VERSION_MAJOR 1
#define NEXUS_RAG_VERSION_MINOR 0
#define NEXUS_RAG_VERSION_PATCH 0

/* =============================================================================
 * Constants
 * ============================================================================= */

/**
 * Maximum supported embedding dimension
 */
#define NEXUS_RAG_MAX_DIMENSION 100000

/**
 * Index types
 */
#define NEXUS_RAG_INDEX_FLAT 0  /**< Flat (exact) index */
#define NEXUS_RAG_INDEX_HNSW 1  /**< HNSW (approximate) index */

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_RAG_H */
