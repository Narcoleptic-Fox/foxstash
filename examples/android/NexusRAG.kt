package com.nexus.rag

/**
 * Kotlin wrapper for Nexus RAG native library
 *
 * Provides idiomatic Kotlin API for Android applications with automatic memory management.
 */
class NexusRAG(
    embeddingDim: Int,
    useHNSW: Boolean = true
) : AutoCloseable {

    private var handle: Long = 0

    init {
        System.loadLibrary("nexus_rag_native")
        handle = create(embeddingDim, if (useHNSW) 1 else 0)
        if (handle == 0L) {
            throw RAGException("Failed to create RAG index")
        }
    }

    /**
     * Search result containing document ID, content, and similarity score
     */
    data class SearchResult(
        val id: String,
        val content: String,
        val score: Float
    )

    /**
     * Add document to the index
     *
     * @param id Unique document identifier
     * @param content Document text content
     * @param embedding Document embedding vector
     * @throws RAGException if adding document fails
     */
    fun add(id: String, content: String, embedding: FloatArray) {
        if (handle == 0L) throw RAGException("Invalid handle")

        val result = addDocument(handle, id, content, embedding)
        if (result != 0) {
            throw RAGException("Failed to add document: ${getLastError()}")
        }
    }

    /**
     * Search for nearest neighbors
     *
     * @param query Query embedding vector
     * @param k Number of results to return
     * @return List of search results ordered by similarity (highest first)
     * @throws RAGException if search fails
     */
    fun search(query: FloatArray, k: Int = 10): List<SearchResult> {
        if (handle == 0L) throw RAGException("Invalid handle")

        return searchNative(handle, query, k)
            ?: throw RAGException("Search failed: ${getLastError()}")
    }

    /**
     * Save index to file
     *
     * @param path File path to save the index
     * @throws RAGException if saving fails
     */
    fun save(path: String) {
        if (handle == 0L) throw RAGException("Invalid handle")

        val result = saveNative(handle, path)
        if (result != 0) {
            throw RAGException("Failed to save: ${getLastError()}")
        }
    }

    /**
     * Get number of documents in index
     *
     * @return Document count
     */
    fun getCount(): Int {
        if (handle == 0L) return 0
        return countNative(handle)
    }

    /**
     * Clear all documents from index
     *
     * @throws RAGException if clearing fails
     */
    fun clear() {
        if (handle == 0L) throw RAGException("Invalid handle")

        val result = clearNative(handle)
        if (result != 0) {
            throw RAGException("Failed to clear")
        }
    }

    /**
     * Close and release native resources
     *
     * This is called automatically when using `use { }` block
     */
    override fun close() {
        if (handle != 0L) {
            destroy(handle)
            handle = 0
        }
    }

    companion object {
        /**
         * Load index from file
         *
         * @param path File path to load the index from
         * @param useHNSW Use HNSW index (approximate) vs Flat (exact)
         * @return Loaded NexusRAG instance
         * @throws RAGException if loading fails
         */
        fun load(path: String, useHNSW: Boolean = true): NexusRAG {
            val handle = loadNative(path, if (useHNSW) 1 else 0)
            if (handle == 0L) {
                throw RAGException("Failed to load: ${getLastError()}")
            }
            return NexusRAG(handle)
        }

        // JNI native method declarations
        @JvmStatic
        private external fun create(embeddingDim: Int, useHNSW: Int): Long

        @JvmStatic
        private external fun destroy(handle: Long)

        @JvmStatic
        private external fun addDocument(
            handle: Long,
            id: String,
            content: String,
            embedding: FloatArray
        ): Int

        @JvmStatic
        private external fun searchNative(
            handle: Long,
            query: FloatArray,
            k: Int
        ): List<SearchResult>?

        @JvmStatic
        private external fun saveNative(handle: Long, path: String): Int

        @JvmStatic
        private external fun loadNative(path: String, useHNSW: Int): Long

        @JvmStatic
        private external fun countNative(handle: Long): Int

        @JvmStatic
        private external fun clearNative(handle: Long): Int

        @JvmStatic
        private external fun getLastError(): String
    }

    /**
     * Private constructor for loading from file
     */
    private constructor(handle: Long) : this(0, true) {
        this.handle = handle
    }
}

/**
 * Exception thrown by RAG operations
 */
class RAGException(message: String) : Exception(message)
