//! Swift wrapper for Nexus RAG native library
//!
//! Provides idiomatic Swift API for iOS applications.

import Foundation

/// Swift wrapper for Nexus RAG
public class NexusRAG {
    private var handle: OpaquePointer?

    /// Create new RAG index
    ///
    /// - Parameters:
    ///   - embeddingDim: Dimension of embeddings (e.g., 384)
    ///   - useHNSW: Use HNSW index (approximate) vs Flat (exact)
    public init(embeddingDim: Int, useHNSW: Bool = true) throws {
        handle = rag_create(embeddingDim, useHNSW ? 1 : 0)
        if handle == nil {
            throw RAGError.initializationFailed
        }
    }

    deinit {
        if let handle = handle {
            rag_destroy(handle)
        }
    }

    /// Add document to index
    public func add(id: String, content: String, embedding: [Float]) throws {
        guard let handle = handle else {
            throw RAGError.invalidHandle
        }

        guard embedding.count > 0 else {
            throw RAGError.invalidEmbedding
        }

        let result = embedding.withUnsafeBufferPointer { embeddingPtr in
            id.withCString { idPtr in
                content.withCString { contentPtr in
                    rag_add_document(
                        handle,
                        idPtr,
                        contentPtr,
                        embeddingPtr.baseAddress,
                        embedding.count
                    )
                }
            }
        }

        if result != 0 {
            throw RAGError.addFailed(getLastError())
        }
    }

    /// Search result
    public struct SearchResult {
        public let id: String
        public let content: String
        public let score: Float
    }

    /// Search for nearest neighbors
    public func search(query: [Float], k: Int = 10) throws -> [SearchResult] {
        guard let handle = handle else {
            throw RAGError.invalidHandle
        }

        guard query.count > 0 else {
            throw RAGError.invalidQuery
        }

        var resultsPtr: UnsafeMutablePointer<CSearchResult>?
        var count: Int = 0

        let result = query.withUnsafeBufferPointer { queryPtr in
            rag_search(
                handle,
                queryPtr.baseAddress,
                query.count,
                k,
                &resultsPtr,
                &count
            )
        }

        guard result == 0, let cResults = resultsPtr else {
            throw RAGError.searchFailed(getLastError())
        }

        defer {
            rag_free_results(cResults, count)
        }

        var searchResults: [SearchResult] = []
        for i in 0..<count {
            let cResult = cResults[i]
            let id = String(cString: cResult.id)
            let content = String(cString: cResult.content)

            searchResults.append(SearchResult(
                id: id,
                content: content,
                score: cResult.score
            ))
        }

        return searchResults
    }

    /// Save index to file
    public func save(to path: String) throws {
        guard let handle = handle else {
            throw RAGError.invalidHandle
        }

        let result = path.withCString { pathPtr in
            rag_save(handle, pathPtr)
        }

        if result != 0 {
            throw RAGError.saveFailed(getLastError())
        }
    }

    /// Load index from file
    public static func load(from path: String, useHNSW: Bool = true) throws -> NexusRAG {
        let handle = path.withCString { pathPtr in
            rag_load(pathPtr, useHNSW ? 1 : 0)
        }

        guard let handle = handle else {
            throw RAGError.loadFailed(getLastError())
        }

        return NexusRAG(handle: handle)
    }

    /// Get document count
    public var count: Int {
        guard let handle = handle else { return 0 }
        return rag_count(handle)
    }

    /// Clear all documents
    public func clear() throws {
        guard let handle = handle else {
            throw RAGError.invalidHandle
        }

        let result = rag_clear(handle)
        if result != 0 {
            throw RAGError.clearFailed
        }
    }

    // Private
    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    private func getLastError() -> String {
        if let errorPtr = rag_last_error() {
            defer { rag_free_error(errorPtr) }
            return String(cString: errorPtr)
        }
        return "Unknown error"
    }
}

/// RAG errors
public enum RAGError: LocalizedError {
    case initializationFailed
    case invalidHandle
    case invalidEmbedding
    case invalidQuery
    case addFailed(String)
    case searchFailed(String)
    case saveFailed(String)
    case loadFailed(String)
    case clearFailed

    public var errorDescription: String? {
        switch self {
        case .initializationFailed:
            return "Failed to initialize RAG index"
        case .invalidHandle:
            return "Invalid RAG handle"
        case .invalidEmbedding:
            return "Invalid embedding array"
        case .invalidQuery:
            return "Invalid query array"
        case .addFailed(let msg):
            return "Failed to add document: \(msg)"
        case .searchFailed(let msg):
            return "Search failed: \(msg)"
        case .saveFailed(let msg):
            return "Failed to save: \(msg)"
        case .loadFailed(let msg):
            return "Failed to load: \(msg)"
        case .clearFailed:
            return "Failed to clear index"
        }
    }
}

// C FFI declarations
private typealias CSearchResult = (
    id: UnsafeMutablePointer<CChar>,
    content: UnsafeMutablePointer<CChar>,
    score: Float
)

@_silgen_name("rag_create")
private func rag_create(_ embeddingDim: Int, _ useHNSW: Int32) -> OpaquePointer?

@_silgen_name("rag_destroy")
private func rag_destroy(_ handle: OpaquePointer)

@_silgen_name("rag_add_document")
private func rag_add_document(
    _ handle: OpaquePointer,
    _ id: UnsafePointer<CChar>,
    _ content: UnsafePointer<CChar>,
    _ embedding: UnsafePointer<Float>?,
    _ embeddingLen: Int
) -> Int32

@_silgen_name("rag_search")
private func rag_search(
    _ handle: OpaquePointer,
    _ query: UnsafePointer<Float>?,
    _ queryLen: Int,
    _ k: Int,
    _ resultsOut: UnsafeMutablePointer<UnsafeMutablePointer<CSearchResult>?>,
    _ countOut: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("rag_free_results")
private func rag_free_results(
    _ results: UnsafeMutablePointer<CSearchResult>,
    _ count: Int
)

@_silgen_name("rag_save")
private func rag_save(_ handle: OpaquePointer, _ path: UnsafePointer<CChar>) -> Int32

@_silgen_name("rag_load")
private func rag_load(_ path: UnsafePointer<CChar>, _ useHNSW: Int32) -> OpaquePointer?

@_silgen_name("rag_count")
private func rag_count(_ handle: OpaquePointer) -> Int

@_silgen_name("rag_clear")
private func rag_clear(_ handle: OpaquePointer) -> Int32

@_silgen_name("rag_last_error")
private func rag_last_error() -> UnsafePointer<CChar>?

@_silgen_name("rag_free_error")
private func rag_free_error(_ error: UnsafePointer<CChar>)
