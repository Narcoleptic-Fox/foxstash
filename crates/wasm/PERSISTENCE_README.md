# IndexedDB Persistence Implementation

This document describes the IndexedDB persistence implementation for the Nexus Local RAG WASM system.

## Overview

The persistence module (`src/persistence.rs`) provides IndexedDB storage capabilities for saving and loading RAG indices in browser environments. It enables:

- **Persistent Storage**: Save indices to IndexedDB for long-term storage
- **Async Operations**: Promise-based API for all database operations
- **Serialization**: JSON-serializable format for both Flat and HNSW indices
- **Management**: List, delete, and clear stored indices

## Architecture

### Core Components

#### 1. IndexedDBStore

The main interface for database operations:

```javascript
const store = new IndexedDBStore();  // Default DB name
// or
const store = new IndexedDBStore("my-custom-db");
```

**Methods:**
- `save(key, data)` - Save an index
- `load(key)` - Load an index
- `delete(key)` - Delete an index
- `list_keys()` - List all saved indices
- `clear()` - Clear all data

#### 2. Serialization Structures

**SerializedFlatIndex:**
```rust
{
    embedding_dim: usize,
    documents: Vec<Document>,
}
```

**SerializedHNSWIndex:**
```rust
{
    embedding_dim: usize,
    config: {
        m, m0, ef_construction, ef_search, ml
    },
    nodes: Vec<{
        id, content, embedding, metadata,
        connections: Vec<Vec<usize>>
    }>,
    entry_point: Option<usize>,
    max_layer: usize,
}
```

**PersistedIndex:**
```rust
{
    metadata: {
        name: String,
        created_at: f64,
        updated_at: f64,
        document_count: usize,
        index_type: "flat" | "hnsw",
    },
    index: SerializedFlatIndex | SerializedHNSWIndex,
}
```

### Database Schema

- **Database Name**: `nexus-rag-db` (configurable)
- **Object Store**: `indices`
- **Storage Type**: Key-value pairs (key = string, value = serialized index)
- **Version**: 1

## Integration with LocalRAG

To integrate the persistence module with the LocalRAG interface, add the following methods to `lib.rs`:

### 1. Export the Module

```rust
pub mod persistence;
```

### 2. Add Serialization Methods

```rust
#[wasm_bindgen]
impl LocalRAG {
    /// Serialize to JsValue
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        // Extract documents and structure from index
        let serialized = match &self.index {
            IndexType::Flat(flat_index) => {
                let documents = self.extract_all_documents()?;
                persistence::create_flat_index(self.embedding_dim, documents)
            }
            IndexType::HNSW(hnsw_index) => {
                self.serialize_hnsw(hnsw_index)?
            }
        };

        let persisted = persistence::create_persisted_index(
            "default".to_string(),
            serialized,
            self.document_count(),
        );

        persistence::to_js_value(&persisted)
    }

    /// Deserialize from JsValue
    #[wasm_bindgen]
    pub fn from_json(data: JsValue) -> Result<LocalRAG, JsValue> {
        let persisted: persistence::PersistedIndex =
            persistence::from_js_value(data)?;

        match persisted.index {
            persistence::SerializedIndex::Flat(flat_data) => {
                let mut flat_index = FlatIndex::new(flat_data.embedding_dim);
                for doc in flat_data.documents {
                    flat_index.add(doc)?;
                }
                Ok(LocalRAG {
                    index: IndexType::Flat(flat_index),
                    embedding_dim: flat_data.embedding_dim,
                })
            }
            persistence::SerializedIndex::Hnsw(hnsw_data) => {
                Self::deserialize_hnsw(hnsw_data)
            }
        }
    }

    /// Save to IndexedDB
    #[wasm_bindgen]
    pub async fn save_to_db(
        &self,
        store: &persistence::IndexedDBStore,
        key: &str,
    ) -> Result<(), JsValue> {
        let data = self.to_json()?;
        store.save(key, data).await
    }

    /// Load from IndexedDB
    #[wasm_bindgen]
    pub async fn load_from_db(
        store: &persistence::IndexedDBStore,
        key: &str,
    ) -> Result<LocalRAG, JsValue> {
        let data = store.load(key).await?;
        Self::from_json(data)
    }
}
```

### 3. Helper Methods Required

To fully support serialization, the core library needs these additions:

#### Option A: Add getter methods to FlatIndex
```rust
impl FlatIndex {
    pub fn get_all_documents(&self) -> Vec<Document> {
        self.documents.values().cloned().collect()
    }
}
```

#### Option B: Add Serialize/Deserialize to indices
```rust
#[derive(Serialize, Deserialize)]
pub struct FlatIndex {
    // ... existing fields
}
```

#### Option C: Add serialization methods to HNSWIndex
```rust
impl HNSWIndex {
    pub fn to_serializable(&self) -> SerializedHNSWIndex { ... }
    pub fn from_serializable(data: SerializedHNSWIndex) -> Self { ... }
}
```

## JavaScript Usage Examples

### Basic Save and Load

```javascript
import init, { LocalRAG, JsDocument, IndexedDBStore } from './nexus_rag_wasm.js';

await init();

// Create and populate an index
const rag = new LocalRAG(384, false);

const doc = new JsDocument(
  "doc1",
  "Hello world",
  new Float32Array(384),
  { source: "test" }
);

rag.add_document(doc);

// Save to IndexedDB
const store = new IndexedDBStore();
await rag.save_to_db(store, "my-index");

console.log("Index saved!");

// Later, load it back
const loadedRag = await LocalRAG.load_from_db(store, "my-index");
console.log(`Loaded ${loadedRag.document_count()} documents`);
```

### Manage Multiple Indices

```javascript
const store = new IndexedDBStore();

// Save multiple indices
await rag1.save_to_db(store, "general-knowledge");
await rag2.save_to_db(store, "code-snippets");
await rag3.save_to_db(store, "documentation");

// List all saved indices
const keys = await store.list_keys();
console.log("Available indices:", keys);
// Output: ["general-knowledge", "code-snippets", "documentation"]

// Load specific index
const codeRag = await LocalRAG.load_from_db(store, "code-snippets");

// Delete an index
await store.delete("documentation");

// Clear all
await store.clear();
```

### Error Handling

```javascript
const store = new IndexedDBStore();

try {
  await rag.save_to_db(store, "my-index");
  console.log("Saved successfully");
} catch (error) {
  if (error.message.includes("quota")) {
    console.error("Storage quota exceeded!");
  } else {
    console.error("Save failed:", error);
  }
}

try {
  const loaded = await LocalRAG.load_from_db(store, "nonexistent");
} catch (error) {
  console.error("Index not found");
}
```

### Using JSON Serialization (Alternative)

```javascript
// Serialize to JSON
const jsonData = rag.to_json();

// Store in localStorage (simpler but size-limited)
localStorage.setItem('rag-index', JSON.stringify(jsonData));

// Load back
const data = JSON.parse(localStorage.getItem('rag-index'));
const restored = LocalRAG.from_json(data);
```

## Implementation Status

### ✅ Completed

1. **IndexedDBStore Implementation**
   - Async database operations
   - CRUD operations (Create, Read, Update, Delete)
   - List and clear functionality
   - Error handling
   - Transaction management

2. **Serialization Structures**
   - SerializedFlatIndex
   - SerializedHNSWIndex
   - PersistedIndex with metadata
   - Helper functions for serialization/deserialization

3. **Documentation**
   - Comprehensive inline documentation
   - Usage examples
   - Integration guide

4. **Testing**
   - Unit tests for serialization
   - WASM tests for IndexedDB operations

### ⚠️ Current Limitations

1. **Core Library Access**
   - FlatIndex's internal HashMap is private
   - HNSWIndex's graph structure is not accessible
   - Need to add getter methods or Serialize derive

2. **HNSW Serialization**
   - Graph structure serialization defined but not integrated
   - Requires core library support for reconstruction

3. **Document Extraction**
   - Cannot extract full documents with embeddings from indices
   - SearchResult doesn't include embeddings

### 🔄 Recommended Core Library Updates

To fully enable persistence, add to `/home/user/nexus/nexus-local-rag/crates/core/src/index/flat.rs`:

```rust
impl FlatIndex {
    /// Get all documents in the index
    pub fn get_all_documents(&self) -> Vec<Document> {
        self.documents.values().cloned().collect()
    }

    /// Get document count
    pub fn len(&self) -> usize {
        self.documents.len()
    }
}
```

For HNSW, add to `/home/user/nexus/nexus-local-rag/crates/core/src/index/hnsw.rs`:

```rust
impl HNSWIndex {
    /// Get read-only access to nodes for serialization
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Get configuration
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    /// Get entry point
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Get max layer
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }
}
```

## Performance Considerations

### Storage Size

- **Flat Index**: ~(embedding_dim * 4 bytes + content size) * num_docs
- **HNSW Index**: Flat size + (m * 8 bytes * avg_layers) * num_docs
- Example: 1000 docs with 384-dim embeddings ≈ 1.5-3 MB

### IndexedDB Limits

- **Chrome**: ~60% of available disk space
- **Firefox**: ~50% of available disk space
- **Safari**: 1 GB (prompts user for more)
- **Typical practical limit**: 50-500 MB

### Optimization Tips

1. **Compression**: Add gzip/brotli before storing large indices
2. **Chunking**: Split large indices into multiple keys
3. **Lazy Loading**: Load documents on-demand
4. **Metadata Only**: Store just document IDs and load content separately

## Browser Compatibility

### IndexedDB Support

- ✅ Chrome 24+
- ✅ Firefox 16+
- ✅ Safari 10+
- ✅ Edge 12+
- ⚠️ Not supported in private/incognito mode in some browsers

### Feature Detection

```javascript
if (!window.indexedDB) {
  console.error("IndexedDB not supported");
  // Fallback to in-memory only
}
```

## Security Considerations

1. **Data Privacy**: IndexedDB data is origin-scoped but not encrypted
2. **Cross-Origin**: Cannot access data from different origins
3. **Sensitive Data**: Consider encryption for sensitive content
4. **Size Limits**: Handle quota exceeded errors gracefully

## Future Enhancements

1. **Compression**: Add automatic compression for large indices
2. **Versioning**: Support multiple schema versions
3. **Migration**: Tools for upgrading stored indices
4. **Encryption**: Optional encryption for sensitive data
5. **Sync**: Cloud sync capabilities
6. **Incremental Updates**: Save only changed documents
7. **Export/Import**: JSON export for backup/transfer

## Files Created

1. `/home/user/nexus/nexus-local-rag/crates/wasm/src/persistence.rs` - Main implementation
2. `/home/user/nexus/nexus-local-rag/crates/wasm/Cargo.toml` - Updated with IndexedDB dependencies
3. `/home/user/nexus/nexus-local-rag/crates/wasm/PERSISTENCE_README.md` - This documentation

## Next Steps

1. **Core Library Updates**: Add document extraction methods to FlatIndex and HNSWIndex
2. **LocalRAG Integration**: Add to_json/from_json/save_to_db/load_from_db methods
3. **Testing**: Add integration tests with LocalRAG
4. **Examples**: Create example application demonstrating persistence
5. **Documentation**: Update main README with persistence examples

## Questions or Issues?

If you encounter issues or have questions about the persistence implementation, please refer to:

- The inline documentation in `persistence.rs`
- The test cases in `persistence.rs`
- This README
- The main project documentation
