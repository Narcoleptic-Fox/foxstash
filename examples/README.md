# Platform Integration Examples

Examples for integrating Nexus RAG into iOS, Android, and Desktop applications.

## iOS (Swift)

### Setup

**1. Add Framework:**
Drag `NexusRAG.xcframework` into your Xcode project.

**2. Add Swift wrapper:**
Include `NexusRAG.swift` in your project.

**3. Configure Xcode:**
- In your target's "General" settings, add the framework to "Frameworks, Libraries, and Embedded Content"
- Set "Embed" to "Embed & Sign"
- Ensure minimum iOS version is 13.0 or higher

### Usage

```swift
import NexusRAG

// Create RAG index
let rag = try NexusRAG(embeddingDim: 384, useHNSW: true)

// Add documents
let embedding = [Float](repeating: 0.1, count: 384)
try rag.add(
    id: "doc1",
    content: "Example document",
    embedding: embedding
)

// Search
let results = try rag.search(query: embedding, k: 5)
for result in results {
    print("\(result.id): \(result.score)")
    print("Content: \(result.content)")
}

// Save/load
try rag.save(to: "/path/to/index.bin")
let loaded = try NexusRAG.load(from: "/path/to/index.bin")

// Get count
print("Documents: \(rag.count)")

// Clear
try rag.clear()
```

### Error Handling

```swift
do {
    let rag = try NexusRAG(embeddingDim: 384)
    try rag.add(id: "doc1", content: "text", embedding: embedding)
} catch RAGError.initializationFailed {
    print("Failed to initialize RAG")
} catch RAGError.addFailed(let message) {
    print("Add failed: \(message)")
} catch {
    print("Unexpected error: \(error)")
}
```

### SwiftUI Example

```swift
import SwiftUI
import NexusRAG

class RAGViewModel: ObservableObject {
    @Published var results: [NexusRAG.SearchResult] = []
    private var rag: NexusRAG?

    init() {
        do {
            rag = try NexusRAG(embeddingDim: 384)
        } catch {
            print("Failed to initialize: \(error)")
        }
    }

    func addDocument(id: String, content: String, embedding: [Float]) {
        do {
            try rag?.add(id: id, content: content, embedding: embedding)
        } catch {
            print("Add failed: \(error)")
        }
    }

    func search(query: [Float]) {
        do {
            results = try rag?.search(query: query, k: 10) ?? []
        } catch {
            print("Search failed: \(error)")
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = RAGViewModel()

    var body: some View {
        List(viewModel.results, id: \.id) { result in
            VStack(alignment: .leading) {
                Text(result.id).font(.headline)
                Text(result.content).font(.body)
                Text("Score: \(result.score)").font(.caption)
            }
        }
    }
}
```

## Android (Kotlin)

### Setup

**1. Add JNI libraries:**
Copy native libraries to `app/src/main/jniLibs/`:
```
jniLibs/
├── arm64-v8a/
│   └── libnexus_rag_native.so
├── armeabi-v7a/
│   └── libnexus_rag_native.so
├── x86/
│   └── libnexus_rag_native.so
└── x86_64/
    └── libnexus_rag_native.so
```

**2. Add Kotlin wrapper:**
Include `NexusRAG.kt` in your project (e.g., `app/src/main/java/com/nexus/rag/NexusRAG.kt`).

**3. Configure Gradle:**
```gradle
android {
    defaultConfig {
        minSdk 21
        targetSdk 34
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64'
        }
    }
}
```

### Usage

```kotlin
import com.nexus.rag.NexusRAG

// Create RAG index
val rag = NexusRAG(embeddingDim = 384, useHNSW = true)

// Add documents
val embedding = FloatArray(384) { 0.1f }
rag.add(
    id = "doc1",
    content = "Example document",
    embedding = embedding
)

// Search
val results = rag.search(query = embedding, k = 5)
for (result in results) {
    println("${result.id}: ${result.score}")
    println("Content: ${result.content}")
}

// Save/load
rag.save("/path/to/index.bin")
val loaded = NexusRAG.load("/path/to/index.bin")

// Get count
println("Documents: ${rag.getCount()}")

// Clear
rag.clear()

// Clean up
rag.close()
```

### Error Handling

```kotlin
try {
    val rag = NexusRAG(embeddingDim = 384)
    rag.add("doc1", "text", embedding)
} catch (e: RAGException) {
    Log.e("RAG", "Operation failed: ${e.message}")
}
```

### Using with AutoCloseable

```kotlin
NexusRAG(embeddingDim = 384).use { rag ->
    rag.add("doc1", "content", embedding)
    val results = rag.search(embedding, k = 10)
    // Automatically closed after block
}
```

### Android ViewModel Example

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RAGViewModel : ViewModel() {
    private val _results = MutableStateFlow<List<NexusRAG.SearchResult>>(emptyList())
    val results: StateFlow<List<NexusRAG.SearchResult>> = _results

    private var rag: NexusRAG? = null

    init {
        try {
            rag = NexusRAG(embeddingDim = 384)
        } catch (e: RAGException) {
            Log.e("RAG", "Failed to initialize: ${e.message}")
        }
    }

    fun addDocument(id: String, content: String, embedding: FloatArray) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                rag?.add(id, content, embedding)
            } catch (e: RAGException) {
                Log.e("RAG", "Add failed: ${e.message}")
            }
        }
    }

    fun search(query: FloatArray) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val searchResults = rag?.search(query, k = 10) ?: emptyList()
                withContext(Dispatchers.Main) {
                    _results.value = searchResults
                }
            } catch (e: RAGException) {
                Log.e("RAG", "Search failed: ${e.message}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        rag?.close()
    }
}
```

### Jetpack Compose Example

```kotlin
@Composable
fun SearchScreen(viewModel: RAGViewModel = viewModel()) {
    val results by viewModel.results.collectAsState()

    LazyColumn {
        items(results) { result ->
            Card(modifier = Modifier.padding(8.dp)) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(text = result.id, style = MaterialTheme.typography.titleMedium)
                    Text(text = result.content, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = "Score: ${result.score}",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}
```

## Desktop (C/C++)

### Setup

**1. Link library:**
```bash
# Linux/macOS
gcc -o app app.c -L./build/desktop -lnexus_rag_native

# Windows (MSVC)
cl app.c /link nexus_rag_native.lib
```

**2. Include header:**
```c
#include "nexus_rag.h"
```

### Usage

```c
#include "nexus_rag.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Create RAG index
    RagHandle* rag = rag_create(384, 1);
    if (!rag) {
        fprintf(stderr, "Failed to create RAG: %s\n", rag_last_error());
        return 1;
    }

    // Add document
    float embedding[384];
    for (int i = 0; i < 384; i++) {
        embedding[i] = 0.1f;
    }

    int result = rag_add_document(
        rag,
        "doc1",
        "Example document",
        embedding,
        384
    );

    if (result != 0) {
        fprintf(stderr, "Failed to add document: %s\n", rag_last_error());
        rag_destroy(rag);
        return 1;
    }

    // Search
    SearchResult* results = NULL;
    size_t count = 0;

    result = rag_search(rag, embedding, 384, 5, &results, &count);
    if (result == 0) {
        for (size_t i = 0; i < count; i++) {
            printf("%s: %.3f\n", results[i].id, results[i].score);
            printf("Content: %s\n", results[i].content);
        }
        rag_free_results(results, count);
    } else {
        fprintf(stderr, "Search failed: %s\n", rag_last_error());
    }

    // Save
    result = rag_save(rag, "/path/to/index.bin");
    if (result != 0) {
        fprintf(stderr, "Save failed: %s\n", rag_last_error());
    }

    // Load
    RagHandle* loaded = rag_load("/path/to/index.bin", 1);
    if (!loaded) {
        fprintf(stderr, "Load failed: %s\n", rag_last_error());
    } else {
        printf("Documents: %zu\n", rag_count(loaded));
        rag_destroy(loaded);
    }

    // Clean up
    rag_destroy(rag);
    return 0;
}
```

### C++ Wrapper Example

```cpp
#include "nexus_rag.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

class NexusRAG {
private:
    struct RAGDeleter {
        void operator()(RagHandle* rag) const {
            if (rag) rag_destroy(rag);
        }
    };

    std::unique_ptr<RagHandle, RAGDeleter> handle_;

public:
    struct SearchResult {
        std::string id;
        std::string content;
        float score;
    };

    NexusRAG(int embeddingDim, bool useHNSW = true) {
        handle_.reset(rag_create(embeddingDim, useHNSW ? 1 : 0));
        if (!handle_) {
            throw std::runtime_error(std::string("Failed to create RAG: ") + rag_last_error());
        }
    }

    void add(const std::string& id, const std::string& content, const std::vector<float>& embedding) {
        int result = rag_add_document(
            handle_.get(),
            id.c_str(),
            content.c_str(),
            embedding.data(),
            embedding.size()
        );
        if (result != 0) {
            throw std::runtime_error(std::string("Failed to add: ") + rag_last_error());
        }
    }

    std::vector<SearchResult> search(const std::vector<float>& query, int k = 10) {
        ::SearchResult* results = nullptr;
        size_t count = 0;

        int result = rag_search(
            handle_.get(),
            query.data(),
            query.size(),
            k,
            &results,
            &count
        );

        if (result != 0) {
            throw std::runtime_error(std::string("Search failed: ") + rag_last_error());
        }

        std::vector<SearchResult> searchResults;
        for (size_t i = 0; i < count; i++) {
            searchResults.push_back({
                results[i].id,
                results[i].content,
                results[i].score
            });
        }

        rag_free_results(results, count);
        return searchResults;
    }

    void save(const std::string& path) {
        if (rag_save(handle_.get(), path.c_str()) != 0) {
            throw std::runtime_error(std::string("Save failed: ") + rag_last_error());
        }
    }

    static NexusRAG load(const std::string& path, bool useHNSW = true) {
        RagHandle* handle = rag_load(path.c_str(), useHNSW ? 1 : 0);
        if (!handle) {
            throw std::runtime_error(std::string("Load failed: ") + rag_last_error());
        }
        return NexusRAG(handle);
    }

    size_t count() const {
        return rag_count(handle_.get());
    }

private:
    NexusRAG(RagHandle* handle) {
        handle_.reset(handle);
    }
};
```

## Best Practices

### Memory Management

**iOS (Swift):**
- `NexusRAG` automatically releases resources in `deinit`
- Use `defer` for cleanup in error paths
- Avoid circular references when storing in properties

**Android (Kotlin):**
- Always call `close()` when done
- Use `use { }` block for automatic cleanup
- Consider using ViewModel lifecycle for long-lived instances

**Desktop (C/C++):**
- Always call `rag_destroy()` for each `rag_create()`/`rag_load()`
- Use RAII wrappers in C++ (unique_ptr, shared_ptr)
- Free search results with `rag_free_results()`

### Error Handling

**iOS (Swift):**
```swift
do {
    try rag.add(id: "doc1", content: "text", embedding: embedding)
} catch RAGError.addFailed(let message) {
    // Handle specific error
} catch {
    // Handle general error
}
```

**Android (Kotlin):**
```kotlin
try {
    rag.add("doc1", "text", embedding)
} catch (e: RAGException) {
    Log.e("RAG", "Error: ${e.message}")
}
```

**Desktop (C):**
```c
if (result != 0) {
    const char* error = rag_last_error();
    fprintf(stderr, "Error: %s\n", error);
    rag_free_error(error);
}
```

### Thread Safety

**Important:** RAG handles are NOT thread-safe. Follow these guidelines:

1. **Single-threaded access:** Use one handle per thread
2. **Synchronization:** If sharing is necessary, use platform-specific locks:
   - iOS: `NSLock`, `DispatchQueue`
   - Android: `synchronized`, `ReentrantLock`
   - Desktop: `pthread_mutex`, `std::mutex`

**Example (iOS):**
```swift
class ThreadSafeRAG {
    private let rag: NexusRAG
    private let queue = DispatchQueue(label: "com.nexus.rag")

    func search(query: [Float], k: Int) throws -> [NexusRAG.SearchResult] {
        try queue.sync {
            try rag.search(query: query, k: k)
        }
    }
}
```

### Performance

**Index Selection:**
- Use **Flat** index for < 1,000 documents (exact search)
- Use **HNSW** index for > 1,000 documents (approximate, faster)

**Batching:**
```swift
// Bad: Individual adds
for doc in documents {
    try rag.add(id: doc.id, content: doc.content, embedding: doc.embedding)
}

// Better: Batch operations (if supported in future)
// try rag.addBatch(documents)
```

**Caching:**
```kotlin
class RAGCache {
    private val cache = LruCache<String, List<SearchResult>>(100)

    fun search(query: FloatArray): List<SearchResult> {
        val key = query.contentHashCode().toString()
        return cache.get(key) ?: rag.search(query).also {
            cache.put(key, it)
        }
    }
}
```

### Persistence

**Save/Load Best Practices:**

1. **Background threads:**
   ```swift
   DispatchQueue.global(qos: .background).async {
       try? rag.save(to: path)
   }
   ```

2. **Error recovery:**
   ```kotlin
   fun saveWithBackup(path: String) {
       try {
           rag.save("$path.tmp")
           File("$path.tmp").renameTo(File(path))
       } catch (e: RAGException) {
           File("$path.tmp").delete()
           throw e
       }
   }
   ```

3. **Atomic writes:**
   - Save to temporary file first
   - Rename to final path on success
   - Keep backup of previous version

### Mobile-Specific Considerations

**iOS:**
- Store indexes in app's Documents directory
- Handle app backgrounding (save state)
- Monitor memory warnings

```swift
NotificationCenter.default.addObserver(
    forName: UIApplication.didEnterBackgroundNotification,
    object: nil,
    queue: nil
) { _ in
    try? rag.save(to: indexPath)
}
```

**Android:**
- Use app's internal storage for indexes
- Handle configuration changes (ViewModel)
- Consider WorkManager for background operations

```kotlin
override fun onStop() {
    super.onStop()
    viewModel.saveIndex()
}
```

## Troubleshooting

### Common Issues

**"Library not found" (iOS):**
- Verify framework is in "Frameworks, Libraries, and Embedded Content"
- Check "Embed & Sign" is selected
- Clean build folder (Cmd+Shift+K)

**"UnsatisfiedLinkError" (Android):**
- Verify .so files are in correct jniLibs directories
- Check ABI filters match device architecture
- Ensure library name matches `System.loadLibrary("nexus_rag_native")`

**"Invalid handle" errors:**
- Don't use RAG after calling `close()`/`destroy()`
- Check initialization succeeded before use
- Verify no double-free issues

**Memory leaks:**
- Always call cleanup methods
- Free search results after use
- Check for retain cycles (iOS)

## Performance Benchmarks

Expected performance on typical devices:

| Operation | iOS (iPhone 12) | Android (Pixel 5) | Desktop (M1 Mac) |
|-----------|-----------------|-------------------|------------------|
| Add document | ~1ms | ~1ms | <1ms |
| Search (HNSW, 1K docs) | ~5ms | ~8ms | ~3ms |
| Search (HNSW, 10K docs) | ~15ms | ~20ms | ~10ms |
| Save index (1K docs) | ~50ms | ~80ms | ~30ms |
| Load index (1K docs) | ~40ms | ~60ms | ~25ms |

*Benchmarks with 384-dimensional embeddings, k=10 search*

## License

MIT - See project LICENSE file for details.

## Support

For issues and questions:
- File an issue on GitHub
- Check the main project README
- Review phase summaries for implementation details
