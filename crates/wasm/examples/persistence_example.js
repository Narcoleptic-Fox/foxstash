/**
 * IndexedDB Persistence Example
 *
 * This example demonstrates how to use the IndexedDB persistence module
 * with the Nexus Local RAG WASM bindings.
 */

import init, { LocalRAG, JsDocument, IndexedDBStore } from './nexus_rag_wasm.js';

// Initialize WASM module
await init();

// ============================================================================
// Example 1: Basic Save and Load
// ============================================================================

console.log("=== Example 1: Basic Save and Load ===");

// Create a RAG instance
const rag = new LocalRAG(384, false); // 384-dim embeddings, flat index

// Add some documents
const doc1 = new JsDocument(
  "doc1",
  "Machine learning is a subset of artificial intelligence.",
  new Float32Array(384).fill(0.1),
  { category: "AI", difficulty: "beginner" }
);

const doc2 = new JsDocument(
  "doc2",
  "Deep learning uses neural networks with multiple layers.",
  new Float32Array(384).fill(0.2),
  { category: "AI", difficulty: "intermediate" }
);

rag.add_document(doc1);
rag.add_document(doc2);

console.log(`Created index with ${rag.document_count()} documents`);

// Create IndexedDB store
const store = new IndexedDBStore();

// Save to IndexedDB
try {
  await rag.save_to_db(store, "ai-knowledge");
  console.log("✓ Index saved to IndexedDB");
} catch (error) {
  console.error("✗ Save failed:", error);
}

// Load from IndexedDB
try {
  const loadedRag = await LocalRAG.load_from_db(store, "ai-knowledge");
  console.log(`✓ Loaded index with ${loadedRag.document_count()} documents`);

  // Verify we can search
  const query = new Float32Array(384).fill(0.15);
  const results = loadedRag.search(query, 2);
  console.log(`  Found ${results.length} results`);
} catch (error) {
  console.error("✗ Load failed:", error);
}

// ============================================================================
// Example 2: Managing Multiple Indices
// ============================================================================

console.log("\n=== Example 2: Managing Multiple Indices ===");

// Create different indices for different topics
const techRag = new LocalRAG(384, false);
const scienceRag = new LocalRAG(384, true); // Using HNSW

// Add documents to tech index
techRag.add_document(new JsDocument(
  "tech1",
  "JavaScript is a versatile programming language.",
  new Float32Array(384).fill(0.3),
  null
));

techRag.add_document(new JsDocument(
  "tech2",
  "Rust provides memory safety without garbage collection.",
  new Float32Array(384).fill(0.4),
  null
));

// Add documents to science index
scienceRag.add_document(new JsDocument(
  "sci1",
  "Quantum mechanics describes nature at the smallest scales.",
  new Float32Array(384).fill(0.5),
  null
));

// Save multiple indices
await techRag.save_to_db(store, "tech-knowledge");
await scienceRag.save_to_db(store, "science-knowledge");

console.log("✓ Saved multiple indices");

// List all saved indices
const keys = await store.list_keys();
console.log(`✓ Available indices: ${keys.join(", ")}`);

// Load specific index
const loadedTech = await LocalRAG.load_from_db(store, "tech-knowledge");
console.log(`✓ Loaded tech index with ${loadedTech.document_count()} documents`);

// ============================================================================
// Example 3: Update Existing Index
// ============================================================================

console.log("\n=== Example 3: Update Existing Index ===");

// Load existing index
const existingRag = await LocalRAG.load_from_db(store, "ai-knowledge");
console.log(`Loaded index with ${existingRag.document_count()} documents`);

// Add more documents
const doc3 = new JsDocument(
  "doc3",
  "Natural language processing enables computers to understand human language.",
  new Float32Array(384).fill(0.25),
  { category: "AI", difficulty: "intermediate" }
);

existingRag.add_document(doc3);
console.log(`Added document, now have ${existingRag.document_count()} documents`);

// Save updated index (overwrites previous version)
await existingRag.save_to_db(store, "ai-knowledge");
console.log("✓ Updated index saved");

// ============================================================================
// Example 4: Error Handling
// ============================================================================

console.log("\n=== Example 4: Error Handling ===");

// Try to load non-existent index
try {
  const nonexistent = await LocalRAG.load_from_db(store, "does-not-exist");
  console.log("This shouldn't print");
} catch (error) {
  console.log("✓ Correctly caught error for non-existent index");
  console.log(`  Error: ${error.message}`);
}

// Handle quota exceeded (simulated)
try {
  // This would happen with very large indices
  await rag.save_to_db(store, "test-index");
  console.log("✓ Save successful");
} catch (error) {
  if (error.message.includes("quota") || error.message.includes("QuotaExceeded")) {
    console.log("Storage quota exceeded! Consider:");
    console.log("  - Clearing old indices");
    console.log("  - Using compression");
    console.log("  - Splitting into smaller indices");
  } else {
    console.error("Save failed:", error);
  }
}

// ============================================================================
// Example 5: Cleanup Operations
// ============================================================================

console.log("\n=== Example 5: Cleanup Operations ===");

// Delete specific index
await store.delete("science-knowledge");
console.log("✓ Deleted science-knowledge index");

// List remaining indices
const remainingKeys = await store.list_keys();
console.log(`✓ Remaining indices: ${remainingKeys.join(", ")}`);

// Clear all data (use with caution!)
const shouldClearAll = false; // Set to true to actually clear
if (shouldClearAll) {
  await store.clear();
  console.log("✓ Cleared all data");

  const afterClear = await store.list_keys();
  console.log(`✓ Indices after clear: ${afterClear.length}`);
}

// ============================================================================
// Example 6: JSON Serialization (Alternative Method)
// ============================================================================

console.log("\n=== Example 6: JSON Serialization ===");

// Serialize to JSON (useful for localStorage or manual storage)
const jsonData = rag.to_json();
console.log("✓ Serialized to JSON");
console.log(`  Data type: ${typeof jsonData}`);

// Store in localStorage (size-limited, but simpler)
try {
  localStorage.setItem('rag-backup', JSON.stringify(jsonData));
  console.log("✓ Saved to localStorage");
} catch (error) {
  console.error("localStorage save failed:", error);
}

// Load from JSON
try {
  const storedData = JSON.parse(localStorage.getItem('rag-backup'));
  const restoredRag = LocalRAG.from_json(storedData);
  console.log(`✓ Restored from JSON: ${restoredRag.document_count()} documents`);
} catch (error) {
  console.error("JSON restore failed:", error);
}

// ============================================================================
// Example 7: Browser Compatibility Check
// ============================================================================

console.log("\n=== Example 7: Browser Compatibility ===");

function checkIndexedDBSupport() {
  if (!window.indexedDB) {
    console.log("✗ IndexedDB not supported in this browser");
    console.log("  Falling back to in-memory only mode");
    return false;
  }

  console.log("✓ IndexedDB is supported");

  // Check if we're in private/incognito mode
  try {
    // Some browsers block IndexedDB in private mode
    const testDB = window.indexedDB.open('test');
    testDB.onsuccess = () => {
      console.log("✓ IndexedDB is accessible");
      testDB.result.close();
      window.indexedDB.deleteDatabase('test');
    };
    testDB.onerror = () => {
      console.log("⚠ IndexedDB blocked (private mode?)");
    };
  } catch (error) {
    console.log("⚠ IndexedDB test failed:", error.message);
  }

  return true;
}

checkIndexedDBSupport();

// ============================================================================
// Example 8: Storage Size Estimation
// ============================================================================

console.log("\n=== Example 8: Storage Size Estimation ===");

if (navigator.storage && navigator.storage.estimate) {
  const estimate = await navigator.storage.estimate();
  const usedMB = (estimate.usage / (1024 * 1024)).toFixed(2);
  const quotaMB = (estimate.quota / (1024 * 1024)).toFixed(2);
  const percentUsed = ((estimate.usage / estimate.quota) * 100).toFixed(2);

  console.log(`Storage used: ${usedMB} MB / ${quotaMB} MB (${percentUsed}%)`);

  if (percentUsed > 80) {
    console.log("⚠ Warning: Storage is running low!");
  }
} else {
  console.log("Storage estimation not available in this browser");
}

// ============================================================================
// Example 9: Advanced Usage - Custom Database Name
// ============================================================================

console.log("\n=== Example 9: Custom Database Name ===");

// Create store with custom database name
const customStore = new IndexedDBStore("my-custom-rag-db");

await rag.save_to_db(customStore, "custom-index");
console.log("✓ Saved to custom database");

const loadedCustom = await LocalRAG.load_from_db(customStore, "custom-index");
console.log(`✓ Loaded from custom database: ${loadedCustom.document_count()} documents`);

// ============================================================================
// Example 10: Performance Comparison
// ============================================================================

console.log("\n=== Example 10: Performance Test ===");

// Create a larger index for performance testing
const perfRag = new LocalRAG(128, false); // Smaller embeddings for faster test

console.log("Creating 100 documents...");
for (let i = 0; i < 100; i++) {
  const doc = new JsDocument(
    `doc${i}`,
    `This is test document number ${i}`,
    new Float32Array(128).fill(Math.random()),
    { index: i }
  );
  perfRag.add_document(doc);
}

// Measure save time
console.time("Save time");
await perfRag.save_to_db(store, "perf-test");
console.timeEnd("Save time");

// Measure load time
console.time("Load time");
const loadedPerf = await LocalRAG.load_from_db(store, "perf-test");
console.timeEnd("Load time");

console.log(`✓ Performance test complete: ${loadedPerf.document_count()} documents`);

// Cleanup performance test
await store.delete("perf-test");

// ============================================================================
// Summary
// ============================================================================

console.log("\n=== Summary ===");
console.log("✓ All examples completed successfully!");
console.log("\nKey takeaways:");
console.log("  - Use IndexedDBStore for persistent storage");
console.log("  - Save/load operations are async (use await)");
console.log("  - Multiple indices can be stored with different keys");
console.log("  - JSON serialization available for localStorage");
console.log("  - Always handle errors (quota, non-existent keys, etc.)");
console.log("  - Check browser compatibility before using");
console.log("\nFor more information, see PERSISTENCE_README.md");
