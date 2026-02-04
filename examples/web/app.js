/**
 * Nexus RAG Demo Application
 *
 * Main application logic for the Nexus RAG web demo.
 * Handles WASM initialization, document management, search, and persistence.
 */

// Global state
let wasmModule = null;
let ragInstance = null;
let storageInstance = null;
let currentConfig = null;

// Sample data for demonstration
const SAMPLE_DOCUMENTS = [
    {
        id: "rust-intro",
        content: "Rust is a systems programming language focused on safety, speed, and concurrency.",
        metadata: { category: "programming", language: "rust" }
    },
    {
        id: "wasm-overview",
        content: "WebAssembly (WASM) enables near-native performance for web applications.",
        metadata: { category: "web", technology: "wasm" }
    },
    {
        id: "rag-definition",
        content: "Retrieval-Augmented Generation combines information retrieval with text generation.",
        metadata: { category: "ai", type: "rag" }
    },
    {
        id: "vector-search",
        content: "Vector search enables semantic similarity search using embeddings.",
        metadata: { category: "ai", type: "search" }
    },
    {
        id: "hnsw-algorithm",
        content: "HNSW is a graph-based algorithm for approximate nearest neighbor search.",
        metadata: { category: "algorithms", type: "ann" }
    },
    {
        id: "privacy-first",
        content: "Local-first applications process data on device without sending it to servers.",
        metadata: { category: "privacy", benefit: "security" }
    },
    {
        id: "indexeddb-storage",
        content: "IndexedDB provides persistent client-side storage for web applications.",
        metadata: { category: "web", storage: "client-side" }
    },
    {
        id: "cosine-similarity",
        content: "Cosine similarity measures the angle between vectors to determine similarity.",
        metadata: { category: "math", type: "metric" }
    },
    {
        id: "embeddings-ml",
        content: "Embeddings are dense vector representations of text learned by machine learning models.",
        metadata: { category: "ml", type: "representation" }
    },
    {
        id: "rust-performance",
        content: "Rust provides memory safety without garbage collection, enabling high performance.",
        metadata: { category: "programming", benefit: "performance" }
    }
];

/**
 * RAG Application Class
 * Manages all application state and interactions
 */
class RAGApp {
    constructor() {
        this.documents = new Map();
        this.currentDimension = 0;
        this.currentIndexType = 'hnsw';
        this.isInitialized = false;
    }

    /**
     * Initialize the application
     */
    async initialize() {
        try {
            console.log('Initializing RAG application...');
            this.showLoading('Loading WASM module...');

            // Import and initialize WASM module
            // Note: The actual WASM module will be built by other agents
            // This is a placeholder that will be replaced with the real module
            try {
                wasmModule = await import('./pkg/nexus_rag_wasm.js');
                await wasmModule.default(); // Initialize WASM
                console.log('WASM module loaded successfully');
            } catch (error) {
                console.warn('WASM module not yet available:', error);
                this.showError('WASM module not built yet. Run "npm run build" first.');
                this.hideLoading();
                return;
            }

            // Set up event listeners
            this.setupEventListeners();

            // Hide loading overlay
            this.hideLoading();

            this.showSuccess('Application initialized! Configure and initialize the RAG system.');
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError(`Initialization failed: ${error.message}`);
            this.hideLoading();
        }
    }

    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // Configuration
        document.getElementById('initButton').addEventListener('click', () => this.initializeRAG());

        // Document management
        document.getElementById('generateRandomBtn').addEventListener('click', () => this.generateRandomEmbedding());
        document.getElementById('loadSampleDataBtn').addEventListener('click', () => this.loadSampleData());
        document.getElementById('addDocButton').addEventListener('click', () => this.addDocument());

        // Search
        document.getElementById('useSampleQueryBtn').addEventListener('click', () => this.useSampleQuery());
        document.getElementById('generateRandomQueryBtn').addEventListener('click', () => this.generateRandomQuery());
        document.getElementById('searchButton').addEventListener('click', () => this.performSearch());

        // Persistence
        document.getElementById('saveButton').addEventListener('click', () => this.saveToIndexedDB());
        document.getElementById('loadButton').addEventListener('click', () => this.loadFromIndexedDB());
        document.getElementById('listIndicesButton').addEventListener('click', () => this.listSavedIndices());
        document.getElementById('clearAllButton').addEventListener('click', () => this.clearAllData());
    }

    /**
     * Initialize the RAG system with user configuration
     */
    async initializeRAG() {
        try {
            const dimension = parseInt(document.getElementById('embeddingDim').value);
            const indexType = document.getElementById('indexType').value;

            if (!dimension || dimension < 1 || dimension > 2048) {
                this.showError('Please enter a valid dimension (1-2048)');
                return;
            }

            this.showLoading('Initializing RAG system...');

            // Store configuration
            this.currentDimension = dimension;
            this.currentIndexType = indexType;
            currentConfig = { dimension, indexType };

            // Initialize RAG instance (will be implemented by WASM binding agent)
            if (wasmModule && wasmModule.LocalRAG) {
                ragInstance = new wasmModule.LocalRAG(dimension, indexType);
            } else {
                // Fallback for demo without WASM
                ragInstance = new MockRAG(dimension, indexType);
            }

            // Initialize storage (will be implemented by IndexedDB agent)
            if (wasmModule && wasmModule.IndexedDBStore) {
                storageInstance = new wasmModule.IndexedDBStore();
            } else {
                // Fallback for demo without WASM
                storageInstance = new MockStorage();
            }

            this.isInitialized = true;

            // Update UI
            this.updateStatus('ready', 'Ready');
            this.enableControls();
            this.updateMetrics();

            this.hideLoading();
            this.showSuccess(`RAG system initialized with ${indexType.toUpperCase()} index (dim: ${dimension})`);
        } catch (error) {
            console.error('RAG initialization error:', error);
            this.showError(`Failed to initialize RAG: ${error.message}`);
            this.hideLoading();
        }
    }

    /**
     * Generate random embedding vector
     */
    generateRandomEmbedding() {
        if (!this.currentDimension) {
            this.showError('Please initialize the RAG system first');
            return;
        }

        const embedding = Array.from({ length: this.currentDimension }, () =>
            (Math.random() * 2 - 1).toFixed(4)
        );

        document.getElementById('docEmbedding').value = embedding.join(', ');
        this.showInfo('Random embedding generated');
    }

    /**
     * Generate random query vector
     */
    generateRandomQuery() {
        if (!this.currentDimension) {
            this.showError('Please initialize the RAG system first');
            return;
        }

        const query = Array.from({ length: this.currentDimension }, () =>
            (Math.random() * 2 - 1).toFixed(4)
        );

        document.getElementById('queryVector').value = query.join(', ');
        this.showInfo('Random query generated');
    }

    /**
     * Use sample query (first document's embedding)
     */
    useSampleQuery() {
        if (this.documents.size === 0) {
            this.showError('No documents added yet');
            return;
        }

        const firstDoc = this.documents.values().next().value;
        document.getElementById('queryVector').value = firstDoc.embedding.join(', ');
        this.showInfo('Using first document embedding as query');
    }

    /**
     * Load sample data
     */
    async loadSampleData() {
        if (!this.isInitialized) {
            this.showError('Please initialize the RAG system first');
            return;
        }

        this.showLoading('Loading sample data...');

        try {
            for (const sample of SAMPLE_DOCUMENTS) {
                // Generate random embedding for each sample
                const embedding = Array.from({ length: this.currentDimension }, () =>
                    Math.random() * 2 - 1
                );

                const doc = {
                    id: sample.id,
                    content: sample.content,
                    embedding: embedding,
                    metadata: sample.metadata
                };

                // Add to RAG instance
                await ragInstance.addDocument(doc);

                // Add to local state
                this.documents.set(doc.id, doc);
            }

            this.updateDocumentsList();
            this.updateMetrics();
            this.hideLoading();
            this.showSuccess(`Loaded ${SAMPLE_DOCUMENTS.length} sample documents`);
        } catch (error) {
            console.error('Error loading sample data:', error);
            this.showError(`Failed to load sample data: ${error.message}`);
            this.hideLoading();
        }
    }

    /**
     * Add a document
     */
    async addDocument() {
        try {
            const id = document.getElementById('docId').value.trim();
            const content = document.getElementById('docContent').value.trim();
            const embeddingStr = document.getElementById('docEmbedding').value.trim();
            const metadataStr = document.getElementById('docMetadata').value.trim();

            // Validation
            if (!id) {
                this.showError('Please enter a document ID');
                return;
            }

            if (!content) {
                this.showError('Please enter document content');
                return;
            }

            if (!embeddingStr) {
                this.showError('Please enter or generate an embedding');
                return;
            }

            // Parse embedding
            const embedding = this.parseEmbeddingInput(embeddingStr);
            if (!embedding) {
                this.showError('Invalid embedding format. Use comma-separated numbers.');
                return;
            }

            if (embedding.length !== this.currentDimension) {
                this.showError(`Embedding must have ${this.currentDimension} dimensions`);
                return;
            }

            // Parse metadata if provided
            let metadata = null;
            if (metadataStr) {
                try {
                    metadata = JSON.parse(metadataStr);
                } catch (e) {
                    this.showError('Invalid JSON in metadata field');
                    return;
                }
            }

            const doc = { id, content, embedding, metadata };

            // Add to RAG instance
            await ragInstance.addDocument(doc);

            // Add to local state
            this.documents.set(id, doc);

            // Update UI
            this.updateDocumentsList();
            this.updateMetrics();

            // Clear form
            document.getElementById('docId').value = '';
            document.getElementById('docContent').value = '';
            document.getElementById('docEmbedding').value = '';
            document.getElementById('docMetadata').value = '';

            this.showSuccess(`Document "${id}" added successfully`);
        } catch (error) {
            console.error('Error adding document:', error);
            this.showError(`Failed to add document: ${error.message}`);
        }
    }

    /**
     * Remove a document
     */
    async removeDocument(id) {
        try {
            // Remove from RAG instance
            await ragInstance.removeDocument(id);

            // Remove from local state
            this.documents.delete(id);

            // Update UI
            this.updateDocumentsList();
            this.updateMetrics();

            this.showSuccess(`Document "${id}" removed`);
        } catch (error) {
            console.error('Error removing document:', error);
            this.showError(`Failed to remove document: ${error.message}`);
        }
    }

    /**
     * Perform search
     */
    async performSearch() {
        try {
            const queryStr = document.getElementById('queryVector').value.trim();
            const k = parseInt(document.getElementById('kResults').value);

            if (!queryStr) {
                this.showError('Please enter a query vector');
                return;
            }

            const query = this.parseEmbeddingInput(queryStr);
            if (!query) {
                this.showError('Invalid query format. Use comma-separated numbers.');
                return;
            }

            if (query.length !== this.currentDimension) {
                this.showError(`Query must have ${this.currentDimension} dimensions`);
                return;
            }

            if (!k || k < 1) {
                this.showError('Please enter a valid k value');
                return;
            }

            // Measure performance
            const startTime = performance.now();

            // Perform search
            const results = await ragInstance.search(query, k);

            const endTime = performance.now();
            const searchTime = (endTime - startTime).toFixed(2);

            // Display results
            this.displaySearchResults(results, searchTime);

            this.showSuccess(`Found ${results.length} results in ${searchTime}ms`);
        } catch (error) {
            console.error('Search error:', error);
            this.showError(`Search failed: ${error.message}`);
        }
    }

    /**
     * Save index to IndexedDB
     */
    async saveToIndexedDB() {
        try {
            const key = document.getElementById('saveKey').value.trim();

            if (!key) {
                this.showError('Please enter a save key');
                return;
            }

            this.showLoading('Saving to IndexedDB...');

            // Save index state
            const state = await ragInstance.serialize();
            await storageInstance.save(key, state);

            this.hideLoading();
            this.showSuccess(`Index saved as "${key}"`);

            // Refresh saved indices list
            await this.listSavedIndices();
        } catch (error) {
            console.error('Save error:', error);
            this.showError(`Failed to save: ${error.message}`);
            this.hideLoading();
        }
    }

    /**
     * Load index from IndexedDB
     */
    async loadFromIndexedDB() {
        try {
            const key = document.getElementById('saveKey').value.trim();

            if (!key) {
                this.showError('Please enter a load key');
                return;
            }

            this.showLoading('Loading from IndexedDB...');

            // Load index state
            const state = await storageInstance.load(key);

            if (!state) {
                this.showError(`No saved index found with key "${key}"`);
                this.hideLoading();
                return;
            }

            // Deserialize and restore RAG instance
            ragInstance = await wasmModule.LocalRAG.deserialize(state);

            // Update configuration
            this.currentDimension = state.dimension;
            this.currentIndexType = state.indexType;
            document.getElementById('embeddingDim').value = this.currentDimension;
            document.getElementById('indexType').value = this.currentIndexType;

            // Rebuild local documents map
            this.documents.clear();
            const docs = await ragInstance.getAllDocuments();
            docs.forEach(doc => this.documents.set(doc.id, doc));

            // Update UI
            this.isInitialized = true;
            this.updateStatus('ready', 'Ready');
            this.enableControls();
            this.updateDocumentsList();
            this.updateMetrics();

            this.hideLoading();
            this.showSuccess(`Loaded index "${key}" with ${this.documents.size} documents`);
        } catch (error) {
            console.error('Load error:', error);
            this.showError(`Failed to load: ${error.message}`);
            this.hideLoading();
        }
    }

    /**
     * List saved indices
     */
    async listSavedIndices() {
        try {
            const keys = await storageInstance.listKeys();

            const listContainer = document.getElementById('savedIndicesList');

            if (keys.length === 0) {
                listContainer.innerHTML = '<p class="empty-state">No saved indices</p>';
                return;
            }

            listContainer.innerHTML = keys.map(key => `
                <div class="saved-index-item">
                    <span class="index-name">${this.escapeHtml(key)}</span>
                    <div class="index-actions">
                        <button class="btn btn-secondary btn-small" onclick="app.loadSpecificIndex('${this.escapeHtml(key)}')">
                            Load
                        </button>
                        <button class="btn btn-danger btn-small" onclick="app.deleteIndex('${this.escapeHtml(key)}')">
                            Delete
                        </button>
                    </div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Error listing indices:', error);
            this.showError(`Failed to list indices: ${error.message}`);
        }
    }

    /**
     * Load a specific index by key
     */
    async loadSpecificIndex(key) {
        document.getElementById('saveKey').value = key;
        await this.loadFromIndexedDB();
    }

    /**
     * Delete an index from storage
     */
    async deleteIndex(key) {
        if (!confirm(`Are you sure you want to delete index "${key}"?`)) {
            return;
        }

        try {
            await storageInstance.delete(key);
            this.showSuccess(`Deleted index "${key}"`);
            await this.listSavedIndices();
        } catch (error) {
            console.error('Delete error:', error);
            this.showError(`Failed to delete: ${error.message}`);
        }
    }

    /**
     * Clear all data
     */
    async clearAllData() {
        if (!confirm('Are you sure you want to clear all documents? This cannot be undone.')) {
            return;
        }

        try {
            // Clear RAG instance
            await ragInstance.clear();

            // Clear local state
            this.documents.clear();

            // Update UI
            this.updateDocumentsList();
            this.updateMetrics();
            document.getElementById('searchResults').innerHTML = '<p class="empty-state">No search performed yet</p>';

            this.showSuccess('All documents cleared');
        } catch (error) {
            console.error('Clear error:', error);
            this.showError(`Failed to clear data: ${error.message}`);
        }
    }

    /**
     * Update documents list UI
     */
    updateDocumentsList() {
        const listContainer = document.getElementById('documentsList');

        if (this.documents.size === 0) {
            listContainer.innerHTML = '<p class="empty-state">No documents added yet</p>';
            return;
        }

        listContainer.innerHTML = Array.from(this.documents.values())
            .map(doc => `
                <div class="document-item">
                    <div class="document-info">
                        <div class="document-id">${this.escapeHtml(doc.id)}</div>
                        <div class="document-content">${this.escapeHtml(doc.content)}</div>
                        ${doc.metadata ? `<div class="document-meta">Metadata: ${this.escapeHtml(JSON.stringify(doc.metadata))}</div>` : ''}
                    </div>
                    <button class="delete-btn" onclick="app.removeDocument('${this.escapeHtml(doc.id)}')">×</button>
                </div>
            `)
            .join('');
    }

    /**
     * Display search results
     */
    displaySearchResults(results, searchTime) {
        const resultsContainer = document.getElementById('searchResults');
        const metricsBox = document.getElementById('performanceMetrics');

        // Update metrics
        document.getElementById('searchTime').textContent = `${searchTime}ms`;
        document.getElementById('resultsCount').textContent = results.length;
        metricsBox.style.display = 'block';

        if (results.length === 0) {
            resultsContainer.innerHTML = '<p class="empty-state">No results found</p>';
            return;
        }

        resultsContainer.innerHTML = results
            .map((result, index) => `
                <div class="result-item" style="animation: slideIn 0.3s ease-out ${index * 0.05}s both;">
                    <div class="result-header">
                        <span class="result-id">${this.escapeHtml(result.id)}</span>
                        <span class="result-score">Score: ${result.score.toFixed(4)}</span>
                    </div>
                    <div class="result-content">${this.escapeHtml(result.content)}</div>
                    ${result.metadata ? `<div class="result-metadata">${this.escapeHtml(JSON.stringify(result.metadata))}</div>` : ''}
                </div>
            `)
            .join('');
    }

    /**
     * Update status indicator
     */
    updateStatus(status, text) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const dot = indicator.querySelector('.status-dot');

        dot.className = `status-dot status-${status}`;
        statusText.textContent = text;
    }

    /**
     * Update metrics display
     */
    updateMetrics() {
        document.getElementById('docCount').textContent = this.documents.size;
        document.getElementById('currentDim').textContent = this.currentDimension || '-';
        document.getElementById('currentIndexType').textContent = this.currentIndexType ? this.currentIndexType.toUpperCase() : '-';
    }

    /**
     * Enable controls after initialization
     */
    enableControls() {
        const controls = [
            'docId', 'docContent', 'docEmbedding', 'docMetadata',
            'generateRandomBtn', 'loadSampleDataBtn', 'addDocButton',
            'queryVector', 'kResults', 'useSampleQueryBtn', 'generateRandomQueryBtn', 'searchButton',
            'saveKey', 'saveButton', 'loadButton', 'listIndicesButton', 'clearAllButton'
        ];

        controls.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.disabled = false;
        });
    }

    /**
     * Parse embedding input string
     */
    parseEmbeddingInput(input) {
        try {
            const values = input.split(',').map(s => parseFloat(s.trim()));
            if (values.some(v => isNaN(v))) return null;
            return values;
        } catch (e) {
            return null;
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Show loading overlay
     */
    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.querySelector('p').textContent = message;
            overlay.style.display = 'flex';
        }
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    showSuccess(message) { this.showToast(message, 'success'); }
    showError(message) { this.showToast(message, 'error'); }
    showInfo(message) { this.showToast(message, 'info'); }
    showWarning(message) { this.showToast(message, 'warning'); }
}

/**
 * Mock RAG implementation for testing without WASM
 */
class MockRAG {
    constructor(dimension, indexType) {
        this.dimension = dimension;
        this.indexType = indexType;
        this.documents = [];
    }

    async addDocument(doc) {
        // Check if document already exists
        const existingIndex = this.documents.findIndex(d => d.id === doc.id);
        if (existingIndex >= 0) {
            this.documents[existingIndex] = doc;
        } else {
            this.documents.push(doc);
        }
    }

    async removeDocument(id) {
        this.documents = this.documents.filter(d => d.id !== id);
    }

    async search(query, k) {
        // Simple cosine similarity search
        const results = this.documents.map(doc => {
            const score = this.cosineSimilarity(query, doc.embedding);
            return {
                id: doc.id,
                content: doc.content,
                score: score,
                metadata: doc.metadata
            };
        });

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);

        // Return top k
        return results.slice(0, k);
    }

    async clear() {
        this.documents = [];
    }

    async serialize() {
        return {
            dimension: this.dimension,
            indexType: this.indexType,
            documents: this.documents
        };
    }

    async getAllDocuments() {
        return this.documents;
    }

    cosineSimilarity(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA === 0 || normB === 0) return 0;

        return dotProduct / (normA * normB);
    }
}

/**
 * Mock Storage implementation for testing without WASM
 */
class MockStorage {
    constructor() {
        this.storage = new Map();
    }

    async save(key, data) {
        this.storage.set(key, JSON.stringify(data));
        // Also save to localStorage as backup
        try {
            localStorage.setItem(`nexus-rag-${key}`, JSON.stringify(data));
        } catch (e) {
            console.warn('localStorage not available');
        }
    }

    async load(key) {
        const data = this.storage.get(key);
        if (data) return JSON.parse(data);

        // Try loading from localStorage
        try {
            const lsData = localStorage.getItem(`nexus-rag-${key}`);
            if (lsData) {
                const parsed = JSON.parse(lsData);
                this.storage.set(key, lsData);
                return parsed;
            }
        } catch (e) {
            console.warn('localStorage not available');
        }

        return null;
    }

    async delete(key) {
        this.storage.delete(key);
        try {
            localStorage.removeItem(`nexus-rag-${key}`);
        } catch (e) {
            console.warn('localStorage not available');
        }
    }

    async listKeys() {
        const keys = Array.from(this.storage.keys());

        // Also check localStorage
        try {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('nexus-rag-')) {
                    const shortKey = key.replace('nexus-rag-', '');
                    if (!keys.includes(shortKey)) {
                        keys.push(shortKey);
                    }
                }
            }
        } catch (e) {
            console.warn('localStorage not available');
        }

        return keys;
    }
}

// Initialize application when DOM is ready
const app = new RAGApp();

document.addEventListener('DOMContentLoaded', () => {
    app.initialize();
});

// Add slideIn animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
`;
document.head.appendChild(style);
