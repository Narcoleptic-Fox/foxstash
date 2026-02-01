# Contributing to Foxstash

Thank you for your interest in contributing to Foxstash! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/foxstash.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `cargo test -p foxstash-core`
6. Submit a pull request

## Development Setup

### Prerequisites

- Rust 1.70+ (stable)
- Cargo

### Optional

- LZ4/Zstd libraries for compression features
- ONNX Runtime for embedding features
- wasm-pack for WASM builds

### Building

```bash
# Build core library
cargo build -p foxstash-core

# Build with all compression codecs
cargo build -p foxstash-core --features compression-all

# Run tests
cargo test -p foxstash-core

# Run benchmarks
cargo bench -p foxstash-core
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Add tests for new functionality
- Update documentation for public APIs

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Add tests**: New code should have test coverage
3. **Update docs**: Public API changes need doc updates
4. **Describe changes**: Explain what and why in the PR description
5. **Reference issues**: Link related issues if applicable

## Testing

```bash
# Run all tests
cargo test -p foxstash-core

# Run specific test
cargo test -p foxstash-core test_hnsw_search

# Run doctests
cargo test -p foxstash-core --doc

# Run with coverage
cargo llvm-cov -p foxstash-core --lib
```

## Architecture

```
crates/
├── core/           # Main library
│   ├── embedding/  # ONNX Runtime integration
│   ├── index/      # HNSW, Flat, Quantized indexes
│   ├── storage/    # File persistence, WAL
│   └── vector/     # SIMD operations, quantization
├── wasm/           # WebAssembly bindings
├── native/         # Native bindings
└── benches/        # Benchmarks
```

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `test:` Adding missing tests
- `perf:` Performance improvement
- `chore:` Maintenance tasks

Example: `feat: Add binary quantization with Hamming distance`

## Questions?

Open an issue or discussion on GitHub.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
