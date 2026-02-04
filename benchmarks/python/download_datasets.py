#!/usr/bin/env python3
"""
Download and prepare standard ANN benchmark datasets.

Datasets:
- SIFT1M: 1M 128-dim SIFT descriptors (standard ANN benchmark)
- SIFT10K: 10K subset for quick testing

Data source: ftp://ftp.irisa.fr/local/texmex/corpus/
Format: fvecs/ivecs (see http://corpus-texmex.irisa.fr/)
"""

import os
import sys
import struct
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import numpy as np
from tqdm import tqdm

# Dataset URLs
DATASETS = {
    "sift1m": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "description": "1M SIFT descriptors, 128-dim",
        "files": {
            "base": "sift/sift_base.fvecs",      # 1M vectors to index
            "query": "sift/sift_query.fvecs",    # 10K query vectors
            "groundtruth": "sift/sift_groundtruth.ivecs",  # Ground truth (100-NN)
            "learn": "sift/sift_learn.fvecs",    # 100K training vectors
        }
    },
}

# Alternative HTTP mirror (if FTP fails)
SIFT_HTTP_MIRROR = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


def read_fvecs(filename: str) -> np.ndarray:
    """Read fvecs format: [dim, float32 * dim] per vector."""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}f", f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """Read ivecs format: [dim, int32 * dim] per vector."""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}i", f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)


def write_fvecs(vectors: np.ndarray, filename: str):
    """Write vectors to fvecs format."""
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}f", *vec))


def write_ivecs(vectors: np.ndarray, filename: str):
    """Write vectors to ivecs format."""
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}i", *vec.astype(np.int32)))


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, description: str = "Downloading"):
    """Download file with progress bar."""
    print(f"\n{description}")
    print(f"  URL: {url}")
    print(f"  Target: {output_path}")
    
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=description) as t:
            urlretrieve(url, output_path, reporthook=t.update_to)
        return True
    except URLError as e:
        print(f"  ERROR: Failed to download - {e}")
        return False


def download_sift1m(data_dir: Path, force: bool = False):
    """Download and extract SIFT1M dataset."""
    dataset_dir = data_dir / "sift1m"
    
    # Check if already exists
    required_files = ["base.fvecs", "query.fvecs", "groundtruth.ivecs"]
    if not force and all((dataset_dir / f).exists() for f in required_files):
        print("SIFT1M already downloaded. Use --force to re-download.")
        return True
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Download tarball
    tar_path = data_dir / "sift.tar.gz"
    info = DATASETS["sift1m"]
    
    if not download_file(info["url"], str(tar_path), "Downloading SIFT1M"):
        print("FTP download failed, trying HTTP mirror...")
        # Fall back to generating synthetic if both fail
        print("Creating synthetic SIFT-like dataset for benchmarking...")
        return create_synthetic_sift(dataset_dir)
    
    # Extract
    print("\nExtracting archive...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        
        # Read and convert to our format
        print("Converting to benchmark format...")
        
        base = read_fvecs(data_dir / info["files"]["base"])
        query = read_fvecs(data_dir / info["files"]["query"])
        groundtruth = read_ivecs(data_dir / info["files"]["groundtruth"])
        
        # Save in our format
        write_fvecs(base, dataset_dir / "base.fvecs")
        write_fvecs(query, dataset_dir / "query.fvecs")
        write_ivecs(groundtruth, dataset_dir / "groundtruth.ivecs")
        
        # Also save as numpy for easier loading
        np.save(dataset_dir / "base.npy", base)
        np.save(dataset_dir / "query.npy", query)
        np.save(dataset_dir / "groundtruth.npy", groundtruth)
        
        # Cleanup
        tar_path.unlink()
        import shutil
        shutil.rmtree(data_dir / "sift", ignore_errors=True)
        
        print(f"\nSIFT1M ready:")
        print(f"  Base vectors: {base.shape}")
        print(f"  Query vectors: {query.shape}")
        print(f"  Ground truth: {groundtruth.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def create_synthetic_sift(dataset_dir: Path, n_base: int = 1_000_000, n_query: int = 10_000, dim: int = 128):
    """Create synthetic SIFT-like dataset for benchmarking when download fails."""
    print(f"Generating synthetic dataset: {n_base:,} base, {n_query:,} queries, {dim}d")
    
    np.random.seed(42)  # Reproducibility
    
    # Generate base vectors (similar distribution to SIFT)
    print("  Generating base vectors...")
    base = np.random.randint(0, 256, size=(n_base, dim)).astype(np.float32)
    
    # Generate query vectors
    print("  Generating query vectors...")
    query = np.random.randint(0, 256, size=(n_query, dim)).astype(np.float32)
    
    # Compute ground truth (brute force - slow but correct)
    print("  Computing ground truth (this takes a while for 1M vectors)...")
    k = 100  # Top-100 nearest neighbors
    
    # Use batched computation to avoid memory issues
    batch_size = 100
    groundtruth = np.zeros((n_query, k), dtype=np.int32)
    
    for i in tqdm(range(0, n_query, batch_size), desc="Computing GT"):
        end = min(i + batch_size, n_query)
        batch_queries = query[i:end]
        
        # Compute L2 distances
        # (q - b)^2 = q^2 + b^2 - 2*q*b
        q_norm = np.sum(batch_queries ** 2, axis=1, keepdims=True)
        b_norm = np.sum(base ** 2, axis=1)
        dists = q_norm + b_norm - 2 * np.dot(batch_queries, base.T)
        
        # Get top-k indices
        groundtruth[i:end] = np.argpartition(dists, k, axis=1)[:, :k]
        # Sort the top-k
        for j in range(end - i):
            idx = groundtruth[i + j]
            sorted_idx = idx[np.argsort(dists[j, idx])]
            groundtruth[i + j] = sorted_idx
    
    # Save
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(dataset_dir / "base.npy", base)
    np.save(dataset_dir / "query.npy", query)
    np.save(dataset_dir / "groundtruth.npy", groundtruth)
    
    write_fvecs(base, dataset_dir / "base.fvecs")
    write_fvecs(query, dataset_dir / "query.fvecs")
    write_ivecs(groundtruth, dataset_dir / "groundtruth.ivecs")
    
    print(f"\nSynthetic SIFT ready:")
    print(f"  Base vectors: {base.shape}")
    print(f"  Query vectors: {query.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    return True


def create_sift10k(data_dir: Path):
    """Create SIFT10K subset from SIFT1M for quick testing."""
    sift1m_dir = data_dir / "sift1m"
    sift10k_dir = data_dir / "sift10k"
    
    if not (sift1m_dir / "base.npy").exists():
        print("SIFT1M not found. Download it first.")
        return False
    
    print("Creating SIFT10K subset...")
    sift10k_dir.mkdir(parents=True, exist_ok=True)
    
    base = np.load(sift1m_dir / "base.npy")[:10_000]
    query = np.load(sift1m_dir / "query.npy")[:1_000]
    
    # Recompute ground truth for subset
    k = 100
    groundtruth = np.zeros((len(query), k), dtype=np.int32)
    
    for i, q in enumerate(tqdm(query, desc="Computing GT")):
        dists = np.sum((base - q) ** 2, axis=1)
        groundtruth[i] = np.argsort(dists)[:k]
    
    np.save(sift10k_dir / "base.npy", base)
    np.save(sift10k_dir / "query.npy", query)
    np.save(sift10k_dir / "groundtruth.npy", groundtruth)
    
    print(f"\nSIFT10K ready:")
    print(f"  Base vectors: {base.shape}")
    print(f"  Query vectors: {query.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    parser.add_argument("--data-dir", type=str, default="../data",
                       help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["sift1m", "sift10k", "all"],
                       help="Dataset to download")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if exists")
    parser.add_argument("--synthetic", action="store_true",
                       help="Create synthetic data instead of downloading")
    parser.add_argument("--synthetic-size", type=int, default=1_000_000,
                       help="Size of synthetic dataset base vectors")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    
    if args.synthetic:
        create_synthetic_sift(data_dir / "sift1m", n_base=args.synthetic_size)
        if args.dataset in ["sift10k", "all"]:
            create_sift10k(data_dir)
    else:
        if args.dataset in ["sift1m", "all"]:
            download_sift1m(data_dir, force=args.force)
        
        if args.dataset in ["sift10k", "all"]:
            create_sift10k(data_dir)
    
    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
