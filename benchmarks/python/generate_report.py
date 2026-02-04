#!/usr/bin/env python3
"""
Generate markdown benchmark reports from JSON results.

Usage:
    python generate_report.py                          # Use latest results
    python generate_report.py --input results.json     # Specific file
    python generate_report.py --compare file1 file2    # Compare runs
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    library: str
    algorithm: str
    dataset: str
    index_size: int
    build_time_sec: float
    peak_memory_mb: float
    recall_at_10: float
    recall_at_100: float
    qps: float
    parameters: Dict[str, Any]
    notes: str = ""


def load_results(path: Path) -> List[BenchmarkResult]:
    """Load results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkResult(**r) for r in data]


def format_number(n: float, precision: int = 2) -> str:
    """Format number with appropriate precision."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return f"{n:.{precision}f}"


def generate_summary_table(results: List[BenchmarkResult]) -> str:
    """Generate main summary table."""
    lines = [
        "## Summary",
        "",
        "| Library | Algorithm | Build Time | Memory | Recall@10 | Recall@100 | QPS |",
        "|---------|-----------|------------|--------|-----------|------------|-----|",
    ]
    
    # Group by library/algorithm, pick best recall@10 config
    best_by_algo = {}
    for r in results:
        key = (r.library, r.algorithm)
        if key not in best_by_algo or r.recall_at_10 > best_by_algo[key].recall_at_10:
            best_by_algo[key] = r
    
    for (lib, algo), r in sorted(best_by_algo.items()):
        lines.append(
            f"| {lib} | {algo} | {r.build_time_sec:.2f}s | {r.peak_memory_mb:.0f}MB | "
            f"{r.recall_at_10:.4f} | {r.recall_at_100:.4f} | {format_number(r.qps)} |"
        )
    
    return "\n".join(lines)


def generate_recall_qps_table(results: List[BenchmarkResult]) -> str:
    """Generate recall vs QPS tradeoff table."""
    lines = [
        "## Recall vs QPS Tradeoff",
        "",
        "Best configurations at different recall targets:",
        "",
        "### Recall@10 ≥ 0.95",
        "",
        "| Library | Algorithm | Recall@10 | QPS | Parameters |",
        "|---------|-----------|-----------|-----|------------|",
    ]
    
    # Filter for recall >= 0.95
    high_recall = [r for r in results if r.recall_at_10 >= 0.95]
    high_recall.sort(key=lambda r: r.qps, reverse=True)
    
    for r in high_recall[:5]:
        params_str = ", ".join(f"{k}={v}" for k, v in r.parameters.items())
        lines.append(
            f"| {r.library} | {r.algorithm} | {r.recall_at_10:.4f} | {format_number(r.qps)} | {params_str} |"
        )
    
    lines.extend([
        "",
        "### Recall@10 ≥ 0.99",
        "",
        "| Library | Algorithm | Recall@10 | QPS | Parameters |",
        "|---------|-----------|-----------|-----|------------|",
    ])
    
    # Filter for recall >= 0.99
    very_high_recall = [r for r in results if r.recall_at_10 >= 0.99]
    very_high_recall.sort(key=lambda r: r.qps, reverse=True)
    
    for r in very_high_recall[:5]:
        params_str = ", ".join(f"{k}={v}" for k, v in r.parameters.items())
        lines.append(
            f"| {r.library} | {r.algorithm} | {r.recall_at_10:.4f} | {format_number(r.qps)} | {params_str} |"
        )
    
    return "\n".join(lines)


def generate_build_comparison(results: List[BenchmarkResult]) -> str:
    """Generate build time and memory comparison."""
    lines = [
        "## Index Construction",
        "",
        "| Library | Algorithm | Build Time | Memory | Notes |",
        "|---------|-----------|------------|--------|-------|",
    ]
    
    # Deduplicate (same library/algo have same build metrics)
    seen = set()
    for r in results:
        key = (r.library, r.algorithm)
        if key in seen:
            continue
        seen.add(key)
        
        notes = r.notes or "-"
        lines.append(
            f"| {r.library} | {r.algorithm} | {r.build_time_sec:.2f}s | {r.peak_memory_mb:.0f}MB | {notes} |"
        )
    
    return "\n".join(lines)


def generate_detailed_results(results: List[BenchmarkResult]) -> str:
    """Generate detailed results per library."""
    lines = ["## Detailed Results"]
    
    # Group by library
    by_library = defaultdict(list)
    for r in results:
        by_library[r.library].append(r)
    
    for lib, lib_results in sorted(by_library.items()):
        lines.extend([
            "",
            f"### {lib}",
            "",
        ])
        
        # Group by algorithm
        by_algo = defaultdict(list)
        for r in lib_results:
            by_algo[r.algorithm].append(r)
        
        for algo, algo_results in sorted(by_algo.items()):
            lines.extend([
                f"#### {algo}",
                "",
                "| Parameters | Recall@10 | Recall@100 | QPS |",
                "|------------|-----------|------------|-----|",
            ])
            
            for r in sorted(algo_results, key=lambda r: r.qps, reverse=True):
                params_str = ", ".join(f"{k}={v}" for k, v in r.parameters.items())
                lines.append(
                    f"| {params_str} | {r.recall_at_10:.4f} | {r.recall_at_100:.4f} | {format_number(r.qps)} |"
                )
            
            lines.append("")
    
    return "\n".join(lines)


def generate_analysis(results: List[BenchmarkResult]) -> str:
    """Generate analysis and recommendations."""
    
    # Find winners in different categories
    best_recall = max(results, key=lambda r: r.recall_at_10)
    best_qps = max(results, key=lambda r: r.qps)
    best_memory = min((r for r in results if r.recall_at_10 > 0.9), 
                      key=lambda r: r.peak_memory_mb, default=None)
    fastest_build = min(results, key=lambda r: r.build_time_sec)
    
    # Best balanced (high recall + good QPS)
    balanced = [r for r in results if r.recall_at_10 >= 0.95]
    if balanced:
        best_balanced = max(balanced, key=lambda r: r.qps)
    else:
        best_balanced = best_recall
    
    lines = [
        "## Analysis",
        "",
        "### Winners by Category",
        "",
        f"- **Best Recall@10**: {best_recall.library}/{best_recall.algorithm} ({best_recall.recall_at_10:.4f})",
        f"- **Best QPS**: {best_qps.library}/{best_qps.algorithm} ({format_number(best_qps.qps)} queries/sec)",
        f"- **Fastest Build**: {fastest_build.library}/{fastest_build.algorithm} ({fastest_build.build_time_sec:.2f}s)",
    ]
    
    if best_memory:
        lines.append(f"- **Lowest Memory** (≥0.9 recall): {best_memory.library}/{best_memory.algorithm} ({best_memory.peak_memory_mb:.0f}MB)")
    
    lines.extend([
        f"- **Best Balanced** (≥0.95 recall): {best_balanced.library}/{best_balanced.algorithm} "
        f"(R@10={best_balanced.recall_at_10:.4f}, {format_number(best_balanced.qps)} QPS)",
        "",
        "### Recommendations",
        "",
        "- **For maximum accuracy**: Use brute force (FAISS flat) if dataset fits in memory",
        "- **For balanced performance**: HNSW variants (hnswlib or FAISS HNSW) with ef_search=100-200",
        "- **For maximum throughput**: IVF with lower nprobe, or Annoy with more trees",
        "- **For memory-constrained environments**: Consider Annoy or quantized FAISS indices",
    ])
    
    return "\n".join(lines)


def generate_report(results: List[BenchmarkResult], dataset_name: str) -> str:
    """Generate complete markdown report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    index_size = results[0].index_size if results else 0
    
    sections = [
        f"# ANN Benchmark Report",
        "",
        f"**Dataset**: {dataset_name}",
        f"**Index Size**: {format_number(index_size)} vectors",
        f"**Generated**: {timestamp}",
        "",
        generate_summary_table(results),
        "",
        generate_recall_qps_table(results),
        "",
        generate_build_comparison(results),
        "",
        generate_analysis(results),
        "",
        generate_detailed_results(results),
        "",
        "---",
        "",
        "*Report generated by Foxstash benchmark suite*",
    ]
    
    return "\n".join(sections)


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """Find the most recent results file."""
    json_files = list(results_dir.glob("results_*.json"))
    if not json_files:
        return None
    return max(json_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--input", type=str, help="Input JSON results file")
    parser.add_argument("--results-dir", type=str, default="../results",
                       help="Directory containing results")
    parser.add_argument("--output", type=str, help="Output markdown file")
    parser.add_argument("--dataset", type=str, help="Dataset name (auto-detected from file)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).resolve()
    
    # Find input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_results(results_dir)
        if not input_path:
            print(f"No results found in {results_dir}")
            print("Run benchmarks first: python run_benchmarks.py")
            return
        print(f"Using latest results: {input_path}")
    
    # Load results
    results = load_results(input_path)
    print(f"Loaded {len(results)} benchmark results")
    
    # Determine dataset name
    if args.dataset:
        dataset_name = args.dataset
    elif results:
        dataset_name = results[0].dataset
    else:
        dataset_name = "unknown"
    
    # Generate report
    report = generate_report(results, dataset_name)
    
    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / f"report_{dataset_name}.md"
    
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    
    # Also print to stdout
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main()
