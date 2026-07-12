#!/usr/bin/env bash
# The scoreboard: foxstash vs hnswlib vs faiss, at every scale, on identical data.
#
# Runs STRICTLY SERIALLY, and waits for the machine to go idle before each run.
# Two benchmarks sharing 16 cores measure each other's load, not their own speed:
# running the Python harness while a 1M Rust build was in flight halved hnswlib's
# apparent QPS (5,478 vs 10,850 at ef=100). Concurrency is not a shortcut here,
# it is a way to generate confident nonsense.
#
# Usage: benchmarks/run-scoreboard.sh [sift10k sift100k sift1m ...]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then DATASETS=("$@"); else DATASETS=(sift10k sift100k sift1m); fi

OUT="benchmarks/scoreboard-$(date +%Y%m%d-%H%M%S).txt"

# Wait until the 1-minute load average drops below ~1 core of activity.
idle() {
  echo "    (waiting for idle...)"
  for _ in $(seq 1 120); do
    load=$(awk '{print int($1)}' /proc/loadavg)
    [ "$load" -lt 1 ] && { sleep 3; return 0; }
    sleep 5
  done
  echo "    WARNING: machine never went idle; numbers below may be contaminated" | tee -a "$OUT"
}

echo "foxstash scoreboard — $(date -u +%FT%TZ)"          | tee    "$OUT"
echo "host: $(uname -sr), $(nproc) threads"              | tee -a "$OUT"
echo "NOTE: recall@10 is NOT comparable across datasets" | tee -a "$OUT"
echo "      (see the per-dataset difficulty line)"       | tee -a "$OUT"

cargo build --release -p foxstash-benches -q

for ds in "${DATASETS[@]}"; do
  echo ""                                          | tee -a "$OUT"
  echo "################ $ds ################"     | tee -a "$OUT"

  idle
  echo ""                                          | tee -a "$OUT"
  echo "=== foxstash ==="                          | tee -a "$OUT"
  cargo run --release -q -p foxstash-benches --example pareto -- "$ds" 2>&1 | tee -a "$OUT"

  idle
  echo ""                                          | tee -a "$OUT"
  ( cd benchmarks/python && ./venv/bin/python competitors.py "$ds" ) 2>&1 | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "written to $OUT" | tee -a "$OUT"
