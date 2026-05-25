#!/usr/bin/env bash
# Run this on your multi-core machine to get thread-scaling results
# Requirements: gcc with OpenMP, Python 3 with matplotlib + numpy
#   pip install matplotlib numpy

set -e
cd "$(dirname "$0")"
gcc -O2 -fopenmp -march=native -std=c99 -o dgemm_sweep src/dgemm_sweep.c -lm
echo "Build OK"

# CSV with header
OMP_NUM_THREADS=1 ./dgemm_sweep > results/sweep_all.csv

# Append remaining thread counts (skip header)
for T in 2 4 8; do
    echo "--- Running T=$T ---"
    OMP_NUM_THREADS=$T ./dgemm_sweep | tail -n +2 >> results/sweep_all.csv
done

echo "All done. Running plots..."
python3 scripts/plot_results.py
echo "Figures saved to results/"
