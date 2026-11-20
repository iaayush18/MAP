#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j
mkdir -p "$ROOT/results"
for t in 1 2 4 8; do
  "$BUILD/heat_solver" --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads "$t" --algorithm openmp-jacobi --schedule static --csv "$ROOT/results/benchmark_results.csv"
  "$BUILD/heat_solver" --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads "$t" --algorithm openmp-gauss-seidel --csv "$ROOT/results/benchmark_results.csv"
done
