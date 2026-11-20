# 2D/3D Heat Equation — Shared-Memory HPC Stencil Project

This project implements and benchmarks double-precision Jacobi and Red–Black Gauss–Seidel heat-equation solvers using sequential C++, OpenMP, spatial blocking, and POSIX Pthreads.

## Mathematical formulation

For a 2D interior cell:

$$u^{k+1}_{i,j}=\frac14(u^k_{i-1,j}+u^k_{i+1,j}+u^k_{i,j-1}+u^k_{i,j+1})$$

For 3D:

$$u^{k+1}_{i,j,k}=\frac16(u^k_{i-1,j,k}+u^k_{i+1,j,k}+u^k_{i,j-1,k}+u^k_{i,j+1,k}+u^k_{i,j,k-1}+u^k_{i,j,k+1})$$

Boundary values remain fixed. The project uses row-major contiguous grids and two grids for Jacobi.

## Build

```bash
cmake -S project -B project/build -DCMAKE_BUILD_TYPE=Release
cmake --build project/build -j
ctest --test-dir project/build --output-on-failure
```

OpenMP is optional during compilation but required for OpenMP/Pthreads comparison features.

## Run

```bash
project/build/heat_solver --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads 4 --algorithm openmp-jacobi --schedule static --block-size 64 --csv project/results/benchmark_results.csv
project/build/heat_solver --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads 4 --algorithm openmp-gauss-seidel
project/build/heat_solver --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads 4 --algorithm openmp-blocked --block-size 32
project/build/heat_solver --dimension 2 --nx 512 --ny 512 --iterations 1000 --threads 4 --algorithm pthreads-jacobi
project/build/heat_solver --dimension 3 --nx 128 --ny 128 --nz 128 --iterations 100 --threads 4 --algorithm openmp-jacobi
```

Use `--help` for the list of options. Grid dimensions should not exceed available RAM.

## Benchmarks

The CSV accumulates real benchmark runs with execution time, updates, Mcell/s, thread count, schedule, convergence iterations, and correctness fields.

```bash
bash project/scripts/run_benchmarks.sh
python3 project/scripts/plot_results.py
```

## Metrics

- Mcell/s = `(interior_updates / seconds) / 10^6`
- Speedup = `sequential_seconds / parallel_seconds`
- Efficiency = `speedup / threads`

Each parallel run in the CLI is compared with a separate measured sequential Jacobi run unless running the sequential algorithm itself.

## Memory guidance

2D 8192² and 16384² grids require approximately 1 GB and 4 GB respectively for two double grids. 3D 512³ and 1024³ require approximately 4 GB and 32 GB. The runtime performs a conservative memory estimate and rejects allocations when it is unsafe.

## Methodology and limitations

Red–Black Gauss–Seidel uses independent-color sweeps. Each color can be parallelized because no same-color neighbors are directly adjacent. Spatial blocking improves cache reuse but block-size performance is architecture-dependent. Temporal blocking is not implemented: the repeated-neighbor dependency across iterations is practical to tile only with halo machinery and has to preserve arithmetic and boundary semantics.

Thread pinning is platform-specific and intentionally left to system launchers (`taskset`, `numactl`, or scheduler affinity). Parallel initialization uses OpenMP first touch in 2D and 3D initialization loops.
