# Academic Methodology Report

## Experimental problem

The tests solve the discrete steady-state limit of a 2D or 3D heat equation by repeated stencil updates on fixed-boundary grids. The initial boundary is set to 100 and interior points to 0.

## Implementations

- **Sequential Jacobi:** two complete grids and a single thread. The old grid is never modified during an iteration.
- **OpenMP Jacobi:** row-parallel updates, static/dynamic scheduling, and an OpenMP max reduction for residual calculation.
- **Red–Black Gauss–Seidel:** red checkerboard cells update, followed by black cells. Only cells of one color are active during a sweep; therefore same-color neighbors cannot race.
- **Spatial blocking:** square row-column tiles improve access locality for the old-to-new Jacobi stencil.
- **Pthreads:** row ownership partitions interior rows among worker threads. A barrier is implemented implicitly with `pthread_join` at each iteration.

## Validation

Tests check boundary preservation, one-step execution, 2D/3D Red–Black operation, thread-count execution, and CSV output. Numerical equivalence is determined by maximum grid difference; benchmark runs report a difference field where a reference comparison is requested. A tolerance below `1e-6` is used as the correctness requirement.

## Benchmark protocol

Use one process, repeated fixed iterations, a warm-up run, and report median or minimum over several repetitions. Rebooting or pinning can be used to reduce noise. Measure CPU time only for the stencil phase, not grid allocation or initialization.

## Interpretation

Jacobi is memory-bandwidth limited. OpenMP scaling should be judged alongside NUMA placement and cache size. A faster implementation is supported only by measured Mcell/s and speedup. Red–Black GS has a larger convergence advantage for near-steady residual convergence but writes only one grid and its two-color synchronization differs from Jacobi.

## Roofline model

Each Jacobi cell performs approximately arithmetic operations on four or six neighbors and requires reading 5-point/7-point data plus writing one double. Plot the measured Mcell/s against the machine-specific arithmetic-intensity point. Ridge point, memory bandwidth, and vectorization vary by hardware.
