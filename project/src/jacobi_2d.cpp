#include "solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace heat {
Result jacobi_2d_seq(const Config& cfg) {
    const std::size_t n = cfg.nx * cfg.ny;
    if (!check_memory(n * sizeof(double) * 2)) return Result{};
    Grid1D a(n, 0.0), b(n, 0.0);
    initialize_2d(a, cfg.nx, cfg.ny); initialize_2d(b, cfg.nx, cfg.ny);
    std::size_t converged = cfg.iterations;
    bool conv = false;
    auto begin = std::chrono::steady_clock::now();
    for (std::size_t it = 0; it < cfg.iterations; ++it) {
        for (std::size_t i = 1; i + 1 < cfg.nx; ++i)
            for (std::size_t j = 1; j + 1 < cfg.ny; ++j) {
                const std::size_t p = i * cfg.ny + j;
                b[p] = 0.25 * (a[p-cfg.ny]+a[p+cfg.ny]+a[p-1]+a[p+1]);
            }
        double residual = 0.0;
        for (std::size_t i = 1; i + 1 < cfg.nx && residual <= cfg.tolerance; ++i)
            for (std::size_t j = 1; j + 1 < cfg.ny && residual <= cfg.tolerance; ++j) {
                std::size_t p=i*cfg.ny+j; residual=std::max(residual,std::abs(a[p]-b[p]));
            }
        std::swap(a,b);
        if (residual <= cfg.tolerance) { conv=true; converged=it+1; break; }
    }
    auto end = std::chrono::steady_clock::now();
    Result r; r.algorithm="jacobi-seq"; r.dimension=2; r.nx=cfg.nx;r.ny=cfg.ny;r.nz=1;
    r.iterations=cfg.iterations;r.converged_iterations=converged;r.threads=1;r.block_size=0;
    r.schedule="none";r.seconds=std::chrono::duration<double>(end-begin).count();
    r.mcell_s=(static_cast<double>((converged)*(cfg.nx-2)*(cfg.ny-2))/r.seconds)/1e6;
    r.speedup=1.0;r.efficiency=1.0;r.converged=conv;r.correct=true;
    return r;
}
}
