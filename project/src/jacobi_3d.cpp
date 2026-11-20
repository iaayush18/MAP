#include "solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace heat {
Result jacobi_3d_seq(const Config& cfg) {
    const std::size_t n = cfg.nx * cfg.ny * cfg.nz;
    if (!check_memory(n * sizeof(double) * 2)) return Result{};
    Grid1D a(n), b(n);
    initialize_3d(a, cfg.nx, cfg.ny, cfg.nz);
    initialize_3d(b, cfg.nx, cfg.ny, cfg.nz);

    std::size_t converged = cfg.iterations;
    bool conv = false;
    auto begin = std::chrono::steady_clock::now();
    for (std::size_t it = 0; it < cfg.iterations; ++it) {
        for (std::size_t i = 1; i + 1 < cfg.nx; ++i)
            for (std::size_t j = 1; j + 1 < cfg.ny; ++j)
                for (std::size_t k = 1; k + 1 < cfg.nz; ++k) {
                    const std::size_t p = (i * cfg.ny + j) * cfg.nz + k;
                    b[p] = (a[p-cfg.ny*cfg.nz] + a[p+cfg.ny*cfg.nz]
                          + a[p-cfg.nz] + a[p+cfg.nz] + a[p-1] + a[p+1]) / 6.0;
                }

        double residual = 0.0;
        for (std::size_t i = 1; i + 1 < cfg.nx; ++i)
            for (std::size_t j = 1; j + 1 < cfg.ny; ++j)
                for (std::size_t k = 1; k + 1 < cfg.nz; ++k) {
                    const std::size_t p = (i * cfg.ny + j) * cfg.nz + k;
                    residual = std::max(residual, std::abs(a[p] - b[p]));
                }
        std::swap(a, b);
        if (residual <= cfg.tolerance) {
            conv = true;
            converged = it + 1;
            break;
        }
    }
    auto end = std::chrono::steady_clock::now();

    Result r;
    r.algorithm = "jacobi-seq";
    r.dimension = 3;
    r.nx = cfg.nx;
    r.ny = cfg.ny;
    r.nz = cfg.nz;
    r.iterations = cfg.iterations;
    r.converged_iterations = converged;
    r.threads = 1;
    r.schedule = "none";
    r.seconds = std::chrono::duration<double>(end-begin).count();
    r.mcell_s = static_cast<double>(converged * (cfg.nx-2) * (cfg.ny-2) * (cfg.nz-2)) / r.seconds / 1e6;
    r.speedup = 1.0;
    r.efficiency = 1.0;
    r.converged = conv;
    r.correct = true;
    return r;
}
}
