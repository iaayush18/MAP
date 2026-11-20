#include "solver.h"
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <new>

namespace heat {
bool check_memory(std::size_t bytes) {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return true;
    const std::size_t available = static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
    if (bytes > available / kMemorySafetyFraction) {
        std::cerr << "Memory requirement " << (bytes / 1000000.0)
                  << " MB exceeds safe available-memory estimate "
                  << (available / kMemorySafetyFraction / 1000000.0) << " MB.\n";
        return false;
    }
    return true;
}

void initialize_2d(Grid1D& g, std::size_t nx, std::size_t ny, double hot, double cold) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(nx); ++i)
        for (std::size_t j = 0; j < ny; ++j)
            g[i * ny + j] = ((i == 0 || i + 1 == nx || j == 0 || j + 1 == ny) ? hot : cold);
}

void initialize_3d(Grid1D& g, std::size_t nx, std::size_t ny, std::size_t nz, double hot, double cold) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(nx); ++i)
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t k = 0; k < nz; ++k)
                g[(i * ny + j) * nz + k] = (i == 0 || i + 1 == nx || j == 0 || j + 1 == ny || k == 0 || k + 1 == nz) ? hot : cold;
}

double max_difference_2d(const Grid1D& a, const Grid1D& b, std::size_t nx, std::size_t ny) {
    double d = 0.0;
#pragma omp parallel for reduction(max:d)
    for (long i = 0; i < static_cast<long>(nx); ++i)
        for (std::size_t j = 0; j < ny; ++j)
            d = std::max(d, std::abs(a[i * ny + j] - b[i * ny + j]));
    return d;
}
}
