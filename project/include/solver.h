#pragma once
#include "config.h"
#include <cstddef>
#include <vector>

namespace heat {
using Grid1D = std::vector<double>;

bool check_memory(std::size_t bytes);
void initialize_2d(Grid1D& grid, std::size_t nx, std::size_t ny,
                  double hot = 100.0, double cold = 0.0);
void initialize_3d(Grid1D& grid, std::size_t nx, std::size_t ny, std::size_t nz,
                  double hot = 100.0, double cold = 0.0);

Result jacobi_2d_seq(const Config& cfg);
Result jacobi_2d_openmp(const Config& cfg);
Result jacobi_2d_openmp_blocked(const Config& cfg);
Result gauss_seidel_2d_openmp(const Config& cfg);
Result pthread_2d(const Config& cfg);

Result jacobi_3d_seq(const Config& cfg);
Result jacobi_3d_openmp(const Config& cfg);
Result gauss_seidel_3d_openmp(const Config& cfg);

double max_difference_2d(const Grid1D& a, const Grid1D& b,
                         std::size_t nx, std::size_t ny);
}
