#pragma once
#include <cstddef>
#include <string>

namespace heat {
constexpr double kTolerance = 1e-6;
constexpr std::size_t kMemorySafetyFraction = 2; // require at least 2 arrays of grid memory

struct Config {
    int dimension = 2;
    std::size_t nx = 512, ny = 512, nz = 1;
    std::size_t iterations = 1000;
    double tolerance = kTolerance;
    int threads = 1;
    int block_size = 64;
    std::string schedule = "static";
    std::string algorithm = "jacobi-seq";
    std::string csv_path = "results/benchmark_results.csv";
};

struct Result {
    std::string algorithm;
    int dimension = 0;
    std::size_t nx = 0, ny = 0, nz = 0;
    std::size_t iterations = 0, converged_iterations = 0;
    int threads = 1;
    int block_size = 0;
    std::string schedule;
    double seconds = 0.0;
    double mcell_s = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double max_difference = 0.0;
    bool converged = false;
    bool correct = false;
};
}
