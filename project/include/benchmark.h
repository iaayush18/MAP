#pragma once
#include "config.h"
#include <string>
#include <vector>

namespace heat {
bool append_csv(const std::string& path, const std::vector<Result>& results);
void print_result(const Result& r);
Result run_solver(const Config& cfg);
}
