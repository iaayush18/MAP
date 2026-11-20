cmake_minimum_required(VERSION 3.16)
project(HeatStencil LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP QUIET)

add_library(heat_solvers
    src/initialization.cpp
    src/jacobi_2d.cpp
    src/jacobi_3d.cpp
    src/gauss_seidel.cpp
    src/openmp_solver.cpp
    src/pthread_solver.cpp
    src/blocking.cpp
    src/benchmark.cpp
)
target_include_directories(heat_solvers PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(heat_solvers PUBLIC OpenMP::OpenMP_CXX)
endif()
find_package(Threads REQUIRED)
target_link_libraries(heat_solvers PUBLIC Threads::Threads)

add_executable(heat_solver src/main.cpp)
target_link_libraries(heat_solver PRIVATE heat_solvers)

enable_testing()
add_executable(test_heat src/test_heat.cpp)
target_link_libraries(test_heat PRIVATE heat_solvers)
add_test(NAME heat_tests COMMAND test_heat)
