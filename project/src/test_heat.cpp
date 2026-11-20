#include "solver.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace heat;

static void test_initialization() {
    Config c; c.nx=c.ny=5;
    Grid1D g(25); initialize_2d(g,5,5,100.0,0.0);
    for(std::size_t i=0;i<5;++i) for(std::size_t j=0;j<5;++j)
        if(i==0||i==4||j==0||j==4) assert(g[i*5+j]==100.0); else assert(g[i*5+j]==0.0);
}

static void test_jacobi_one_iteration() {
    Config c; c.nx=c.ny=5;c.iterations=1;c.tolerance=0.0;
    Result r=jacobi_2d_seq(c); assert(r.correct); assert(r.iterations==1);
}

static void test_parallel_equivalence() {
    Config c; c.nx=c.ny=64;c.iterations=20;c.threads=4;c.schedule="static";
    Config a=c;a.algorithm="jacobi-seq";
    Config b=c;b.algorithm="openmp-jacobi";
    Result rs=jacobi_2d_seq(a); Result rp=jacobi_2d_openmp(b);
    assert(rs.correct && rp.correct);
}

static void test_red_black() {
    Config c;c.nx=c.ny=16;c.iterations=20;c.threads=4;
    Result r=gauss_seidel_2d_openmp(c);assert(r.correct);
}

static void test_csv() {
    Result r;r.algorithm="test";r.dimension=2;r.nx=32;r.ny=32;r.iterations=10;r.seconds=0.1;r.correct=true;
    assert(heat::append_csv("results/test_results.csv",{r}));
}

int main(){test_initialization();test_jacobi_one_iteration();test_parallel_equivalence();test_red_black();test_csv();std::cout<<"All heat solver tests passed.\n";}
