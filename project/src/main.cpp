#include "solver.h"
#include "benchmark.h"
#include <iostream>
#include <cstring>
#include <vector>

static void usage(){std::cout<<"heat_solver --dimension 2|3 [--nx N --ny N --nz N] [--iterations N] [--threads N] [--algorithm NAME] [--schedule static|dynamic] [--block-size N] [--csv PATH]\nAlgorithms: jacobi-seq, openmp-jacobi, openmp-gauss-seidel, openmp-blocked, pthreads-jacobi\n";}

int main(int argc,char**argv){heat::Config c;if(argc==1){usage();return 0;}for(int i=1;i<argc;++i){std::string a=argv[i];auto need=[&](int& i){if(i+1>=argc){throw std::invalid_argument("missing value");}return std::string(argv[++i]);};try{if(a=="--dimension")c.dimension=std::stoi(need(i));else if(a=="--nx")c.nx=std::stoul(need(i));else if(a=="--ny")c.ny=std::stoul(need(i));else if(a=="--nz")c.nz=std::stoul(need(i));else if(a=="--iterations")c.iterations=std::stoul(need(i));else if(a=="--threads")c.threads=std::stoi(need(i));else if(a=="--algorithm")c.algorithm=need(i);else if(a=="--schedule")c.schedule=need(i);else if(a=="--block-size")c.block_size=std::stoi(need(i));else if(a=="--csv")c.csv_path=need(i);else if(a=="--help") {usage();return 0;}}catch(const std::exception&e){std::cerr<<"Argument error: "<<e.what()<<"\n";return 2;}}
    if(c.dimension==2)c.nz=1;if(c.dimension==3&&c.nz==1)c.nz=c.nx;
    if(c.nx<3||c.ny<3||(c.dimension==3&&c.nz<3)){std::cerr<<"Grid dimensions must be at least 3 in every axis.\n";return 2;}if(c.threads<1||c.iterations<1){std::cerr<<"Threads and iterations must be positive.\n";return 2;}
    heat::Result r=heat::run_solver(c);if(r.seconds==0.0){return 1;}heat::print_result(r);heat::append_csv(c.csv_path,{r});return 0;
}
