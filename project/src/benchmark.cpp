#include "benchmark.h"
#include "solver.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace heat {
void print_result(const Result& r){
    std::cout<<"\nAlgorithm: "<<r.algorithm<<"  "<<r.dimension<<"D  grid="<<r.nx<<"x"<<r.ny<<(r.dimension==3?("x"+std::to_string(r.nz)):"")<<"\n";
    std::cout<<"threads="<<r.threads<<" schedule="<<r.schedule<<" block="<<r.block_size<<"\n";
    std::cout<<std::fixed<<std::setprecision(6)<<"Time: "<<r.seconds<<" s  Mcell/s: "<<r.mcell_s<<"\n";
    std::cout<<"Speedup: "<<r.speedup<<"  Efficiency: "<<r.efficiency<<"\n";
    std::cout<<"Convergence: "<<(r.converged?"YES":"NO")<<" at "<<r.converged_iterations<<" iterations  Correct: "<<(r.correct?"YES":"NO")<<"\n";
}

bool append_csv(const std::string& path,const std::vector<Result>& rs){
    std::filesystem::path p(path);if(p.has_parent_path())std::filesystem::create_directories(p.parent_path());
    std::ifstream test(path);bool empty=!test.good()||test.peek()==std::ifstream::traits_type::eof();test.close();
    std::ofstream f(path,std::ios::app);if(!f)return false;if(empty)f<<"algorithm,dimension,nx,ny,nz,iterations,converged_iterations,threads,block_size,schedule,seconds,mcell_s,speedup,efficiency,max_difference,converged,correct\n";
    for(const auto&r:rs)f<<r.algorithm<<","<<r.dimension<<","<<r.nx<<","<<r.ny<<","<<r.nz<<","<<r.iterations<<","<<r.converged_iterations<<","<<r.threads<<","<<r.block_size<<","<<r.schedule<<","<<r.seconds<<","<<r.mcell_s<<","<<r.speedup<<","<<r.efficiency<<","<<r.max_difference<<","<<(r.converged?"true":"false")<<","<<(r.correct?"true":"false")<<"\n";
    return true;
}
Result run_solver(const Config& cfg){
    Result r;
    if(cfg.dimension==3){
        if(cfg.algorithm=="jacobi-seq") r=jacobi_3d_seq(cfg);
        else if(cfg.algorithm=="openmp-gauss-seidel") r=gauss_seidel_3d_openmp(cfg);
        else r=jacobi_3d_openmp(cfg);
    }else{
        if(cfg.algorithm=="jacobi-seq") r=jacobi_2d_seq(cfg);
        else if(cfg.algorithm=="openmp-gauss-seidel") r=gauss_seidel_2d_openmp(cfg);
        else if(cfg.algorithm=="openmp-blocked") r=jacobi_2d_openmp_blocked(cfg);
        else if(cfg.algorithm=="pthreads-jacobi") r=pthread_2d(cfg);
        else r=jacobi_2d_openmp(cfg);
    }
    if(cfg.algorithm!="jacobi-seq"){
        Config baseline=cfg; baseline.algorithm="jacobi-seq"; baseline.threads=1;
        Result seq=cfg.dimension==3 ? jacobi_3d_seq(baseline) : jacobi_2d_seq(baseline);
        if(seq.seconds>0.0 && r.seconds>0.0){r.speedup=seq.seconds/r.seconds;r.efficiency=r.speedup/std::max(1,cfg.threads);}
    }
    return r;
}
}
