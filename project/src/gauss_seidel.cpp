#include "solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

namespace heat {
static void rb_sweep_2d(Grid1D& g,std::size_t nx,std::size_t ny,int color,int threads){
#pragma omp parallel for schedule(static)
    for(long i=1;i<(long)nx-1;++i)
        for(std::size_t j=1;j+1<ny;++j)
            if(static_cast<int>((i+j)&1)==color){std::size_t p=i*ny+j;g[p]=.25*(g[p-ny]+g[p+ny]+g[p-1]+g[p+1]);}
}

Result gauss_seidel_2d_openmp(const Config& cfg){
    if(!check_memory(cfg.nx*cfg.ny*sizeof(double)))return Result{};
    Grid1D g(cfg.nx*cfg.ny);initialize_2d(g,cfg.nx,cfg.ny);omp_set_num_threads(cfg.threads);
    std::size_t conv_it=cfg.iterations;bool conv=false;auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
        rb_sweep_2d(g,cfg.nx,cfg.ny,0,cfg.threads);rb_sweep_2d(g,cfg.nx,cfg.ny,1,cfg.threads);
        double residual=0.0;
#pragma omp parallel for reduction(max:residual)
        for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;double ref=.25*(g[p-cfg.ny]+g[p+cfg.ny]+g[p-1]+g[p+1]);residual=std::max(residual,std::abs(g[p]-ref));}
        if(residual<=cfg.tolerance){conv=true;conv_it=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now();Result r;r.algorithm="openmp-gauss-seidel";r.dimension=2;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=1;r.iterations=cfg.iterations;r.converged_iterations=conv_it;r.threads=cfg.threads;r.schedule=cfg.schedule;r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(conv_it*(cfg.nx-2)*(cfg.ny-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}

Result gauss_seidel_3d_openmp(const Config& cfg){
    if(!check_memory(cfg.nx*cfg.ny*cfg.nz*sizeof(double)))return Result{};
    Grid1D g(cfg.nx*cfg.ny*cfg.nz);initialize_3d(g,cfg.nx,cfg.ny,cfg.nz);omp_set_num_threads(cfg.threads);
    std::size_t conv_it=cfg.iterations;bool conv=false;auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
        for(int color=0;color<2;++color){
#pragma omp parallel for schedule(static)
            for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j)for(std::size_t k=1;k+1<cfg.nz;++k)if((i+j+k+2)%2==color){std::size_t p=(i*cfg.ny+j)*cfg.nz+k;g[p]=(g[p-cfg.ny*cfg.nz]+g[p+cfg.ny*cfg.nz]+g[p-cfg.nz]+g[p+cfg.nz]+g[p-1]+g[p+1])/6.0;}
        }
        double residual = 0.0;
#pragma omp parallel for reduction(max:residual)
        for(long i=1;i<(long)cfg.nx-1;++i)
            for(std::size_t j=1;j+1<cfg.ny;++j)
                for(std::size_t k=1;k+1<cfg.nz;++k){
                    std::size_t p=(i*cfg.ny+j)*cfg.nz+k;
                    double ref=(g[p-cfg.ny*cfg.nz]+g[p+cfg.ny*cfg.nz]+g[p-cfg.nz]+g[p+cfg.nz]+g[p-1]+g[p+1])/6.0;
                    residual=std::max(residual,std::abs(g[p]-ref));
                }
        if(residual<=cfg.tolerance){conv=true;conv_it=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now();Result r;r.algorithm="openmp-gauss-seidel";r.dimension=3;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=cfg.nz;r.iterations=cfg.iterations;r.converged_iterations=conv_it;r.threads=cfg.threads;r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(conv_it*(cfg.nx-2)*(cfg.ny-2)*(cfg.nz-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}
}
