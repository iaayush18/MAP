#include "solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

namespace heat {
Result jacobi_2d_openmp(const Config& cfg) {
    if (!check_memory(cfg.nx*cfg.ny*sizeof(double)*2)) return Result{};
    Grid1D a(cfg.nx*cfg.ny), b(cfg.nx*cfg.ny); initialize_2d(a,cfg.nx,cfg.ny);initialize_2d(b,cfg.nx,cfg.ny);
    omp_set_num_threads(cfg.threads); std::size_t converged=cfg.iterations; bool conv=false;
    auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
        if(cfg.schedule=="dynamic"){
#pragma omp parallel for schedule(dynamic)
            for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;b[p]=.25*(a[p-cfg.ny]+a[p+cfg.ny]+a[p-1]+a[p+1]);}
        } else {
#pragma omp parallel for schedule(static)
            for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;b[p]=.25*(a[p-cfg.ny]+a[p+cfg.ny]+a[p-1]+a[p+1]);}
        }
        double residual=0.0;
#pragma omp parallel for reduction(max:residual) schedule(static)
        for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;residual=std::max(residual,std::abs(a[p]-b[p]));}
        std::swap(a,b); if(residual<=cfg.tolerance){conv=true;converged=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now(); Result r; r.algorithm="openmp-jacobi";r.dimension=2;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=1;r.iterations=cfg.iterations;r.converged_iterations=converged;r.threads=cfg.threads;r.block_size=0;r.schedule=cfg.schedule;r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(converged*(cfg.nx-2)*(cfg.ny-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}

Result jacobi_3d_openmp(const Config& cfg) {
    if(!check_memory(cfg.nx*cfg.ny*cfg.nz*sizeof(double)*2))return Result{};
    Grid1D a(cfg.nx*cfg.ny*cfg.nz),b(cfg.nx*cfg.ny*cfg.nz);initialize_3d(a,cfg.nx,cfg.ny,cfg.nz);initialize_3d(b,cfg.nx,cfg.ny,cfg.nz);omp_set_num_threads(cfg.threads);
    std::size_t converged=cfg.iterations;bool conv=false;auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
#pragma omp parallel for schedule(static)
        for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j)for(std::size_t k=1;k+1<cfg.nz;++k){std::size_t p=(i*cfg.ny+j)*cfg.nz+k;b[p]=(a[p-cfg.ny*cfg.nz]+a[p+cfg.ny*cfg.nz]+a[p-cfg.nz]+a[p+cfg.nz]+a[p-1]+a[p+1])/6.0;}
        double residual=0.0;
#pragma omp parallel for reduction(max:residual)
        for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j)for(std::size_t k=1;k+1<cfg.nz;++k){std::size_t p=(i*cfg.ny+j)*cfg.nz+k;residual=std::max(residual,std::abs(a[p]-b[p]));}
        std::swap(a,b);if(residual<=cfg.tolerance){conv=true;converged=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now();Result r;r.algorithm="openmp-jacobi";r.dimension=3;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=cfg.nz;r.iterations=cfg.iterations;r.converged_iterations=converged;r.threads=cfg.threads;r.schedule=cfg.schedule;r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(converged*(cfg.nx-2)*(cfg.ny-2)*(cfg.nz-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}
}
