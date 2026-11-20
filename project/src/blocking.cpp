#include "solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

namespace heat {
Result jacobi_2d_openmp_blocked(const Config& cfg) {
    if(!check_memory(cfg.nx*cfg.ny*sizeof(double)*2))return Result{};
    Grid1D a(cfg.nx*cfg.ny),b(cfg.nx*cfg.ny);initialize_2d(a,cfg.nx,cfg.ny);initialize_2d(b,cfg.nx,cfg.ny);omp_set_num_threads(cfg.threads);
    int bs=std::max(1,cfg.block_size);std::size_t converged=cfg.iterations;bool conv=false;auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for(std::size_t ib=1;ib<cfg.nx-1;ib+=bs)
                for(std::size_t jb=1;jb+1<cfg.ny;jb+=bs)
                    for(std::size_t i=ib;i<std::min(ib+(std::size_t)bs,cfg.nx-1);++i)
                        for(std::size_t j=jb;j<std::min(jb+(std::size_t)bs,cfg.ny-1);++j){std::size_t p=i*cfg.ny+j;b[p]=.25*(a[p-cfg.ny]+a[p+cfg.ny]+a[p-1]+a[p+1]);}
        }
        double residual=0.0;
#pragma omp parallel for reduction(max:residual)
        for(long i=1;i<(long)cfg.nx-1;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;residual=std::max(residual,std::abs(a[p]-b[p]));}
        std::swap(a,b);if(residual<=cfg.tolerance){conv=true;converged=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now();Result r;r.algorithm="openmp-blocked";r.dimension=2;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=1;r.iterations=cfg.iterations;r.converged_iterations=converged;r.threads=cfg.threads;r.block_size=bs;r.schedule=cfg.schedule;r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(converged*(cfg.nx-2)*(cfg.ny-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}
}
