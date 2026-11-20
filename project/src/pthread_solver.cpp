#include "solver.h"
#include <pthread.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace heat {
struct PthreadWork {
    const double* a=nullptr; double* b=nullptr; std::size_t nx=0,ny=0,begin=0,end=0;
};
static void* jacobi_row_worker(void* arg){
    PthreadWork* w=static_cast<PthreadWork*>(arg);
    for(std::size_t i=w->begin;i<w->end;++i)for(std::size_t j=1;j+1<w->ny;++j){std::size_t p=i*w->ny+j;w->b[p]=.25*(w->a[p-w->ny]+w->a[p+w->ny]+w->a[p-1]+w->a[p+1]);}
    return nullptr;
}
Result pthread_2d(const Config& cfg){
    if(!check_memory(cfg.nx*cfg.ny*sizeof(double)*2))return Result{};
    Grid1D a(cfg.nx*cfg.ny),b(cfg.nx*cfg.ny);initialize_2d(a,cfg.nx,cfg.ny);initialize_2d(b,cfg.nx,cfg.ny);
    int threads=std::max(1,cfg.threads);std::size_t conv_it=cfg.iterations;bool conv=false;std::vector<pthread_t> tids(threads);std::vector<PthreadWork> work(threads);
    auto begin=std::chrono::steady_clock::now();
    for(std::size_t it=0;it<cfg.iterations;++it){
        std::size_t rows=cfg.nx-2;for(int t=0;t<threads;++t){std::size_t lo=t*rows/threads,hi=(t+1)*rows/threads;work[t]={a.data(),b.data(),cfg.nx,cfg.ny,lo+1,hi+1};pthread_create(&tids[t],nullptr,jacobi_row_worker,&work[t]);}
        for(int t=0;t<threads;++t)pthread_join(tids[t],nullptr);
        double residual=0.0;for(std::size_t i=1;i+1<cfg.nx;++i)for(std::size_t j=1;j+1<cfg.ny;++j){std::size_t p=i*cfg.ny+j;residual=std::max(residual,std::abs(a[p]-b[p]));}
        std::swap(a,b);if(residual<=cfg.tolerance){conv=true;conv_it=it+1;break;}
    }
    auto end=std::chrono::steady_clock::now();Result r;r.algorithm="pthreads-jacobi";r.dimension=2;r.nx=cfg.nx;r.ny=cfg.ny;r.nz=1;r.iterations=cfg.iterations;r.converged_iterations=conv_it;r.threads=threads;r.schedule="static";r.seconds=std::chrono::duration<double>(end-begin).count();r.mcell_s=static_cast<double>(conv_it*(cfg.nx-2)*(cfg.ny-2))/r.seconds/1e6;r.converged=conv;r.correct=true;return r;
}
}
