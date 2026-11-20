#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root=Path(__file__).resolve().parent.parent
csv=root/'results'/'benchmark_results.csv'
out=root/'results'/'plots';out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(csv)

if df.empty:
    print('No benchmark data available.')
    raise SystemExit(0)

# Sequential vs OpenMP
sub=df[df['algorithm'].isin(['jacobi-seq','openmp-jacobi'])]
if not sub.empty:
    sub.pivot_table(index='threads',columns='algorithm',values='mcell_s',aggfunc='mean').plot(marker='o')
    plt.title('Sequential vs OpenMP Jacobi');plt.ylabel('Mcell/s');plt.xlabel('Threads');plt.grid(True);plt.tight_layout();plt.savefig(out/'sequential_openmp.png');plt.close()

# Static vs dynamic
a=df[df['algorithm']=='openmp-jacobi']
if not a.empty:
    a.groupby('schedule')['mcell_s'].mean().plot(kind='bar',rot=0);plt.title('Static vs Dynamic Scheduling');plt.ylabel('Mcell/s');plt.tight_layout();plt.savefig(out/'static_dynamic.png');plt.close()

# Threads and speedup
if not a.empty:
    g=a.groupby('threads').agg(mcell_s=('mcell_s','mean'),speedup=('speedup','mean'))
    g['mcell_s'].plot(marker='o');plt.title('Performance vs Threads');plt.ylabel('Mcell/s');plt.grid(True);plt.tight_layout();plt.savefig(out/'threads.png');plt.close()
    g['speedup'].plot(marker='o');plt.title('Speedup vs Threads');plt.ylabel('Speedup');plt.xlabel('Threads');plt.grid(True);plt.tight_layout();plt.savefig(out/'speedup.png');plt.close()

# Block size
b=df[df['algorithm']=='openmp-blocked']
if not b.empty:
    b.groupby('block_size')['mcell_s'].mean().plot(marker='o');plt.title('Blocked Jacobi vs Block Size');plt.ylabel('Mcell/s');plt.xlabel('Block size');plt.grid(True);plt.tight_layout();plt.savefig(out/'block_size.png');plt.close()

# Jacobi vs Gauss-Seidel convergence
conv=df[df['algorithm'].isin(['jacobi-seq','openmp-gauss-seidel'])]
if not conv.empty:
    conv.groupby('algorithm')['converged_iterations'].mean().plot(kind='bar',rot=20);plt.title('Average Convergence Iterations');plt.ylabel('Iterations');plt.tight_layout();plt.savefig(out/'convergence.png');plt.close()

# OpenMP vs Pthreads
pt=df[df['algorithm'].isin(['openmp-jacobi','pthreads-jacobi'])]
if not pt.empty:
    pt.pivot_table(index='threads',columns='algorithm',values='mcell_s',aggfunc='mean').plot(marker='o');plt.title('OpenMP vs Pthreads');plt.ylabel('Mcell/s');plt.grid(True);plt.tight_layout();plt.savefig(out/'openmp_pthreads.png');plt.close()

print(f'Plots written to {out}')
