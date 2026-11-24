#!/bin/bash
# Static terms of the baseline representations (baselines/static_terms.py) on a Perlmutter CPU
# node: Perlmutter layout /pscratch/sd/m/mwest/aeromeld/baselines/static/{repo,venv}, repo/ a copy
# of baselines/, scripts/{paperfigs,extract,vendor}, training/paper_errors.py and the small
# inputs; repo/data/transport/all1_12h -> ../../../../transport/runs/all1_12h (the replay).
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --job-name aeromeld_base_static
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/static_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd $S/repo/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINES_PROCS=120
md5sum ../data/transport/all1_12h/surface.npz
timeout 28m $S/venv/bin/python ../baselines/static_terms.py
echo "exit $?"
